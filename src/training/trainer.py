import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np

from src.training.metrics import compute_per_snr_accuracy, compute_overall_accuracy


class Trainer:
    def __init__(self, model, trainDataset, valDataset, config, device=None):
        self.model = model
        self.trainDataset = trainDataset
        self.valDataset = valDataset
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.001),
            weight_decay=config.get("weight_decay", 1e-4)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 100)
        )

        batchSize = config.get("batch_size", 256)
        numWorkers = config.get("num_workers", 0)

        self.trainLoader = DataLoader(
            trainDataset,
            batch_size=batchSize,
            shuffle=True,
            num_workers=numWorkers,
            pin_memory=True
        )

        self.valLoader = DataLoader(
            valDataset,
            batch_size=batchSize,
            shuffle=False,
            num_workers=numWorkers,
            pin_memory=True
        )

        self.useMixedPrecision = config.get("mixed_precision", True) and self.device.type == "cuda"
        if self.useMixedPrecision:
            self.scaler = GradScaler("cuda")

        self.bestValAcc = 0.0
        self.patience = config.get("patience", 15)
        self.patienceCounter = 0
        self.checkpointDir = config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpointDir, exist_ok=True)

        self.wandbEnabled = config.get("wandb", False)

    def train_one_epoch(self, epoch):
        self.model.train()
        runningLoss = 0.0
        correctCount = 0
        totalCount = 0

        for batchIdx, (features, labels, snrs) in enumerate(self.trainLoader):
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.useMixedPrecision:
                with autocast("cuda"):
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            runningLoss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)
            correctCount += (preds == labels).sum().item()
            totalCount += features.size(0)

        epochLoss = runningLoss / totalCount
        epochAcc = correctCount / totalCount
        return epochLoss, epochAcc

    def validate(self):
        self.model.eval()
        runningLoss = 0.0
        correctCount = 0
        totalCount = 0

        allPreds = []
        allLabels = []
        allSnrs = []

        with torch.no_grad():
            for features, labels, snrs in self.valLoader:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.useMixedPrecision:
                    with autocast("cuda"):
                        outputs = self.model(features)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)

                runningLoss += loss.item() * features.size(0)
                preds = outputs.argmax(dim=1)
                correctCount += (preds == labels).sum().item()
                totalCount += features.size(0)

                allPreds.extend(preds.cpu().numpy())
                allLabels.extend(labels.cpu().numpy())
                allSnrs.extend(snrs)

        valLoss = runningLoss / totalCount
        valAcc = correctCount / totalCount

        snrAcc = compute_per_snr_accuracy(allPreds, allLabels, allSnrs)

        return valLoss, valAcc, snrAcc

    def train(self):
        epochs = self.config.get("epochs", 100)
        modelName = self.config.get("model_name", "model")

        print(f"training {modelName} for {epochs} epochs on {self.device}")
        print(f"mixed precision: {self.useMixedPrecision}")
        print(f"train samples: {len(self.trainDataset)}, val samples: {len(self.valDataset)}")
        print("-" * 60)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "snr_acc": [],
            "lr": []
        }

        for epoch in range(epochs):
            startTime = time.time()

            trainLoss, trainAcc = self.train_one_epoch(epoch)
            valLoss, valAcc, snrAcc = self.validate()

            self.scheduler.step()
            currentLR = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - startTime

            history["train_loss"].append(trainLoss)
            history["train_acc"].append(trainAcc)
            history["val_loss"].append(valLoss)
            history["val_acc"].append(valAcc)
            history["snr_acc"].append(snrAcc)
            history["lr"].append(currentLR)

            print(f"epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss: {trainLoss:.4f} train_acc: {trainAcc:.4f} | "
                  f"val_loss: {valLoss:.4f} val_acc: {valAcc:.4f} | "
                  f"lr: {currentLR:.6f} | {elapsed:.1f}s")

            if self.wandbEnabled:
                try:
                    import wandb
                    logData = {
                        "train/loss": trainLoss,
                        "train/accuracy": trainAcc,
                        "val/loss": valLoss,
                        "val/accuracy": valAcc,
                        "lr": currentLR,
                        "epoch": epoch + 1
                    }
                    for snrVal, acc in snrAcc.items():
                        logData[f"val/snr_{snrVal}dB"] = acc
                    wandb.log(logData)
                except ImportError:
                    pass

            if valAcc > self.bestValAcc:
                self.bestValAcc = valAcc
                self.patienceCounter = 0
                savePath = os.path.join(self.checkpointDir, f"{modelName}_best.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc": valAcc,
                    "val_loss": valLoss,
                    "config": self.config
                }, savePath)
                print(f"  -> saved best model (val_acc: {valAcc:.4f})")
            else:
                self.patienceCounter += 1
                if self.patienceCounter >= self.patience:
                    print(f"early stopping at epoch {epoch+1}")
                    break

        print("-" * 60)
        print(f"best val accuracy: {self.bestValAcc:.4f}")
        return history
