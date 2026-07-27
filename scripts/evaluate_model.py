import os
import sys
import torch
import pickle
import numpy as np
import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import RadioMLDataset
from scripts.train import build_model
from src.training.metrics import compute_per_snr_accuracy

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    modelName = cfg.experiment.get("name", "model")
    ckptDir = cfg.training.get("checkpoint_dir", "checkpoints")
    ckptPath = os.path.join(ckptDir, f"{modelName}_best.pth")
    
    if not os.path.exists(ckptPath):
        print(f"could not find checkpoint at {ckptPath}")
        return
        
    print(f"Loading trained model from {ckptPath}...")
    checkpoint = torch.load(ckptPath, weights_only=True)
    
    model = build_model(cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    dataDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    splitFile = os.path.join(dataDir, "radioml_splits.pkl")
    
    with open(splitFile, "rb") as f:
        splits = pickle.load(f)
        
    reprType = cfg.data.get("representation", "iq")
    testDataset = RadioMLDataset(splits["test"]["X"], splits["test"]["labels"], representation=reprType)
    
    # We will run the entire test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    from torch.utils.data import DataLoader
    testLoader = DataLoader(testDataset, batch_size=512, shuffle=False)
    
    allPreds = []
    allLabels = []
    allSnrs = []
    
    print("Evaluating entire test set...")
    with torch.no_grad():
        for features, labels, snrs in testLoader:
            features = features.to(device)
            outputs = model(features)
            preds = outputs.argmax(dim=1)
            
            allPreds.extend(preds.cpu().numpy())
            allLabels.extend(labels.cpu().numpy())
            
            # Extract standard integers from the SNR tensors so they group correctly
            if isinstance(snrs, torch.Tensor):
                allSnrs.extend(snrs.cpu().numpy())
            else:
                allSnrs.extend([s.item() if hasattr(s, "item") else s for s in snrs])
            
    snrAcc = compute_per_snr_accuracy(allPreds, allLabels, allSnrs)
    
    print("\n" + "="*40)
    print("   ACCURACY PER SNR (dB)")
    print("="*40)
    for snr in sorted(snrAcc.keys()):
        acc = snrAcc[snr]
        print(f"SNR {snr:>3} dB : {acc*100:>5.1f}%")
    print("="*40)

if __name__ == "__main__":
    main()
