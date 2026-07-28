import os
import sys
import torch
import pickle
import numpy as np
import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import RadioMLDataset, MOD_CLASSES
from scripts.train import build_model

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
    
    testDataset = RadioMLDataset(
        splits["test"]["X"], 
        splits["test"]["labels"],
        representation=reprType
    )
    
    print("\n" + "="*50)
    print("   TESTING INFERENCE (10 RANDOM SAMPLES)")
    print("="*50)
    print(f"{'SNR':>5} | {'TRUE MODULATION':>15} | {'PREDICTED MODULATION':>20} | {'RESULT':>6}")
    print("-" * 55)
    
    indices = np.random.choice(len(testDataset), 10, replace=False)
    
    correctCount = 0
    with torch.no_grad():
        for idx in indices:
            feat, labelIdx, snr = testDataset[idx]
            
            feat = feat.unsqueeze(0)
            
            out = model(feat)
            predIdx = out.argmax(dim=1).item()
            
            trueMod = MOD_CLASSES[labelIdx.item()]
            predMod = MOD_CLASSES[predIdx]
            
            isCorrect = "MATCH" if trueMod == predMod else "FAIL"
            if isCorrect == "MATCH":
                correctCount += 1
                
            print(f"{snr:>5} | {trueMod:>15} | {predMod:>20} | {isCorrect:>6}")
            
    print("-" * 55)
    print(f"Accuracy on this random batch: {correctCount}/10")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
