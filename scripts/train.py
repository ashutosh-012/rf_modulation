import os
import sys
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import RadioMLDataset
from src.training.trainer import Trainer
from src.models.basic_cnn import BasicCNN
from src.models.resnet1d import ResNet1D
from src.models.cldnn import CLDNN
from src.models.cldnn_se import CLDNN_SE
from src.models.lstm_model import LSTMClassifier
from src.models.efficientnet_wrapper import EfficientNetWrapper

def build_model(modelConfig):
    arch = modelConfig.get("architecture", "basic_cnn")
    numClasses = modelConfig.get("num_classes", 8)
    
    if arch == "basic_cnn":
        return BasicCNN(
            numClasses=numClasses,
            inputChannels=modelConfig.get("input_channels", 2),
            seqLen=modelConfig.get("seq_len", 128)
        )
    elif arch == "resnet1d":
        return ResNet1D(
            numClasses=numClasses,
            inputChannels=modelConfig.get("input_channels", 2),
            numBlocks=modelConfig.get("num_blocks", 6),
            hiddenDim=modelConfig.get("hidden_dim", 64)
        )
    elif arch == "cldnn":
        return CLDNN(
            numClasses=numClasses,
            inputChannels=modelConfig.get("input_channels", 2),
            seqLen=modelConfig.get("seq_len", 128)
        )
    elif arch == "cldnn_se":
        return CLDNN_SE(
            numClasses=numClasses,
            inputChannels=modelConfig.get("input_channels", 2),
            seqLen=modelConfig.get("seq_len", 128)
        )
    elif arch == "lstm":
        return LSTMClassifier(
            numClasses=numClasses,
            inputSize=modelConfig.get("input_size", 2),
            hiddenSize=modelConfig.get("hidden_size", 128),
            numLayers=modelConfig.get("num_layers", 2)
        )
    elif arch == "efficientnetv2":
        return EfficientNetWrapper(
            numClasses=numClasses,
            inputChannels=modelConfig.get("input_channels", 1),
            pretrained=modelConfig.get("pretrained", True)
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("=== Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.training.get("wandb", False):
        try:
            import wandb
            wandb.init(
                project="rf-modulation-classification",
                name=cfg.experiment.get("name", "experiment"),
                config=OmegaConf.to_container(cfg, resolve=True)
            )
        except ImportError:
            print("wandb not installed. run: pip install wandb")
            
    dataDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    splitFile = os.path.join(dataDir, "radioml_splits.pkl")
    
    if not os.path.exists(splitFile):
        print(f"Error: {splitFile} not found.")
        print("Please run scripts/prepare_data.py first.")
        return
        
    print("Loading data splits...")
    with open(splitFile, "rb") as f:
        splits = pickle.load(f)
        
    reprType = cfg.data.get("representation", "iq")
    stftParams = OmegaConf.to_container(cfg.data.get("stft_params", {}))
    constelParams = OmegaConf.to_container(cfg.data.get("constel_params", {}))
    
    trainDataset = RadioMLDataset(
        splits["train"]["X"], 
        splits["train"]["labels"],
        representation=reprType,
        stftParams=stftParams,
        constelParams=constelParams
    )
    
    valDataset = RadioMLDataset(
        splits["val"]["X"], 
        splits["val"]["labels"],
        representation=reprType,
        stftParams=stftParams,
        constelParams=constelParams
    )
    
    print("Building model...")
    model = build_model(cfg.model)
    
    # Enable PyTorch 2.0 compilation if requested
    if cfg.training.get("compile_model", False):
        print("Compiling model with torch.compile()...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Failed to compile model: {e}")
            print("Falling back to eager mode.")
            
    # Combine training config with experiment info for the trainer
    trainConfig = OmegaConf.to_container(cfg.training, resolve=True)
    trainConfig["model_name"] = cfg.experiment.get("name", "model")
    
    trainer = Trainer(model, trainDataset, valDataset, trainConfig)
    
    print("Starting training...")
    history = trainer.train()
    
    if cfg.training.get("wandb", False):
        import wandb
        
        # log the best model as an artifact
        bestModelPath = os.path.join(trainConfig["checkpoint_dir"], f"{trainConfig['model_name']}_best.pth")
        if os.path.exists(bestModelPath):
            artifact = wandb.Artifact(f"{trainConfig['model_name']}_model", type="model")
            artifact.add_file(bestModelPath)
            wandb.log_artifact(artifact)
            
        wandb.finish()

if __name__ == "__main__":
    main()
