import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.export.onnx_export import export_to_onnx
from src.export.trt_convert import convert_onnx_to_trt
from scripts.train import build_model


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    modelName = cfg.experiment.get("name", "model")
    ckptDir = cfg.training.get("checkpoint_dir", "checkpoints")
    ckptPath = os.path.join(ckptDir, f"{modelName}_best.pth")
    
    if not os.path.exists(ckptPath):
        print(f"could not find checkpoint at {ckptPath}")
        return
        
    print(f"loading model from {ckptPath}")
    checkpoint = torch.load(ckptPath)
    
    model = build_model(cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    reprType = cfg.data.get("representation", "iq")
    
    if reprType in ["iq", "ap"]:
        seqLen = cfg.model.get("seq_len", 128)
        dummyInput = torch.randn(1, 2, seqLen)
    elif reprType == "stft":
        dummyInput = torch.randn(1, 1, 64, 33)
    elif reprType == "constellation":
        dummyInput = torch.randn(1, 1, 64, 64)
    else:
        dummyInput = torch.randn(1, 2, 128)
        
    onnxPath = os.path.join("outputs", "models", f"{modelName}.onnx")
    export_to_onnx(model, dummyInput, onnxPath)
    
    try:
        import tensorrt
        print("tensorrt found. building trt engine...")
        trtPath = os.path.join("outputs", "models", f"{modelName}_fp16.engine")
        convert_onnx_to_trt(onnxPath, trtPath, useFp16=True)
    except ImportError:
        print("tensorrt not installed, skipping engine build")
        print("to build on jetson, copy the .onnx file to the device")


if __name__ == "__main__":
    main()
