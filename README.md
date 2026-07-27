# RF Modulation Classification (2026)

This project tackles Automatic Modulation Classification (AMC) on RF signals, focusing on edge deployment and ablation across architectures and representations.

It builds upon baseline 1D models (like CLDNN/ResNet) and modernizes them with PyTorch 2.5+, mixed precision, SE-attention, and EfficientNetV2 on spectrograms, culminating in an ONNX to TensorRT 11.x deployment pipeline.

## Features
- **Signal Representations:** IQ time-series, Amplitude-Phase, STFT Spectrograms, Constellation diagrams.
- **Architectures:** Basic CNN, 1D ResNet, CLDNN (with Squeeze-and-Excitation), Bidirectional LSTM, EfficientNetV2-S (via `timm`).
- **Data Pipeline:** SNR-stratified train/val/test splits preventing data leakage on correlated time-series data.
- **MLOps:** Config-driven via Hydra, tracked via Weights & Biases (W&B).
- **Edge Deployment:** PyTorch `torch.onnx.export` (opset 17) -> TensorRT 11.x (`IBuilderConfig`, `execute_async_v3`).

## Setup

```bash
pip install -r requirements.txt
```

Download and prep the RadioML 2016.10A dataset:
```bash
python scripts/prepare_data.py
```

## Training

Train the default model (CLDNN-SE on Amplitude-Phase):
```bash
python scripts/train.py
```

Train EfficientNet on STFT spectrograms:
```bash
python scripts/train.py model=efficientnet data.representation=stft
```

## Deployment

Export the trained model to ONNX and TensorRT:
```bash
python scripts/export_onnx.py
```

## Notes on Architecture Choices
While SSM/Mamba-based models (like MAMCA) are currently SOTA for linear complexity on long IQ sequences, this project intentionally focuses on CNN-based architectures (VGG-style, ResNet, EfficientNet) and their edge deployment paths, as these have highly mature and predictable quantization behaviors in TensorRT for latency-critical environments.
