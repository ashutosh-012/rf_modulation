# System Architecture & Flow

This document details the architectural decisions and deployment pipeline for the AMC (Automatic Modulation Classification) Edge System.

## High-Level System Architecture

```mermaid
graph TD
    subgraph Data Pipeline
        A[Raw I/Q Signal] -->|Preprocessing| B(Feature Extraction)
        B --> C1(Amplitude-Phase)
        B --> C2(STFT Spectrogram)
        B --> C3(Constellation Diagram)
    end
    
    subgraph Training Framework
        C1 --> M1[1D CNN / ResNet1D]
        C1 --> M2[CLDNN w/ SE Attention]
        C2 --> M3[EfficientNetV2-S]
        C3 --> M3
        
        M1 -->|PyTorch 2.5| T(Loss / Optimizer)
        M2 --> T
        M3 --> T
        
        T -->|Export| O(ONNX Opset 17)
    end
    
    subgraph Edge Deployment
        O -->|trt_convert.py| TRT[TensorRT 11.x Engine]
        TRT -->|execute_async_v3| INF[Inference Node Jetson/Orin]
        INF --> OUT[Predicted Modulation Class]
    end
```

## Model Architectures

### CLDNN with Squeeze-and-Excitation (SE)
This is our primary 1D architecture optimized for edge execution.

```mermaid
graph TD
    I[Input 2x128 I/Q] --> C1[Conv1D k=8]
    C1 --> C2[Conv1D k=5]
    C2 --> SE[Squeeze-and-Excitation Block]
    SE --> P1[MaxPool1D]
    
    SE -->|Skip Connection| S1[AdaptiveAvgPool1D]
    
    P1 --> L[LSTM 2-Layer]
    L --> F1[Flatten Last Step]
    
    F1 --> CONCAT
    S1 --> CONCAT
    
    CONCAT --> D1[Dense 256 + Dropout]
    D1 --> OUT[Dense 8 Softmax]
```

## SNR Stratification Strategy
To prevent data leakage, samples are grouped by their `(Modulation, SNR)` tuple before random assignment to Train/Val/Test splits. This guarantees the model sees varying noise floors consistently without inadvertently testing on adjacent time-windows from the training set.
