import os
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="RF Modulation Classification API",
    description="Production ONNX Inference Server for RadioML",
    version="1.0.0"
)

# Global variables for model session and classes
ort_session = None
MOD_CLASSES = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK']

class InferenceRequest(BaseModel):
    # Expecting a flat list of 256 floats (2 channels x 128 seq_len)
    iq_data: list[float]

class InferenceResponse(BaseModel):
    prediction: str
    confidence: float
    snr_warning: bool

@app.on_event("startup")
def load_model():
    global ort_session
    # server.py is in src/deployment/, so we need to go up two levels to reach the root
    rootDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(rootDir, "outputs", "models", "default_experiment.onnx")
    
    if not os.path.exists(model_path):
        print(f"WARNING: ONNX model not found at {model_path}. Please run ONNX export script first.")
        return
        
    print(f"Loading ONNX model from {model_path}")
    
    # Configure ONNX Runtime for max performance on CPU/Edge
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 2
    
    # Try TensorRT provider first, fallback to CPU
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    
    ort_session = ort.InferenceSession(model_path, sess_options, providers=providers)
    print("Model loaded successfully!")

@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    if len(request.iq_data) != 256:
        raise HTTPException(status_code=400, detail=f"Expected 256 features, got {len(request.iq_data)}")
        
    # Reshape input to (Batch=1, Channels=2, SeqLen=128)
    input_array = np.array(request.iq_data, dtype=np.float32).reshape(1, 2, 128)
    
    # Run ONNX Inference
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_array})
    
    # Calculate Softmax for confidence
    logits = outputs[0][0]
    exp_preds = np.exp(logits - np.max(logits))
    probs = exp_preds / np.sum(exp_preds)
    
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    
    return InferenceResponse(
        prediction=MOD_CLASSES[pred_idx],
        confidence=confidence,
        snr_warning=(confidence < 0.3)
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": ort_session is not None}

if __name__ == "__main__":
    print("Starting Production Inference Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
