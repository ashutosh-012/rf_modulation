import torch
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    trt = None

class TRTInferenceWrapper:
    def __init__(self, enginePath):
        if trt is None:
            raise ImportError("tensorrt is not installed")
            
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(enginePath, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        self.inputName = self.engine.get_tensor_name(0)
        self.outputName = self.engine.get_tensor_name(1)
        
    def predict(self, inputData):
        if not isinstance(inputData, torch.Tensor):
            inputData = torch.tensor(inputData, dtype=torch.float32)
            
        if not inputData.is_cuda:
            inputData = inputData.cuda()
            
        outShape = self.context.get_tensor_shape(self.outputName)
        if outShape[0] == -1:
            outShape = list(outShape)
            outShape[0] = inputData.size(0)
            
        outputData = torch.empty(tuple(outShape), device=inputData.device, dtype=torch.float32)
        
        self.context.set_tensor_address(self.inputName, inputData.data_ptr())
        self.context.set_tensor_address(self.outputName, outputData.data_ptr())
        
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle=stream)
        
        return outputData
