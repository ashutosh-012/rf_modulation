import os

try:
    import tensorrt as trt
except ImportError:
    trt = None


def convert_onnx_to_trt(onnxPath, trtPath, useFp16=True, maxWorkspace=1<<30):
    if trt is None:
        print("tensorrt is not installed. cannot build engine.")
        return None
        
    logger = trt.Logger(trt.Logger.WARNING)
    
    explicitBatch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    builder = trt.Builder(logger)
    network = builder.create_network(explicitBatch)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, maxWorkspace)
    
    if useFp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    with open(onnxPath, "rb") as f:
        if not parser.parse(f.read()):
            print("failed to parse onnx file:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    print("building tensorrt engine...")
    serializedEngine = builder.build_serialized_network(network, config)
    
    if serializedEngine is None:
        print("failed to build engine.")
        return None
        
    os.makedirs(os.path.dirname(trtPath), exist_ok=True)
    with open(trtPath, "wb") as f:
        f.write(serializedEngine)
        
    print(f"saved tensorrt engine to {trtPath}")
    return trtPath
