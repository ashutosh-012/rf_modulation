import torch
import os


def export_to_onnx(model, dummyInput, outputPath, opsetVersion=17):
    model.eval()

    os.makedirs(os.path.dirname(outputPath), exist_ok=True)

    torch.onnx.export(
        model,
        dummyInput,
        outputPath,
        export_params=True,
        opset_version=opsetVersion,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"exported onnx model to {outputPath}")
    return outputPath
