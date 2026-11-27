"""Export the Multilingual-v2 weight-only INT4 encoder from its FP32 ONNX graph."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="FP32 encoder with its external data")
    parser.add_argument("output", type=Path, help="Destination INT4 ONNX file")
    args = parser.parse_args()

    if args.source.resolve() == args.output.resolve():
        parser.error("Choose an output path different from the FP32 source")
    model = onnx.load(args.source, load_external_data=False)
    # The source export stores its parameters in files next to the graph.
    destination = args.output.resolve()
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        if onnx.external_data_helper.uses_external_data(tensor):
            sidecar = onnx.external_data_helper.ExternalDataInfo(tensor).location
            if (args.source.parent / sidecar).resolve() == destination:
                parser.error("The output path would overwrite FP32 tensor data")
    onnx.external_data_helper.load_external_data_for_model(model, str(args.source.parent))
    if [value.name for value in model.graph.input] != ["audio_signal", "length", "language_vector"]:
        parser.error("Expected the full-context Multilingual-v2 encoder")
    settings = DefaultWeightOnlyQuantConfig(
        bits=4, block_size=32, is_symmetric=True,
        accuracy_level=4, op_types_to_quantize=("MatMul",),
    )
    quantizer = MatMulNBitsQuantizer(model, algo_config=settings)
    quantizer.process()
    packed = [
        node for node in quantizer.model.model.graph.node
        if node.domain == "com.microsoft" and node.op_type == "MatMulNBits"
    ]
    expected = {"bits": 4, "block_size": 32, "accuracy_level": 4}
    if not packed:
        parser.error("No constant MatMul weights were quantized")
    for node in packed:
        attrs = {attr.name: attr.i for attr in node.attribute}
        if any(attrs.get(key) != value for key, value in expected.items()):
            parser.error("Unexpected MatMulNBits quantization settings")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        pending = Path(temp) / args.output.name
        quantizer.model.save_model_to_file(str(pending), use_external_data_format=False)
        onnx.checker.check_model(str(pending))
        pending.replace(args.output)


if __name__ == "__main__":
    main()
