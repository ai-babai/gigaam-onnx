"""Build the CPU dynamic W8A8 package from a full-context upstream ONNX export."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from math import prod
from pathlib import Path
from shutil import copyfile

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper
from omegaconf import OmegaConf

from multi_export import MODEL_REVISION, PREFIX, SOURCE_DIGESTS, UPSTREAM_REVISION


def promote_fp16(graph, base):
    """Use FP32 for all surviving float operations after an FP16 export."""
    for value in [*graph.input, *graph.output, *graph.value_info]:
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type == TensorProto.FLOAT16:
            tensor_type.elem_type = TensorProto.FLOAT
    for node in graph.node:
        for attr in node.attribute:
            if attr.name in ("to", "dtype") and attr.type == AttributeProto.INT:
                if attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
            if (
                attr.type == AttributeProto.TENSOR
                and attr.t.data_type == TensorProto.FLOAT16
            ):
                array = numpy_helper.to_array(attr.t, base_dir=base).astype(np.float32)
                attr.t.CopyFrom(numpy_helper.from_array(array, attr.t.name))
            if attr.type == AttributeProto.GRAPH:
                for tensor in attr.g.initializer:
                    if tensor.data_type == TensorProto.FLOAT16:
                        array = numpy_helper.to_array(tensor, base_dir=base).astype(
                            np.float32
                        )
                        tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))
                promote_fp16(attr.g, base)


def quantize_encoder(source, output):
    model = onnx.load(source, load_external_data=False)
    graph = model.graph
    if len(graph.input) != 3 or graph.input[2].name != "language_vector":
        raise ValueError("Expected a full-context encoder with language conditioning")
    base = str(source.parent)
    initializers = {value.name: value for value in graph.initializer}
    retained_uses = Counter(name for node in graph.node for name in node.input)
    external_count = sum(bool(tensor.external_data) for tensor in graph.initializer)
    fp_elements = sum(
        prod(tensor.dims)
        for tensor in graph.initializer
        if tensor.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16)
    )
    converted = {}
    activations = {}
    quantized_params = 0
    nodes = []
    tensors = []

    for index, node in enumerate(graph.node):
        if node.op_type != "MatMul" or node.domain not in ("", "ai.onnx"):
            nodes.append(node)
            continue
        activations_name, weight_name = node.input
        weight = initializers.get(weight_name)
        if (
            weight is None
            or weight.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16)
            or len(weight.dims) != 2
        ):
            nodes.append(node)
            continue
        tag = f"/dynamic_int8/{index}"
        if weight_name not in converted:
            matrix = numpy_helper.to_array(weight, base_dir=base).astype(
                np.float32, copy=False
            )
            peak = np.max(np.abs(matrix), axis=0)
            scale = np.where(peak == 0, 1.0, peak / 127).astype(np.float32)
            quantized = np.clip(np.rint(matrix / scale), -127, 127).astype(np.int8)
            q_weight, w_scale, w_zero = tag + "/weight", tag + "/scale", tag + "/zero"
            tensors.extend(
                numpy_helper.from_array(array, name)
                for name, array in (
                    (q_weight, quantized),
                    (w_scale, scale),
                    (w_zero, np.zeros(scale.shape, dtype=np.int8)),
                )
            )
            converted[weight_name] = (q_weight, w_scale, w_zero)
            quantized_params += quantized.size
        q_weight, w_scale, w_zero = converted[weight_name]
        retained_uses[weight_name] -= 1

        if activations_name not in activations:
            outputs = (
                tag + "/activation",
                tag + "/activation_scale",
                tag + "/activation_zero",
            )
            nodes.append(
                helper.make_node(
                    "DynamicQuantizeLinear",
                    [activations_name],
                    outputs,
                    name=tag + "/quantize",
                )
            )
            activations[activations_name] = outputs
        q_input, a_scale, a_zero = activations[activations_name]
        integer, cast, scale = tag + "/integer", tag + "/cast", tag + "/product_scale"
        nodes.extend(
            [
                helper.make_node("Mul", [a_scale, w_scale], [scale], name=scale),
                helper.make_node(
                    "MatMulInteger",
                    [q_input, q_weight, a_zero, w_zero],
                    [integer],
                    name=integer,
                ),
                helper.make_node(
                    "Cast", [integer], [cast], name=cast, to=TensorProto.FLOAT
                ),
                helper.make_node("Mul", [cast, scale], node.output, name=node.name),
            ]
        )

    if quantized_params == 0:
        raise RuntimeError("No static MatMul weights were eligible for INT8")
    for tensor in graph.initializer:
        if tensor.name in converted and retained_uses[tensor.name] == 0:
            continue
        if tensor.data_type == TensorProto.FLOAT16 or tensor.external_data:
            array = numpy_helper.to_array(tensor, base_dir=base)
            if array.dtype == np.float16:
                array = array.astype(np.float32)
            tensor = numpy_helper.from_array(array, tensor.name)
        tensors.append(tensor)

    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(tensors)
    promote_fp16(graph, base)
    onnx.save_model(model, output)
    return {
        "scheme": "Dynamic UINT8 activations, symmetric per-output-channel INT8 MatMul weights",
        "quantized_weight_elements": quantized_params,
        "source_float_initializer_elements": fp_elements,
        "quantized_initializers": len(converted),
        "matmul_integer_nodes": sum(
            node.op_type == "MatMulInteger" for node in graph.node
        ),
        "upstream_external_initializers": external_count,
        "encoder_bytes": output.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--fp-dir", default="fp")
    args = parser.parse_args()
    fp = args.work / args.fp_dir
    source = args.work / "source"
    package = args.work / "package"
    package.mkdir(parents=True, exist_ok=True)
    output = package / f"{PREFIX}_encoder.int8.onnx"
    pending = output.with_suffix(".pending.onnx")
    stats = quantize_encoder(fp / f"{PREFIX}_encoder.onnx", pending)
    onnx.checker.check_model(str(pending))
    pending.replace(output)
    advance = fp / f"{PREFIX}_advance.onnx"
    onnx.checker.check_model(str(advance))
    decoder_types = {
        tensor.data_type
        for tensor in onnx.load(advance, load_external_data=False).graph.initializer
    }
    if TensorProto.FLOAT not in decoder_types or decoder_types & {
        TensorProto.FLOAT16,
        TensorProto.INT8,
        TensorProto.UINT8,
    }:
        raise ValueError("Expected the original FP32 RNNT advance graph")
    stats["decoder"] = "FP32"
    for suffix in (
        "advance.onnx",
        "raw_tokenizer.model",
        "e2e_tokenizer.model",
        "mel.npz",
    ):
        filename = f"{PREFIX}_{suffix}"
        copyfile(fp / filename, package / filename)
    copyfile(fp / f"{PREFIX}.yaml", package / f"{PREFIX}.yaml")
    for name in ("config.json", "LICENSE", "LICENSE-APACHE-2.0"):
        copyfile(source / name, package / name)
    copyfile(Path(__file__).with_name("multi_client.py"), package / "multi_client.py")
    (package / "requirements.txt").write_text(
        "numpy>=2,<3\nonnxruntime==1.23.2\nsentencepiece>=0.2,<0.3\n"
    )
    (package / "README.md").write_text(
        "# GigaAM Multilingual-v2 RNN-T ONNX INT8\n\n"
        "This package runs full-context ASR on CPU and returns formatted `e2e` and plain `raw` text. "
        "It requires Python 3.10+, FFmpeg on PATH, and the packages in requirements.txt.\n\n"
        "```sh\n"
        "python -m pip install -r requirements.txt\n"
        "python multi_client.py /path/to/clip.wav --language kk\n"
        "```\n\n"
        "Use a clip up to 30 seconds. Set a language code or pass `auto`. The client runs offline "
        "with CPUExecutionProvider and three intra-op threads. Keep the package files together.\n\n"
        "Original model and weights © 2026 Giga under MIT. ONNX conversion and INT8 packaging by Maksim Popkov. "
        "The upstream encoder incorporates fragments adapted from Xiaomi Corp., Daniel Povey, and Zengwei Yao "
        "under Apache-2.0. The package includes the license texts and revision IDs.\n"
    )
    cfg = OmegaConf.load(fp / f"{PREFIX}.yaml")
    if cfg.model_name != PREFIX or cfg.onnx_future_context_ms is not None:
        raise ValueError("Expected an upstream full-context export")
    metadata = {
        "model_revision": MODEL_REVISION,
        "original_sha256": SOURCE_DIGESTS,
        "upstream_revision": UPSTREAM_REVISION,
        "model_name": PREFIX,
        "context": "full",
        "head_order": list(cfg.onnx_fc),
        "encoder_source": args.fp_dir,
        "quantization": stats,
        "files": {},
    }
    for file in sorted(package.iterdir()):
        if file.is_file() and file.name != "manifest.json":
            with file.open("rb") as stream:
                metadata["files"][file.name] = {
                    "bytes": file.stat().st_size,
                    "sha256": file_digest(stream, "sha256").hexdigest(),
                }
    (package / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
