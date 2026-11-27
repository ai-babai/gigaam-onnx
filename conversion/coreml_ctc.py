"""Export the GigaAM-v4 CTC E2E encoder and head as a native Core ML MLProgram."""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil.frontend.torch.ops import _get_inputs, logical_or, mb
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op

from conversion.ctc_export import CtcGraph, load_source

FRAMES = 1600


def clamp_min(context, node):
    data, bound = _get_inputs(context, node, expected=2)
    # PyTorch permits an integer scalar bound for a floating-point tensor.
    if data.dtype != bound.dtype:
        bound = mb.cast(x=bound, dtype=data.dtype.__name__)
    context.add(mb.maximum(x=data, y=bound, name=node.name))


def export(source, output):
    # TorchScript exports the boolean attention mask as aten::__or__.
    register_torch_op(logical_or, torch_alias=["or"], override=True)
    register_torch_op(clamp_min, override=True)

    model = load_source(source)
    example = torch.zeros((1, 64, FRAMES)), torch.tensor([FRAMES - 45], dtype=torch.int32)
    with model.encoder.onnx_export_mode(), torch.no_grad():
        traced = torch.jit.trace(CtcGraph(model).eval(), example, check_trace=False)

    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        inputs=[
            ct.TensorType(name="features", shape=example[0].shape, dtype=np.float32),
            ct.TensorType(name="feature_lengths", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="log_probs", dtype=np.float32),
            ct.TensorType(name="encoded_lengths", dtype=np.int32),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    converted.user_defined_metadata["gigaam_model_type"] = "v4-ctc-e2e"
    converted.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="GigaAM source config and safetensors directory")
    parser.add_argument("output", type=Path, help="New output .mlpackage path")
    args = parser.parse_args()
    if args.output.suffix != ".mlpackage" or args.output.exists():
        parser.error("Expected a new output path ending in .mlpackage")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    export(args.source, args.output)


if __name__ == "__main__":
    main()
