"""Export only the full-context GigaAM RNN-T encoder to Core ML."""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil.frontend.torch.ops import logical_or
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    register_torch_op,
)
from gigaam.decoder import DualASRExport

from conversion.coreml_ctc import FRAMES, clamp_min
from conversion.multi_export import load_reference
from conversion.rnnt_export import load_torch


def export(source, output, multilingual):
    register_torch_op(logical_or, torch_alias=["or"], override=True)
    register_torch_op(clamp_min, override=True)

    model = load_reference(source) if multilingual else load_torch(source)
    model.encoder._onnx_context = None
    example = [
        torch.zeros((1, 64, FRAMES)),
        torch.tensor([FRAMES - 45], dtype=torch.int32),
    ]
    names = ["features", "feature_lengths"]
    if multilingual:
        language = torch.zeros((1, 32))
        language[0, 3] = 1.0
        example.append(language)
        names.append("language_vector")
    dtypes = {"feature_lengths": np.int32}
    with model.encoder.onnx_export_mode(), torch.no_grad():
        traced = torch.jit.trace(
            DualASRExport(model).eval(), tuple(example), check_trace=False
        )

    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        inputs=[
            ct.TensorType(
                name=name, shape=tensor.shape, dtype=dtypes.get(name, np.float32)
            )
            for name, tensor in zip(names, example)
        ],
        outputs=[
            ct.TensorType(name="encoded_lengths", dtype=np.int32),
            ct.TensorType(name="head_outputs", dtype=np.float32),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    converted.user_defined_metadata["gigaam_model_type"] = (
        "multilingual-v2-rnnt" if multilingual else "v4-rnnt"
    )
    converted.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local upstream model files")
    parser.add_argument("output", type=Path)
    parser.add_argument("--multilingual", action="store_true")
    args = parser.parse_args()
    if args.output.suffix != ".mlpackage" or args.output.exists():
        parser.error("Expected a new output path ending in .mlpackage")
    if args.multilingual:
        torch.set_num_threads(3)
        torch.set_num_interop_threads(1)
    export(args.source, args.output, args.multilingual)


if __name__ == "__main__":
    main()
