"""Build the full-context GigaAM-v4 RNN-T CPU ONNX INT8 package."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from math import prod
import os
from pathlib import Path
from shutil import copyfile

os.environ["OMP_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "3"
os.environ["VECLIB_MAXIMUM_THREADS"] = "3"

PREFIX = "v4_rnnt"
UPSTREAM = "e9bbf2e773eedd217c6f0d3b89b9728c41332b7d"
HF_REVISION = "9fe3ebf50c9e0cce5eed60d546938632238b6a96"
SOURCE_MD5 = "af1027b4fa8b306c7aa1974e4b740fe0"
TOKENIZERS = {
    "raw": "dcbe2598d1b374065091832970593a06065d498a43fbb1787de73e180d0556bc",
    "e2e": "f547bcb38e071694ef1ddb25d9d40c86b4ab88cd0b16ab9fd52f27dd52c48e39",
}


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return file_digest(stream, algorithm).hexdigest()


def load_torch(source):
    import gigaam
    import torch

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    for head, expected in TOKENIZERS.items():
        path = source / f"{PREFIX}_{head}_tokenizer.model"
        if digest(path) != expected:
            raise ValueError(f"Tokenizer SHA-256 mismatch: {path}")
    if not (source / f"{PREFIX}.ckpt").is_file():
        raise FileNotFoundError(
            f"Expected the verified upstream checkpoint in {source}"
        )
    return gigaam.load_model(
        PREFIX, device="cpu", fp16_encoder=False, download_root=str(source)
    )


def export_fp(source, fp):
    import numpy as np
    import torch

    fp.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261127)
    model = load_torch(source)
    mel = model.preprocessor.featurizer[0]
    np.savez(
        fp / f"{PREFIX}_mel.npz",
        window=mel.spectrogram.window.numpy(),
        filterbank=mel.mel_scale.fb.numpy(),
    )
    model.to_onnx(str(fp))


def build_package(fp, package, upstream_license, apache_license):
    import onnx
    from onnx import TensorProto
    from omegaconf import OmegaConf
    from onnxruntime.quantization import QuantType, quantize_dynamic

    config = OmegaConf.load(fp / f"{PREFIX}.yaml")
    if config.model_name != PREFIX or config.onnx_future_context_ms is not None:
        raise ValueError("A true full-context upstream export is required")
    package.mkdir(parents=True, exist_ok=True)
    source_encoder = fp / f"{PREFIX}_encoder.onnx"
    output = package / f"{PREFIX}_encoder.int8.onnx"
    pending = output.with_suffix(".pending.onnx")
    print(f"Quantizing {source_encoder.name}", flush=True)
    quantize_dynamic(
        str(source_encoder),
        str(pending),
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={"MatMulConstBOnly": True},
    )
    onnx.checker.check_model(str(pending))
    encoder = onnx.load(pending, load_external_data=False).graph
    operations = Counter(node.op_type for node in encoder.node)
    int8 = [
        tensor for tensor in encoder.initializer if tensor.data_type == TensorProto.INT8
    ]
    if len(encoder.input) != 2 or not operations["MatMulInteger"] or not int8:
        raise ValueError(
            "Expected a two-input full-context encoder with real integer MatMul"
        )
    if any(tensor.external_data for tensor in encoder.initializer):
        raise ValueError("Expected a self-contained encoder file")
    int8_elements = sum(prod(tensor.dims) for tensor in int8)
    del encoder, int8
    pending.replace(output)

    advance = fp / f"{PREFIX}_advance.onnx"
    onnx.checker.check_model(str(advance))
    graph = onnx.load(advance, load_external_data=False).graph
    if not any(tensor.data_type == TensorProto.FLOAT for tensor in graph.initializer):
        raise ValueError("Expected an FP32 advance graph")
    if any(
        tensor.data_type in (TensorProto.FLOAT16, TensorProto.INT8, TensorProto.UINT8)
        for tensor in graph.initializer
    ):
        raise ValueError("Unexpected reduced-precision initializer in FP32 advance")
    for name in (
        f"{PREFIX}.yaml",
        f"{PREFIX}_advance.onnx",
        f"{PREFIX}_e2e_tokenizer.model",
        f"{PREFIX}_raw_tokenizer.model",
        f"{PREFIX}_mel.npz",
    ):
        copyfile(fp / name, package / name)
    copyfile(upstream_license, package / "LICENSE")
    copyfile(apache_license, package / "LICENSE-APACHE-2.0")
    copyfile(
        Path(__file__).with_name("rnnt_transcribe.py"), package / "rnnt_transcribe.py"
    )
    (package / "requirements.txt").write_text(
        "numpy>=2,<3\nonnxruntime==1.23.2\nsentencepiece>=0.2,<0.3\n"
    )
    (package / "README.md").write_text(
        "# GigaAM-v4 RNN-T ONNX INT8 (full context)\n\n"
        "CPU speech recognition in Russian and English, with formatted `e2e` and plain `raw` output. "
        "Use Python 3.10+ and FFmpeg. Keep the package files together.\n\n"
        "```sh\n"
        "python -m pip install -r requirements.txt\n"
        "python rnnt_transcribe.py /path/to/clip.wav\n"
        "```\n\n"
        "After installing the dependencies, inference runs offline in ONNX Runtime with three CPU threads. "
        "Use individual clips of up to 30 seconds. This model has no language selector.\n\n"
        "Original model and checkpoint © 2026 Giga, MIT. ONNX conversion and INT8 packaging by Maksim Popkov. "
        "The upstream encoder incorporates fragments adapted from Xiaomi Corp., Daniel Povey and Zengwei Yao "
        "under Apache-2.0. License texts are included.\n"
    )
    metadata = {
        "model": "GigaAM-v4",
        "model_name": PREFIX,
        "source": "official CDN v4_rnnt.ckpt",
        "source_checkpoint_md5": SOURCE_MD5,
        "hf_rnnt_revision": HF_REVISION,
        "upstream_code_revision": UPSTREAM,
        "context": "full",
        "head_order": list(config.onnx_fc),
        "supported_languages": ["ru", "en"],
        "preprocessor": OmegaConf.to_container(config.preprocessor, resolve=True),
        "quantization": {
            "scheme": "Dynamic UINT8 activations, per-output-channel signed INT8 MatMul weights",
            "matmul_integer_nodes": operations["MatMulInteger"],
            "int8_initializer_elements": int8_elements,
            "decoder": "FP32",
        },
        "files": {
            file.name: {"bytes": file.stat().st_size, "sha256": digest(file)}
            for file in sorted(package.iterdir())
            if file.is_file() and file.name != "manifest.json"
        },
    }
    (package / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["quantization"], indent=2), flush=True)
    print(f"{output.name}: {output.stat().st_size:,} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("upstream_license", type=Path)
    parser.add_argument("apache_license", type=Path)
    parser.add_argument("--quantize-only", action="store_true")
    args = parser.parse_args()
    fp = args.work / "fp"
    if not args.quantize_only:
        export_fp(args.source, fp)
    build_package(fp, args.work / "package", args.upstream_license, args.apache_license)


if __name__ == "__main__":
    main()
