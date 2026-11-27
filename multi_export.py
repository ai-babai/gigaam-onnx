"""Export the pinned GigaAM Multilingual-v2 checkpoints via upstream GigaAM.

Usage: PYTHONPATH=/path/to/upstream python multi_export.py /path/to/multi
"""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

PREFIX = "multilingual_v2_rnnt"
MODEL_REVISION = "c21483b5e185ff32f6fb2e2c801f92f4e7fbfee8"
UPSTREAM_REVISION = "e9bbf2e773eedd217c6f0d3b89b9728c41332b7d"
SOURCE_DIGESTS = {
    "model.safetensors": "1ea6b24faa308791b02e6916450a63b2a2ff401a7d890e08deaf34fc2acfe869",
    "raw_tokenizer.model": "c51fe228b747218891e3facdaeda12957065f4238452c377f38f76d3c2d4c423",
    "e2e_tokenizer.model": "3b049685e51a893350452f31613b47f4bd7640857f53facdf32266de82df3c0a",
}


def load_reference(source):
    from gigaam.model import GigaAMDualASR
    from omegaconf import OmegaConf
    from safetensors import safe_open

    config = json.loads((source / "config.json").read_text())["cfg"]
    config["model_name"] = PREFIX
    for decoding in config["decodings"].values():
        decoding["model_path"] = str((source / decoding["model_path"]).resolve())
    model = GigaAMDualASR(OmegaConf.create(config))
    with safe_open(
        source / "model.safetensors", framework="pt", device="cpu"
    ) as weights:
        state = {
            name.removeprefix("model."): weights.get_tensor(name)
            for name in weights.keys()
        }
        model.load_state_dict(state, strict=True)
    return model.eval()


def main():
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument(
        "--export-half", action="store_true", help="Use an FP16 encoder for export"
    )
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    source = args.work / "source"
    export = args.work / ("fp16" if args.export_half else "fp")
    export.mkdir(parents=True, exist_ok=True)
    print("Verifying the source files", flush=True)
    for name, digest in SOURCE_DIGESTS.items():
        with (source / name).open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"SHA-256 mismatch for {source / name}")
    print("Loading the pinned reference", flush=True)
    model = load_reference(source)
    mel = model.preprocessor.featurizer[0]
    np.savez(
        export / f"{PREFIX}_mel.npz",
        window=mel.spectrogram.window.numpy(),
        filterbank=mel.mel_scale.fb.numpy(),
    )
    print(f"Exporting full-context model into {export}", flush=True)
    if args.export_half:
        model.encoder.half()
    # Same graph construction as upstream to_onnx, with this disposable CPU model.
    model._to_onnx(export, dtype=torch.float16 if args.export_half else torch.float32)
    print(f"Export complete: {export}", flush=True)


if __name__ == "__main__":
    main()
