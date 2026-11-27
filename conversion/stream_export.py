"""Export the fixed-context GigaAM-v4 RNN-T stream as an ONNX CPU package."""
import argparse
import json
from pathlib import Path
import shutil

import gigaam
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Upstream checkpoint cache")
    parser.add_argument("--work", type=Path, required=True, help="Floating point ONNX export")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.work / "streaming.json").is_file() or not list(args.work.glob("*.yaml")):
        model = gigaam.load_model("v4_rnnt", device="cpu", fp16_encoder=False,
                                  download_root=str(args.source))
        model.to_onnx(str(args.work), streaming=True, future_context_ms=640, dynamic_context=False)
        del model
    manifest = json.loads((args.work / "streaming.json").read_text())
    step = args.work / manifest["step"]["file"]
    quant = args.output / (step.stem + ".int8.onnx")
    quantize_dynamic(str(step), str(quant), op_types_to_quantize=["MatMul"],
                     weight_type=QuantType.QInt8, per_channel=True)
    manifest["step"]["file"] = quant.name
    # This package provides only streaming recognition. The upstream whole-file
    # graph duplicates the encoder weights and is unused for incremental decoding.
    whole = manifest.pop("whole")["file"]
    advance = manifest.pop("advance", {}).get("file")
    for path in args.work.iterdir():
        if path.suffix in {".onnx", ".model", ".yaml"} and path.name not in {step.name, whole, advance}:
            shutil.copy2(path, args.output / path.name)
    for filename in ("streaming.json", manifest["model_name"] + ".streaming.json"):
        (args.output / filename).write_text(json.dumps(manifest, indent=2) + "\n")
    print("Exported", quant, quant.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
