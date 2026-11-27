"""Paired CPU speed with a floating point and quantized graph in the same runtime."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile
import torch


def get_decoders(base, name):
    if name == "v4":
        from clients.ctc_transcribe import OrtCtc
        package = base / "package"
        return (OrtCtc(package, base / "work/model.fp32.onnx").recognize,
                OrtCtc(package).recognize)
    if name == "rnnt":
        from clients.rnnt_transcribe import GigaAMRNNT
        package = base / "package"
        return (GigaAMRNNT(package, encoder=base / "fp/v4_rnnt_encoder.onnx").transcribe,
                GigaAMRNNT(package).transcribe)
    from clients.stream_transcribe import decode, load_stream
    fp = load_stream(base / "fp")
    quantized = load_stream(base / "package")
    return (lambda audio: decode(fp, torch.from_numpy(audio)),
            lambda audio: decode(quantized, torch.from_numpy(audio)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="Variant workspace with reference graph and package")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("variant", choices=("v4", "rnnt", "stream"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    selected = []
    # The input is already seed-shuffled within language. Select four per language.
    for language in ("ru", "en"):
        clips = [r for r in rows if r["lang"] == language][:4]
        if len(clips) != 4:
            raise ValueError(language)
        selected.extend(clips)
    audio = []
    for row in selected:
        wave, rate = soundfile.read(args.manifest.parent / row["path"], dtype="float32")
        assert rate == 16000 and wave.ndim == 1
        audio.append(wave)
    fp, quantized = get_decoders(args.work, args.variant)
    fp(audio[0])
    quantized(audio[0])
    totals = defaultdict(lambda: {"audio_seconds": 0.0, "seconds_fp": 0.0, "seconds_int8": 0.0})
    for i, (row, wave) in enumerate(zip(selected, audio, strict=True)):
        order = [("fp", fp), ("int8", quantized)]
        if i % 2:
            order.reverse()
        measured = {}
        for label, fn in order:
            before = perf_counter()
            fn(wave)
            measured[label] = perf_counter() - before
        for group in (row["lang"], "all"):
            totals[group]["audio_seconds"] += len(wave)/16000
            for label, seconds in measured.items():
                totals[group]["seconds_" + label] += seconds
    for s in totals.values():
        s["speedup"] = s["seconds_fp"] / s["seconds_int8"]
        for label in ("fp", "int8"):
            s["rtfx_" + label] = s["audio_seconds"] / s["seconds_" + label]
    result = {"variant": args.variant, "sample_ids": [r["id"] for r in selected],
              "warm": True, "threads": 3, "platform": "Apple M4 Pro CPU", "stats": totals}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.variant, totals["all"], flush=True)


if __name__ == "__main__":
    main()
