"""Compare timestamped incremental calls against the matching upstream PyTorch stream."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import gigaam
import torch
from gigaam.streaming import GigaAMStreaming

from stream_transcribe import decode, load_stream, read_wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "package", "manifest", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    reference = gigaam.load_model("v4_rnnt", device="cpu", fp16_encoder=False,
                                  download_root=str(args.source))
    streams = {"fp": GigaAMStreaming(reference, 640), "int8": load_stream(args.package)}
    clips = [item for line in args.manifest.read_text().splitlines()
             if line.strip() and (item := json.loads(line))["lang"] in {"ru", "en"}]
    if not clips:
        raise ValueError(f"No Russian or English audio in {args.manifest}")
    for stream in streams.values():
        decode(stream, read_wave(clips[0]["path"]))
    with args.output.open("w") as dst:
        for index, clip in enumerate(clips):
            wave = read_wave(clip["path"])
            row = {key: clip[key] for key in ("lang", "id", "ref", "audio_seconds")}
            for name in list(streams)[::1 if index % 2 == 0 else -1]:
                before = perf_counter()
                out = decode(streams[name], wave)
                row["sec_" + name] = perf_counter() - before
                row["hyp_" + name] = out["e2e"]
                row["raw_" + name] = out["raw"]
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            dst.flush()


if __name__ == "__main__":
    main()
