"""Run paired GigaAM-v4 CTC FP32 and INT8 decoding on Russian and English clips."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from clients.ctc_transcribe import OrtCtc, load_audio

LANGUAGES = ("ru", "en")


def manifest_rows(manifest):
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["lang"] in LANGUAGES:
            row["path"] = manifest.parent / row["path"]
            yield row


def benchmark(source, package, manifest, output):
    import torch

    from conversion.ctc_export import load_source

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model = load_source(source)
    quantized = OrtCtc(package)
    rows = list(manifest_rows(manifest))
    if not rows:
        raise ValueError(f"No Russian or English audio in {manifest}")

    @torch.inference_mode()
    def source_transcript(samples):
        features, lengths = model(
            torch.from_numpy(samples)[None, :], torch.tensor([samples.size])
        )
        return model.decodings["e2e"].decode(model.heads["e2e"], features, lengths)[0][0]

    first_clip = load_audio(rows[0]["path"])
    source_transcript(first_clip)
    quantized.recognize(first_clip)

    staging = output.with_suffix(".partial.jsonl")
    with staging.open("w", encoding="utf-8") as report:
        for row in rows:
            samples = load_audio(row["path"])
            start = perf_counter()
            hyp_fp = source_transcript(samples)
            sec_fp = perf_counter() - start
            start = perf_counter()
            hyp_int8 = quantized.recognize(samples)
            sec_int8 = perf_counter() - start
            item = {
                "lang": row["lang"], "id": row["id"], "ref": row["ref"],
                "hyp_fp": hyp_fp, "hyp_int8": hyp_int8,
                "sec_fp": sec_fp, "sec_int8": sec_int8,
                "audio_seconds": row["audio_seconds"],
            }
            report.write(json.dumps(item, ensure_ascii=False) + "\n")
            report.flush()
            print(f'{row["lang"]}/{row["id"]}: FP32 {sec_fp:.3f} s / INT8 {sec_int8:.3f} s', flush=True)
    staging.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    benchmark(args.source, args.package, args.manifest, args.output)


if __name__ == "__main__":
    main()
