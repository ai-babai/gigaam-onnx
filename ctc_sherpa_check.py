"""Verify the published graph with sherpa-onnx and compare with the measured ORT run."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from ctc_benchmark import LANGUAGES, manifest_rows
from ctc_transcribe import SherpaCtc, load_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("ort_results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = list(manifest_rows(args.manifest))
    by_clip = {}
    for line in args.ort_results.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["lang"] in LANGUAGES:
            by_clip[row["lang"], row["id"]] = row["hyp_int8"]
    keys = {(row["lang"], row["id"]) for row in rows}
    if not keys or keys != set(by_clip):
        raise ValueError("The manifest and ORT predictions must contain the same RU/EN clips")
    recognizer = SherpaCtc(args.package)
    staging = args.output.with_suffix(".partial.jsonl")
    exact = 0
    with staging.open("w", encoding="utf-8") as output:
        for row in rows:
            samples = load_audio(row["path"])
            start = perf_counter()
            text = recognizer.recognize(samples)
            seconds = perf_counter() - start
            matches_ort = text == by_clip[row["lang"], row["id"]]
            exact += matches_ort
            output.write(json.dumps({
                "lang": row["lang"], "id": row["id"], "hyp_sherpa": text,
                "matches_ort": matches_ort, "sec_sherpa": seconds,
            }, ensure_ascii=False) + "\n")
    staging.replace(args.output)
    print(f"Sherpa vs ORT exact matches: {exact}/{len(by_clip)}")


if __name__ == "__main__":
    main()
