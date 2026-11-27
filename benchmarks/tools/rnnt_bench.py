"""Check full-context parity and benchmark v4 RNN-T on paired clips."""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

from clients.rnnt_transcribe import GigaAMRNNT, read_audio

import numpy as np
import torch

from conversion.rnnt_export import PREFIX, load_torch

LANGUAGES = ("ru", "en")


@torch.inference_mode()
def torch_text(model, audio):
    samples = torch.from_numpy(audio)[None, :]
    lengths = torch.tensor([audio.size])
    encoded, lengths = model(samples, lengths)
    return {head: items[0] for head, items in model.decode(encoded, lengths).items()}


def timed(call, *args):
    started = perf_counter()
    text = call(*args)["e2e"]
    return text, perf_counter() - started


def check_parity(reference, package, encoder, work, records):
    export = GigaAMRNNT(package, encoder=encoder)
    checks = []
    for language in LANGUAGES:
        row = next(row for row in records if row["lang"] == language)
        audio = read_audio(row["path"], export.sample_rate)
        with torch.inference_mode():
            expected_features, _ = reference.preprocessor(
                torch.from_numpy(audio)[None, :], torch.tensor([audio.size])
            )
        error = float(
            np.max(np.abs(export.features(audio) - expected_features.numpy()))
        )
        native = torch_text(reference, audio)
        onnx = export.transcribe(audio)
        matches = {head: native[head] == onnx[head] for head in native}
        checks.append(
            {
                "lang": language,
                "id": row["id"],
                "max_abs_log_mel_error": error,
                "exact_match": matches,
                "torch": native,
                "onnx_fp32": onnx,
            }
        )
        print(
            f"Full-context FP32 parity {language}: {matches}, log-mel max error {error:.6g}",
            flush=True,
        )
    (work / "parity_fp_onnx.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n"
    )
    if not all(
        all(row["exact_match"].values()) and row["max_abs_log_mel_error"] < 0.003
        for row in checks
    ):
        raise RuntimeError(
            "The packaged frontend or full-context FP32 graph changed the reference output"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    records = [
        row
        for line in args.manifest.read_text().splitlines()
        if line.strip() and (row := json.loads(line))["lang"] in LANGUAGES
    ]
    if {row["lang"] for row in records} != set(LANGUAGES):
        raise ValueError(f"Expected evaluation clips for {LANGUAGES}")
    if len({(row["lang"], row["id"]) for row in records}) != len(records):
        raise ValueError("Duplicate language/id in the manifest")

    package = args.work / "package"
    reference = load_torch(args.source)
    check_parity(
        reference,
        package,
        args.work / "fp" / f"{PREFIX}_encoder.onnx",
        args.work,
        records,
    )
    gc.collect()
    int8 = GigaAMRNNT(package)
    warm = sorted(records, key=lambda row: row["audio_seconds"])[len(records) // 2]
    audio = read_audio(warm["path"], int8.sample_rate)
    torch_text(reference, audio)
    int8.transcribe(audio)

    pending = args.work / "results.pending.jsonl"
    with pending.open("w", encoding="utf-8") as results:
        for index, row in enumerate(records):
            audio = read_audio(row["path"], int8.sample_rate)
            audio_seconds = audio.size / int8.sample_rate
            if abs(audio_seconds - row["audio_seconds"]) > 0.002:
                raise ValueError(f"Duration differs from the manifest for {row['id']}")
            if index % 2:
                hyp_int8, sec_int8 = timed(int8.transcribe, audio)
                hyp_fp, sec_fp = timed(torch_text, reference, audio)
            else:
                hyp_fp, sec_fp = timed(torch_text, reference, audio)
                hyp_int8, sec_int8 = timed(int8.transcribe, audio)
            record = {
                "lang": row["lang"],
                "id": row["id"],
                "ref": row["ref"],
                "hyp_fp": hyp_fp,
                "hyp_int8": hyp_int8,
                "sec_fp": sec_fp,
                "sec_int8": sec_int8,
                "audio_seconds": audio_seconds,
            }
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()
            print(
                f"{index + 1}/{len(records)} {row['lang']}: {sec_fp:.3f}s FP32, {sec_int8:.3f}s INT8",
                flush=True,
            )
    pending.replace(args.work / "results.jsonl")


if __name__ == "__main__":
    main()
