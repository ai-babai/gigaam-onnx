"""Evaluate PyTorch FP32 and the packaged ONNX INT8 on the same manifest."""

import argparse
from collections import defaultdict
from functools import partial
import json
from pathlib import Path
from time import perf_counter
import unicodedata

from multi_client import MultilingualRNNT, read_audio

import numpy as np
import torch

from gigaam.utils import normalize_raw_text
from multi_export import load_reference

REQUIRED_LANGUAGES = {"ru", "en", "kk", "ky", "uz"}


def distance(left, right):
    row = list(range(len(right) + 1))
    for i, word in enumerate(left, 1):
        next_row = [i]
        for j, other in enumerate(right, 1):
            next_row.append(
                min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (word != other))
            )
        row = next_row
    return row[-1]


def words(text, language):
    return normalize_raw_text(unicodedata.normalize("NFC", text), language).split()


def summarize(rows):
    totals = defaultdict(lambda: defaultdict(float))
    for row in rows:
        language = row["lang"]
        ref = words(row["ref"], language)
        fp = words(row["hyp_fp"], language)
        int8 = words(row["hyp_int8"], language)
        for key in (language, "all"):
            acc = totals[key]
            acc["clips"] += 1
            acc["words"] += len(ref)
            acc["fp_errors"] += distance(ref, fp)
            acc["int8_errors"] += distance(ref, int8)
            acc["int8_vs_fp_errors"] += distance(fp, int8)
            acc["fp_words"] += len(fp)
            acc["exact_e2e_matches"] += row["hyp_fp"] == row["hyp_int8"]
            for field in ("audio_seconds", "sec_fp", "sec_int8"):
                acc[field] += row[field]
    return {
        lang: {
            **total,
            "wer_fp_pct": 100 * total["fp_errors"] / total["words"],
            "wer_int8_pct": 100 * total["int8_errors"] / total["words"],
            "pairwise_wer_pct": 100 * total["int8_vs_fp_errors"] / total["fp_words"]
            if total["fp_words"]
            else None,
            "rtfx_fp": total["audio_seconds"] / total["sec_fp"],
            "rtfx_int8": total["audio_seconds"] / total["sec_int8"],
            "int8_speedup": total["sec_fp"] / total["sec_int8"],
        }
        for lang, total in totals.items()
    }


@torch.inference_mode()
def pytorch_text(model, audio, language):
    wav = torch.from_numpy(audio)[None, :]
    lengths = torch.tensor([audio.size])
    encoded, encoded_lengths = model(wav, lengths, language=language)
    return {
        head: values[0]
        for head, values in model.decode(encoded, encoded_lengths).items()
    }


def timed(call, *args):
    start = perf_counter()
    text = call(*args)
    return text, perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--fp-onnx",
        type=Path,
        help="Compare both ONNX encoders using the same FP32 advance session",
    )
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    records = [
        json.loads(line) for line in args.manifest.read_text().splitlines() if line
    ]
    languages = {row["lang"] for row in records}
    if not REQUIRED_LANGUAGES <= languages:
        raise ValueError(
            f"Evaluation manifest is missing {sorted(REQUIRED_LANGUAGES - languages)}"
        )
    if len({(row["lang"], row["id"]) for row in records}) != len(records):
        raise ValueError("Duplicate language/id in evaluation manifest")

    reference = load_reference(args.work / "source")
    package = args.work / "package"
    print("Loading the INT8 ONNX encoder", flush=True)
    quantized = MultilingualRNNT(package)
    runners = {
        "torch": partial(pytorch_text, reference),
        "int8": quantized.transcribe,
    }
    if args.fp_onnx:
        print("Loading the FP32 ONNX encoder with external weights", flush=True)
        fp_onnx = MultilingualRNNT(
            package, encoder=args.fp_onnx, advance=quantized.advance
        )
        runners["fp_onnx"] = fp_onnx.transcribe
    order = list(runners)
    warm = sorted(records, key=lambda row: row["audio_seconds"])[len(records) // 2]
    audio = read_audio(warm["path"], quantized.sample_rate)
    for run in runners.values():
        run(audio, warm["lang"])
    with torch.inference_mode():
        want, _ = reference.preprocessor(
            torch.from_numpy(audio)[None, :], torch.tensor([audio.size])
        )
    actual = quantized.features(audio)
    feature_error = float(np.max(np.abs(want.numpy() - actual)))
    print(f"Warmup features max absolute error: {feature_error:.6g}", flush=True)
    if feature_error > 0.003:
        raise RuntimeError("Package log-mel frontend differs from the reference")

    rows = []
    ort_rows = []
    parity = []
    pending = args.work / "results.pending.jsonl"
    with pending.open("w", encoding="utf-8") as result:
        for index, row in enumerate(records):
            language = row["lang"]
            audio = read_audio(row["path"], quantized.sample_rate)
            audio_seconds = audio.size / quantized.sample_rate
            if abs(audio_seconds - row["audio_seconds"]) > 0.002:
                raise ValueError(f"Duration differs from the manifest for {row['id']}")
            start = index % len(order)
            runs = {
                name: timed(runners[name], audio, language)
                for name in order[start:] + order[:start]
            }
            hyp_fp, sec_fp = runs["torch"]
            hyp_int8, sec_int8 = runs["int8"]
            record = {
                "lang": language,
                "id": row["id"],
                "ref": row["ref"],
                "hyp_fp": hyp_fp["e2e"],
                "hyp_int8": hyp_int8["e2e"],
                "sec_fp": sec_fp,
                "sec_int8": sec_int8,
                "audio_seconds": audio_seconds,
            }
            result.write(json.dumps(record, ensure_ascii=False) + "\n")
            result.flush()
            rows.append(record)
            if args.fp_onnx:
                hyp_ort, sec_ort = runs["fp_onnx"]
                ort_rows.append({**record, "hyp_fp": hyp_ort["e2e"], "sec_fp": sec_ort})
                parity.append(
                    {
                        "lang": language,
                        "id": row["id"],
                        "exact_match": {
                            head: hyp_fp[head] == hyp_ort[head] for head in hyp_fp
                        },
                        "torch": hyp_fp,
                        "onnx_fp32": hyp_ort,
                    }
                )
            times = " ".join(f"{name}={secs:.3f}s" for name, (_, secs) in runs.items())
            print(
                f"{index + 1}/{len(records)} {language} {row['id']} {times}", flush=True
            )
    pending.replace(args.work / "results.jsonl")
    summary = summarize(rows)
    (args.work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    if args.fp_onnx:
        (args.work / "results_ort.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ort_rows)
        )
        (args.work / "summary_ort.json").write_text(
            json.dumps(summarize(ort_rows), indent=2) + "\n"
        )
        (args.work / "parity_fp_onnx.json").write_text(
            json.dumps(parity, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
