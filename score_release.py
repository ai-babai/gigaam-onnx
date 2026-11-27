"""Paired WER and throughput on the exact same random audio clips.

Two independent seeded paired clip bootstraps. RTF = wall seconds / audio seconds.
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from gigaam.utils import normalize_raw_text


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cleaned(text, lang):
    # NFC is material for stressed/combined characters and Latin apostrophes.
    import unicodedata
    return normalize_raw_text(unicodedata.normalize("NFC", text), language=lang)


def word_errors(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, source_word in enumerate(reference, start=1):
        current = [i]
        for j, word in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (source_word != word)))
        previous = current
    return previous[-1]


def bootstrap(words, deltas):
    intervals = []
    for seed in [20261127, 20261128]:
        rng = np.random.default_rng(seed)
        draws = rng.integers(len(words), size=(2000, len(words)))
        proportions = 100.0 * deltas[draws].sum(axis=1) / words[draws].sum(axis=1)
        bounds = np.quantile(proportions, [0.025, 0.975]).tolist()
        intervals.append({"seed": seed, "resamples": len(draws), "lower_pp": bounds[0], "upper_pp": bounds[1]})
    return intervals


def score_language(rows, lang):
    refs = [cleaned(row["ref"], lang) for row in rows]
    counts = np.array([len(ref.split()) for ref in refs])
    if counts.min() == 0:
        raise ValueError(f"Empty normalized reference in {lang}")
    total_audio = sum(row["audio_seconds"] for row in rows)
    score = {"clips": len(rows), "reference_words": int(counts.sum()), "audio_seconds": total_audio}
    errors = {}
    for name in ("fp", "int8"):
        texts = [cleaned(row["hyp_" + name], lang) for row in rows]
        errors[name] = np.array([word_errors(r.split(), h.split()) for r, h in zip(refs, texts)])
        runtimes = [row["sec_" + name] for row in rows]
        if any(not math.isfinite(t) or t <= 0 for t in runtimes):
            raise ValueError(f"Invalid measured runtime for {lang}/{name}")
        wall = sum(runtimes)
        score[name] = {
            "wer_pct": float(100 * errors[name].sum() / counts.sum()),
            "word_errors": int(errors[name].sum()),
            "empty_outputs": sum(not txt for txt in texts),
            "wall_seconds": wall,
            "rtf": wall / total_audio,
            "times_realtime": total_audio / wall,
            "median_seconds_per_clip": statistics.median(runtimes),
        }
    score["delta_wer_pp"] = score["int8"]["wer_pct"] - score["fp"]["wer_pct"]
    score["speed_vs_reference"] = score["fp"]["wall_seconds"] / score["int8"]["wall_seconds"]
    score["verbatim_matches_to_reference_model"] = sum(r["hyp_fp"].strip() == r["hyp_int8"].strip() for r in rows)
    score["normalized_matches_to_reference_model"] = sum(cleaned(r["hyp_fp"], lang) == cleaned(r["hyp_int8"], lang) for r in rows)
    score["delta_wer_paired_bootstrap_ci95"] = bootstrap(counts, errors["int8"] - errors["fp"])
    return score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("predictions", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--languages", nargs="+", required=True)
    args = p.parse_args()
    manifest = {(r["lang"], str(r["id"])): r for r in read_jsonl(args.manifest) if r["lang"] in args.languages}
    by_lang = defaultdict(list)
    seen = Counter()
    for row in read_jsonl(args.predictions):
        key = row["lang"], str(row["id"])
        source = manifest[key]
        if row["ref"] != source["ref"] or abs(row["audio_seconds"] - source["audio_seconds"]) > 0.002:
            raise ValueError(f"Manifest mismatch at {key}")
        seen[key] += 1
        by_lang[row["lang"]].append(row)
    if any(c != 1 for c in seen.values()) or set(seen) != set(manifest):
        raise ValueError(f"Missing or duplicate prediction keys: expected {len(manifest)}, got {len(seen)}")
    result = {lang: score_language(by_lang[lang], lang) for lang in args.languages}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({l: [round(r["fp"]["wer_pct"], 2), round(r["int8"]["wer_pct"], 2),
                          round(r["int8"]["rtf"], 4)] for l, r in result.items()}))


if __name__ == "__main__":
    main()
