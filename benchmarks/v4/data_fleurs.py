#!/usr/bin/env python3
"""Build the 32-clip Russian and English FLEURS test sample for GigaAM-v4."""

import argparse
import csv
import hashlib
import io
import json
import subprocess
import tarfile
import wave
from pathlib import Path

DEFAULT_ROOT = Path.cwd() / "fleurs-work"
DATASET = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
HUB = f"https://huggingface.co/datasets/{DATASET}"
LICENSE = "CC-BY-4.0"
SEED = 20261127
COUNT = 16
MIN_SECONDS, MAX_SECONDS = 2, 15
SAMPLE_RATE = 16_000
# HF Git blob SHA-1 for test.tsv and LFS SHA-256 for the original audio archive.
SOURCES = {
    "ru": (
        "ru_ru", "1b1b37112dcbc6089433084255785f03f9fce445",
        "8a0d6a0d23c3421f50c575bcf65d875cd19dadae2f7cabb415024f81b65178b1",
    ),
    "en": (
        "en_us", "bcd7e3e4ecb8ad2a4d9754fc6be14dd6d3364b54",
        "d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356",
    ),
}


def fetch(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(
        [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "5", "--retry-max-time", "300",
            "--connect-timeout", "15", "--max-time", "1800",
            "--continue-at", "-", "--output", str(temporary), url,
        ],
        check=True,
    )
    temporary.replace(destination)


def random_order(config: str, filename: str) -> bytes:
    return hashlib.sha256(f"{SEED}:{config}:{filename}".encode()).digest()


def decode_pcm(payload: bytes) -> bytes:
    return subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-threads", "1",
            "-filter_threads", "1", "-i", "pipe:0", "-map", "0:a:0",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
        ],
        check=True,
        input=payload,
        stdout=subprocess.PIPE,
    ).stdout


def sample_language(lang: str, output: Path, cache: Path) -> tuple[list[dict], dict]:
    config, tsv_oid, archive_sha256 = SOURCES[lang]
    source_url = f"{HUB}/resolve/{REVISION}/data/{config}"
    tsv_url = f"{source_url}/test.tsv"
    archive_url = f"{source_url}/audio/test.tar.gz"
    tsv = cache / f"{config}.test.tsv"
    archive = cache / f"{config}.test.tar.gz"
    fetch(tsv_url, tsv)
    fetch(f"{archive_url}?download=true", archive)
    tsv_bytes = tsv.read_bytes()
    actual_oid = hashlib.sha1(f"blob {len(tsv_bytes)}\0".encode() + tsv_bytes).hexdigest()
    with archive.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_oid != tsv_oid:
        raise ValueError(f"Unexpected Git blob SHA-1: {tsv}")
    if actual_sha256 != archive_sha256:
        raise ValueError(f"Unexpected SHA-256: {archive}")

    texts = {}
    rows = csv.reader(io.StringIO(tsv_bytes.decode()), delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in rows:
        _, filename, raw_ref, normalized_ref, *_ = row
        if raw_ref.strip() and normalized_ref.strip() and not any(ch.isdecimal() for ch in raw_ref):
            texts[filename] = (raw_ref.strip(), " ".join(normalized_ref.casefold().split()))

    # Keep the first COUNT eligible distinct texts by seeded hash without buffering the archive.
    selected = {}
    seen = set()
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            filename = Path(member.name).name
            if not member.isfile() or filename not in texts or filename in seen:
                continue
            seen.add(filename)
            reference, key = texts[filename]
            rank = (random_order(config, filename), filename)
            if key in selected and selected[key][0] <= rank:
                continue
            if len(selected) == COUNT and rank >= max(item[0] for item in selected.values()):
                continue
            with source.extractfile(member) as clip:
                pcm = decode_pcm(clip.read())
            frames = len(pcm) // 2
            if not MIN_SECONDS * SAMPLE_RATE <= frames <= MAX_SECONDS * SAMPLE_RATE:
                continue
            selected[key] = (rank, frames, reference, pcm)
            if len(selected) > COUNT:
                del selected[max(selected, key=lambda item: selected[item][0])]

    if len(selected) != COUNT:
        raise ValueError(f"{config}: expected {COUNT} clips, got {len(selected)}")

    audio_dir = output / "wav" / lang
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for (_, filename), frames, reference, pcm in sorted(selected.values()):
        destination = audio_dir / filename
        temporary = destination.with_suffix(".wav.part")
        with wave.open(str(temporary), "wb") as wav:
            wav.setparams((1, 2, SAMPLE_RATE, frames, "NONE", "not compressed"))
            wav.writeframes(pcm)
        temporary.replace(destination)
        manifest.append({
            "lang": lang,
            "id": f"{config}:{Path(filename).stem}",
            "ref": reference,
            "path": str(destination),
            "audio_seconds": frames / SAMPLE_RATE,
            "source": tsv_url,
        })

    print(f"{lang}: {len(manifest)} clips, {sum(row['audio_seconds'] for row in manifest):.1f} s", flush=True)
    return manifest, {
        "config": config,
        "split": "test",
        "tsv_url": tsv_url,
        "tsv_git_blob_sha1": tsv_oid,
        "archive_url": archive_url,
        "archive_sha256": archive_sha256,
        "transcript_eligible_filenames": len(texts),
        "transcript_eligible_filenames_present_in_archive": len(seen),
    }


def write_atomic(destination: Path, text: str) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "eval")
    parser.add_argument("--cache", type=Path, default=DEFAULT_ROOT / "cache" / "fleurs-sample")
    args = parser.parse_args()
    output, cache = args.output.resolve(), args.cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    rows, sources = [], {}
    for lang in SOURCES:
        clips, sources[lang] = sample_language(lang, output, cache)
        rows.extend(clips)

    write_atomic(output / "provenance.json", json.dumps({
        "dataset": DATASET,
        "revision": REVISION,
        "dataset_card": f"{HUB}/blob/{REVISION}/README.md",
        "license": LICENSE,
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "seed": SEED,
        "clips_per_language": COUNT,
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        "sample_order": "Ascending SHA-256(seed:config:filename), with filename tie-break.",
        "eligibility": {
            "split": "test",
            "wave": "Decoded with FFmpeg to PCM signed 16-bit mono at 16000 Hz. No trimming or loudness filtering.",
            "duration_seconds_inclusive": [MIN_SECONDS, MAX_SECONDS],
            "reference": "Non-empty original human reference (TSV column 3), without decimal digits.",
            "unique_reference": "First eligible clip per case-folded, space-normalized TSV column 4.",
            "filename": "Matched by WAV basename between the original TSV and tar.gz.",
        },
        "sources": sources,
    }, indent=2, ensure_ascii=False) + "\n")
    write_atomic(output / "manifest.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(output / "manifest.jsonl", flush=True)


if __name__ == "__main__":
    main()
