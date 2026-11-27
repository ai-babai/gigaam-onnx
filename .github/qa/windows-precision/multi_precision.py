"""Run the two Multilingual-v2 precision packages in separate CPU ORT processes."""

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
import time
import unicodedata
import urllib.request
from pathlib import Path
from urllib.parse import quote

REPO = "ai-babai/gigaam-multilingual-v2-onnx"
PREFIX = "multilingual_v2_rnnt"
FLEURS = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_BASE = (
    f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS}/data/kk_kz"
)
WAV_MEMBER = "test/10521119105984172730.wav"
WAV_SHA = "3e3b0546be66fa6aaf532322cbfed58b3ff9c20be1b1aa5230dd7b6e174317cc"
TSV_OID = "8dabc83e97e1916f47c1b7e5150aa04bb033aacd"
WAV_REFERENCE = "Жапонияның ядролық агенттігінің хабарлауынша, зауытта радиоактивті цезий мен йод анықталған."
REQUIRED = {
    "manifest.json",
    "config.json",
    f"{PREFIX}_mel.npz",
    f"{PREFIX}_advance.onnx",
    f"{PREFIX}_e2e_tokenizer.model",
    f"{PREFIX}_raw_tokenizer.model",
}
VARIANTS = ("fp32", "fp16-storage")


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_lock(lock):
    source = lock["source_commit"]
    if not re.fullmatch(r"[a-f0-9]{40}", source):
        raise ValueError("Use a full pinned loader commit")
    if set(lock["variants"]) != set(VARIANTS):
        raise ValueError("The hosted job accepts only Multi FP32 and FP16-storage")
    for name, package in lock["variants"].items():
        if not re.fullmatch(r"[a-f0-9]{40}", package["revision"]):
            raise ValueError(f"{name}: pin an immutable model revision")
        directory = package["directory"]
        if directory and not re.fullmatch(
            r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", directory
        ):
            raise ValueError(f"{name}: expected a relative model directory")
        encoder, sidecar = package["encoder"], package["sidecar"]
        if (
            not encoder.endswith(".onnx")
            or encoder == sidecar
            or {encoder, sidecar} & REQUIRED
        ):
            raise ValueError(
                f"{name}: expected a single ONNX encoder and external-data sidecar"
            )
        files = package["sha256"]
        if set(files) != REQUIRED | {encoder, sidecar}:
            raise ValueError(
                f"{name}: expected just the encoder, sidecar and loader assets"
            )
        for filename, digest in files.items():
            if not re.fullmatch(r"[\w.-]+", filename) or not re.fullmatch(
                r"[a-f0-9]{64}", digest
            ):
                raise ValueError(f"{name}: invalid file or digest {filename!r}")
    return source


def fixture(destination):
    with urllib.request.urlopen(f"{FLEURS_BASE}/test.tsv", timeout=60) as response:
        tsv = response.read()
    oid = hashlib.sha1(f"blob {len(tsv)}\0".encode() + tsv).hexdigest()
    if oid != TSV_OID:
        raise ValueError("FLEURS reference file differs from the pinned Git blob")
    rows = csv.reader(io.StringIO(tsv.decode()), delimiter="\t", quoting=csv.QUOTE_NONE)
    found = [row[2].strip() for row in rows if row[1] == Path(WAV_MEMBER).name]
    if found != [WAV_REFERENCE]:
        raise ValueError("FLEURS reference text mismatch")
    request = urllib.request.Request(
        f"{FLEURS_BASE}/audio/test.tar.gz?download=true",
        headers={"Range": "bytes=0-16777215"},
    )
    with (
        urllib.request.urlopen(request, timeout=90) as response,
        tarfile.open(fileobj=response, mode="r|gz") as archive,
    ):
        for item in archive:
            if item.name == WAV_MEMBER and item.isfile():
                with archive.extractfile(item) as file:
                    wav = file.read()
                if hashlib.sha256(wav).hexdigest() != WAV_SHA:
                    raise ValueError("FLEURS WAV SHA-256 mismatch")
                destination.write_bytes(wav)
                return
    raise ValueError(f"Missing FLEURS WAV {WAV_MEMBER}")


def stage(spec, destination):
    destination.mkdir()
    base = f"https://huggingface.co/{REPO}/resolve/{spec['revision']}"
    if spec["directory"]:
        base += "/" + spec["directory"]
    for name, expected in spec["sha256"].items():
        print(f"Download {name}", flush=True)
        path = destination / name
        temporary = path.with_suffix(path.suffix + ".part")
        subprocess.run(
            [
                "curl.exe" if sys.platform == "win32" else "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "4",
                "--retry-all-errors",
                "--continue-at",
                "-",
                "--connect-timeout",
                "20",
                "--max-time",
                "1800",
                "--output",
                str(temporary),
                f"{base}/{quote(name)}?download=true",
            ],
            check=True,
        )
        if sha256(temporary) != expected:
            raise ValueError(f"Model file SHA-256 mismatch: {name}")
        temporary.replace(path)


def peak_working_set():
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("size", wintypes.DWORD),
            ("page_faults", wintypes.DWORD),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "peak_ws",
                    "ws",
                    "quota_peak_paged",
                    "quota_paged",
                    "quota_peak_nonpaged",
                    "quota_nonpaged",
                    "pagefile",
                    "peak_pagefile",
                    "private",
                )
            ],
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    counters = Counters()
    counters.size = ctypes.sizeof(counters)
    if not kernel.K32GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.size
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak_ws


def words(text):
    return "".join(
        ch if unicodedata.category(ch)[0] in "LN" else " " for ch in text.casefold()
    ).split()


def word_errors(reference, actual):
    expected, actual = words(reference), words(actual)
    prior = list(range(len(actual) + 1))
    for index, word in enumerate(expected, 1):
        row = [index]
        for col, result in enumerate(actual, 1):
            row.append(
                min(prior[col] + 1, row[-1] + 1, prior[col - 1] + (word != result))
            )
        prior = row
    return {"edits": prior[-1], "words": len(expected)}


def infer(args):
    import numpy
    import onnxruntime
    import sentencepiece
    import soundfile

    sys.path.insert(0, str(Path.cwd()))
    from clients.multi_client import MultilingualRNNT

    audio, rate = soundfile.read(args.wav, dtype="float32")
    if rate != 16_000 or audio.ndim != 1 or sha256(args.wav) != WAV_SHA:
        raise ValueError("Expected the locked mono 16 kHz FLEURS WAV")
    manifest = json.loads((args.package / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("encoder_filename") != args.encoder:
        raise ValueError("Published manifest does not select the locked encoder")
    started = time.monotonic()
    model = MultilingualRNNT(args.package)
    load_seconds = time.monotonic() - started
    providers = {
        "encoder": model.encoder.get_providers(),
        "advance": model.advance.get_providers(),
    }
    if any(value != ["CPUExecutionProvider"] for value in providers.values()):
        raise ValueError("Both sessions must load exclusively on CPU")
    options = model.encoder.get_session_options()
    started = time.monotonic()
    result = {}
    for hint in ("auto", "kk"):
        output = model.transcribe(audio, language=hint)
        if set(output) != {"raw", "e2e"} or not all(
            text.strip() for text in output.values()
        ):
            raise ValueError(f"Expected both nonempty heads for {hint}")
        result[hint] = {
            head: {"text": text, **word_errors(WAV_REFERENCE, text)}
            for head, text in output.items()
        }
    write_json(
        args.output,
        {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__,
            "numpy": numpy.__version__,
            "sentencepiece": sentencepiece.__version__,
            "soundfile": soundfile.__version__,
            "providers": providers,
            "encoder_threads": {
                "intra_op": options.intra_op_num_threads,
                "inter_op": options.inter_op_num_threads,
            },
            "client_sha256": sha256(Path.cwd() / "clients/multi_client.py"),
            "encoder_input_types": [node.type for node in model.encoder.get_inputs()],
            "load_seconds": load_seconds,
            "infer_seconds": time.monotonic() - started,
            "peak_working_set_bytes": peak_working_set(),
            "fleurs_revision": FLEURS,
            "wav_member": WAV_MEMBER,
            "wav_sha256": WAV_SHA,
            "reference": WAV_REFERENCE,
            "reference_sha256": hashlib.sha256(WAV_REFERENCE.encode()).hexdigest(),
            "heads_by_hint": result,
        },
    )


def run(args):
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    source = validate_lock(lock)
    changes = subprocess.check_output(
        ["git", "diff", "--name-only", source, "HEAD"], text=True
    ).splitlines()
    if not changes or any(
        name != ".github/workflows/qa-windows-multi-precision.yml"
        and not name.startswith(".github/qa/windows-precision/")
        for name in changes
    ):
        raise ValueError(
            "Only the QA workflow and evidence runner may differ from the locked source"
        )
    args.output.mkdir()
    args.scratch.mkdir()
    audio = args.scratch / "fleurs-kk.wav"
    summary = {"source_commit": source, "wav_sha256": WAV_SHA, "variants": {}}
    try:
        fixture(audio)
    except (OSError, ValueError, tarfile.TarError) as error:
        summary["fixture_error"] = str(error)
        write_json(args.output / "summary.json", summary)
        raise
    for variant in VARIANTS:
        print(f"Starting {variant}", flush=True)
        spec = lock["variants"][variant]
        folder = args.scratch / variant
        try:
            stage(spec, folder)
            receipt = args.output / f"{variant}.json"
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "infer",
                    "--package",
                    str(folder),
                    "--encoder",
                    spec["encoder"],
                    "--wav",
                    str(audio),
                    "--output",
                    str(receipt),
                ],
                check=True,
            )
            result = json.loads(receipt.read_text(encoding="utf-8"))
            result.update({"source_commit": source, "model": spec, "model_repo": REPO})
            write_json(receipt, result)
            summary["variants"][variant] = {"status": "pass", "receipt": receipt.name}
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            summary["variants"][variant] = {"status": "fail", "error": str(error)}
        write_json(args.output / "summary.json", summary)
    return int(any(item["status"] != "pass" for item in summary["variants"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    control, worker = sub.add_parser("run"), sub.add_parser("infer")
    control.add_argument("--lock", required=True, type=Path)
    control.add_argument("--output", required=True, type=Path)
    control.add_argument("--scratch", required=True, type=Path)
    worker.add_argument("--package", required=True, type=Path)
    worker.add_argument("--wav", required=True, type=Path)
    worker.add_argument("--output", required=True, type=Path)
    worker.add_argument("--encoder", required=True)
    args = parser.parse_args()
    if args.mode == "infer":
        infer(args)
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
