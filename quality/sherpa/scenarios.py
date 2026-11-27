"""Run native GigaAM offline and streaming release scenarios with Sherpa ONNX."""

import argparse
import hashlib
import json
import platform
import re
import tempfile
import traceback
import unicodedata
import wave
from itertools import zip_longest
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

RATE = 16_000
WINDOW = 320
MAX_SAMPLES = 30 * RATE
PACKET = 10_240
HEADS = ("raw", "e2e")
STREAM_PREFIX = "v4_rnnt_640ms"
OFFLINE_PREFIXES = {"v4": "v4_rnnt", "multi": "multilingual_v2_rnnt"}
WER_LIMIT = 0.30


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_file(path, expected):
    actual = sha256(path)
    require(actual == expected, f"{path}: SHA-256 {actual}, expected {expected}")
    return actual


def read_pcm(path):
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (
            RATE,
            1,
            2,
        ) or audio.getcomptype() != "NONE":
            raise ValueError(f"{path}: expected uncompressed 16 kHz mono PCM16")
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def clip_path(stored, audio_root, lang):
    posix = PurePosixPath(stored.replace("\\", "/"))
    if posix.is_absolute() or PureWindowsPath(stored).is_absolute():
        rebased = audio_root / "wav" / lang / posix.name
        return rebased if rebased.is_file() else Path(stored)
    return audio_root.joinpath(*posix.parts)


def load_clips(manifest, audio_root, pins):
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            require(row["id"] not in rows, f"Duplicate manifest ID: {row['id']}")
            rows[row["id"]] = row
    clips = {}
    for lang, fixture in pins.items():
        row = rows[fixture["id"]]
        require(row["lang"] == lang, f"{row['id']}: unexpected language")
        path = clip_path(row["path"], audio_root, lang)
        verify_file(path, fixture["wav_sha256"])
        ref = row["ref"]
        require(
            hashlib.sha256(ref.encode("utf-8")).hexdigest()
            == fixture["reference_sha256"],
            f"{row['id']}: reference string changed",
        )
        audio = read_pcm(path)
        require(WINDOW <= len(audio) <= MAX_SAMPLES, f"{row['id']}: invalid duration")
        clips[lang] = (audio, ref)
    return clips


def words(text):
    text = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, flags=re.UNICODE)


def quality(text, ref):
    expected, actual = words(ref), words(text)
    if not expected:
        raise ValueError("Empty normalized reference")
    previous = list(range(len(actual) + 1))
    for i, word in enumerate(expected, start=1):
        current = [i]
        for j, token in enumerate(actual, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token != word))
            )
        previous = current
    errors = previous[-1]
    wer = errors / len(expected)
    require(
        bool(actual) and wer <= WER_LIMIT,
        f"WER {wer:.1%} exceeds {WER_LIMIT:.0%} or empty transcript: {text!r}",
    )
    return {
        "text": text,
        "word_errors": errors,
        "reference_words": len(expected),
        "wer": wer,
    }


def rejects(call, message):
    try:
        call()
    except (ValueError, RuntimeError) as exc:
        require(
            re.search(message, str(exc), re.IGNORECASE), f"Unexpected rejection: {exc}"
        )
        return str(exc)
    raise AssertionError("Invalid input unexpectedly succeeded")


def offline_engine(sherpa, package, family, head, threads, *, tokenizer_head=None):
    prefix = OFFLINE_PREFIXES[family]
    return sherpa.OfflineRecognizer.from_gigaam(
        encoder=str(package / f"{prefix}_encoder.int8.onnx"),
        advance=str(package / f"{prefix}_advance.onnx"),
        tokenizer=str(package / f"{prefix}_{tokenizer_head or head}_tokenizer.model"),
        language="auto",
        head=head,
        num_threads=threads,
    )


def offline_stream(engine, hint):
    stream = engine.create_stream()
    if hint != "auto":
        stream.set_option("language", hint)
    return stream


def snapshot(result):
    return result.text, tuple(result.tokens)


def offline_single(engine, audio, hint="auto"):
    stream = offline_stream(engine, hint)
    stream.accept_waveform(RATE, audio)
    engine.decode_stream(stream)
    return snapshot(stream.result)


def offline_pair(sherpa, root, family, head, threads, clips):
    engine = offline_engine(sherpa, root, family, head, threads)
    hints = {"ru": "auto", **({"en": "auto"} if family == "v4" else {"kk": "kk"})}
    isolated = {
        lang: offline_single(engine, clips[lang][0], hint)
        for lang, hint in hints.items()
    }
    scores = {
        lang: quality(result[0], clips[lang][1]) for lang, result in isolated.items()
    }
    streams = {lang: offline_stream(engine, hint) for lang, hint in hints.items()}
    for half in (0, 1):
        for lang, stream in streams.items():
            data = clips[lang][0]
            middle = len(data) // 2
            stream.accept_waveform(RATE, data[:middle] if half == 0 else data[middle:])
    engine.decode_streams(list(reversed(streams.values())))
    for lang, stream in streams.items():
        require(
            snapshot(stream.result) == isolated[lang], f"{lang}: paired result changed"
        )
    if family == "multi":
        scores["kk_auto"] = quality(
            offline_single(engine, clips["kk"][0])[0], clips["kk"][1]
        )
    return {"isolated_and_paired": scores, "hints": hints}


def offline_edges(sherpa, v4, multi, head, threads, ru):
    bad_head = rejects(
        lambda: offline_engine(
            sherpa, v4, "v4", "unsupported", threads, tokenizer_head=head
        ),
        r"head|requires valid model",
    )
    engine = offline_engine(sherpa, v4, "v4", head, threads)
    bad_v4_language = rejects(
        lambda: offline_single(engine, ru, "ru"), r"does not accept a language"
    )
    for length in (0, WINDOW - 1):
        require(
            offline_single(engine, np.zeros(length, dtype=np.float32)) == ("", ()),
            f"Unexpected decoding output for {length} samples",
        )
    offline_single(engine, np.zeros(WINDOW, dtype=np.float32))
    boundary = np.pad(ru, (0, MAX_SAMPLES - len(ru)))
    require(
        bool(offline_single(engine, boundary)[0]),
        "No text for a 30 s clip containing speech",
    )
    too_long = rejects(
        lambda: offline_single(engine, np.pad(boundary, (0, 1))),
        r"up to 30 seconds",
    )
    del engine
    engine = offline_engine(sherpa, multi, "multi", head, threads)
    bad_multi_language = rejects(
        lambda: offline_single(engine, ru, "unsupported"), r"unknown GigaAM language"
    )
    return {
        "accepted_samples": [0, WINDOW - 1, WINDOW, MAX_SAMPLES],
        "unsupported_head": bad_head,
        "v4_language": bad_v4_language,
        "multi_language": bad_multi_language,
        "over_duration": too_long,
    }


def stream_tokens(package, destination, head):
    import sentencepiece

    tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=str(package / f"{STREAM_PREFIX}_{head}_tokenizer.model")
    )
    require(tokenizer.get_piece_size() == 1024, f"{head}: expected blank ID 1024")
    lines = [f"{tokenizer.id_to_piece(i)} {i}" for i in range(1024)]
    destination.write_bytes(("\n".join([*lines, "<blk> 1024"]) + "\n").encode("utf-8"))


def online_engine(sherpa, package, head, table, threads, *, joint_head=None):
    return sherpa.OnlineRecognizer.from_transducer(
        tokens=str(table),
        encoder=str(package / f"{STREAM_PREFIX}_stream_step.int8.onnx"),
        decoder=str(package / f"{STREAM_PREFIX}_{head}_decoder.onnx"),
        joiner=str(package / f"{STREAM_PREFIX}_{joint_head or head}_joint.onnx"),
        model_type="gigaam_v4",
        provider="cpu",
        num_threads=threads,
        enable_endpoint_detection=False,
    )


def reject_stream_head(sherpa, package, tables, threads, pair):
    head, token_head, joint_head = pair
    return {
        "rejection": rejects(
            lambda: online_engine(
                sherpa,
                package,
                head,
                tables[token_head],
                threads,
                joint_head=joint_head,
            ),
            r"must use the same head",
        )
    }


def drain(engine, stream):
    for count in range(1024):
        if not engine.is_ready(stream):
            return count
        engine.decode_stream(stream)
    raise AssertionError("Stream kept reporting readiness after 1024 decode calls")


def packets(audio):
    for start in range(0, len(audio), PACKET):
        yield audio[start : start + PACKET]


def online_result(engine, stream):
    return snapshot(engine.get_result_all(stream))


def accept_packet(engine, stream, audio):
    stream.accept_waveform(RATE, audio)
    drain(engine, stream)
    return online_result(engine, stream)


def finish_stream(engine, stream):
    stream.input_finished()
    drain(engine, stream)
    result = online_result(engine, stream)
    require(drain(engine, stream) == 0, "Second EOF drain decoded another packet")
    require(
        online_result(engine, stream) == result,
        "Second EOF drain changed text or tokens",
    )
    return result


def online_single(engine, stream, audio):
    first_partial = None
    for index, data in enumerate(packets(audio), start=1):
        text, _ = accept_packet(engine, stream, data)
        if text and first_partial is None:
            first_partial = index
    require(first_partial is not None, "No partial transcript before EOF")
    return finish_stream(engine, stream), first_partial


def empty_stream(engine):
    stream = engine.create_stream()
    require(
        online_result(engine, stream) == ("", ()),
        "Fresh stream retained text or tokens",
    )
    return stream


def interleaved_results(engine, clips):
    streams = {lang: empty_stream(engine) for lang in ("ru", "en")}
    for ru, en in zip_longest(packets(clips["ru"][0]), packets(clips["en"][0])):
        for lang, data in (("ru", ru), ("en", en)):
            if data is not None:
                accept_packet(engine, streams[lang], data)
    return {lang: finish_stream(engine, stream) for lang, stream in streams.items()}


def reset_result(engine, before, after):
    stream = empty_stream(engine)
    for data in packets(before):
        if accept_packet(engine, stream, data)[0]:
            break
    require(
        bool(online_result(engine, stream)[0]),
        "Reset was never exercised on live speech",
    )
    accept_packet(engine, stream, before[:159])
    engine.reset(stream)
    require(online_result(engine, stream) == ("", ()), "Reset retained text or tokens")
    return online_single(engine, stream, after)[0]


def online_lifecycle(sherpa, package, head, tables, threads, clips):
    engine = online_engine(sherpa, package, head, tables[head], threads)
    isolated, scores = {}, {}
    for lang in ("ru", "en"):
        audio, ref = clips[lang]
        isolated[lang], first_partial = online_single(
            engine, empty_stream(engine), audio
        )
        scores[lang] = {
            **quality(isolated[lang][0], ref),
            "first_partial_packet": first_partial,
        }
    for lang, result in interleaved_results(engine, clips).items():
        require(result == isolated[lang], f"{lang}: interleaving changed output")
    require(
        reset_result(engine, clips["ru"][0], clips["en"][0]) == isolated["en"],
        "Stream reuse after reset changed output",
    )
    return {"isolated_interleaved_and_reset": scores, "second_drain_is_idle": True}


def mono_contract(directory):
    path = directory / "stereo.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(WINDOW * 4))
    return {"stereo_rejection": rejects(lambda: read_pcm(path), r"mono PCM16")}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-package", type=Path, required=True)
    parser.add_argument("--multi-package", type=Path, required=True)
    parser.add_argument("--stream-package", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument(
        "--stream-tokens",
        type=Path,
        help="Directory with tokens.raw.txt and tokens.e2e.txt",
    )
    parser.add_argument("--head", choices=(*HEADS, "both"), default="both")
    parser.add_argument(
        "--backend", choices=("offline", "stream", "both"), default="both"
    )
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def write_report(output, report):
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def run_case(output, report, name, check, *arguments):
    print(f"RUN {name}", flush=True)
    try:
        case = {"passed": True, "details": check(*arguments)}
    except Exception as exc:
        case = {"passed": False, "error": str(exc), "traceback": traceback.format_exc()}
    report["cases"][name] = case
    write_report(output, report)
    print(f"{'PASS' if case['passed'] else 'FAIL'} {name}", flush=True)
    return case["passed"]


def prepare(args, pins):
    import sherpa_onnx
    from sherpa_onnx.lib import _sherpa_onnx

    packages = {}
    if args.backend != "stream":
        require(
            hasattr(sherpa_onnx.OfflineRecognizer, "from_gigaam"),
            "Offline GigaAM API missing",
        )
        packages.update(v4=args.v4_package, multi=args.multi_package)
    if args.backend != "offline":
        packages["stream"] = args.stream_package
    details = {
        kind: {
            file: verify_file(package / file, digest)
            for file, digest in pins["models"][kind].items()
        }
        for kind, package in packages.items()
    }
    clips = load_clips(
        args.manifest, args.audio_root or args.manifest.parent, pins["audio"]
    )
    native = {
        "sherpa": sherpa_onnx.version,
        "onnxruntime": sherpa_onnx.onnxruntime_version,
        "module_sha256": sha256(Path(_sherpa_onnx.__file__)),
    }
    return sherpa_onnx, clips, details, native


def run_offline(args, report, sherpa, clips, heads):
    for family, root in (("v4", args.v4_package), ("multi", args.multi_package)):
        for head in heads:
            run_case(
                args.output,
                report,
                f"offline_{family}_{head}_paired",
                offline_pair,
                sherpa,
                root,
                family,
                head,
                args.threads,
                clips,
            )
    run_case(
        args.output,
        report,
        "offline_validation_and_duration",
        offline_edges,
        sherpa,
        args.v4_package,
        args.multi_package,
        heads[-1],
        args.threads,
        clips["ru"][0],
    )


def check_tables(args, pins, tables):
    if args.stream_tokens is None:
        for head, path in tables.items():
            stream_tokens(args.stream_package, path, head)
    return {
        head: verify_file(path, pins["stream_token_tables"][head])
        for head, path in tables.items()
    }


def run_stream(args, report, sherpa, clips, heads, directory, pins):
    tables = {
        head: (args.stream_tokens or directory) / f"tokens.{head}.txt" for head in HEADS
    }
    if not run_case(
        args.output, report, "stream_token_tables", check_tables, args, pins, tables
    ):
        return
    for head in heads:
        run_case(
            args.output,
            report,
            f"stream_{head}_lifecycle",
            online_lifecycle,
            sherpa,
            args.stream_package,
            head,
            tables,
            args.threads,
            clips,
        )
    for part, pair in (
        ("tokens", ("raw", "e2e", "raw")),
        ("joiner", ("e2e", "e2e", "raw")),
    ):
        run_case(
            args.output,
            report,
            f"stream_rejects_mismatched_{part}",
            reject_stream_head,
            sherpa,
            args.stream_package,
            tables,
            args.threads,
            pair,
        )


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pins_path = Path(__file__).with_name("fixtures.json")
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "threads": args.threads,
        "head": args.head,
        "backend": args.backend,
        "fixtures_sha256": sha256(pins_path),
        "fixtures": pins["audio"],
        "cases": {},
    }
    try:
        sherpa, clips, details, report["native"] = prepare(args, pins)
    except Exception as exc:
        report["cases"]["pinned_inputs_and_native_import"] = {
            "passed": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        report["passed"] = False
        write_report(args.output, report)
        print(f"FAIL pinned_inputs_and_native_import: {exc}", flush=True)
        return 1
    report["cases"]["pinned_inputs_and_native_import"] = {
        "passed": True,
        "details": details,
    }
    heads = HEADS if args.head == "both" else (args.head,)
    with tempfile.TemporaryDirectory(
        prefix="sherpa-scenarios-", dir=args.output.parent
    ) as temp:
        directory = Path(temp)
        run_case(args.output, report, "mono_pcm16", mono_contract, directory)
        if args.backend != "stream":
            run_offline(args, report, sherpa, clips, heads)
        if args.backend != "offline":
            run_stream(args, report, sherpa, clips, heads, directory, pins)
    report["passed"] = all(case["passed"] for case in report["cases"].values())
    write_report(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
