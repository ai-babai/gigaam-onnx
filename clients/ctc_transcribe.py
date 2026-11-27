"""Transcribe a short audio file with a GigaAM-v4 CTC e2e ONNX package."""

import argparse
import os
import subprocess
import wave
from itertools import groupby
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "3"

import numpy as np

SAMPLE_RATE = 16_000
MAX_SAMPLES = 30 * SAMPLE_RATE


def read_pcm16_wav(path):
    """Read mono 16 kHz PCM16 directly when the WAV needs no conversion."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as source, wave.open(source) as wav:
            if wav.getparams()[:3] != (1, 2, SAMPLE_RATE):
                return None
            frames = wav.getnframes()
            if frames > MAX_SAMPLES:
                return None
            pcm = wav.readframes(frames)
            if len(pcm) != 2 * frames or source.read(1):
                return None
    except (EOFError, OSError, RuntimeError, wave.Error):
        return None
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) * (1 / 32768)


def ffmpeg_samples(path):
    pcm = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-threads", "3",
            "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return np.frombuffer(pcm, dtype="<f4").copy()


def load_audio(path):
    local = Path(path)
    samples = read_pcm16_wav(local) if local.suffix.lower() == ".wav" else None
    if samples is None:
        samples = ffmpeg_samples(path)
    if not 320 <= samples.size <= MAX_SAMPLES:
        raise ValueError("Expected 20 ms to 30 s of audio. Split long recordings at pauses.")
    return samples


def cpu_session(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def select_model(package):
    default = package / "model.int8.onnx"
    if default.is_file():
        return default
    names = ("model.onnx", "model.fp32.onnx", "model.fp16.onnx", "model.int4.onnx")
    candidates = [package / name for name in names if (package / name).is_file()]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError("Multiple CTC graphs in package. Select one with --model.")
    raise FileNotFoundError(default)


class OrtCtc:
    def __init__(self, package, model=None):
        self.frontend = cpu_session(package / "preprocessor.onnx")
        self.asr = cpu_session(model or select_model(package))
        self.blank = int(self.asr.get_modelmeta().custom_metadata_map["blank_id"])
        self.tokens = {}
        for line in (package / "tokens.txt").read_text(encoding="utf-8").splitlines():
            token, index = line.rsplit(" ", 1)
            self.tokens[int(index)] = token.replace("▁", " ")

    def recognize(self, samples):
        features, lengths = self.frontend.run(
            None,
            {"waveforms": samples[None, :], "waveform_lengths": np.array([samples.size], dtype=np.int64)},
        )
        log_probs, encoded_lengths = self.asr.run(
            None, {"features": features, "feature_lengths": lengths},
        )
        labels = log_probs[0, : int(encoded_lengths[0])].argmax(axis=-1)
        return "".join(self.tokens[label] for label, _ in groupby(labels) if label != self.blank)


class SherpaCtc:
    def __init__(self, package, model=None, *, provider="cpu"):
        import sherpa_onnx

        model = model or select_model(package)
        for path in (model, package / "tokens.txt"):
            if not path.is_file():
                raise FileNotFoundError(path)
        tokens = (package / "tokens.txt").read_text(encoding="utf-8").splitlines()
        if not any(line.split()[:1] == ["<blk>"] for line in tokens):
            raise ValueError("tokens.txt must contain the CTC blank token <blk>")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(model),
            tokens=str(package / "tokens.txt"),
            provider=provider,
            num_threads=3,
            sample_rate=SAMPLE_RATE,
            feature_dim=64,
        )

    def recognize(self, samples):
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text


def parse_sherpa_provider(value):
    provider, separator, config = value.partition(":")
    if provider not in ("cpu", "coreml", "cuda") or (separator and not config.strip()):
        raise argparse.ArgumentTypeError("expected cpu, coreml or cuda, optionally followed by :CONFIG")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model", type=Path, help="Path to the CTC ONNX graph")
    parser.add_argument("--sherpa", action="store_true")
    parser.add_argument(
        "--sherpa-provider", type=parse_sherpa_provider, metavar="PROVIDER[:CONFIG]",
        help="Sherpa execution provider: cpu (default), coreml or cuda. Requires --sherpa.",
    )
    args = parser.parse_args()
    if args.sherpa:
        decoder = SherpaCtc(args.package, args.model, provider=args.sherpa_provider or "cpu")
    else:
        if args.sherpa_provider is not None:
            parser.error("--sherpa-provider requires --sherpa")
        decoder = OrtCtc(args.package, args.model)
    print(decoder.recognize(load_audio(args.audio)))


if __name__ == "__main__":
    main()
