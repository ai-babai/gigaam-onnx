"""Transcribe a short audio file with a GigaAM-v4 CTC e2e ONNX package."""

import argparse
import os
import subprocess
from itertools import groupby
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "3"

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16_000
MAX_SAMPLES = 30 * SAMPLE_RATE


def load_audio(path):
    pcm = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-threads", "3",
            "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    samples = np.frombuffer(pcm, dtype="<f4")
    if not 320 <= samples.size <= MAX_SAMPLES:
        raise ValueError("Expected 20 ms to 30 s of audio. Split long recordings at pauses.")
    return samples.copy()


def cpu_session(path):
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
    def __init__(self, package, model=None):
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
            provider="cpu",
            num_threads=3,
            sample_rate=SAMPLE_RATE,
            feature_dim=64,
        )

    def recognize(self, samples):
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model", type=Path, help="Path to the CTC ONNX graph")
    parser.add_argument("--sherpa", action="store_true")
    args = parser.parse_args()
    decoder_type = SherpaCtc if args.sherpa else OrtCtc
    decoder = decoder_type(args.package, args.model)
    print(decoder.recognize(load_audio(args.audio)))


if __name__ == "__main__":
    main()
