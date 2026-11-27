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


class OrtCtc:
    def __init__(self, package, model=None):
        self.frontend = cpu_session(package / "preprocessor.onnx")
        self.asr = cpu_session(model or package / "model.int8.onnx")
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
    def __init__(self, package):
        import sherpa_onnx

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(package / "model.int8.onnx"),
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
    parser.add_argument("--sherpa", action="store_true")
    args = parser.parse_args()
    decoder = SherpaCtc(args.package) if args.sherpa else OrtCtc(args.package)
    print(decoder.recognize(load_audio(args.audio)))


if __name__ == "__main__":
    main()
