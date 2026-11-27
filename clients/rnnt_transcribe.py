"""Transcribe a clip with the packaged GigaAM-v4 RNN-T ONNX models."""

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ["OMP_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "3"
os.environ["VECLIB_MAXIMUM_THREADS"] = "3"

import numpy as np
import onnxruntime as ort
import sentencepiece as spm


def read_audio(path, sample_rate=16000):
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "3",
            "-i",
            str(path),
            "-threads",
            "3",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def cpu_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def select_encoder(package, prefix, manifest):
    if "encoder_filename" in manifest:
        return package / manifest["encoder_filename"]
    default = package / f"{prefix}_encoder.int8.onnx"
    if default.is_file():
        return default
    names = (f"{prefix}_encoder.onnx", f"{prefix}_encoder.int4.onnx")
    candidates = [package / name for name in names if (package / name).is_file()]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError("Multiple RNN-T encoders in package. Select one with --encoder.")
    raise FileNotFoundError(default)


class GigaAMRNNT:
    def __init__(self, package, *, encoder=None):
        package = Path(package)
        manifest = json.loads((package / "manifest.json").read_text())
        prefix = manifest["model_name"]
        self.heads = manifest["head_order"]
        self.preprocessor = manifest["preprocessor"]
        with np.load(package / f"{prefix}_mel.npz") as mel:
            self.window = mel["window"]
            self.filterbank = mel["filterbank"]
        self.tokenizers = {
            head: spm.SentencePieceProcessor(
                model_file=str(package / f"{prefix}_{head}_tokenizer.model")
            )
            for head in self.heads
        }
        self.encoder = cpu_session(encoder or select_encoder(package, prefix, manifest))
        self.advance = cpu_session(package / f"{prefix}_advance.onnx")
        self.encoder_inputs = [node.name for node in self.encoder.get_inputs()]
        if len(self.encoder_inputs) != 2:
            raise ValueError(
                "Expected a full-context encoder taking log-mel features and lengths"
            )

    @property
    def sample_rate(self):
        return self.preprocessor["sample_rate"]

    def features(self, audio):
        window = self.preprocessor["win_length"]
        if not window <= audio.size <= 30 * self.sample_rate:
            raise ValueError(f"Expected a single clip from {1000 * window / self.sample_rate:g} ms to 30 s")
        frames = np.lib.stride_tricks.sliding_window_view(audio, window)[
            :: self.preprocessor["hop_length"]
        ]
        spectrum = np.fft.rfft(frames * self.window, n=self.preprocessor["n_fft"])
        power = spectrum.real**2 + spectrum.imag**2
        mel = np.log(np.clip(power @ self.filterbank, 1e-9, 1e9))
        return mel.T[None, :, :]

    def transcribe(self, audio):
        features = self.features(audio)
        sizes = np.array([features.shape[-1]], dtype=np.int64)
        lengths, projected = self.encoder.run(
            None, dict(zip(self.encoder_inputs, (features, sizes), strict=True))
        )
        (tokens,) = self.advance.run(None, {"projected": projected, "lengths": lengths})
        rows = tokens.transpose(1, 2, 0)[:, 0, :]
        return {
            head: self.tokenizers[head].decode(row[row >= 0].tolist())
            for head, row in zip(self.heads, rows, strict=True)
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--encoder", type=Path, help="Path to the encoder ONNX graph")
    args = parser.parse_args()
    model = GigaAMRNNT(args.package, encoder=args.encoder)
    try:
        result = model.transcribe(read_audio(args.audio, model.sample_rate))
    except subprocess.CalledProcessError as error:
        parser.exit(1, error.stderr.decode(errors="replace"))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
