"""Run the Multilingual-v2 RNN-T full-context ONNX package on CPU."""

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


def onnx_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    session.disable_fallback()
    return session


class MultilingualRNNT:
    def __init__(self, package, *, encoder=None, advance=None):
        package = Path(package)
        manifest = json.loads((package / "manifest.json").read_text())
        config = json.loads((package / "config.json").read_text())["cfg"]
        prefix = manifest["model_name"]
        self.heads = manifest["head_order"]
        self.preprocessor = config["preprocessor"]
        language = config["language_conditioning"]
        self.slots = {
            code: index for index, code in enumerate(["auto", *language["languages"]])
        }
        self.num_slots = language["num_slots"]
        with np.load(package / f"{prefix}_mel.npz") as mel:
            self.window = mel["window"]
            self.filterbank = mel["filterbank"]
        self.tokenizers = {
            head: spm.SentencePieceProcessor(
                model_file=str(package / f"{prefix}_{head}_tokenizer.model")
            )
            for head in self.heads
        }
        if encoder is None:
            encoder = package / manifest.get("encoder_filename", f"{prefix}_encoder.int8.onnx")
        self.encoder = onnx_session(encoder)
        self.advance = (
            advance
            if advance is not None
            else onnx_session(package / f"{prefix}_advance.onnx")
        )
        inputs = self.encoder.get_inputs()
        if len(inputs) != 3 or inputs[2].name != "language_vector":
            raise ValueError(
                "Expected the full-context encoder with language conditioning"
            )
        self.encoder_inputs = [node.name for node in inputs]
        self.features_dtype = (
            np.float16 if inputs[0].type == "tensor(float16)" else np.float32
        )
        self.language_dtype = (
            np.float16 if inputs[2].type == "tensor(float16)" else np.float32
        )

    @property
    def sample_rate(self):
        return self.preprocessor["sample_rate"]

    def features(self, audio):
        length = self.preprocessor["win_length"]
        if not length <= audio.size <= 30 * self.sample_rate:
            raise ValueError("Expected a single clip from 20 ms to 30 s")
        frames = np.lib.stride_tricks.sliding_window_view(audio, length)[
            :: self.preprocessor["hop_length"]
        ]
        spectrum = np.fft.rfft(frames * self.window, n=self.preprocessor["n_fft"])
        power = spectrum.real**2 + spectrum.imag**2
        mel = np.log(np.clip(power @ self.filterbank, 1e-9, 1e9))
        return mel.T[None, :, :]

    def transcribe(self, audio, language="auto"):
        features = self.features(audio).astype(self.features_dtype, copy=False)
        lang_vector = np.zeros((1, self.num_slots), dtype=self.language_dtype)
        language = "kk" if language == "kz" else language
        lang_vector[0, self.slots[language]] = 1
        sizes = np.array([features.shape[-1]], dtype=np.int64)
        lengths, projected = self.encoder.run(
            None,
            dict(zip(self.encoder_inputs, (features, sizes, lang_vector), strict=True)),
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
    parser.add_argument(
        "--language", default="auto", help="Single upstream language code, or auto"
    )
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    model = MultilingualRNNT(args.package)
    if args.language not in model.slots and args.language != "kz":
        parser.error(
            f"unknown language {args.language!r}; choose from {', '.join(model.slots)}"
        )
    try:
        result = model.transcribe(
            read_audio(args.audio, model.sample_rate), args.language
        )
    except subprocess.CalledProcessError as error:
        parser.exit(1, error.stderr.decode(errors="replace"))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
