"""Export pinned GigaAM-v4 CTC e2e weights for ORT and sherpa-onnx."""

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "3"

import onnx
import torch
from omegaconf import OmegaConf
from onnxruntime.quantization import QuantType, quantize_dynamic
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from gigaam.model import GigaAMDualASR


def load_source(source):
    config = OmegaConf.create(json.loads((source / "config.json").read_text())["cfg"])
    model = GigaAMDualASR(config).eval()
    state = load_file(source / "model.safetensors")
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()})
    return model


class CtcGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.head = model.heads["e2e"]

    def forward(self, features, feature_lengths):
        encoded, lengths = self.encoder.forward_for_export(features, feature_lengths)
        return self.head(encoded), lengths


class OnnxFrontend(nn.Module):
    """The frozen upstream Hann window and HTK mel bank, with a real-valued DFT."""

    def __init__(self, feature_extractor):
        super().__init__()
        transform = feature_extractor.featurizer[0]
        n_fft = transform.spectrogram.n_fft
        window = transform.spectrogram.window
        assert len(window) == n_fft and not transform.spectrogram.center
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        angles = (2 * torch.pi / n_fft) * k[:, None] * n[None, :]
        dft = torch.cat((angles.cos(), angles.sin())).float() * window
        self.register_buffer("dft", dft[:, None, :])
        self.register_buffer("mel", transform.mel_scale.fb.T.clone())
        self.hop = transform.spectrogram.hop_length
        self.n_fft = n_fft

    def forward(self, waveforms, waveform_lengths):
        real, imag = F.conv1d(waveforms[:, None, :], self.dft, stride=self.hop).chunk(2, dim=1)
        power = real.square() + imag.square()
        features = torch.log(torch.matmul(self.mel, power).clamp(1e-9, 1e9))
        lengths = (waveform_lengths - self.n_fft) // self.hop + 1
        return features, lengths


def export_frontend(model, output):
    front = OnnxFrontend(model.preprocessor).eval()
    waveforms = torch.randn(1, 16000, generator=torch.Generator().manual_seed(42))
    lengths = torch.tensor([waveforms.shape[1]])
    with torch.no_grad():
        for actual, expected in zip(front(waveforms, lengths), model.preprocessor(waveforms, lengths)):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-4)
    torch.onnx.export(
        front,
        (waveforms, lengths),
        output,
        input_names=["waveforms", "waveform_lengths"],
        output_names=["features", "feature_lengths"],
        dynamic_axes={
            "waveforms": {0: "batch", 1: "samples"},
            "waveform_lengths": {0: "batch"},
            "features": {0: "batch", 2: "frames"},
            "feature_lengths": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )


def export_ctc(model, output):
    with model.encoder.onnx_export_mode(), torch.no_grad():
        torch.onnx.export(
            CtcGraph(model).eval(),
            model.encoder.input_example(batch_size=1, seqlen=128),
            output,
            input_names=["features", "feature_lengths"],
            output_names=["log_probs", "encoded_lengths"],
            dynamic_axes={
                "features": {0: "batch", 2: "frames"},
                "feature_lengths": {0: "batch"},
                "log_probs": {0: "batch", 1: "encoded_frames"},
                "encoded_lengths": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    graph = onnx.load(output)
    vocab = model.decodings["e2e"].tokenizer
    onnx.helper.set_model_props(
        graph,
        {
            "model_type": "EncDecCTCModel",
            "version": "1",
            "vocab_size": str(len(vocab) + 1),
            "blank_id": str(len(vocab)),
            "subsampling_factor": str(model.encoder.subsampling_factor),
            "normalize_type": "",
            "is_giga_am": "1",
            "language": "Russian, English",
            "model_author": "https://github.com/salute-developers/GigaAM",
            "source_revision": "5b81216afa29c8b5a1a3e64c424c8416e18c1e5b",
            "license": "MIT",
        },
    )
    onnx.save(graph, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("--quantize-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    args.work.mkdir(parents=True, exist_ok=True)
    args.package.mkdir(parents=True, exist_ok=True)

    work_fp = args.work / "model.fp32.onnx"
    quantized = args.work / "model.int8.onnx"
    if not args.quantize_only:
        model = load_source(args.source)
        print("Exporting frontend", flush=True)
        export_frontend(model, args.package / "preprocessor.onnx")
        tokenizer = model.decodings["e2e"].tokenizer
        tokens = [tokenizer.id_to_str(index).replace(" ", "▁") for index in range(len(tokenizer))]
        tokens.append("<blk>")
        (args.package / "tokens.txt").write_text(
            "".join(f"{value} {index}\n" for index, value in enumerate(tokens)), encoding="utf-8"
        )
        print("Exporting FP32 CTC e2e", flush=True)
        export_ctc(model, work_fp)
        del model
    print("Quantizing CTC e2e", flush=True)
    quantize_dynamic(
        work_fp,
        quantized,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
        weight_type=QuantType.QUInt8,
        per_channel=True,
    )
    onnx.checker.check_model(quantized)
    quantized.replace(args.package / "model.int8.onnx")
    shutil.copyfile(args.source / "LICENSE", args.package / "LICENSE")
    print("INT8 package ready", flush=True)


if __name__ == "__main__":
    main()
