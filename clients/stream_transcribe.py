"""Decode 16 kHz mono WAV audio using the quantized streaming encoder."""
import argparse
from pathlib import Path

import soundfile as sf
import torch
from gigaam.onnx_utils import OnnxGigaAMStreaming, load_onnx


def read_wave(path):
    wave, rate = sf.read(path, dtype="float32")
    if rate != 16000 or wave.ndim != 1:
        raise ValueError("Expected a 16 kHz mono audio file")
    return torch.from_numpy(wave)


def load_stream(directory, threads=3):
    torch.set_num_threads(threads)
    sessions, cfg = load_onnx(directory, provider="CPUExecutionProvider", num_threads=threads)
    return OnnxGigaAMStreaming(cfg, sessions, future_context_ms=640)


@torch.inference_mode()
def decode(stream, wave):
    stream.reset_state()
    pieces = {"e2e": [], "raw": []}
    for packet in wave.split(10240):
        for head, result in stream.transcribe_partial(packet).items():
            pieces[head].append(result.text)
    for head, result in stream.flush().items():
        pieces[head].append(result.text)
    return {head: "".join(parts).strip() for head, parts in pieces.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    print(decode(load_stream(args.directory), read_wave(args.audio))["e2e"])
