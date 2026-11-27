# macOS and Linux speech recognition compatibility

All four release clients completed 16 paired cases on **Apple Silicon macOS 15.8.1** and **x86_64 Ubuntu 26.04.1 LTS**. They used the CPU execution provider in ONNX Runtime 1.23.2. Model and audio hashes matched across hosts. This is a loader and transcript smoke test. The [quality and throughput benchmarks](../README.md) use the larger FLEURS sample on Mac.

| Weights | ONNX Runtime | sherpa-onnx | transcribe-rs | onnx-asr | GigaAM API | Note |
|---|:---:|:---:|:---:|:---:|:---:|---|
| v4 CTC | ✓ | ✓ | ✓ | | | NeMo CTC or GigaAM loader |
| v4 RNN-T | ✓ | | | | | Offline, both text outputs |
| v4 streaming RNN-T | ✓ | | | | ✓ | 640 ms context |
| Multilingual-v2 RNN-T | ✓ | | | | ✓ | GigaAM API on Mac with filename alias |

✓ = verified with the released weights. Empty cells have no verified integration. Checks cover macOS arm64 and Linux x86-64 unless the note specifies one platform.

Versions tested: ONNX Runtime 1.23.2 (CPU), sherpa-onnx 1.13.2, transcribe-rs 0.3.11 and the [GigaAM release commit](https://github.com/salute-developers/GigaAM/tree/e9bbf2e773eedd217c6f0d3b89b9728c41332b7d). We tested onnx-asr 0.12.0 on both platforms. Its built-in loaders do not support these graph contracts. “ONNX Runtime” includes the preprocessing and decoder required by each graph.

Overall, 28 of 30 texts matched byte for byte across platforms. Both one-character differences repeated on rerun. The CLI checks include the five explicit language hints and auto on the same five recordings. The CTC integrations with sherpa-onnx and transcribe-rs were checked independently. See [v4 details](v4-platforms.md), [multilingual details](multilingual-platforms.md), [v4 machine-readable results](linux-v4.json) and [multilingual machine-readable results](linux-multi.json).

The v4 streaming weights also returned per-word start and end times through the GigaAM API with `word_timestamps=True` on both platforms. The bundled CLI prints text. On the 13.8-second Russian clip, both platforms returned 25 formatted words. The first word started at 1.36 s and ended at 1.68 s. Timing accuracy against human word alignments was not evaluated.
