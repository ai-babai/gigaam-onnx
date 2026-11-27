# macOS and Linux speech recognition compatibility

All four release clients completed 16 paired cases on **Apple Silicon macOS 15.8.1** and **x86_64 Ubuntu 26.04.1 LTS**. They used the CPU execution provider in ONNX Runtime 1.23.2. Model and audio hashes matched across hosts. This is a loader and transcript smoke test. The [quality and throughput benchmarks](../README.md) use the larger FLEURS sample on Mac.

| Package or library | Mac arm64 | Linux x86_64 | Cross-platform comparison |
|---|---|---|---|
| GigaAM-v4 CTC, packaged ONNX Runtime client | Passed | Passed | RU and EN text matched |
| GigaAM-v4 CTC, sherpa-onnx 1.13.2 | Passed | Passed | RU and EN text matched |
| GigaAM-v4 CTC, transcribe-rs 0.3.11 | Passed | Passed | RU and EN text matched |
| GigaAM-v4 offline RNN-T, both heads | Passed | Passed | RU and EN text matched |
| GigaAM-v4 streaming RNN-T, both heads | Passed | Passed | Partial text before flush on RU and EN; one character differed in formatted RU |
| Multilingual-v2 RNN-T, both heads | Passed | Passed | Five explicit hints matched; one character differed in Russian raw text in auto mode |
| onnx-asr 0.12.0, stock loaders for these packages | Unsupported | Unsupported | v3 loaders use different graph contracts or audio features |

Overall, 28 of 30 texts matched byte for byte across platforms. Both one-character differences repeated on rerun. The CLI checks include the five explicit language hints and auto on the same five recordings. The CTC integrations with sherpa-onnx and transcribe-rs were checked independently. See [v4 details](v4-platforms.md), [multilingual details](multilingual-platforms.md), [v4 machine-readable results](linux-v4.json) and [multilingual machine-readable results](linux-multi.json).
