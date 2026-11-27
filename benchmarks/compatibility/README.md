# macOS and Linux speech recognition compatibility

All four release clients completed 16 paired cases on **Apple Silicon macOS 15.8.1** and **x86_64 Ubuntu 26.04.1 LTS**. They used the CPU execution provider in ONNX Runtime 1.23.2. Model and audio hashes matched across hosts. This is a loader and transcript smoke test. The [quality and throughput benchmarks](../README.md) use the larger FLEURS sample on Mac.

| Weights | ONNX Runtime | sherpa-onnx | transcribe-rs | onnx-asr | GigaAM API | Note |
|---|:---:|:---:|:---:|:---:|:---:|---|
| v4 CTC | ✓ | ✓ | ✓ | | | NeMo CTC or GigaAM loader |
| v4 RNN-T | ✓ | | | | | Offline, both text outputs |
| v4 streaming RNN-T | ✓ | | | | ✓ | 640 ms context |
| Multilingual-v2 INT8 | ✓ | | | | ✓ | GigaAM API on Mac with filename alias |
| Multilingual-v2 INT4 | ✓ | | | | | Lower peak memory |

✓ = verified with the released weights. Empty cells have no verified integration. Checks cover macOS arm64 and Linux x86-64 unless the note specifies one platform.

Versions tested: ONNX Runtime 1.23.2 (CPU), sherpa-onnx 1.13.2, transcribe-rs 0.3.11 and the [GigaAM release commit](https://github.com/salute-developers/GigaAM/tree/e9bbf2e773eedd217c6f0d3b89b9728c41332b7d). We tested onnx-asr 0.12.0 on both platforms. Its built-in loaders do not support these graph contracts. “ONNX Runtime” includes the preprocessing and decoder required by each graph.

An additional macOS arm64 run used the sherpa-onnx 2.0.3 wheel with its 2.0.3 native core and Python 3.11.15. The offline NeMo CTC loader returned text on 32/32 RU and EN clips. On 24/32 clips the text matched the bundled ONNX frontend exactly.

Overall, 28 of 30 texts matched byte for byte across platforms. Both one-character differences repeated on rerun. The CLI checks include the five explicit language hints and auto on the same five recordings. The CTC integrations with sherpa-onnx and transcribe-rs were checked independently. See [v4 details](v4-platforms.md), [multilingual details](multilingual-platforms.md), [v4 machine-readable results](linux-v4.json) and [multilingual machine-readable results](linux-multi.json).

The v4 streaming weights also returned per-word start and end times through the GigaAM API with `word_timestamps=True` on both platforms. The bundled CLI prints text. On the 13.8-second Russian clip, both platforms returned 25 formatted words. The first word started at 1.36 s and ended at 1.68 s. Timing accuracy against human word alignments was not evaluated.

The Multilingual-v2 INT4 variant passed ten Linux hint cases on five recordings. Compared with INT8, 19 of 20 raw/formatted outputs matched exactly. [INT4 Linux results](../multilingual-v2/int4/linux.json)

## Tested dependency versions

| Component | Apple Silicon macOS | Linux x86-64 |
|---|---|---|
| Python | 3.12.11 | 3.12.16 |
| NumPy | 2.5.1 | 2.5.5 |
| ONNX Runtime, Python | 1.23.2 | 1.23.2 |
| SentencePiece | 0.2.2 | 0.2.3 |
| SoundFile | 0.14.0 | 0.14.0 |
| PyTorch / torchaudio, streaming | 2.10.0 | 2.10.0+cpu |
| FFmpeg | 8.1 | n9.0.2-17-g2a571b6068 |

The examples pin the macOS Python package versions. Linux native CTC loaders were tested with ORT 1.24.4. Model downloads were checked with the Hugging Face CLI from `huggingface_hub==1.3.5`.

## Use the v4 CTC weights in an application

The [v4 CTC directory on Hugging Face](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/ctc-e2e) contains the files for both loaders. In an existing application, point its loader at the files shown below. The released `ctc_transcribe.py` is optional.

| Integration | Files to download | Entry point |
|---|---|---|
| Python: sherpa-onnx 1.13.2 | `model.int8.onnx`, `tokens.txt` | `OfflineRecognizer.from_nemo_ctc` with 16 kHz and 64 mel bins |
| Rust: transcribe-rs 0.3.11 | `model.int8.onnx`, `vocab.txt` | `GigaAMModel::load(&model_dir, &Quantization::Int8)` |

Minimal Python example for a 16 kHz mono WAV. Install `sherpa-onnx==1.13.2` and `soundfile==0.14.0`.

```python
from pathlib import Path

import sherpa_onnx
import soundfile as sf

for path in (Path("ctc-e2e/model.int8.onnx"), Path("ctc-e2e/tokens.txt")):
    if not path.is_file():
        raise FileNotFoundError(path)

asr = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
    model="ctc-e2e/model.int8.onnx", tokens="ctc-e2e/tokens.txt",
    sample_rate=16000, feature_dim=64, num_threads=3, provider="cpu",
)
audio, rate = sf.read("speech.wav", dtype="float32")
assert rate == 16000 and audio.ndim == 1
stream = asr.create_stream()
stream.accept_waveform(rate, audio)
asr.decode_stream(stream)
print(stream.result.text)
```

For a Rust integration, use the GigaAM module and ONNX Runtime setup from the [transcribe-rs 0.3.11 example](https://docs.rs/crate/transcribe-rs/0.3.11/source/examples/gigaam.rs). The check covered the native engine used by AutoSubs. Its GUI catalog was outside the test.

**LM Studio 0.4.25+1 (Mac).** The stock importer warned about the ONNX extension. After importing the v4 CTC file as a symbolic link, the model catalog returned `Model not found`. We found no direct ASR loader for these weights in this version. [Supported imports](https://lmstudio.ai/docs/app/advanced/import-model)
