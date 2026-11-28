# GigaAM-v4 and GigaAM Multilingual-v2 ONNX

The [GigaAM team](https://github.com/salute-developers/GigaAM) developed and trained both models. The ONNX releases provide CPU inference for both models in FP32, FP16 storage, INT8 and INT4.

Use GigaAM-v4 for Russian–English dictation or streaming, and GigaAM Multilingual-v2 for offline transcription in 21 languages, including Bashkir and Tatar. Sherpa also supports NVIDIA CUDA on Linux and experimental native Core ML for v4 CTC and streaming on Apple Silicon. [GPU setup and benchmarks](benchmarks/compatibility/sherpa-gpu.md) · [Apple offline RNN-T preview for both models](benchmarks/compatibility/sherpa-gpu.md#separate-offline-rnn-t-preview-for-apple-silicon)

[GigaAM-v4 on Hugging Face](https://huggingface.co/ai-babai/gigaam-v4-onnx) · [Multilingual-v2 on Hugging Face](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx) · [Benchmarks](benchmarks/README.md) · [Hugging Face collection](https://huggingface.co/collections/ai-babai/gigaam-onnx-offline-and-streaming-speech-recognition-6b0973ce369990a8a4c53c43)

[Compare weight formats, sizes and accuracy](benchmarks/precision/README.md) · [Release quality checks](quality/README.md) · [Agent guide and command map](AGENTS.md)

## Model files on Hugging Face

All four weight formats are available in the **Files** tab on `main`. CTC links open encoder files. RNN-T links open packages containing the encoder and its matching decoder files. FP16 stores weights in FP16 and computes in FP32 on CPU.

| Model | FP32 | FP16 storage | INT8 (default) | INT4 |
|---|---|---|---|---|
| v4 CTC | [FP32](https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/ctc-e2e/model.onnx) | [FP16](https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/ctc-e2e/model.fp16.onnx) | [INT8](https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/ctc-e2e/model.int8.onnx) | [INT4](https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/ctc-e2e/model.int4.onnx) |
| v4 offline RNN-T | [FP32](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/main/rnnt/fp32) | [FP16](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/main/rnnt/fp16) | [INT8](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/main/rnnt) | [INT4](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/main/rnnt/int4) |
| v4 streaming RNN-T | [FP32](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/main/rnnt-streaming-640ms/fp32) | [FP16](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/main/rnnt-streaming-640ms/fp16) | [INT8](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/main/rnnt-streaming-640ms) | [INT4](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/main/rnnt-streaming-640ms/int4) |
| Multilingual-v2 | [FP32](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/main/fp32) | [FP16](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/main/fp16) | [INT8](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/main) | [INT4](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/main/int4) |

## Tested compatibility

The table covers the original INT8 packages and Multilingual-v2 INT4. For results by precision and operating system, see the [16-package platform matrix](benchmarks/compatibility/precision-platforms.md).

| Weights | ONNX Runtime | Sherpa release | [Sherpa preview](benchmarks/compatibility/sherpa-preview.md) | transcribe-rs | onnx-asr | GigaAM API | Note |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| v4 CTC | ✓ | ✓ | ✓ | ✓ | | | Offline NeMo CTC |
| v4 RNN-T | ✓ | | ✓ | | | | Raw or formatted text |
| v4 streaming RNN-T | ✓ | | ✓ | | | ✓ | Partial and final text, 640 ms |
| Multilingual-v2 INT8 | ✓ | | ✓ | | | ✓ | GigaAM API tested on Mac with a filename alias |
| Multilingual-v2 INT4 | ✓ | | ✓ | | | | Same decoder, smaller encoder |

✓ = verified with the released weights. Empty cells have no verified integration. Checks cover macOS arm64 and Linux x86-64 unless the note specifies one platform. ONNX Runtime 1.23.2, sherpa-onnx 1.13.2 and 2.0.3 and transcribe-rs 0.3.11 were tested. onnx-asr 0.12.0 has no compatible built-in loader. Windows 11 x64 passed all five weight variants with the inference engines listed in the platform matrix. [Platform versions and test results](benchmarks/compatibility/README.md)

## Choose an INT8 model package

| Download model files | Use case | Output | Package | CPU speed¹ |
|---|---|---|---:|---:|
| [v4 CTC E2E](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.3.1/ctc-e2e) | Offline transcription with sherpa-onnx, transcribe-rs or ONNX Runtime | Formatted text | 242.5 MB | 79.8× |
| [v4 RNN-T](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.3.1/rnnt) | Offline Russian and English | Raw and formatted text | 255.4 MB | 70.9× |
| [v4 streaming RNN-T](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.3.1/rnnt-streaming-640ms) | Incremental dictation with 640 ms future context | Partial and final text | 259.0 MB | 32.1× |
| [Multilingual-v2 RNN-T](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/v0.3.1) | Offline transcription in 21 languages | Raw and formatted text | 811.0 MB | 15.5× |

The optional [Multilingual-v2 INT4](#lower-memory-multilingual-v2-int4) package is 519.4 MB.

¹ Audio seconds per wall-clock second on an Apple M4 Pro, three CPU threads, after warmup. Compared with FP32 ONNX, throughput improved by 2.42× for CTC, 2.40× for offline RNN-T, 1.84× for streaming and 2.89× for Multilingual-v2. [Measurement setup and accuracy by language](benchmarks/README.md)

## Download model files and run a client

Reference setup: Python 3.12.11 and FFmpeg 8.1 on macOS. Linux validation used Python 3.12.16 and FFmpeg n9.0.2 (build 17-g2a571b6068). Clone the clients, then run the commands for one model below. Each download selects just its runtime data. Offline clients take speech clips up to 30 seconds.

```sh
git clone --branch v0.4.0 https://github.com/ai-babai/gigaam-onnx.git
cd gigaam-onnx
python -m pip install huggingface_hub==1.3.5 hf-xet==1.8.0 typer-slim==0.21.1 click==8.2.1
weights=../gigaam-weights
```

### GigaAM-v4 CTC E2E

```sh
hf download ai-babai/gigaam-v4-onnx --revision v0.3.1 \
  --include 'ctc-e2e/model.int8.onnx' --include 'ctc-e2e/preprocessor.onnx' --include 'ctc-e2e/tokens.txt' --include 'ctc-e2e/vocab.txt' \
  --local-dir "$weights/v4"
python -m pip install numpy==2.5.1 onnxruntime==1.23.2
python clients/ctc_transcribe.py clip.wav --package "$weights/v4/ctc-e2e"
```

To try FP32, replace `ctc-e2e/model.int8.onnx` in the download command with `ctc-e2e/model.onnx`, then add `--model "$weights/v4/ctc-e2e/model.onnx"` to the client command. All four graphs use the same preprocessor and tokens.

This package also supports the sherpa-onnx offline NeMo CTC loader. Install `sherpa-onnx==1.13.2` and add `--sherpa` to the same command. On macOS arm64, the GigaAM loader in `transcribe-rs` 0.3.11 and 0.3.12 loaded all four CTC graphs by filename and decoded a Russian test clip with the shared `vocab.txt`. Both integrations use their own audio frontends. [CTC input and output contract](https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/ctc-e2e/USAGE.md)

The CTC client reads standard 16 kHz mono PCM16 WAV files directly and uses FFmpeg for other audio. It uses CPU by default. With `--sherpa`, `--sherpa-provider` selects a native execution provider. See the [Sherpa GPU setup and compatibility table](benchmarks/compatibility/sherpa-gpu.md) for verified combinations of model, runtime and hardware.

### GigaAM-v4 offline RNN-T

```sh
hf download ai-babai/gigaam-v4-onnx --revision v0.3.1 \
  --include 'rnnt/v4_rnnt_*' --include 'rnnt/manifest.json' \
  --local-dir "$weights/v4"
python -m pip install numpy==2.5.1 onnxruntime==1.23.2 sentencepiece==0.2.2
python clients/rnnt_transcribe.py clip.wav --package "$weights/v4/rnnt"
```

### GigaAM-v4 streaming RNN-T

The example feeds 640 ms packets and flushes the decoder at the end. The 640 ms figure is the model's future context. Application latency includes buffering and decoding. Word start and end times are available through the upstream streaming API with `word_timestamps=True`. The example CLI prints the final transcript.

```sh
hf download ai-babai/gigaam-v4-onnx --revision v0.3.1 \
  --include 'rnnt-streaming-640ms/v4_rnnt_640ms*' --include 'rnnt-streaming-640ms/streaming.json' \
  --include LICENSE --include LICENSE-APACHE-2.0 --include NOTICE \
  --local-dir "$weights/v4"
python -m pip install onnxruntime==1.23.2 torch==2.10.0 torchaudio==2.10.0 soundfile==0.14.0 \
  'gigaam @ git+https://github.com/salute-developers/GigaAM.git@e9bbf2e773eedd217c6f0d3b89b9728c41332b7d'
ffmpeg -i clip.wav -ac 1 -ar 16000 "$weights/clip-16k.wav"
python clients/stream_transcribe.py "$weights/v4/rnnt-streaming-640ms" "$weights/clip-16k.wav"
```

### GigaAM Multilingual-v2 offline RNN-T

Use `--language ba` for Bashkir or `--language tt` for Tatar. Each code provides an explicit language hint. With `--language auto` (the default), the model transcribes without an explicit hint. [Upstream model card](https://huggingface.co/ai-sage/GigaAM-Multilingual-v2#usage) · [Pinned language configuration](https://huggingface.co/ai-sage/GigaAM-Multilingual-v2/blob/c21483b5e185ff32f6fb2e2c801f92f4e7fbfee8/config.json)

The CLI accepts one language code per clip. It prints JSON with formatted `e2e` and plain `raw` text. Split at long pauses even within the 30-second limit.

```sh
hf download ai-babai/gigaam-multilingual-v2-onnx --revision v0.3.1 \
  --include 'multilingual_v2_rnnt_*.onnx' --include 'multilingual_v2_rnnt_*.model' \
  --include 'multilingual_v2_rnnt_mel.npz' --include config.json --include manifest.json \
  --local-dir "$weights/multilingual-v2"
python -m pip install numpy==2.5.1 onnxruntime==1.23.2 sentencepiece==0.2.2
python clients/multi_client.py clip.wav --package "$weights/multilingual-v2" --language kk
```

Supported codes are `ru`, `en`, `kk`, `ky`, `uz`, `be`, `fr`, `es`, `ba`, `ka`, `hy`, `ro`, `tk`, `az`, `tg`, `de`, `it`, `nl`, `pl`, `pt`, and `tt`.

### Lower-memory Multilingual-v2 INT4

Use the 519.4 MB INT4 package when memory or download size matters. In a paired M4 Pro run, peak process memory fell from 1.81 to 1.36 GiB and throughput fell from 16.34× to 12.46× real time. Use the same Python dependencies as the offline RNN-T client above. [INT4 weights](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/v0.3.1/int4) · [Five-language comparison](benchmarks/multilingual-v2/int4/README.md)

```sh
hf download ai-babai/gigaam-multilingual-v2-onnx --revision v0.3.1 --include 'int4/*' --local-dir "$weights/multilingual-v2"
python clients/multi_client.py clip.wav --package "$weights/multilingual-v2/int4" --language kk
```

If you prefer a single download, each [GitHub Release archive](https://github.com/ai-babai/gigaam-onnx/releases/tag/v0.2.0) bundles one model with its matching client.

## Benchmarks and conversion

The [benchmark tools](benchmarks/tools/) run from the repository root, for example `python -m benchmarks.tools.score_release --help`. The [benchmark report](benchmarks/README.md) includes sample selection, per-language WER, paired bootstrap intervals and CPU timings. The [runtime matrix](benchmarks/compatibility/README.md) records which third-party loaders actually opened each model.

To rebuild the models, follow the matching recipe from a Hugging Face snapshot. The [conversion tools](conversion/) run from the repository root as Python modules, for example `python -m conversion.multi_quantize_int4 --help`. [GigaAM-v4 conversion](https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/conversion/README.md) · [Multilingual-v2 conversion](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/main/conversion/README.md)

The original models and this port are MIT-licensed. The encoder includes Apache-2.0 components. [License](LICENSE) · [Third-party notices](NOTICE)

## Citation and attribution

Original model architecture, training and checkpoints by the GigaAM team. [Original GigaAM repository](https://github.com/salute-developers/GigaAM) · [Authors’ v4 and Multilingual-v2 release announcement](https://habr.com/ru/companies/sberbank/articles/1112846/)

Pinned source weights for this release. [v4 CTC checkpoint](https://huggingface.co/ai-sage/GigaAM-v4/tree/5b81216afa29c8b5a1a3e64c424c8416e18c1e5b) · [v4 RNN-T checkpoint](https://huggingface.co/ai-sage/GigaAM-v4/tree/9fe3ebf50c9e0cce5eed60d546938632238b6a96) · [Multilingual-v2 RNN-T checkpoint](https://huggingface.co/ai-sage/GigaAM-Multilingual-v2/tree/c21483b5e185ff32f6fb2e2c801f92f4e7fbfee8)

Audio and reference transcripts for the original v0.2.0 benchmark come from Google FLEURS under CC BY 4.0. [Pinned dataset](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd) · [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

Please cite the model papers and this ONNX release. [GigaAM, Interspeech 2025](https://doi.org/10.21437/Interspeech.2025-1616) · [GigaAM Multilingual, 2026](https://arxiv.org/abs/2607.10371)

<details>
<summary>BibTeX for the model papers and this ONNX release</summary>

```bibtex
@inproceedings{kutsakov2025gigaam,
  author = {Aleksandr Kutsakov and Alexandr Maximenko and Georgii Gospodinov and Pavel Bogomolov and Fyodor Minkin},
  title = {{GigaAM}: Efficient Self-Supervised Learner for Speech Recognition},
  booktitle = {Interspeech 2025},
  year = {2025},
  pages = {1213--1217},
  doi = {10.21437/Interspeech.2025-1616}
}

@misc{kuzmenko2026gigaam,
  author = {Andrei Kuzmenko and Alexandr Maximenko and Aleksandr Kutsakov and Georgii Gospodinov and Dmitrii Bolotov and Oleg Kutuzov and Pavel Bogomolov and Fyodor Minkin},
  title = {{GigaAM Multilingual}: Foundation Model for Underrepresented Languages},
  year = {2026},
  eprint = {2607.10371},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2607.10371}
}

@software{popkov2026gigaamonnx,
  author = {Maksim Popkov},
  title = {{GigaAM-v4 and GigaAM Multilingual-v2 ONNX for CPU}},
  version = {0.4.0},
  year = {2026},
  url = {https://github.com/ai-babai/gigaam-onnx/releases/tag/v0.4.0}
}
```

</details>

ONNX conversion, quantization and cross-platform evaluation by **[Maksim Popkov](https://popkov.dev/)**.
