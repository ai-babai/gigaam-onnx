# GigaAM-v4 and GigaAM Multilingual-v2 ONNX for CPU

Local speech recognition on CPU. Use GigaAM-v4 for Russian–English dictation or streaming, and GigaAM Multilingual-v2 for offline transcription in 21 languages.

[GigaAM-v4 on Hugging Face](https://huggingface.co/ai-babai/gigaam-v4-onnx) · [Multilingual-v2 on Hugging Face](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx) · [Benchmarks](benchmarks/README.md) · [Hugging Face collection](https://huggingface.co/collections/ai-babai/gigaam-onnx-offline-and-streaming-speech-recognition-6b0973ce369990a8a4c53c43)

## Tested compatibility

| Weights | ONNX Runtime | sherpa-onnx | transcribe-rs | onnx-asr | GigaAM API | Note |
|---|:---:|:---:|:---:|:---:|:---:|---|
| v4 CTC | ✓ | ✓ | ✓ | | | NeMo CTC or GigaAM loader |
| v4 RNN-T | ✓ | | | | | Offline, both text outputs |
| v4 streaming RNN-T | ✓ | | | | ✓ | 640 ms context |
| Multilingual-v2 INT8 | ✓ | | | | ✓ | GigaAM API on Mac with filename alias |
| Multilingual-v2 INT4 | ✓ | | | | | Lower peak memory |

✓ = verified with the released weights. Empty cells have no verified integration. Checks cover macOS arm64 and Linux x86-64 unless the note specifies one platform. ONNX Runtime 1.23.2, sherpa-onnx 1.13.2 and transcribe-rs 0.3.11 were tested. onnx-asr 0.12.0 has no compatible built-in loader. [Platform versions and test results](benchmarks/compatibility/README.md)

## Choose a model package

| Download model files | Use case | Output | Package | CPU speed¹ |
|---|---|---|---:|---:|
| [v4 CTC E2E](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/ctc-e2e) | Offline transcription with sherpa-onnx, transcribe-rs or ONNX Runtime | Formatted text | 242.5 MB | 79.8× |
| [v4 RNN-T](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/rnnt) | Offline Russian and English | Raw and formatted text | 255.4 MB | 70.9× |
| [v4 streaming RNN-T](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/rnnt-streaming-640ms) | Incremental dictation with 640 ms future context | Partial and final text | 259.0 MB | 32.1× |
| [Multilingual-v2 RNN-T](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/v0.2.0) | Offline transcription in 21 languages | Raw and formatted text | 811.0 MB | 15.5× |

The optional [Multilingual-v2 INT4](#lower-memory-multilingual-v2-int4) package is 519.4 MB.

¹ Audio seconds per wall-clock second on an Apple M4 Pro, three CPU threads, after warmup. Compared with FP32 ONNX, throughput improved by 2.42× for CTC, 2.40× for offline RNN-T, 1.84× for streaming and 2.89× for Multilingual-v2. [Measurement setup and accuracy by language](benchmarks/README.md)

## Download model files and run a client

Reference setup: Python 3.12.11 and FFmpeg 8.1 on macOS. Linux validation used Python 3.12.16 and FFmpeg n9.0.2 (build 17-g2a571b6068). Clone the clients, then run the commands for one model below. Each download selects just its runtime data. Offline clients take speech clips up to 30 seconds.

```sh
git clone --branch v0.2.1 https://github.com/ai-babai/gigaam-onnx.git
cd gigaam-onnx
python -m pip install huggingface_hub==1.3.5
weights=../gigaam-weights
```

### GigaAM-v4 CTC E2E

```sh
hf download ai-babai/gigaam-v4-onnx --revision v0.2.0 \
  --include 'ctc-e2e/*.onnx' --include 'ctc-e2e/tokens.txt' --include 'ctc-e2e/vocab.txt' \
  --local-dir "$weights/v4"
python -m pip install numpy==2.5.1 onnxruntime==1.23.2
python ctc_transcribe.py clip.wav --package "$weights/v4/ctc-e2e"
```

This package also supports the sherpa-onnx offline NeMo CTC loader. Install `sherpa-onnx==1.13.2` and add `--sherpa` to the same command. The `transcribe-rs` GigaAM loader uses the included `vocab.txt`. Both integrations use their own audio frontends. [CTC input and output contract](https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/ctc-e2e/USAGE.md)

### GigaAM-v4 offline RNN-T

```sh
hf download ai-babai/gigaam-v4-onnx --revision v0.2.0 \
  --include 'rnnt/*.onnx' --include 'rnnt/*.model' --include 'rnnt/*.npz' --include 'rnnt/manifest.json' \
  --local-dir "$weights/v4"
python -m pip install numpy==2.5.1 onnxruntime==1.23.2 sentencepiece==0.2.2
python rnnt_transcribe.py clip.wav --package "$weights/v4/rnnt"
```

### GigaAM-v4 streaming RNN-T

The example feeds 640 ms packets and flushes the decoder at the end. The 640 ms figure is the model's future context. Application latency includes buffering and decoding. Word start and end times are available through the upstream streaming API with `word_timestamps=True`. The example CLI prints the final transcript.

```sh
hf download ai-babai/gigaam-v4-onnx --revision v0.2.0 \
  --include 'rnnt-streaming-640ms/*.onnx' --include 'rnnt-streaming-640ms/*.model' --include 'rnnt-streaming-640ms/*.json' --include 'rnnt-streaming-640ms/*.yaml' \
  --include LICENSE --include LICENSE-APACHE-2.0 --include NOTICE \
  --local-dir "$weights/v4"
python -m pip install onnxruntime==1.23.2 torch==2.10.0 torchaudio==2.10.0 soundfile==0.14.0 \
  'gigaam @ git+https://github.com/salute-developers/GigaAM.git@e9bbf2e773eedd217c6f0d3b89b9728c41332b7d'
ffmpeg -i clip.wav -ac 1 -ar 16000 "$weights/clip-16k.wav"
python stream_transcribe.py "$weights/v4/rnnt-streaming-640ms" "$weights/clip-16k.wav"
```

### GigaAM Multilingual-v2 offline RNN-T

The CLI accepts `auto` or one supported language code, such as `kk` for Kazakh. It prints JSON with formatted `e2e` and plain `raw` text. Split at long pauses even within the 30-second limit.

```sh
hf download ai-babai/gigaam-multilingual-v2-onnx --revision v0.2.0 \
  --include 'multilingual_v2_rnnt_*.onnx' --include 'multilingual_v2_rnnt_*.model' \
  --include 'multilingual_v2_rnnt_mel.npz' --include config.json --include manifest.json \
  --local-dir "$weights/multilingual-v2"
python -m pip install numpy==2.5.1 onnxruntime==1.23.2 sentencepiece==0.2.2
python multi_client.py clip.wav --package "$weights/multilingual-v2" --language kk
```

Supported codes are `ru`, `en`, `kk`, `ky`, `uz`, `be`, `fr`, `es`, `ba`, `ka`, `hy`, `ro`, `tk`, `az`, `tg`, `de`, `it`, `nl`, `pl`, `pt`, and `tt`.

### Lower-memory Multilingual-v2 INT4

Use the 519.4 MB INT4 package when memory or download size matters. In a paired M4 Pro run, peak process memory fell from 1.81 to 1.36 GiB and throughput fell from 16.34× to 12.46× real time. Use the same Python dependencies as the offline RNN-T client above. [INT4 weights](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/v0.2.1/int4) · [Five-language comparison](benchmarks/multilingual-v2/int4/README.md)

```sh
hf download ai-babai/gigaam-multilingual-v2-onnx --revision v0.2.1 --include 'int4/*' --local-dir "$weights/multilingual-v2"
python multi_client.py clip.wav --package "$weights/multilingual-v2/int4" --language kk
```

If you prefer a single download, each [GitHub Release archive](https://github.com/ai-babai/gigaam-onnx/releases/tag/v0.2.0) bundles one model with its matching client.

## Benchmarks and conversion

The [benchmark report](benchmarks/README.md) includes sample selection, per-language WER, paired bootstrap intervals and CPU timings. The [runtime matrix](benchmarks/compatibility/README.md) records which third-party loaders actually opened each model.

To rebuild the models, follow the matching recipe from a Hugging Face snapshot. The converter sources are also in this checkout. [GigaAM-v4 conversion](https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/conversion/README.md) · [Multilingual-v2 conversion](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/main/conversion/README.md)

The original models and this port are MIT-licensed. The encoder includes Apache-2.0 components. [License](LICENSE) · [Third-party notices](NOTICE)

## Citation and attribution

Original model architecture, training and checkpoints by the GigaAM team. [Original GigaAM repository](https://github.com/salute-developers/GigaAM) · [Authors’ v4 and Multilingual-v2 release announcement](https://habr.com/ru/companies/sberbank/articles/1112846/)

Pinned source weights for this release. [v4 CTC checkpoint](https://huggingface.co/ai-sage/GigaAM-v4/tree/5b81216afa29c8b5a1a3e64c424c8416e18c1e5b) · [v4 RNN-T checkpoint](https://huggingface.co/ai-sage/GigaAM-v4/tree/9fe3ebf50c9e0cce5eed60d546938632238b6a96) · [Multilingual-v2 RNN-T checkpoint](https://huggingface.co/ai-sage/GigaAM-Multilingual-v2/tree/c21483b5e185ff32f6fb2e2c801f92f4e7fbfee8)

Benchmark audio and reference transcripts are from [Google FLEURS](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd), distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

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
  version = {0.2.1},
  year = {2026},
  url = {https://github.com/ai-babai/gigaam-onnx/releases/tag/v0.2.1}
}
```

</details>

ONNX conversion, quantization and cross-platform evaluation by **[Maksim Popkov](https://huggingface.co/ai-babai)**.
