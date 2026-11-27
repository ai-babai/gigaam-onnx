# GigaAM-v4 and GigaAM Multilingual-v2 ONNX INT8

Run GigaAM speech-to-text locally on a CPU with ONNX Runtime. This repository contains the ONNX conversion and quantization code, Python inference clients, and benchmark tools. Download individual model files from Hugging Face and run them with the clients in this repository.

GigaAM-v4 supports Russian and English. Its three packages cover offline CTC, full-context RNN-T, and streaming RNN-T. The GigaAM Multilingual-v2 package provides full-context RNN-T for 21 languages, including Russian, English, Kazakh, Kyrgyz, and Uzbek.

## Tested compatibility

| Runtime or loader | macOS 15.8.1, Apple Silicon | Ubuntu 26.04.1, x86-64 | Models tested |
|---|---|---|---|
| ONNX Runtime 1.23.2, CPU | Pass | Pass | All four inference clients |
| sherpa-onnx 1.13.2, NeMo CTC | Pass | Pass | GigaAM-v4 offline CTC |
| transcribe-rs 0.3.11, GigaAM loader used by AutoSubs | Pass | Pass | GigaAM-v4 offline CTC |
| onnx-asr 0.12.0, built-in loaders | Unsupported | Unsupported | Both model families |

All 16 test cases ran on each platform. Across 30 text outputs, 28 matched exactly. Two Russian outputs differed by one character each, one in the v4 streaming formatted text and one in Multilingual-v2 raw text with automatic language selection. With explicit language hints, the multilingual outputs matched on all five tested languages. The streaming example uses the upstream GigaAM wrapper with PyTorch and torchaudio 2.10.0. [Full compatibility results](benchmarks/compatibility/README.md)

## Choose a model package

| Model | Recognition mode | Output | Model files on Hugging Face |
|---|---|---|---|
| GigaAM-v4 CTC E2E | Offline | Formatted text | [ctc-e2e/](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/ctc-e2e) |
| GigaAM-v4 RNN-T | Offline, full context | Formatted and raw text | [rnnt/](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/rnnt) |
| GigaAM-v4 Streaming RNN-T | Streaming, fixed 640 ms future context | Incremental decoding, with final formatted text printed by the CLI | [rnnt-streaming-640ms/](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/rnnt-streaming-640ms) |
| GigaAM Multilingual-v2 RNN-T | Offline, full context, optional language hint | Formatted and raw text | [Model directory](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/v0.2.0) |

The packages apply 8-bit quantization to selected operations. Graphs and operations that use floating point are included where required by the decoder or frontend.

## Measured on Apple M4 Pro

| CPU package | Package size | Audio seconds per wall second | Speedup over FP32 ONNX |
|---|---:|---:|---:|
| GigaAM-v4 CTC E2E | 242.5 MB | 79.8× | 2.42× |
| GigaAM-v4 offline RNN-T | 255.4 MB | 70.9× | 2.40× |
| GigaAM-v4 streaming RNN-T | 259.0 MB | 32.1× | 1.84× |
| GigaAM Multilingual-v2 | 811.0 MB | 15.5× | 2.89× |

At 811 MB, the multilingual package is about half the size of the 1.575 GB source checkpoint. On 80 FLEURS utterances, corpus WER was 4.99% with the original PyTorch model and 5.20% with INT8. See the [per-language results and confidence intervals](benchmarks/README.md).

## Download model files and run a client

Clone the inference code once. Install the Hugging Face CLI and FFmpeg, then choose one of the four examples below. The download filters select only the graphs, vocabulary and configuration needed for that model. The Python clients run from the source checkout. Use a speech clip of up to 30 seconds for offline recognition.

```sh
git clone --branch v0.2.0 https://github.com/ai-babai/gigaam-onnx.git
cd gigaam-onnx
python -m pip install huggingface_hub
weights=../gigaam-weights
```

### GigaAM-v4 CTC E2E

```sh
hf download ai-babai/gigaam-v4-onnx --revision v0.2.0 \
  --include 'ctc-e2e/*.onnx' --include 'ctc-e2e/tokens.txt' --include 'ctc-e2e/vocab.txt' \
  --local-dir "$weights/v4"
python -m pip install numpy onnxruntime==1.23.2
python ctc_transcribe.py clip.wav --package "$weights/v4/ctc-e2e"
```

This package also supports the sherpa-onnx offline NeMo CTC loader. Install `sherpa-onnx==1.13.2` and add `--sherpa` to the same command. The `transcribe-rs` GigaAM loader uses the included `vocab.txt`. Both integrations use their own audio frontends. [CTC input and output contract](https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/v0.2.0/ctc-e2e/USAGE.md)

### GigaAM-v4 offline RNN-T

```sh
hf download ai-babai/gigaam-v4-onnx --revision v0.2.0 \
  --include 'rnnt/*.onnx' --include 'rnnt/*.model' --include 'rnnt/*.npz' --include 'rnnt/manifest.json' \
  --local-dir "$weights/v4"
python -m pip install 'numpy>=2,<3' onnxruntime==1.23.2 'sentencepiece>=0.2,<0.3'
python rnnt_transcribe.py clip.wav --package "$weights/v4/rnnt"
```

### GigaAM-v4 streaming RNN-T

The example feeds 640 ms packets and flushes the decoder at the end. The 640 ms figure is the model's future context. Application latency includes audio buffering and decoding.

```sh
hf download ai-babai/gigaam-v4-onnx --revision v0.2.0 \
  --include 'rnnt-streaming-640ms/*.onnx' --include 'rnnt-streaming-640ms/*.model' --include 'rnnt-streaming-640ms/*.json' --include 'rnnt-streaming-640ms/*.yaml' \
  --local-dir "$weights/v4"
python -m pip install onnxruntime==1.23.2 torch==2.10.0 torchaudio==2.10.0 soundfile \
  'gigaam @ git+https://github.com/salute-developers/GigaAM.git@e9bbf2e773eedd217c6f0d3b89b9728c41332b7d'
ffmpeg -i clip.wav -ac 1 -ar 16000 "$weights/clip-16k.wav"
python stream_transcribe.py "$weights/v4/rnnt-streaming-640ms" "$weights/clip-16k.wav"
```

### GigaAM Multilingual-v2 offline RNN-T

The CLI accepts `auto` or one supported language code, such as `kk` for Kazakh. It prints JSON with formatted `e2e` and plain `raw` text.

```sh
hf download ai-babai/gigaam-multilingual-v2-onnx --revision v0.2.0 \
  --include 'multilingual_v2_rnnt_*.onnx' --include 'multilingual_v2_rnnt_*.model' \
  --include 'multilingual_v2_rnnt_mel.npz' --include config.json --include manifest.json \
  --local-dir "$weights/multilingual-v2"
python -m pip install 'numpy>=2,<3' onnxruntime==1.23.2 'sentencepiece>=0.2,<0.3'
python multi_client.py clip.wav --package "$weights/multilingual-v2" --language kk
```

Supported codes are `ru`, `en`, `kk`, `ky`, `uz`, `be`, `fr`, `es`, `ba`, `ka`, `hy`, `ro`, `tk`, `az`, `tg`, `de`, `it`, `nl`, `pl`, `pt`, and `tt`.

If you prefer a single download, each [GitHub Release archive](https://github.com/ai-babai/gigaam-onnx/releases/tag/v0.2.0) bundles one model with its matching client.

## Benchmarks and conversion

See [benchmarks](benchmarks/README.md) for the evaluation setup and per-model results. The [v4 Hugging Face snapshot](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/evaluation) includes FLEURS sample IDs, reference text, and per-clip predictions. Recreate the five-language Multilingual-v2 sample with [data_fleurs.py](data_fleurs.py), or the Russian/English v4 sample with [its two-language sampler](benchmarks/v4/data_fleurs.py).

```sh
python data_fleurs.py --output ../gigaam-onnx-data/eval --cache ../gigaam-onnx-data/cache/fleurs-sample
```

The v4 converters are [ctc_export.py](ctc_export.py), [rnnt_export.py](rnnt_export.py), and [stream_export.py](stream_export.py). Their arguments and checkpoint setup are documented in the [pinned v4 conversion guide](https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/v0.2.0/conversion/README.md). Run its commands from this checkout with the `conversion/` prefix removed and point `work/` paths at a separate data directory.

For Multilingual-v2, use the same conversion environment and populate the converter's `source/` directory with the pinned checkpoint.

```sh
conversion_dir=../gigaam-onnx-data/multilingual-v2
mkdir -p "$conversion_dir/source"
hf download ai-sage/GigaAM-Multilingual-v2 config.json model.safetensors \
  raw_tokenizer.model e2e_tokenizer.model LICENSE \
  --revision c21483b5e185ff32f6fb2e2c801f92f4e7fbfee8 --local-dir "$conversion_dir/source"
cp LICENSE-APACHE-2.0 "$conversion_dir/source/"
python multi_export.py "$conversion_dir"
python multi_quantize.py "$conversion_dir"
```

The converted Multilingual-v2 package is written to `$conversion_dir/package/`.

## Credits and licenses

Original [GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4) and [GigaAM Multilingual-v2](https://huggingface.co/ai-sage/GigaAM-Multilingual-v2) models by Giga. ONNX conversion, INT8 packaging, and evaluation by Maksim Popkov. See [LICENSE](LICENSE), [NOTICE](NOTICE), and the [pinned upstream implementation](https://github.com/salute-developers/GigaAM/tree/e9bbf2e773eedd217c6f0d3b89b9728c41332b7d).

## Citation and attribution

Original model architecture, training and checkpoints by the GigaAM team. [Original GigaAM repository](https://github.com/salute-developers/GigaAM) · [Authors’ v4 and Multilingual-v2 release announcement](https://habr.com/ru/companies/sberbank/articles/1112846/)

Pinned source weights for this release. [v4 CTC checkpoint](https://huggingface.co/ai-sage/GigaAM-v4/tree/5b81216afa29c8b5a1a3e64c424c8416e18c1e5b) · [v4 RNN-T checkpoint](https://huggingface.co/ai-sage/GigaAM-v4/tree/9fe3ebf50c9e0cce5eed60d546938632238b6a96) · [Multilingual-v2 RNN-T checkpoint](https://huggingface.co/ai-sage/GigaAM-Multilingual-v2/tree/c21483b5e185ff32f6fb2e2c801f92f4e7fbfee8)

Benchmark audio and reference transcripts are from [Google FLEURS](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd), distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Please cite the original model research and the ONNX release when reporting results. [GigaAM, Interspeech 2025](https://doi.org/10.21437/Interspeech.2025-1616) · [GigaAM Multilingual, 2026](https://arxiv.org/abs/2607.10371)

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
  title = {{GigaAM-v4 and GigaAM Multilingual-v2 ONNX INT8}},
  version = {0.2.0},
  year = {2026},
  url = {https://github.com/ai-babai/gigaam-onnx/releases/tag/v0.2.0}
}
```

ONNX conversion, INT8 quantization and cross-platform evaluation by **[Maksim Popkov](https://huggingface.co/ai-babai)**.
