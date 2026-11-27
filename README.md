# GigaAM-v4 and GigaAM Multilingual-v2 ONNX INT8

Run GigaAM speech-to-text locally on a CPU with ONNX Runtime. This repository contains the ONNX conversion and quantization code, Python inference clients, and benchmark tools. The model packages are distributed through GitHub Releases and Hugging Face.

GigaAM-v4 supports Russian and English. Its three packages cover offline CTC, full-context RNN-T, and streaming RNN-T. The GigaAM Multilingual-v2 package provides full-context RNN-T for 21 languages, including Russian, English, Kazakh, Kyrgyz, and Uzbek.


The four release clients were tested on Apple Silicon macOS and x86_64 Ubuntu. See the [macOS and Linux compatibility matrix](benchmarks/compatibility/README.md) for loader versions, cross-platform transcript differences and integration limits.

## Choose a model package

| Model | Recognition mode | Output | GitHub Release asset |
|---|---|---|---|
| GigaAM-v4 CTC E2E | Offline | Formatted text | [ctc-e2e-int8.zip](https://github.com/ai-babai/gigaam-onnx/releases/download/v0.2.0/ctc-e2e-int8.zip) |
| GigaAM-v4 RNN-T | Offline, full context | Formatted and raw text | [rnnt-int8.zip](https://github.com/ai-babai/gigaam-onnx/releases/download/v0.2.0/rnnt-int8.zip) |
| GigaAM-v4 Streaming RNN-T | Streaming, fixed 640 ms future context | Incremental decoding, with final formatted text printed by the CLI | [rnnt-streaming-640ms-int8.zip](https://github.com/ai-babai/gigaam-onnx/releases/download/v0.2.0/rnnt-streaming-640ms-int8.zip) |
| GigaAM Multilingual-v2 RNN-T | Offline, full context, optional language hint | Formatted and raw text | [multilingual-v2-rnnt-int8.zip](https://github.com/ai-babai/gigaam-onnx/releases/download/v0.2.0/multilingual-v2-rnnt-int8.zip) |

The packages apply 8-bit quantization to selected operations. Graphs and operations that use floating point are included where required by the decoder or frontend.

## Measured on Apple M4 Pro

| CPU package | Package size | Audio seconds per wall second | Speedup over FP32 ONNX |
|---|---:|---:|---:|
| GigaAM-v4 CTC E2E | 242.5 MB | 79.8× | 2.42× |
| GigaAM-v4 offline RNN-T | 255.4 MB | 70.9× | 2.40× |
| GigaAM-v4 streaming RNN-T | 259.0 MB | 32.1× | 1.84× |
| GigaAM Multilingual-v2 | 811.0 MB | 15.5× | 2.89× |

At 811 MB, the multilingual package is about half the size of the 1.575 GB source checkpoint. On 80 FLEURS utterances, corpus WER was 4.99% with the original PyTorch model and 5.20% with INT8. See the [per-language results and confidence intervals](benchmarks/README.md).

## Download

Set the package directory from the source checkout. These examples keep the model files next to the checkout.

```sh
package_root=../gigaam-onnx-data/models
mkdir -p "$package_root"
```

Each ZIP from [GitHub Release v0.2.0](https://github.com/ai-babai/gigaam-onnx/releases/tag/v0.2.0) extracts to a directory matching its filename without `.zip`. Use the GitHub CLI with repository access to download and extract the package you need.

```sh
gh release download v0.2.0 --repo ai-babai/gigaam-onnx --pattern ctc-e2e-int8.zip --dir "$package_root"
unzip "$package_root/ctc-e2e-int8.zip" -d "$package_root"
```

The [GigaAM-v4 Hugging Face repository](https://huggingface.co/ai-babai/gigaam-v4-onnx) uses the `ctc-e2e/`, `rnnt/`, and `rnnt-streaming-640ms/` directories. The [GigaAM Multilingual-v2 Hugging Face repository](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx) has its client and graphs at the root. Install the Hugging Face CLI with `python -m pip install huggingface_hub` and run `hf auth login` if access requires it.

```sh
hf download ai-babai/gigaam-v4-onnx --revision v0.2.0 --include 'ctc-e2e/*' --local-dir "$package_root/gigaam-v4"
hf download ai-babai/gigaam-multilingual-v2-onnx --revision v0.2.0 --local-dir "$package_root/gigaam-multilingual-v2"
```

To download a different v4 variant, change the directory in the include pattern.

## Transcribe audio

Run these commands from the source checkout with Python 3.11+ and FFmpeg on PATH. The offline clients accept a single clip from 20 ms to 30 seconds. For longer offline recordings, split the audio at speech boundaries. Keep the `package_root` variable from the download step. If you used Hugging Face, substitute the corresponding package directory.

### GigaAM-v4 CTC E2E

```sh
python -m pip install numpy onnxruntime==1.23.2
python ctc_transcribe.py clip.wav --package "$package_root/ctc-e2e-int8"
```

This package also supports the sherpa-onnx offline NeMo CTC loader. Install `sherpa-onnx==1.13.2` and add `--sherpa` to the same command. The `transcribe-rs` GigaAM loader uses the included `vocab.txt`. Both integrations use their own audio frontends. [CTC input and output contract](https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/v0.2.0/ctc-e2e/USAGE.md)

### GigaAM-v4 offline RNN-T

```sh
python -m pip install -r "$package_root/rnnt-int8/requirements.txt"
python rnnt_transcribe.py clip.wav --package "$package_root/rnnt-int8"
```

### GigaAM-v4 streaming RNN-T

The example feeds 640 ms packets and flushes the decoder at the end. The 640 ms figure is the model's future context. Application latency includes audio buffering and decoding.

```sh
python -m pip install torch==2.10.0 torchaudio==2.10.0 soundfile \
  'gigaam @ git+https://github.com/salute-developers/GigaAM.git@e9bbf2e773eedd217c6f0d3b89b9728c41332b7d'
ffmpeg -i clip.wav -ac 1 -ar 16000 "$package_root/clip-16k.wav"
python stream_transcribe.py "$package_root/rnnt-streaming-640ms-int8" "$package_root/clip-16k.wav"
```

### GigaAM Multilingual-v2 offline RNN-T

The CLI accepts `auto` or one supported language code, such as `kk` for Kazakh. It prints JSON with formatted `e2e` and plain `raw` text.

```sh
python -m pip install -r "$package_root/multilingual-v2-rnnt-int8/requirements.txt"
python multi_client.py clip.wav --package "$package_root/multilingual-v2-rnnt-int8" --language kk
```

Supported codes are `ru`, `en`, `kk`, `ky`, `uz`, `be`, `fr`, `es`, `ba`, `ka`, `hy`, `ro`, `tk`, `az`, `tg`, `de`, `it`, `nl`, `pl`, `pt`, and `tt`.

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
