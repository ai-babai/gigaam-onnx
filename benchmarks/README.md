# GigaAM ONNX INT8 benchmarks

Paired speech recognition quality and CPU performance for the v0.2.0 release. Each language contributes 16 seeded Google FLEURS test clips. GigaAM-v4 is evaluated on Russian and English. GigaAM Multilingual-v2 is evaluated on Russian, Kazakh, Uzbek, Kyrgyz and English, with explicit language hints.

## CPU throughput

| CPU package | Package size | Audio seconds per wall second | Speedup over FP32 ONNX |
|---|---:|---:|---:|
| GigaAM-v4 CTC E2E | 242.5 MB | 79.8× | 2.42× |
| GigaAM-v4 offline RNN-T | 255.4 MB | 70.9× | 2.40× |
| GigaAM-v4 streaming RNN-T | 259.0 MB | 32.1× | 1.84× |
| GigaAM Multilingual-v2 | 811.0 MB | 15.5× | 2.89× |

Measurements use an Apple M4 Pro with three CPU threads and ONNX Runtime 1.23.2. The v4 same-runtime speed comparison uses eight clips (four per supported language). Multilingual-v2 uses all 80 clips. Both precisions use the same frontend and decoder, including feature extraction. Timings exclude file reads and startup, with sessions warmed in advance. Streaming throughput measures compute time on prerecorded audio. Live delivery also requires buffering and future context.

## Recognition quality

| Model | Language | Words | Upstream FP32 WER | INT8 WER | Change [paired 95% interval], pp |
|---|---|---:|---:|---:|---:|
| GigaAM-v4 CTC E2E | Russian | 277 | 2.89% | 2.53% | -0.36 [-1.15, +0.00] |
| GigaAM-v4 CTC E2E | English | 381 | 6.30% | 5.51% | -0.79 [-2.10, +0.00] |
| GigaAM-v4 offline RNN-T | Russian | 277 | 2.17% | 2.53% | +0.36 [-0.75, +1.65] |
| GigaAM-v4 offline RNN-T | English | 381 | 6.82% | 6.56% | -0.26 [-2.31, +1.49] |
| GigaAM-v4 streaming RNN-T | Russian | 277 | 3.25% | 2.89% | -0.36 [-1.15, +0.00] |
| GigaAM-v4 streaming RNN-T | English | 381 | 6.82% | 6.56% | -0.26 [-1.59, +0.76] |
| GigaAM Multilingual-v2 | Russian | 277 | 1.81% | 1.81% | +0.00 [+0.00, +0.00] |
| GigaAM Multilingual-v2 | Kazakh | 241 | 4.56% | 4.56% | +0.00 [+0.00, +0.00] |
| GigaAM Multilingual-v2 | Uzbek | 271 | 7.75% | 8.49% | +0.74 [+0.00, +1.69] |
| GigaAM Multilingual-v2 | Kyrgyz | 252 | 6.35% | 6.75% | +0.40 [+0.00, +1.37] |
| GigaAM Multilingual-v2 | English | 381 | 4.72% | 4.72% | +0.00 [+0.00, +0.00] |

**Method.** We compare the formatted transcript from the original checkpoint in PyTorch FP32 with the quantized runtime on the same clip. WER is the sum of word substitutions, insertions and deletions divided by all reference words, after NFC and GigaAM text normalization. Confidence intervals use paired resampling of whole clips. We ran 2,000 bootstrap resamples with each of two seeds (20261127 and 20261128). The table shows the first run. Both intervals are in the per-model metrics.

**Sample.** Recordings come from the [Google FLEURS test set](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd), CC BY 4.0. The fixed sample excludes references with digits, empty references and duplicate texts within one language, and selects clips of 2–15 seconds with seed 20261127. These results describe the sampled read speech. For confidence intervals, transcripts, IDs and raw times see [v4](v4) and [Multilingual-v2](multilingual-v2). The same table is available as [CSV](quality.csv).

The four release clients were tested on Apple Silicon macOS and x86_64 Ubuntu. See the [macOS and Linux compatibility matrix](compatibility/README.md) for loader versions, cross-platform transcript differences and integration limits.
