---
type: Precision Guide
description: "Choose a GigaAM ONNX precision, find the matching runtime files, and compare paired CPU measurements."
created: "2026-11-27"
---
# GigaAM ONNX precision guide

Select a model path first, then a weight format. Sizes are decimal MB for the ONNX files loaded by the bundled CPU client, including the frontend, decoder or advance graph where applicable and any ONNX external tensor file. A runnable download also needs the matching manifest, tokenizer and other small files listed in each recipe. The four default INT8 quickstarts are in the [project README](../../README.md#download-model-files-and-run-a-client).

| Model path | FP32 ONNX | FP16 storage | INT8 | INT4 |
|---|---:|---:|---:|---:|
| v4 offline CTC E2E | [952.1 MB][ctc-fp32] | [488.1 MB][ctc-fp16] | [242.5 MB][ctc-int8] | [152.4 MB][ctc-int4] |
| v4 offline RNN-T | [966.0 MB][rnnt-fp32] | [502.0 MB][rnnt-fp16] | [255.3 MB][rnnt-int8] | [166.3 MB][rnnt-int4] |
| v4 streaming RNN-T, 640 ms | [966.0 MB][stream-fp32] | [503.6 MB][stream-fp16] | [259.0 MB][stream-int8] | [168.9 MB][stream-int4] |
| Multilingual-v2 offline RNN-T | [3,118.9 MB][multi-fp32] | [1,601.1 MB][multi-fp16] | [810.8 MB][multi-int8] | [519.2 MB][multi-int4] |

The table links to the selected model graph on Hugging Face `main`. v4 CTC keeps its four graphs together in `ctc-e2e/`. Its FP32 graph is `model.onnx` on `main` and `model.fp32.onnx` in the tested snapshot. The v4 offline and streaming RNN-T INT8 packages stay in their existing directories, with the other formats in `fp32/`, `fp16/` and `int4/` subdirectories. Multilingual-v2 keeps INT8 at the root and one directory per other format.

**FP32 ONNX** stores and computes floating-point tensors in FP32. **FP16 storage** stores selected weights in FP16 and casts them to FP32 for inference. **INT8** stores selected weights as INT8 and dynamically quantizes activations for those matrix products. **INT4** packs selected matrix weights for ONNX Runtime `MatMulNBits` (symmetric blocks of 32, `accuracy_level=4`). Eligible CPU kernels can quantize activations to INT8 inside that operator. Floating-point graph I/O stays FP32 in all four formats. [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantize-to-int4uint4)

[Choosing a model by size and accuracy](accuracy-and-size.md) · [Current main paths and exact file lists](main-assets.json) · [Tested snapshots and SHA-256 provenance](assets.json)

## Download one path

Run the command for the model and format you want from the GitHub checkout. These examples pin the `v0.3.1` release and list the selected files explicitly. The [main index](main-assets.json) gives the matching file list for every format. Use the matching client and dependencies in the [project README](../../README.md#download-model-files-and-run-a-client).

```sh
weights=../gigaam-weights

hf download ai-babai/gigaam-v4-onnx --revision v0.3.1 \
  ctc-e2e/model.onnx \
  ctc-e2e/preprocessor.onnx \
  ctc-e2e/tokens.txt \
  ctc-e2e/vocab.txt \
  --local-dir "$weights/v4-ctc-fp32"

hf download ai-babai/gigaam-v4-onnx --revision v0.3.1 \
  rnnt/manifest.json \
  rnnt/v4_rnnt_advance.onnx \
  rnnt/v4_rnnt_e2e_tokenizer.model \
  rnnt/v4_rnnt_encoder.int8.onnx \
  rnnt/v4_rnnt_mel.npz \
  rnnt/v4_rnnt_raw_tokenizer.model \
  --local-dir "$weights/v4-rnnt-int8"

stream=rnnt-streaming-640ms/int4
hf download ai-babai/gigaam-v4-onnx --revision v0.3.1 \
  "$stream/streaming.json" \
  "$stream/tokens.e2e.txt" \
  "$stream/tokens.raw.txt" \
  "$stream/v4_rnnt_640ms.streaming.json" \
  "$stream/v4_rnnt_640ms.yaml" \
  "$stream/v4_rnnt_640ms_e2e_decoder.onnx" \
  "$stream/v4_rnnt_640ms_e2e_joint.onnx" \
  "$stream/v4_rnnt_640ms_e2e_tokenizer.model" \
  "$stream/v4_rnnt_640ms_raw_decoder.onnx" \
  "$stream/v4_rnnt_640ms_raw_joint.onnx" \
  "$stream/v4_rnnt_640ms_raw_tokenizer.model" \
  "$stream/v4_rnnt_640ms_stream_frontend.onnx" \
  "$stream/v4_rnnt_640ms_stream_step.int4.onnx" \
  --local-dir "$weights/v4-stream-int4"

multi=fp16
hf download ai-babai/gigaam-multilingual-v2-onnx --revision v0.3.1 \
  "$multi/config.json" \
  "$multi/manifest.json" \
  "$multi/multilingual_v2_rnnt_advance.onnx" \
  "$multi/multilingual_v2_rnnt_e2e_tokenizer.model" \
  "$multi/multilingual_v2_rnnt_encoder.onnx" \
  "$multi/multilingual_v2_rnnt_encoder.weights.bin" \
  "$multi/multilingual_v2_rnnt_mel.npz" \
  "$multi/multilingual_v2_rnnt_raw_tokenizer.model" \
  --local-dir "$weights/multi-fp16"
```

Pass the selected subdirectory to the client. In the CTC example, pass `--package ../gigaam-weights/v4-ctc-fp32/ctc-e2e` and `--model ../gigaam-weights/v4-ctc-fp32/ctc-e2e/model.onnx`. The CTC file list includes the shared `vocab.txt` for transcribe-rs. Keep the Multilingual-v2 external tensor file next to its encoder.

The graph bytes on `main` match the verified digests in [the historical index](assets.json). The pinned release cards describe the original test packages. [v4 FP32][v4-fp32-card] · [v4 FP16 storage][v4-fp16-card] · [v4 INT4][v4-int4-card] · [Multilingual-v2 FP32][multi-fp32-card] · [Multilingual-v2 FP16 storage][multi-fp16-card]

## Match the source, client and test

| Model path | Source checkpoint | Bundled CPU client and model files | Sherpa check |
|---|---|---|---|
| v4 offline CTC | [v4 CTC `5b81216`][source-ctc] | `ctc_transcribe.py` uses `preprocessor.onnx`, the selected model graph and `tokens.txt`. | [Stock NeMo CTC smoke][stock-ctc] |
| v4 offline RNN-T | [v4 RNN-T `9fe3ebf`][source-rnnt] | `rnnt_transcribe.py` uses log-mel features, the selected encoder, `v4_rnnt_advance.onnx` and both tokenizers. | [Offline CTest][offline-test] |
| v4 streaming RNN-T | [Same v4 RNN-T checkpoint `9fe3ebf`][source-rnnt] | `stream_transcribe.py` uses the YAML and `streaming.json` to load the frontend, selected step graph and both decoder/joiner pairs and tokenizers. | [Streaming CTest][stream-test] |
| Multilingual-v2 RNN-T | [Multilingual-v2 `c21483b`][source-multi] | `multi_client.py` uses mel features and the language vector, the selected encoder, the shared advance and both tokenizers. Keep the FP32 or FP16 external weight file beside the encoder. | [Offline preview scenarios][preview-scenarios] |

The [maintained GitHub clients][maintained-clients] also accept these packages. Select the CTC graph via `--model` or the v4 offline RNN-T graph via `--encoder` if more than one is present. Streaming and Multilingual-v2 use their package manifests.

Stock sherpa-onnx 2.0.3 loads v4 CTC with `from_nemo_ctc`, the 64-mel built-in frontend and `tokens.txt`. The v4 documentation proposals are open. [Python example PR #4327](https://github.com/k2-fsa/sherpa-onnx/pull/4327) · [Docs-site PR #859](https://github.com/k2-fsa/sherpa/pull/859) · [NeMo CTC docs](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/nemo/index.html)

The CTC preprocessor, the RNN-T advances and the streaming frontend, decoder and joiner graphs are byte-identical across formats for each path. Keep the selected core graph with its own manifest.

The [Sherpa source preview][preview] has separate native offline and streaming RNN-T loaders and tests. All four v4 streaming packages on `main` include the matching native token tables. Use the file list for the selected format to download its graph, companions and token tables together.

## Quality and speed

These paired runs use a fixed [FLEURS test sample][fleurs] with 16 clips per language. WER counts normalized word errors in formatted text over all reference words. Warm throughput is audio duration divided by inference time. See the [INT8 report](../README.md) and the [Multilingual-v2 INT4 comparison](../multilingual-v2/int4/README.md) for the original release measurements.


| Formatted WER | FP32 ONNX | FP16 storage | INT8 | INT4 |
|---|---:|---:|---:|---:|
| v4 CTC, RU/EN (32 clips) | 4.86% | 4.86% | 4.26% | 4.71% |
| v4 offline RNN-T, RU/EN (32 clips) | 4.86% | 4.86% | 4.86% | 4.71% |
| v4 streaming RNN-T, RU/EN (32 clips) | 5.32% | 5.32% | 5.02% | 6.38% |
| Multilingual-v2, RU/KK/UZ/KY/EN (80 clips) | 4.99% | 4.99% | 5.20% | 4.85% |

The Multilingual-v2 row uses explicit language hints. The shipped INT4 encoder (SHA-256 `6d6bac76…`, accuracy level 4) made 69 word errors in 1,422 words. The v0.3.0 table used an experimental level-1 encoder for this cell (70 errors, 4.92%). FP16 storage matched all FP32 formatted transcripts in the four model paths.

Warm throughput on Apple M4 Pro (macOS 15.8.1 arm64), ONNX Runtime 1.23.2 CPU, three intra-op threads, 2026-11-27. Each pair comes from one campaign. The Multilingual-v2 INT8/INT4 pair is from the release comparison linked above. Timing excludes model load and file I/O. Streaming measures compute on prerecorded clips with 640 ms future context.

| Workload | FP32 → FP16 storage | INT8 → INT4 |
|---|---:|---:|
| v4 CTC | 27.6× → 32.1× | 66.5× → 34.8× |
| v4 offline RNN-T | 24.1× → 24.1× | 50.9× → 27.8× |
| v4 streaming RNN-T | 17.1× → 16.6× | 17.3× → 11.6× |
| Multilingual-v2 | 4.0× → 3.7× | 16.3× → 12.5× |


## Platform checks

The platform report records CPU smoke tests for the ONNX Runtime clients, stock Sherpa CTC and the Sherpa RNN-T preview on the tested systems. [Platform checks](../compatibility/precision-platforms.md)

[fleurs]: https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd
[source-ctc]: https://huggingface.co/ai-sage/GigaAM-v4/blob/5b81216afa29c8b5a1a3e64c424c8416e18c1e5b/model.safetensors
[source-rnnt]: https://huggingface.co/ai-sage/GigaAM-v4/blob/9fe3ebf50c9e0cce5eed60d546938632238b6a96/model.safetensors
[source-multi]: https://huggingface.co/ai-sage/GigaAM-Multilingual-v2/blob/c21483b5e185ff32f6fb2e2c801f92f4e7fbfee8/model.safetensors
[stock-ctc]: ../compatibility/README.md#tested-platforms-with-released-software
[offline-test]: https://github.com/ai-babai/sherpa-onnx/blob/411c7c3ed6232348add92537fdfd61f7f5b6588d/python-api-examples/gigaam-offline-rnnt.md#offline-regression-test
[stream-test]: https://github.com/ai-babai/sherpa-onnx/blob/411c7c3ed6232348add92537fdfd61f7f5b6588d/scripts/gigaam-v4-streaming/README.md#streaming-regression-test
[preview-scenarios]: ../../quality/sherpa/README.md
[preview]: https://github.com/ai-babai/sherpa-onnx/tree/411c7c3ed6232348add92537fdfd61f7f5b6588d
[maintained-clients]: https://github.com/ai-babai/gigaam-onnx/tree/3a42502cd2b54210e7742d7106afcd0fbfcdb681/clients
[v4-fp32-card]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/3d26669a200b282be6db065c16d30fed5375183f/README.md
[v4-fp16-card]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/aff7eef3c9ff064455154363ba96149594d16ac0/README.md
[v4-int4-card]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/d72ef81e7dfb6a8d9d3ddbe0ad8d0f93dd53cdf6/README.md
[multi-fp32-card]: https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/685d629e8ecc0839938a5b097b6f91caece6c513/README.md
[multi-fp16-card]: https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/2dbc6ad0b9e5ccdeb4d9a8e2e6a5507cc212ecff/README.md
[ctc-fp32]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/ctc-e2e/model.onnx
[ctc-fp16]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/ctc-e2e/model.fp16.onnx
[ctc-int8]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/ctc-e2e/model.int8.onnx
[ctc-int4]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/ctc-e2e/model.int4.onnx
[rnnt-fp32]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/rnnt/fp32/v4_rnnt_encoder.onnx
[rnnt-fp16]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/rnnt/fp16/v4_rnnt_encoder.onnx
[rnnt-int8]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/rnnt/v4_rnnt_encoder.int8.onnx
[rnnt-int4]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/rnnt/int4/v4_rnnt_encoder.int4.onnx
[stream-fp32]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/rnnt-streaming-640ms/fp32/v4_rnnt_640ms_stream_step.onnx
[stream-fp16]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/rnnt-streaming-640ms/fp16/v4_rnnt_640ms_stream_step.onnx
[stream-int8]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/rnnt-streaming-640ms/v4_rnnt_640ms_stream_step.int8.onnx
[stream-int4]: https://huggingface.co/ai-babai/gigaam-v4-onnx/blob/main/rnnt-streaming-640ms/int4/v4_rnnt_640ms_stream_step.int4.onnx
[multi-fp32]: https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/main/fp32/multilingual_v2_rnnt_encoder.onnx
[multi-fp16]: https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/main/fp16/multilingual_v2_rnnt_encoder.onnx
[multi-int8]: https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/main/multilingual_v2_rnnt_encoder.int8.onnx
[multi-int4]: https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/main/int4/multilingual_v2_rnnt_encoder.int4.onnx
