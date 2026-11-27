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

The optional Hub branches are `float`, `fp16` and v4 `int4`. The current v4 INT8 pin includes the streaming token tables. Multilingual-v2 INT8 uses `v0.2.0` and INT4 uses the `int4/` directory at `v0.2.1`.

**FP32 ONNX** stores and computes floating-point tensors in FP32. **FP16 storage** stores selected weights in FP16 and casts them to FP32 for inference. **INT8** stores selected weights as INT8 and dynamically quantizes activations for those matrix products. **INT4** packs selected matrix weights for ONNX Runtime `MatMulNBits` (symmetric blocks of 32, `accuracy_level=4`). Eligible CPU kernels can quantize activations to INT8 inside that operator. Floating-point graph I/O stays FP32 in all four formats. [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantize-to-int4uint4)

[Machine-readable index of weights, companion files and SHA-256 hashes](assets.json)

## Download one path

Run the matching command from the GitHub checkout. Use the client and dependencies listed in the pinned model card. The sibling directories keep separate copies of formats that use the same filenames.

Pinned release cards — [v4 FP32][v4-fp32-card] · [v4 FP16 storage][v4-fp16-card] · [v4 INT4][v4-int4-card] · [Multilingual-v2 FP32][multi-fp32-card] · [Multilingual-v2 FP16 storage][multi-fp16-card].

```sh
weights=../gigaam-weights

hf download ai-babai/gigaam-v4-onnx --revision aff7eef3c9ff064455154363ba96149594d16ac0 \
  --include 'ctc-e2e/*' --local-dir "$weights/v4-ctc-fp16"

hf download ai-babai/gigaam-v4-onnx --revision 9c1f125787db5befcfcd3c8e232a642f9b1531f6 \
  --include 'rnnt/*' --local-dir "$weights/v4-rnnt-int8"

hf download ai-babai/gigaam-v4-onnx --revision 9ff6f9147dd394f239c5a23fc4b0d7d1cb0312ef \
  --include 'rnnt-streaming-640ms/*' --local-dir "$weights/v4-stream-int4"

hf download ai-babai/gigaam-multilingual-v2-onnx --revision f641a9a0b82b95062f495696eedb471c2e3515b2 \
  --include 'multilingual_v2_rnnt*' --include 'config.json' --include 'manifest.json' --include 'multi_client.py' --include 'requirements.txt' --local-dir "$weights/multi-fp32"
```

The v4 INT4 and Multilingual-v2 FP32 commands pin the tested runtime commits. Their [v4 INT4][v4-int4-card] and [Multilingual-v2 FP32][multi-fp32-card] cards retain the same runtime files.

## Match the source, client and test

| Model path | Source checkpoint | Bundled CPU client and model files | Sherpa check |
|---|---|---|---|
| v4 offline CTC | [v4 CTC `5b81216`][source-ctc] | `ctc_transcribe.py` uses `preprocessor.onnx`, `model.<format>.onnx` and `tokens.txt`. | [Stock NeMo CTC smoke][stock-ctc] |
| v4 offline RNN-T | [v4 RNN-T `9fe3ebf`][source-rnnt] | `rnnt_transcribe.py` uses log-mel features, the selected encoder, `v4_rnnt_advance.onnx` and both tokenizers. | [Offline CTest][offline-test] |
| v4 streaming RNN-T | [Same v4 RNN-T checkpoint `9fe3ebf`][source-rnnt] | `stream_transcribe.py` uses the YAML and `streaming.json` to load the frontend, selected step graph and both decoder/joiner pairs and tokenizers. | [Streaming CTest][stream-test] |
| Multilingual-v2 RNN-T | [Multilingual-v2 `c21483b`][source-multi] | `multi_client.py` uses mel features and the language vector, the selected encoder, the shared advance and both tokenizers. Keep the FP32 or FP16 external weight file beside the encoder. | [Offline preview scenarios][preview-scenarios] |

The [maintained GitHub clients][maintained-clients] also accept these packages. Select the CTC graph via `--model` or the v4 offline RNN-T graph via `--encoder` if more than one is present. Streaming and Multilingual-v2 use their package manifests.

Stock sherpa-onnx 2.0.3 loads v4 CTC with `from_nemo_ctc`, the 64-mel built-in frontend and `tokens.txt`. The v4 documentation proposals are open. [Python example PR #4327](https://github.com/k2-fsa/sherpa-onnx/pull/4327) · [Docs-site PR #859](https://github.com/k2-fsa/sherpa/pull/859) · [NeMo CTC docs](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/nemo/index.html)

The CTC preprocessor, the RNN-T advances and the streaming frontend, decoder and joiner graphs are byte-identical across formats for each path. Keep the selected core graph with its own manifest.

The [Sherpa source preview][preview] has separate native offline and streaming RNN-T loaders and tests. All v4 streaming pins in the table include matching native token tables. Select `rnnt-streaming-640ms/*` to get the model and tables in one download.

## Quality and speed

These paired runs use a fixed [FLEURS test sample][fleurs] with 16 clips per language. WER counts normalized word errors in formatted text over all reference words. Warm throughput is audio duration divided by inference time. See the [INT8 report](../README.md) and the [Multilingual-v2 INT4 comparison](../multilingual-v2/int4/README.md) for the original release measurements.


| Formatted WER | FP32 ONNX | FP16 storage | INT8 | INT4 |
|---|---:|---:|---:|---:|
| v4 CTC, RU/EN (32 clips) | 4.86% | 4.86% | 4.26% | 4.71% |
| v4 offline RNN-T, RU/EN (32 clips) | 4.86% | 4.86% | 4.86% | 4.71% |
| v4 streaming RNN-T, RU/EN (32 clips) | 5.32% | 5.32% | 5.02% | 6.38% |
| Multilingual-v2, RU/KK/UZ/KY/EN (80 clips) | 4.99% | 4.99% | 5.20% | 4.92% |

The Multilingual-v2 WER row uses explicit language hints and the four-format run. The separate INT8/INT4 release run reported 4.85% for INT4. FP16 storage matched all FP32 formatted transcripts in the four model paths.

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
[multi-fp16-card]: https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/4e29c392dd7b9fb22b9950703ba8794a4be09bbd/README.md
[ctc-fp32]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/3d26669a200b282be6db065c16d30fed5375183f/ctc-e2e
[ctc-fp16]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/aff7eef3c9ff064455154363ba96149594d16ac0/ctc-e2e
[ctc-int8]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/9c1f125787db5befcfcd3c8e232a642f9b1531f6/ctc-e2e
[ctc-int4]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/d72ef81e7dfb6a8d9d3ddbe0ad8d0f93dd53cdf6/ctc-e2e
[rnnt-fp32]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/3d26669a200b282be6db065c16d30fed5375183f/rnnt
[rnnt-fp16]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/aff7eef3c9ff064455154363ba96149594d16ac0/rnnt
[rnnt-int8]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/9c1f125787db5befcfcd3c8e232a642f9b1531f6/rnnt
[rnnt-int4]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/d72ef81e7dfb6a8d9d3ddbe0ad8d0f93dd53cdf6/rnnt
[stream-fp32]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/3d26669a200b282be6db065c16d30fed5375183f/rnnt-streaming-640ms
[stream-fp16]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/aff7eef3c9ff064455154363ba96149594d16ac0/rnnt-streaming-640ms
[stream-int8]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/9c1f125787db5befcfcd3c8e232a642f9b1531f6/rnnt-streaming-640ms
[stream-int4]: https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/d72ef81e7dfb6a8d9d3ddbe0ad8d0f93dd53cdf6/rnnt-streaming-640ms
[multi-fp32]: https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/685d629e8ecc0839938a5b097b6f91caece6c513
[multi-fp16]: https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/4e29c392dd7b9fb22b9950703ba8794a4be09bbd
[multi-int8]: https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/b37aa1110317e7ffc6d8ba737b94629a3d63d4cc
[multi-int4]: https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/4e0da82538e6fc1fd8f27fe012451f83cc034974/int4
