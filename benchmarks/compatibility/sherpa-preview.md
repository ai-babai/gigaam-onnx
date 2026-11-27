# GigaAM RNN-T in the Sherpa ONNX preview

The [GigaAM preview fork](https://github.com/ai-babai/sherpa-onnx/tree/c8a28e207b87a408ea758347c475b4d1ee82701a) loads the published ONNX weights directly. Use the stock sherpa-onnx 2.0.3 release for v4 CTC. Build the preview for RNN-T. [Apple Silicon source install](sherpa-source-install.md) · [Offline RNN-T example](https://github.com/ai-babai/sherpa-onnx/blob/c8a28e207b87a408ea758347c475b4d1ee82701a/python-api-examples/gigaam-offline-rnnt.md) · [Streaming example](https://github.com/ai-babai/sherpa-onnx/blob/c8a28e207b87a408ea758347c475b4d1ee82701a/scripts/gigaam-v4-streaming/README.md)

## Accuracy on Apple Silicon

Formatted-text WER on 16 FLEURS test clips per language. Both runtimes used the same quantized weights and recordings. The released client used ONNX Runtime 1.23.2 and the Sherpa source build used native ONNX Runtime 1.28.2. Multilingual rows use an explicit language hint. [All languages, raw outputs, auto mode and paired bootstrap intervals](sherpa-preview.json)

| Model | Clips | Release client | Sherpa preview | Note |
|---|---:|---:|---:|---|
| v4 offline INT8 | 32 | 4.86% | 4.26% | Russian and English |
| v4 streaming INT8 | 32 | 5.02% | 5.02% | Same 33 word errors in aggregate, 640 ms packets |
| Multilingual-v2 INT8 | 80 | 5.20% | 5.06% | RU, KK, UZ, KY, EN |
| Multilingual-v2 INT4 | 80 | 4.85% | 4.92% | Same five languages |

The three offline RNN-T variants returned 704 outputs across both heads and language modes. Of these, 653 matched the release client exactly. In a separate nine-scenario streaming check, 316/322 partial and final texts matched the GigaAM API. Reset and end-of-input checks passed, including a final packet containing a single mel frame. Packet boundaries can affect decoding, so comparisons use the same packet sizes on both sides.

### Cross-platform smoke test

The offline recognizer ran on macOS arm64 and Linux x64 with native ONNX Runtime 1.28.2. Both builds returned identical text and token pieces on all 44 matched outputs. The sample covers v4 INT8 in Russian and English and Multilingual-v2 INT8/INT4 in five languages, with explicit and automatic language selection. The builds used different source revisions with matching inference code. The streaming recognizer produced partial and final text and handled reset on both systems.
