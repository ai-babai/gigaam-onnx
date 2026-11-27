# GigaAM RNN-T in the Sherpa ONNX preview

The [GigaAM preview fork](https://github.com/ai-babai/sherpa-onnx/tree/411c7c3ed6232348add92537fdfd61f7f5b6588d) loads the published ONNX weights directly. Use the stock sherpa-onnx 2.0.3 release for v4 CTC. Build the preview for RNN-T. [Apple Silicon source install](sherpa-source-install.md) · [Offline RNN-T example](https://github.com/ai-babai/sherpa-onnx/blob/411c7c3ed6232348add92537fdfd61f7f5b6588d/python-api-examples/gigaam-offline-rnnt.md) · [Streaming example](https://github.com/ai-babai/sherpa-onnx/blob/411c7c3ed6232348add92537fdfd61f7f5b6588d/scripts/gigaam-v4-streaming/README.md)

## Selecting weights in v0.3.1

- **Stock CTC.** Pass the chosen [CTC graph](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.3.1/ctc-e2e) to `OfflineRecognizer.from_nemo_ctc` with `tokens.txt` from the same directory. FP32 uses `model.onnx`.
- **Offline RNN-T preview.** Point `OfflineRecognizer.from_gigaam` to the encoder, advance graph and head-specific tokenizer in the selected [v4](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.3.1/rnnt) or [multilingual](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/v0.3.1) package. Keep a floating-point multilingual encoder beside its `.weights.bin`.
- **Streaming RNN-T preview.** Pass the encoder, decoder, joiner and token table from one [640 ms package](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.3.1/rnnt-streaming-640ms). Sherpa finds the frontend beside its encoder. The decoder, joiner and tokens must belong to the same head.

The pinned INT8 recipes in the [offline PR](https://github.com/k2-fsa/sherpa-onnx/pull/4328) and [streaming PR](https://github.com/k2-fsa/sherpa-onnx/pull/4329) use the same graph bytes.

## Verified native preview

All three operating systems ran the same source revision, ONNX Runtime 1.28.2 CPU, and pinned recordings. A check means every case in that column passed. Both streaming heads returned partial text and a final result. A mixed token table or joiner was rejected at construction. [Scenario runner](../../quality/sherpa/README.md) · [Builds and results](sherpa-scenarios.json)

| Platform | Shared scenarios (12) | Offline INT8 (2) | Streaming INT8 (2) | Streaming INT4 (2) | Note |
|---|:---:|:---:|:---:|:---:|---|
| macOS 15.8.1 arm64 | ✓ | ✓ | ✓ | ✓ | Source-built Python wheel |
| Ubuntu 26.04.1 x64 | ✓ | ✓ | ✓ | ✓ | Native Python module |
| Windows 11 x64 | ✓ | ✓ | ✓ | ✓ | MSVC build, both native CLIs also decoded WAV files |

The preview also reproduced the upstream transcripts for one GigaAM-v3 NeMo CTC recording and all five Chinese–English Zipformer recordings on each OS, including the 8 kHz sample. These six recordings exercise existing recognizers. [GigaAM-v3 example](https://github.com/k2-fsa/sherpa-onnx/pull/2901) · [Upstream Zipformer test](https://github.com/k2-fsa/sherpa-onnx/blob/935e5ded3d949a8b557ac44ef4dad03a7d2038e7/.github/scripts/test-python-asr-streaming.sh#L172-L181)

## Formatted-text accuracy

We measured formatted-text WER on 16 FLEURS test clips per language using the earlier inference builds recorded in the [benchmark data](sherpa-preview.json). Both runtimes used the same quantized weights and recordings. The released client used ONNX Runtime 1.23.2 and the Sherpa source build used native ONNX Runtime 1.28.2. Multilingual rows use an explicit language hint. [All languages, raw outputs, auto mode and paired bootstrap intervals](sherpa-preview.json)

| Model | Clips | Release client on Mac | Sherpa on Mac | Sherpa on Linux | Note |
|---|---:|---:|---:|---:|---|
| v4 offline INT8 | 32 | 4.86% | 4.26% | 3.95% | RU and EN |
| v4 streaming INT8 | 32 | 5.02% | 5.02% | 5.78% | 640 ms packets |
| Multilingual-v2 INT8 | 80 | 5.20% | 5.06% | 5.06% | RU, KK, UZ, KY, EN |
| Multilingual-v2 INT4 | 80 | 4.85% | 4.92% | 4.99% | Same five languages |

The three offline RNN-T variants returned 704 outputs across both heads and language modes. Mac matched the release client on 653/704 outputs and Linux on 648/704. The two native builds matched each other on 635/704. In a separate nine-scenario streaming check on Mac, 316/322 partial and final texts matched the GigaAM API. Reset and end-of-input checks passed, including a final packet containing a single mel frame. Packet boundaries can affect decoding, so comparisons use the same packet sizes on both sides.

### Cross-platform smoke test

The offline recognizer ran on macOS arm64 and Linux x64 with native ONNX Runtime 1.28.2. Both builds returned identical text and token pieces on all 44 matched outputs. The sample covers v4 INT8 in Russian and English and Multilingual-v2 INT8/INT4 in five languages, with explicit and automatic language selection. The builds used different source revisions with matching inference code. The streaming recognizer produced partial and final text and handled reset on both systems.

With the released GigaAM API and the Sherpa preview on the same Linux host, streaming WER was 5.32% and 5.78%, respectively (35 and 38 errors in 658 words). All 32 clips produced formatted text.
