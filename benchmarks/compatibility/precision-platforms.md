---
type: Compatibility Report
description: "CPU inference checks by GigaAM ONNX model path, precision and platform."
created: "2026-11-27"
---
# GigaAM ONNX precision on macOS, Linux and Windows

CPU smoke tests from 2026-11-27. **✓** means the named runtime decoded real speech with the published graph bytes. **—** means no verified result for that cell. A pass confirms decoding on the tested inputs. Quality and throughput measurements use larger samples and separate protocols. [Precision guide](../precision/README.md) · [Machine-readable results](precision-platforms.json)

The snapshot links below identify the weights used in the tests. The [current download table](../../README.md#model-files-on-hugging-face) opens the same weights on each model repository’s default branch.

## Released ONNX Runtime clients

All checks used ONNX Runtime 1.23.2 on CPU. **✓†** marks Windows tests through the released inference class or streaming API. The original offline Windows tests used PCM16 WAV input, one thread and basic graph optimization. The v4 harness used one thread with the published audio loaders. Both disabled the CPU arena and weight prepacking.

| Model path | Precision and weights | macOS arm64 | Linux x64 | Windows x64 | Notes |
|---|---|:---:|:---:|:---:|---|
| v4 CTC E2E | [FP32](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/3d26669a200b282be6db065c16d30fed5375183f/ctc-e2e) | ✓ | ✓ | ✓† | Packaged waveform frontend. |
| v4 CTC E2E | [FP16 storage](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/aff7eef3c9ff064455154363ba96149594d16ac0/ctc-e2e) | ✓ | ✓ | ✓† | Packaged waveform frontend. |
| v4 CTC E2E | [INT8](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/d39e2a200f596315acef7003d246b65a73e9a327/ctc-e2e) | ✓ | ✓ | ✓† | Packaged waveform frontend. |
| v4 CTC E2E | [INT4](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/9ff6f9147dd394f239c5a23fc4b0d7d1cb0312ef/ctc-e2e) | ✓ | ✓ | ✓† | Packaged waveform frontend. |
| v4 offline RNN-T | [FP32](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/3d26669a200b282be6db065c16d30fed5375183f/rnnt) | ✓ | ✓ | ✓† | Raw and formatted text. |
| v4 offline RNN-T | [FP16 storage](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/aff7eef3c9ff064455154363ba96149594d16ac0/rnnt) | ✓ | ✓ | ✓† | Raw and formatted text. |
| v4 offline RNN-T | [INT8](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/d39e2a200f596315acef7003d246b65a73e9a327/rnnt) | ✓ | ✓ | ✓† | Raw and formatted text. |
| v4 offline RNN-T | [INT4](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/9ff6f9147dd394f239c5a23fc4b0d7d1cb0312ef/rnnt) | ✓ | ✓ | ✓† | Raw and formatted text. |
| v4 streaming RNN-T | [FP32](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/3d26669a200b282be6db065c16d30fed5375183f/rnnt-streaming-640ms) | ✓ | ✓ | ✓† | 640 ms future context. |
| v4 streaming RNN-T | [FP16 storage](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/aff7eef3c9ff064455154363ba96149594d16ac0/rnnt-streaming-640ms) | ✓ | ✓ | ✓† | 640 ms future context. |
| v4 streaming RNN-T | [INT8](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/d39e2a200f596315acef7003d246b65a73e9a327/rnnt-streaming-640ms) | ✓ | ✓ | ✓† | 640 ms future context. |
| v4 streaming RNN-T | [INT4](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/9ff6f9147dd394f239c5a23fc4b0d7d1cb0312ef/rnnt-streaming-640ms) | ✓ | ✓ | ✓† | 640 ms future context. |
| Multilingual-v2 offline RNN-T | [FP32](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/f641a9a0b82b95062f495696eedb471c2e3515b2) | ✓ | ✓ | ✓† | Keep the external tensor file beside the encoder. |
| Multilingual-v2 offline RNN-T | [FP16 storage](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/2dbc6ad0b9e5ccdeb4d9a8e2e6a5507cc212ecff) | ✓ | ✓ | ✓† | Keep the external tensor file beside the encoder. |
| Multilingual-v2 offline RNN-T | [INT8](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/b37aa1110317e7ffc6d8ba737b94629a3d63d4cc) | ✓ | ✓ | ✓† | Explicit language hints and auto mode. |
| Multilingual-v2 offline RNN-T | [INT4](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/4e0da82538e6fc1fd8f27fe012451f83cc034974/int4) | ✓ | ✓ | ✓† | Explicit language hints and auto mode. |

### Environments and evidence

- **macOS 15.8.1 arm64.** Python 3.12.11, NumPy 2.5.1, SentencePiece 0.2.2 and three CPU threads. The maintained clients decoded one Russian v4 or Kazakh Multilingual-v2 FLEURS clip per variant. The runner checked both outputs for the offline RNN-T clients. [Pinned clients](https://github.com/ai-babai/gigaam-onnx/tree/3a42502cd2b54210e7742d7106afcd0fbfcdb681/clients)
- **Ubuntu 26.04 x64.** Python 3.12.16, NumPy 2.5.1, SentencePiece 0.2.2 and one CPU thread for the new formats. The packaged clients decoded Russian and English for v4. Multilingual-v2 used Russian, English, Kazakh, Kyrgyz and Uzbek with both explicit hints and auto. Streaming recorded nonempty partial text and a final flush. The original INT8 and Multilingual-v2 INT4 checks used three CPU threads in separate Linux runs. [INT8 on v4](linux-v4.json) · [Multilingual-v2 INT8](linux-multi.json) · [Multilingual-v2 INT4](../multilingual-v2/int4/linux.json)
- **Windows 11 x64.** Python 3.12.10. The original offline harness used NumPy 2.2.6 and SentencePiece 0.2.1. The original streaming run used PyTorch and torchaudio 2.10.0+cpu. The v4 precision harness used NumPy 2.5.1. Windows INT8 CTC decoded four clips with the same acoustic graph, frontend and tokens at a separate pin. [Windows release evidence](windows-cpu.json) · [Tested INT8 CTC package](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/9c1f125787db5befcfcd3c8e232a642f9b1531f6/ctc-e2e)
- **Windows Server 2022 x64.** Multilingual-v2 FP32 and FP16 storage ran with Python 3.12.10, NumPy 2.5.1, SentencePiece 0.2.2 and SoundFile 0.13.1. Both heads returned text with the explicit Kazakh hint and auto on one FLEURS clip. [Windows Multilingual-v2 run](https://github.com/ai-babai/gigaam-onnx/actions/runs/46025144581)
- **Shared inputs.** FLEURS test data at a fixed revision, decoded as 16 kHz mono PCM. The streaming client uses the upstream GigaAM runtime at a fixed source revision. [FLEURS data](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd) · [GigaAM streaming source](https://github.com/salute-developers/GigaAM/tree/e9bbf2e773eedd217c6f0d3b89b9728c41332b7d)
- **Upstream checkpoints.** The v4 offline and streaming RNN-T exports share one original checkpoint. [v4 CTC](https://huggingface.co/ai-sage/GigaAM-v4/tree/5b81216afa29c8b5a1a3e64c424c8416e18c1e5b) · [v4 RNN-T](https://huggingface.co/ai-sage/GigaAM-v4/tree/9fe3ebf50c9e0cce5eed60d546938632238b6a96) · [Multilingual-v2 RNN-T](https://huggingface.co/ai-sage/GigaAM-Multilingual-v2/tree/c21483b5e185ff32f6fb2e2c801f92f4e7fbfee8)
- **v4 INT4 pin.** The decoded runtime snapshot is `9ff6f9147dd394f239c5a23fc4b0d7d1cb0312ef`. The final card at `d72ef81e7dfb6a8d9d3ddbe0ad8d0f93dd53cdf6` updates release metadata and keeps the runtime bytes unchanged. [Final v4 INT4 card](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/d72ef81e7dfb6a8d9d3ddbe0ad8d0f93dd53cdf6)
- **Multilingual-v2 FP32 pin.** The decoded runtime snapshot is `f641a9a0b82b95062f495696eedb471c2e3515b2`. The final card at `685d629e8ecc0839938a5b097b6f91caece6c513` carries the same runtime bytes and adds the clean-download record. [Final verification](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/685d629e8ecc0839938a5b097b6f91caece6c513/evaluation/anonymous-verification.json)

## Stock Sherpa ONNX

Sherpa 2.0.3 uses its built-in 64-mel NeMo CTC frontend on these packages. Its checks use the model graph and `tokens.txt`.

| Model path | Precision | macOS arm64 | Linux x64 | Windows x64 | Notes |
|---|---|:---:|:---:|:---:|---|
| v4 CTC E2E | FP32 | ✓ | ✓ | — | Two macOS clips and one Linux clip. |
| v4 CTC E2E | FP16 storage | ✓ | ✓ | — | Two macOS clips and one Linux clip. |
| v4 CTC E2E | INT8 | ✓ | ✓ | ✓ | 32 paired Mac/Linux clips. Four Windows clips. |
| v4 CTC E2E | INT4 | ✓ | ✓ | — | Two macOS clips and one Linux clip. |

[Stock 2.0.3 release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v2.0.3) · [INT8 CTC per-clip evidence](sherpa-ctc-2.0.3.json)

## Sherpa ONNX source preview

The macOS and newer Linux precision checks used Sherpa 2.0.3 with native ORT 1.28.2 and the pinned source preview. The Linux INT8 checks and Multilingual-v2 INT4 used earlier preview builds. [Pinned preview source](https://github.com/ai-babai/sherpa-onnx/tree/411c7c3ed6232348add92537fdfd61f7f5b6588d) · [Linux preview builds and results](sherpa-preview.json)

| Model path | Precision | macOS arm64 | Linux x64 | Windows x64 | Notes |
|---|---|:---:|:---:|:---:|---|
| v4 offline RNN-T | FP32 | ✓ | ✓ | — | Offline native RNN-T. |
| v4 offline RNN-T | FP16 storage | ✓ | ✓ | — | Offline native RNN-T. |
| v4 offline RNN-T | INT8 | ✓ | ✓ | ✓ | Offline native RNN-T. |
| v4 offline RNN-T | INT4 | ✓ | ✓ | — | Offline native RNN-T. |
| v4 streaming RNN-T | FP32 | ✓ | ✓ | — | Partial output and end-of-input. |
| v4 streaming RNN-T | FP16 storage | ✓ | ✓ | — | Partial output and end-of-input. |
| v4 streaming RNN-T | INT8 | ✓ | ✓ | ✓ | Use the matching native token tables. |
| v4 streaming RNN-T | INT4 | ✓ | ✓ | ✓ | Partial output and end-of-input. |
| Multilingual-v2 offline RNN-T | FP32 | ✓ | ✓ | — | Offline native RNN-T. |
| Multilingual-v2 offline RNN-T | FP16 storage | ✓ | ✓ | — | Offline native RNN-T. |
| Multilingual-v2 offline RNN-T | INT8 | ✓ | ✓ | ✓ | Offline native RNN-T. |
| Multilingual-v2 offline RNN-T | INT4 | ✓ | ✓ | ✓ | Offline native RNN-T. |

The Windows checks used Sherpa 2.0.3 and native ORT 1.28.2. Multilingual-v2 INT4 used an earlier source build. The matching source revisions for each pass are in the JSON.

Use one matching token table, decoder and joiner per native streaming head. The INT8 streaming variants can use the tables from the pinned token snapshot. [Streaming setup](https://github.com/ai-babai/sherpa-onnx/blob/411c7c3ed6232348add92537fdfd61f7f5b6588d/scripts/gigaam-v4-streaming/README.md) · [INT8 token tables](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/00936113c4ccdb679da0212ebdef8e52b73aac48/rnnt-streaming-640ms)
