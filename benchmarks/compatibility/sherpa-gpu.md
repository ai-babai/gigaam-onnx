---
type: Compatibility guide
description: "Select a Sherpa execution provider for GigaAM-v4 CTC and check that inference uses it."
created: "2026-11-28"
---
# GigaAM with Sherpa execution providers

The GigaAM-v4 CTC client passes `--sherpa-provider` to Sherpa's native NeMo CTC recognizer. Use it with `--sherpa`. Omitting the option keeps the CPU default. The direct ONNX Runtime client also uses CPU. Sherpa uses its own 16 kHz, 64-mel frontend. The Sherpa path needs NumPy and FFmpeg with the native Sherpa build.

| Platform | GigaAM v4 CTC graph | Sherpa runtime | Result |
|---|---|---|---|
| Linux x86-64, NVIDIA A40 | Published FP32, FP16 storage, INT8 and INT4 | Sherpa 2.0.3, CUDA 12, cuDNN 9, native ORT 1.28.2 | The stock C API returned the same CPU and CUDA text on both RU/EN clips in each format, twice. The Python client also decoded FP32 with CUDA. |
| macOS 15.8.1 arm64, Apple M4 Pro | Published FP32 and INT8 | Sherpa 2.0.3 CoreML test build | CoreML execution failed at a dynamic reshape. Use CPU for these packages on macOS. |

A native provider request can fall back to CPU when the provider is unavailable. ONNX Runtime can also run individual nodes on CPU. Check the native profile to confirm where inference ran. The process may return a transcript after a CPU fallback.

## Linux CUDA

Use an x86-64 environment with an NVIDIA driver, CUDA 12.8, cuDNN 9, Python 3.12 and FFmpeg. The 2026-11-28 test used Ubuntu 24.04.3, driver 595.112.07, cuDNN 9.8.0, Python 3.12.3 and FFmpeg 6.1.1. Install the CUDA build of Sherpa in a clean Python environment. [Sherpa CUDA wheel index](https://k2-fsa.github.io/sherpa/onnx/cuda.html) · [Verified wheel for CPython 3.12](https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/blob/c9f5ed2cf4b2663d23f4e93ab7f642b176dc62e3/cuda/2.0.3/sherpa_onnx-2.0.3%2Bcuda12.cudnn9.onnxruntime1.28.2-cp312-cp312-linux_x86_64.whl) · [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

Run from the GigaAM ONNX checkout. Install the pinned Hugging Face CLI from the [project quickstart](../../README.md#download-model-files-and-run-a-client) in this environment or use it from your existing environment.

```sh
python -m pip install 'sherpa-onnx==2.0.3+cuda12.cudnn9.onnxruntime1.28.2' 'numpy==2.1.2' \
  -f https://k2-fsa.github.io/sherpa/onnx/cuda.html

weights=../gigaam-weights/v4-gpu
hf download ai-babai/gigaam-v4-onnx --revision v0.3.1 \
  ctc-e2e/model.onnx ctc-e2e/tokens.txt --local-dir "$weights"

profile="$(mktemp -d)"
printf 'LogSeverityLevel=0\nProfilingFilePrefix=%s\n' "$profile/ctc" > "$profile/provider.conf"
python clients/ctc_transcribe.py clip.wav --package "$weights/ctc-e2e" \
  --model "$weights/ctc-e2e/model.onnx" --sherpa \
  --sherpa-provider "cuda:$profile/provider.conf"
```

Inspect the JSON profile in `"$profile"` for node events that ran on `CUDAExecutionProvider`. Copy events alone do not establish GPU compute. With the same native package, `--sherpa-provider cuda` requests CUDA with default options and omits the profile.

The FP32 Python run recorded 2,001 CUDA and 850 CPU node events in ORT and 1,586 GPU kernel launches in Nsight Systems. In the stock C API INT4 run, all 136 `MatMulNBits` events used CUDA. In the INT8 run, 136 `MatMulInteger` events stayed on CPU while other operators ran on CUDA. The INT8 and INT4 runs returned identical text with different token timestamps. The Python CTC client prints text.
