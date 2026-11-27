---
type: Source Install Recipe
description: "Build the GigaAM-enabled sherpa-onnx Python package from source on Apple Silicon and run an offline smoke test."
created: "2026-11-27"
---
# Build sherpa-onnx for GigaAM on macOS arm64

Tested on macOS 15.8.1 with AppleClang 17.0.0, CMake 4.3.2, and CPython 3.12.11. This builds sherpa-onnx 2.0.3 with its bundled CPU ONNX Runtime 1.28.2. The commands below use Git 2.54.0, uv 0.8.4 and the Apple compiler toolchain.

Build from the GigaAM preview fork. The published sherpa-onnx 2.0.3 wheels support the v4 CTC model. The RNN-T loaders in this guide require the preview source.

```bash
revision=411c7c3ed6232348add92537fdfd61f7f5b6588d
scratch="$(mktemp -d /tmp/sherpa-gigaam.XXXXXX)"
mkdir "$scratch/src"
git clone --quiet --filter=blob:none --single-branch --branch preview/gigaam-rnnt https://github.com/ai-babai/sherpa-onnx.git "$scratch/repo"
git -C "$scratch/repo" archive "$revision" | tar -xf - -C "$scratch/src"

uv venv --python 3.12.11 "$scratch/.venv"
python_bin="$scratch/.venv/bin/python"
uv pip install --python "$python_bin" 'pip==26.3' 'setuptools==84.2.1' 'wheel==0.48.0'

(
  cd "$scratch/src"
  unset SHERPA_ONNX_SPLIT_PYTHON_PACKAGE SHERPA_ONNX_IS_FOR_PYPI CMAKE_GENERATOR
  export MACOSX_DEPLOYMENT_TARGET=14.0
  export SHERPA_ONNX_MAKE_ARGS=-j11
  export SHERPA_ONNX_CMAKE_ARGS="
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    '-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -DNDEBUG'
    '-DCMAKE_C_FLAGS_RELWITHDEBINFO=-O2 -DNDEBUG'
    -DCMAKE_OSX_ARCHITECTURES=arm64
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
    -DPython_EXECUTABLE=$python_bin
    -DPYTHON_EXECUTABLE=$python_bin
    -DSHERPA_ONNX_ENABLE_BINARY=OFF
    -DSHERPA_ONNX_ENABLE_C_API=OFF
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF
    -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF
    -DSHERPA_ONNX_ENABLE_TTS=ON
    -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF
    -DSHERPA_ONNX_ENABLE_GPU=OFF
    -DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=OFF
  "
  "$python_bin" -m pip install --no-cache-dir --no-index --no-deps --no-build-isolation .
)
"$python_bin" -m pip check
```

CMake downloads the pinned dependencies. Set both Python executable options so CMake and pybind11 use the venv interpreter. Leave `SHERPA_ONNX_SPLIT_PYTHON_PACKAGE` unset to build a single wheel with the compiled extension and ORT.

## Run an offline smoke test

Set the model directory to the extracted GigaAM-v4 offline RNN-T INT8 package. Use a mono, 16 kHz PCM16 WAV.

```bash
"$python_bin" -I - /path/to/gigaam-v4-rnnt-package /path/to/audio.wav <<'PY'
import sys
import wave
from array import array
from pathlib import Path

import sherpa_onnx

model = Path(sys.argv[1])
with wave.open(sys.argv[2], "rb") as wav:
    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
    pcm = array("h", wav.readframes(wav.getnframes()))
if sys.byteorder != "little":
    pcm.byteswap()

recognizer = sherpa_onnx.OfflineRecognizer.from_gigaam(
    encoder=str(model / "v4_rnnt_encoder.int8.onnx"),
    advance=str(model / "v4_rnnt_advance.onnx"),
    tokenizer=str(model / "v4_rnnt_e2e_tokenizer.model"),
    num_threads=3,
)
stream = recognizer.create_stream()
stream.accept_waveform(16000, [sample / 32768.0 for sample in pcm])
recognizer.decode_stream(stream)
assert stream.result.text.strip()
print(stream.result.text)
PY
```

[Upstream Python source installation](https://k2-fsa.github.io/sherpa/onnx/python/install.html#method-4-from-source) · [GigaAM-v4 CTC recipe](https://github.com/ai-babai/sherpa-onnx/blob/411c7c3ed6232348add92537fdfd61f7f5b6588d/python-api-examples/gigaam-v4-ctc.md) · [Offline RNN-T recipe](https://github.com/ai-babai/sherpa-onnx/blob/411c7c3ed6232348add92537fdfd61f7f5b6588d/python-api-examples/gigaam-offline-rnnt.md) · [Streaming RNN-T recipe](https://github.com/ai-babai/sherpa-onnx/blob/411c7c3ed6232348add92537fdfd61f7f5b6588d/scripts/gigaam-v4-streaming/README.md)
