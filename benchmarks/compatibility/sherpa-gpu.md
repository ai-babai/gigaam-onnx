---
type: Compatibility guide
description: "Select a Sherpa execution provider for GigaAM-v4 CTC and check that inference uses it."
created: "2026-11-28"
---
# GigaAM with Sherpa execution providers

The GigaAM-v4 CTC client passes `--sherpa-provider` to Sherpa's native NeMo CTC recognizer. Use it with `--sherpa`. Omitting the option keeps the CPU default. The direct ONNX Runtime client also uses CPU. Both paths read standard 16 kHz mono PCM16 WAV files directly and use FFmpeg for other inputs. Sherpa uses its own 16 kHz, 64-mel frontend.

| Platform | Published v4 CTC graph | Observed execution |
|---|---|---|
| Linux x86-64, NVIDIA A40 | FP32 | Mixed CUDA and CPU. The Python CLI recorded 1,586 GPU kernel launches. |
| Linux x86-64, NVIDIA A40 | FP16 storage | Mixed CUDA and CPU. ORT recorded 2,001 CUDA and 850 CPU node events. |
| Linux x86-64, NVIDIA A40 | INT8 | Mixed with substantial CPU work. ORT ran 145 `DynamicQuantizeLinear` and 136 `MatMulInteger` events on CPU and recorded 435 CUDA copy events. |
| Linux x86-64, NVIDIA A40 | INT4 | Mixed CUDA and CPU. All 136 `MatMulNBits` events in the profile used CUDA. |
| macOS 15.8.1 arm64, Apple M4 Pro | FP32, FP16 storage, INT8 and INT4 | CoreML inference failed with the published graphs. Use the CPU CTC client on macOS. |

The stock Sherpa 2.0.3 Linux C API returned the same CPU and CUDA text on both RU/EN clips in each format, twice. INT8 and INT4 token timestamps differed. The Python CTC client prints text.

For ONNX models, Sherpa can fall back to CPU when the requested provider is unavailable. ONNX Runtime can also run individual nodes on CPU. Check the native profile to confirm where inference ran. The process may return a transcript after a CPU fallback.

The direct PCM16 reader produced byte-identical samples on 80 checked WAVs. On an M4 Pro with stock Sherpa 2.0.3, INT8 CTC and three threads, a paired warmed run under background load measured a median of 0.37 ms for direct WAV reading and 41.05 ms for FFmpeg. Median file-to-text time was 88.62 ms and 135.39 ms respectively across 12 pairs. These measurements cover audio reading and application latency at the same provider.

## Linux CUDA

Use an x86-64 environment with an NVIDIA driver, CUDA 12.8 and cuDNN 9. Install the CUDA build of Sherpa in a clean Python 3.12 environment. Install FFmpeg if you will use other audio formats. The 2026-11-28 test used Ubuntu 24.04.3, driver 595.112.07, cuDNN 9.8.0, Python 3.12.3 and FFmpeg 6.1.1. [Sherpa CUDA wheel index](https://k2-fsa.github.io/sherpa/onnx/cuda.html) · [Verified wheel for CPython 3.12](https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/blob/c9f5ed2cf4b2663d23f4e93ab7f642b176dc62e3/cuda/2.0.3/sherpa_onnx-2.0.3%2Bcuda12.cudnn9.onnxruntime1.28.2-cp312-cp312-linux_x86_64.whl) · [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

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

The FP32 Python run recorded 2,001 CUDA and 850 CPU node events in ORT. To check physical GPU work, prefix the decode command with `nsys profile --trace=cuda --sample=none -o ctc-cuda` and inspect `nsys stats --report cuda_gpu_kern_sum ctc-cuda.nsys-rep`. Nsight Systems captured the GPU kernel launches in the table.

## Linux CUDA RNN-T preview

The NVIDIA A40 tests used a source build at `b898c1b6` with native ORT 1.28.2, CUDA 12 and cuDNN 9. The v4 and Multilingual-v2 offline recognizers and the v4 streaming recognizer passed 28 native CPU and CUDA cases with the published INT8 packages, including both heads. [Tested Sherpa revision](https://github.com/ai-babai/sherpa-onnx/tree/b898c1b6b56bb0f61c36e29ad96067a4904aee95) · [Exact model file lists](../precision/main-assets.json)

Run this in a Python 3.12 environment with the CUDA dependencies above, CMake and a C++ compiler. The checkout and build directory keep the native runtime separate from the stock Python wheel.

```sh
sherpa_src="$(mktemp -d)/sherpa-onnx"
git clone --branch preview/gigaam-accelerators --single-branch \
  https://github.com/ai-babai/sherpa-onnx.git "$sherpa_src"
git -C "$sherpa_src" checkout --detach b898c1b6b56bb0f61c36e29ad96067a4904aee95
build="$sherpa_src/build-cuda"
cmake -S "$sherpa_src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_GPU=ON -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_PYTHON=ON -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build "$build" --parallel 8 --target _sherpa_onnx sherpa-onnx-offline sherpa-onnx

module="$sherpa_src/python-stage"
mkdir -p "$module"
cp -R "$sherpa_src/sherpa-onnx/python/sherpa_onnx" "$module/"
mkdir -p "$module/sherpa_onnx/lib"
cp "$build"/lib/_sherpa_onnx*.so "$module/sherpa_onnx/lib/"
ln -s "$build"/_deps/onnxruntime-src/lib/libonnxruntime*.so "$module/sherpa_onnx/lib/"
export PYTHONPATH="$module"
export LD_LIBRARY_PATH="$module/sherpa_onnx/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import sherpa_onnx; print(sherpa_onnx.__file__)'
```

Pass `provider="cuda"` in the native Python API or `--provider=cuda` in the built CLI. Use `OfflineRecognizer.from_gigaam` for v4 offline and Multilingual-v2, and `OnlineRecognizer.from_transducer(model_type="gigaam_v4", ...)` for v4 streaming. The recipes show the exact graph and tokenizer arguments. For profiling, use `cuda:/absolute/provider.conf` with the options above. ORT writes one profile per graph. [Offline recipe](https://github.com/ai-babai/sherpa-onnx/blob/b898c1b6b56bb0f61c36e29ad96067a4904aee95/python-api-examples/gigaam-offline-rnnt.md) · [Streaming recipe](https://github.com/ai-babai/sherpa-onnx/blob/b898c1b6b56bb0f61c36e29ad96067a4904aee95/scripts/gigaam-v4-streaming/README.md)

Nsight recorded GPU kernels for each of the 16 model-and-precision combinations on the A40. ORT still mixed CUDA and CPU in the offline encoder and advance sessions. With streaming FP32, the decoder and joiner ran entirely on CUDA in the profiled clip, while the frontend and encoder mixed providers. In 22 streaming encoder calls, INT8 recorded 9,856 copy events and FP32 recorded 1,078. Keep the published INT8 streaming path on CPU on this host when throughput is the priority.

### Paired CUDA measurements on the A40

Both providers ran with three threads on the same Linux host. These formatted FLEURS runs include the first decode and exclude model initialization, file I/O and profiling. Each cell gives the ratio of total CPU decode time to total CUDA decode time, followed by the number of byte-identical CPU and CUDA transcripts.

| Model path | FP32 | FP16 storage | INT8 | INT4 |
|---|---:|---:|---:|---:|
| v4 CTC | 4.70× · 32/32 | 5.90× · 32/32 | 1.03× · 28/32 | 7.29× · 31/32 |
| v4 offline RNN-T | 6.65× · 32/32 | 7.04× · 32/32 | 1.38× · 28/32 | 8.22× · 32/32 |
| v4 streaming RNN-T | 2.99× · 32/32 | 2.96× · 32/32 | 0.82× · 29/32 | 2.23× · 29/32 |
| Multilingual-v2 offline RNN-T, auto | 10.53× · 80/80 | 12.23× · 80/80 | 0.95× · 77/80 | 11.86× · 79/80 |
| Multilingual-v2 offline RNN-T, explicit language | 11.37× · 80/80 | 14.10× · 80/80 | 1.09× · 72/80 | 10.40× · 76/80 |

Each Multilingual-v2 run used 16 recordings per language in Russian, English, Kazakh, Uzbek and Kyrgyz. With an explicit hint, FP16 storage produced 71 word errors out of 1,422 reference words on both CPU and CUDA. [Error counts and timings for all 20 paired runs](sherpa-cuda-a40.json)

For exact transcript agreement on this sample, choose FP32 or FP16 storage with CUDA for CTC and streaming. The offline v4 INT4 pair also agreed on every clip. Keep CPU as the starting point for INT8 CTC, INT8 streaming and Multilingual-v2 INT8 auto on this host.

### CPU thread count for streaming

On a separate Linux VM with three AMD EPYC 7763 vCPUs, one INT8 streaming recognizer ran fastest with `provider="cpu", num_threads=1`. Across six paired runs on 128.52 seconds of audio, median decode time was 19.24 seconds with one thread, 21.64 with two and 34.56 with three. All 220 transcript, timestamp and lifecycle events agreed across the three settings. The profiled build spent 99.47% of decode time in ORT at one thread. Choose the thread count on the deployment host.

## macOS CoreML

All four published v4 CTC graphs failed in native Sherpa at `88b59ec6` with ORT 1.23.2. Both bare `coreml` and explicit `ModelFormat=MLProgram`, `MLComputeUnits=CPUAndGPU` failed on the first Russian decode. The Sherpa preview at `b898c1b6` also failed on stock CTC FP32 with ORT 1.28.2. With MLProgram on that preview, the published v4 offline and Multilingual-v2 INT8 encoders failed at the attention reshape and the v4 streaming INT8 encoder failed CoreML execution-plan creation. The CPU scenario suite passed all 12 cases on that host. [CPU scenario suite](../../quality/sherpa/README.md)

### Experimental native Core ML packages

These packages use separate Apple Silicon Core ML models. Compile the portable `.mlpackage` locally with Xcode. The package recipes pin the native Sherpa build, ORT SDK and model files. Hugging Face `v0.4.0` resolves to `43afef6286ce01771b979dd0c3e817eede012a6b`, the immutable snapshot used below.

#### Build native Sherpa on Apple Silicon

Use macOS 15 on arm64 with Xcode command-line tools, CMake, Git and Python 3.12. The checked Apple build uses native ORT 1.23.2. The temporary build directory keeps the SDK and native artifacts outside the GigaAM checkout. [Combined Sherpa source](https://github.com/ai-babai/sherpa-onnx/tree/4ce634bd4104a586e78c0f368b85e93d41a6cc3d)

```sh
work="$(mktemp -d)"
sherpa_src="$work/sherpa-onnx"
sherpa_build="$work/build"
git clone --branch preview/gigaam-accelerators --single-branch \
  https://github.com/ai-babai/sherpa-onnx.git "$sherpa_src"
git -C "$sherpa_src" checkout --detach 4ce634bd4104a586e78c0f368b85e93d41a6cc3d
python3.12 -m venv "$work/venv"
. "$work/venv/bin/activate"
python -m pip install numpy==2.5.5 huggingface_hub==1.3.5 hf-xet==1.8.0 typer-slim==0.21.1 click==8.2.1

sdk=onnxruntime-osx-arm64-1.23.2
sdk_rev=a0df02c5bc9e712256cbaaaf764db73964525478
curl -fL "https://huggingface.co/csukuangfj/onnxruntime-libs/resolve/$sdk_rev/1.23.2/$sdk.zip" -o "$work/$sdk.zip"
printf '%s  %s\n' 3a7571f0d98f4ff661adedf4ccdb598560fe177d49d4b692b3849e7b841fff70 "$work/$sdk.zip" | shasum -a 256 -c -
unzip -q "$work/$sdk.zip" -d "$work"
ort="$work/$sdk"
codesign --force --sign - "$ort/lib/libonnxruntime.1.23.2.dylib"

cmake -S "$sherpa_src" -B "$sherpa_build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
  -DSHERPA_ONNX_ENABLE_GIGAAM_COREML=ON -DSHERPA_ONNX_ENABLE_PYTHON=ON \
  -DSHERPA_ONNX_ENABLE_BINARY=ON -DSHERPA_ONNX_ENABLE_TESTS=ON -DSHERPA_ONNX_ENABLE_TTS=ON \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=OFF -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=OFF \
  -DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=OFF -DFETCHCONTENT_SOURCE_DIR_ONNXRUNTIME="$ort" \
  -DPython_EXECUTABLE="$(command -v python)" -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build "$sherpa_build" --parallel 8 --target _sherpa_onnx sherpa-onnx sherpa-onnx-offline

pylib="$sherpa_src/sherpa-onnx/python/sherpa_onnx/lib"
mkdir -p "$pylib"
cp "$sherpa_build"/lib/_sherpa_onnx*.so "$pylib/"
cp "$ort/lib/libonnxruntime.1.23.2.dylib" "$pylib/"
export PYTHONPATH="$sherpa_src/sherpa-onnx/python"
python -c 'import sherpa_onnx; print(sherpa_onnx.__file__)'
```

The ad-hoc signature applies to the extracted local dylib. Keep the native extension and the versioned library together. Use `DYLD_LIBRARY_PATH="$ort/lib"` for the native CLI.

#### Offline CTC

Use the compiled T=1600 model with the maintained CTC client. Its limit is 1,600 mel frames, 256,079 samples at 16 kHz. The native bridge rejects 256,080 samples because they produce 1,601 frames. Split longer inputs or use the published CPU ONNX model. Use the bare `coreml` provider for the compiled model. [Native Sherpa build and model package](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/43afef6286ce01771b979dd0c3e817eede012a6b/ctc-e2e/coreml-experimental/mil-fp16)

After installing the native Sherpa build from the package recipe, run from the GigaAM ONNX checkout with its Python environment.

```sh
p=ctc-e2e/coreml-experimental/mil-fp16
hf download ai-babai/gigaam-v4-onnx --revision 43afef6286ce01771b979dd0c3e817eede012a6b \
  --include "$p/*" --local-dir ../gigaam-weights/v4-apple
package="$PWD/../gigaam-weights/v4-apple/$p"
(cd "$package" && shasum -a 256 -c SHA256SUMS)
xcrun coremlcompiler compile "$package/ctc-1600.mlpackage" "$package"
python clients/ctc_transcribe.py speech.wav --package "$package" \
  --model "$package/ctc-1600.mlmodelc" --sherpa --sherpa-provider coreml
```

On the M4 Pro with macOS 15.8.1, the native FP16-compute CTC model matched the published FP32 ONNX CPU text and tokens on all 32 FLEURS clips (4.56% WER). At three threads, three fresh processes each ran 20 warmed decodes on the pinned RU and EN clips. The native Core ML p50 was 41.9 ms RU and 39.3 ms EN, compared with 285.8 ms and 197.8 ms on the published FP32 ONNX CPU route. The Metal trace recorded 55, 54, 53 and 52 GPU-active intervals overlapping warm decode on Sherpa `fdb966f2`. The combined `4ce634bd` build also passed the compiled-client smoke for RU and EN. [CTC benchmark source](https://github.com/ai-babai/sherpa-onnx/tree/fdb966f2be316622cab166ebfbad9ecd68c71dbe)

#### Streaming RNN-T, 640 ms

The fused encoder runs through native Core ML. Keep the ONNX frontend beside the compiled encoder and select one matching decoder, joiner and token table. Use the build from the package recipe on macOS arm64. [Native Sherpa build and model package](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/43afef6286ce01771b979dd0c3e817eede012a6b/rnnt-streaming-640ms/coreml-experimental/mil-fp16)

Run a mono 16 kHz PCM16 WAV with the built CLI. Set `head=raw` for plain text.

```sh
p=rnnt-streaming-640ms/coreml-experimental/mil-fp16
n=v4_rnnt_640ms
hf download ai-babai/gigaam-v4-onnx --revision 43afef6286ce01771b979dd0c3e817eede012a6b \
  --include "$p/*" --local-dir weights
package="$PWD/weights/$p"
(cd "$package" && shasum -a 256 -c SHA256SUMS)
xcrun coremlcompiler compile "$package/${n}_stream_step.mlpackage" "$package"

head=e2e
DYLD_LIBRARY_PATH="$ort/lib" "$sherpa_build/bin/sherpa-onnx" --model-type=gigaam_v4 --provider=coreml_fused --num-threads=3 \
  --left-padding=0 --right-padding=0 --decoding-method=greedy_search --enable-endpoint=false \
  --encoder="$package/${n}_stream_step.mlmodelc" --tokens="$package/tokens.$head.txt" \
  --decoder="$package/${n}_${head}_decoder.onnx" --joiner="$package/${n}_${head}_joint.onnx" speech.wav
```

On the M4 Pro with macOS 15.8.1, the fused bridge completed 322 of 322 lifecycle checks across both heads and logged 1,272, 1,279 and 1,277 recognizer-process Metal GPU execution intervals during three warm decodes. The combined `4ce634bd` build also passed RU and EN with both heads. The throughput experiment used the same fused model with an earlier bridge that retained an ONNX session for schema discovery. The combined release bridge reads that schema directly from Core ML.

| M4 Pro streaming path | Warm throughput | Warm block p50 / p95 | First 13.8 s clip |
|---|---:|---:|---:|
| Released INT8 CPU | 36.98× | 16.13 / 18.24 ms | 0.40 s |
| Compact FP16 ORT CoreML experiment | 13.30× | 46.70 / 58.64 ms | 1.12 s |
| Fused native Core ML, earlier bridge | 37.04× | 16.23 / 18.19 ms | 3.43 s |

The three-thread warm comparison used eight RU/EN clips totaling 81.28 s, two warmup clips, three rotated passes and 390 timed decode calls per path. All eight formatted transcripts matched. The fused prototype was 2.79× faster than the compact ORT CoreML experiment. For lower first-clip latency on this Mac, use the published INT8 CPU package. The earlier folded ONNX CoreML experiment used ORT 1.23.2 and reached 13.61× versus 17.07× for the same graph on CPU. ORT 1.28.2 failed to initialize that folded encoder. The compact and folded experiments use different graphs.

#### Streaming regression and profiling

The registered streaming test in the combined Sherpa source honors `SHERPA_ONNX_GIGAAM_TEST_PROVIDER`. Run this after the fused streaming command above. [Upstream-style streaming cases](https://github.com/ai-babai/sherpa-onnx/blob/4ce634bd4104a586e78c0f368b85e93d41a6cc3d/scripts/gigaam-v4-streaming/README.md#streaming-regression-test)

```sh
test_wav="$work/upstream-ru.wav"
curl -fL https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/2408de4949741cb03b444a8edbc3d38acd769927/test_wavs/russian/test.wav -o "$test_wav"
printf '%s  %s\n' d6e768803b3bc3afcb08326677f3bb872c9beeed29af40d3579bcc14e74484f8 "$test_wav" | shasum -a 256 -c -
DYLD_LIBRARY_PATH="$ort/lib" \
SHERPA_ONNX_GIGAAM_STREAMING_PACKAGE="$package" \
SHERPA_ONNX_GIGAAM_STREAMING_ENCODER="${n}_stream_step.mlmodelc" \
SHERPA_ONNX_GIGAAM_TEST_WAV="$test_wav" \
SHERPA_ONNX_GIGAAM_TEST_PROVIDER=coreml_fused \
  ctest --test-dir "$sherpa_build" -R '^test_online_gigaam_v4_py$' --output-on-failure
```

For the Linux CUDA regression, check out `4ce634bd4104a586e78c0f368b85e93d41a6cc3d` and use the CUDA build commands above with `-DSHERPA_ONNX_ENABLE_TESTS=ON` in a fresh build directory. Use the released INT8 fixture, set `SHERPA_ONNX_GIGAAM_TEST_PROVIDER=cuda` and unset `SHERPA_ONNX_GIGAAM_STREAMING_ENCODER` to select its default encoder. Run `test_online_gigaam_v4_py` for the device tests. Build and run `online-stream-test` for the shared C++ checks, including 48 kHz input at EOF and rejection of `tokens_buf`.

On macOS, use Instruments Metal System Trace to inspect GPU execution intervals for the recognizer process during warm decoding.

### Separate offline RNN-T preview for Apple Silicon

The offline encoders have their own build of Sherpa. Compile the model from the corresponding package and keep the packaged `*_advance.onnx` and tokenizer beside it. Core ML runs the encoder and both output projections on the M4 Pro. ORT runs the token-decoding step on CPU. Both models accept up to 1,600 mel frames. [Offline Sherpa source](https://github.com/ai-babai/sherpa-onnx/tree/d83bc72e344bf8458ecec8b98d02b251187309c4)

| Package and instructions | FLEURS results compared with the published FP32 transcripts |
|---|---|
| [v4 offline RNN-T](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/b383b237364db47ff2e0682cde51be5396c97b7b/rnnt/coreml-experimental/mil-fp16) | All 32 RU/EN transcripts matched on each head. Formatted WER was 4.86% for both implementations. |
| [Multilingual-v2 offline RNN-T](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/6f19a3872d78761756dd2eb9eaa6f4bd07ad22c0/coreml-experimental/mil-fp16) | 80 clips in five languages. Formatted output matched on 79/80 in each language-selection mode, with the same WER as FP32 (4.92% auto, 4.99% explicit). Raw output matched on 80/80 auto and 79/80 explicit. |

Metal traces captured GPU activity during warm offline decoding for both encoders. The two package links pin the preview snapshots separately from the CTC and streaming packages in the HF `v0.4.0` tag.
