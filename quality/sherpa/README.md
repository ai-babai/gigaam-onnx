---
type: QA Runbook
description: "Run the pinned native Sherpa ONNX GigaAM RNN-T scenarios on macOS, Linux, and Windows."
created: "2026-11-27"
---

# Native Sherpa ONNX scenarios

Run these scenarios from the project root with Python 3.12.11, NumPy 2.5.5, and a native Sherpa ONNX build that includes `OfflineRecognizer.from_gigaam` and the `gigaam_v4` online transducer. Use the same Sherpa source revision on each platform when comparing results. Record that revision alongside each JSON report. [Source installation](../../benchmarks/compatibility/sherpa-source-install.md)

## Inputs

Keep models, recordings and reports outside the checkout. Reuse the release directories from the [project quickstart](../../README.md#download-model-files-and-run-a-client).

| Runner flag | Release directory |
| --- | --- |
| `--v4-package` | [v4 INT8 offline RNN-T](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/rnnt) |
| `--multi-package` | [Multilingual-v2 INT8 offline RNN-T](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/v0.2.0) |
| `--stream-package` | [v4 INT8 streaming RNN-T, 640 ms](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/rnnt-streaming-640ms) |

Pass the benchmark's `eval/manifest.jsonl` and keep its `wav/<language>/<file>.wav` layout. The runner selects the three [pinned FLEURS recordings](fixtures.json) by ID and verifies the WAV, reference text and model bytes with SHA-256. To recreate the corpus from a clean checkout, run the existing sampler with curl and FFmpeg on PATH. The macOS toolchain used curl 8.7.1 and FFmpeg 8.1. [FLEURS dataset revision](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)

```sh
python -m benchmarks.tools.data_fleurs --output ../gigaam-qa/eval --cache ../gigaam-qa/cache
```

## Run the suite

Activate the Python environment containing the native build. Install the tested dependencies with `python -m pip install numpy==2.5.5 sentencepiece==0.2.2`. The runner derives streaming token tables in a temporary directory. If the environment already has the verified `tokens.raw.txt` and `tokens.e2e.txt`, pass their directory with `--stream-tokens` to run without SentencePiece.

On macOS and Linux, set `DATA` to an absolute path outside the checkout containing the release packages and corpus. Run this command from the repository root.

```sh
python quality/sherpa/scenarios.py \
  --v4-package "$DATA/weights/v4/rnnt" \
  --multi-package "$DATA/weights/multilingual-v2" \
  --stream-package "$DATA/weights/v4/rnnt-streaming-640ms" \
  --manifest "$DATA/eval/manifest.jsonl" \
  --output "$DATA/scenarios.json"
```

On Windows PowerShell, set `$Data` to the corresponding directory. A single inference thread fits smaller CI hosts.

```powershell
python quality/sherpa/scenarios.py --v4-package "$Data/weights/v4/rnnt" --multi-package "$Data/weights/multilingual-v2" --stream-package "$Data/weights/v4/rnnt-streaming-640ms" --manifest "$Data/eval/manifest.jsonl" --threads 1 --output "$Data/scenarios-windows.json"
```

Both `raw` and `e2e` run by default. Use `--head raw` or `--head e2e`, and `--backend offline` or `--backend stream` for focused runs. A relocated corpus can use `--audio-root`. Exit code 0 means the selected scenarios passed. Exit code 1 records a failed check in the JSON. The report includes the native module SHA-256, runtime versions, pinned assets and scenario details.

## Checks

- The offline v4 engine decodes RU and EN. The offline Multilingual-v2 INT8 engine decodes RU and KK, including automatic and explicit Kazakh. The suite compares isolated one-at-a-time calls with two differently ordered streams whose PCM is fed in alternating halves and decoded through `decode_streams`.
- Offline validation rejects an unsupported head and invalid language options. Zero samples and 319 samples produce empty results. The engine accepts 320 samples and exactly 30 seconds, and rejects one sample beyond 30 seconds. The fixture reader rejects stereo and accepts 16 kHz mono PCM16.
- Each streaming head must yield a nonempty partial before EOF. Two streams with alternating packets match their isolated results within the same run. Reset clears text, tokens and buffered context. A second drain after EOF adds no tokens. Both a wrong token table and a wrong joiner head must be rejected when constructing the recognizer.
- Speech checks require nonempty text and per-clip WER at most 30% after Unicode NFC, case folding and punctuation normalization. Exact text and token equality are required for equivalent schedules within one build. Each OS is evaluated against the same FLEURS references.
- The focused suite uses INT8 packages. Separate full-corpus native offline QA covers Multilingual-v2 INT4 with 80 clips, both heads and both language modes. [INT4 package](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/v0.2.1/int4) · [INT4 evaluation](../../benchmarks/multilingual-v2/int4/README.md)

The real-audio single/list pattern follows Sherpa's offline integration tests. The online lifecycle cases follow its ready/EOF/reset tests. [Offline real WAV](https://github.com/k2-fsa/sherpa-onnx/pull/4263/files) · [Offline CI matrix](https://github.com/k2-fsa/sherpa-onnx/pull/3456/files) · [Streaming lifecycle](https://github.com/k2-fsa/sherpa-onnx/pull/4133/files)

## Native code review

The two patches build on sherpa-onnx `7a924305`. The measurements below cover added functions and executable changes on macOS arm64. [Offline patch](https://github.com/ai-babai/sherpa-onnx/compare/7a924305...cf96b9a3) · [Streaming patch](https://github.com/ai-babai/sherpa-onnx/compare/7a924305...7b8e9cff) · [Measurements and scope](code-metrics.json)

| Check | Target | Offline | Streaming | Result | Note |
|---|---|---:|---:|:---:|---|
| Cyclomatic complexity of new functions | ≤10 | 10 | 10 | 🟢 | Lizard, C++ and Python |
| Cognitive complexity of new functions | ≤15 | 15 | 10 | 🟢 | clang-tidy for C++, complexipy for Python |
| CRAP, new C++ functions (mean / max) | <5 / ≤30 | 4.50 / 10.00 | 2.90 / 10.00 | 🟢 | Based on observed function line coverage |
| CRAP, all touched C++ functions (mean / max) | <5 / ≤30 | 289.17 / 7934.33 | 12.90 / 334.61 | 🔴 | Includes existing model dispatchers under GigaAM-only workloads |
| Changed executable C++ lines covered | ≥80% | 458/460 | 326/326 | 🟢 | LLVM source-based coverage including the native binding |
| New include cycles | 0 | 0 | 0 | 🟢 | Local project headers |
| Selected mutations detected | Each exercised fault | 1/1 | 1/1 | 🟢 | Swapped output head and dropped EOF remainder |

Project clang-format, cpplint 2.0.2, Black and clean patch application passed. The offline Python API covered all 17 changed executable lines. Linux and Windows runtime checks are reported separately in the [preview platform results](../../benchmarks/compatibility/sherpa-preview.md).
