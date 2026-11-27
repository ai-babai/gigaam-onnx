---
type: Agent instructions
description: "Locate the maintained code, select a GigaAM model and run the checks appropriate to your change."
created: "2026-11-27"
---
# Working on GigaAM ONNX

Run `make help` from the repository root to list the checks. The repository contains Python clients, converters and evaluation tools. Download weights separately from the Hugging Face revision in the [quickstart](README.md#download-model-files-and-run-a-client). Keep weights, recordings and generated exports outside the checkout.

## Find the right entry point

| Task | Edit or run | Contract |
|---|---|---|
| Transcribe RU/EN offline with CTC | [clients/ctc_transcribe.py](clients/ctc_transcribe.py) | Plain text on stdout. Pass WAV and `--package DIR`. Add `--sherpa` for the native loader. |
| Transcribe RU/EN offline with RNN-T | [clients/rnnt_transcribe.py](clients/rnnt_transcribe.py) | JSON with `raw` and `e2e`. Pass WAV and `--package DIR`. |
| Stream RU/EN | [clients/stream_transcribe.py](clients/stream_transcribe.py) | Pass model directory, then 16 kHz mono WAV. CLI prints final text. The streaming API also provides partial results. |
| Transcribe with Multilingual-v2 | [clients/multi_client.py](clients/multi_client.py) | JSON with `raw` and `e2e`. Select the precision via `--package DIR` and pass `--language kk` or `auto`. |
| Export or quantize | [conversion/](conversion/) | Run `python -m conversion.<module>`. Use a separate output directory. Keep FP32 external tensor files beside their graph. |
| Sample, score or time a run | [benchmarks/tools/](benchmarks/tools/) | Run `python -m benchmarks.tools.<module>`. Write outputs outside the checkout. |

Start with the dependency block for your model in the [README](README.md#download-model-files-and-run-a-client). Each CLI lists its arguments with `--help` after its dependencies are installed. Use the [precision guide](benchmarks/precision/README.md) to select a pinned snapshot. Keep the external tensor file beside the Multilingual-v2 FP32 or FP16 encoder. The four clients can also be downloaded and run as standalone files. Keep their sibling weights discoverable through the existing CLI arguments.

## Verify a change

**Navigation and syntax.** These checks used Git, Python 3.12 and GNU Make (3.81 on macOS, 4.4.1 on Linux). They work before downloading weights.

```sh
git ls-files 'clients/*.py' 'conversion/*.py' 'benchmarks/tools/*.py'
make check
```

On a minimal Linux image without Make, run `python3 -m compileall -q clients conversion benchmarks/tools quality/sherpa` and `git diff --check HEAD` directly.

**Inference.** Run the matching client on a known speech clip, using the pinned dependencies. For offline models use a single utterance between 20 ms and 30 s. The following command uses the v4 CTC package from the quickstart.

```sh
python clients/ctc_transcribe.py clip.wav --package ../gigaam-weights/v4/ctc-e2e
```

Compare formatted text with the saved output for the same clip and runtime. For RNN-T also compare the raw output where a baseline exists. A multilingual test should pass the same language hint as the reference. Use the [platform matrix](benchmarks/compatibility/README.md) to choose the relevant native loader.

**Sherpa integration.** Use the [native scenario runner](quality/sherpa/README.md) for changes to offline or streaming recognition. It checks pinned WAVs, tokenizers and graphs, exercises both heads, and writes a JSON report.

**Model selection.** All precisions are visible on Hugging Face `main`. CTC shares one directory and defaults to `model.int8.onnx`. Use `--model PATH` to select another graph. FP32 uses `model.onnx` on main and `model.fp32.onnx` in the original export. RNN-T keeps each precision with its companion files. The offline v4 RNN-T client also accepts an explicit `--encoder PATH`. With the pinned inference dependencies installed, run `python -m unittest quality.test_precision_selection` to check legacy defaults and precision selection.

**Scoring.** Use the [pinned evaluation environment](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/v0.2.1/conversion/README.md). Recompute all four published INT8 score reports, including the paired bootstrap intervals, from the tracked references and predictions. This check needs no weights or audio and exits with a nonzero status on a mismatch.

```sh
make verify-scores
```

Set `PYTHON=/path/to/python` when invoking Make to select an existing environment. To check the five-language report directly with Python, run the scorer and compare its output.

```sh
metrics=$(mktemp)
python -m benchmarks.tools.score_release benchmarks/multilingual-v2/sample.jsonl \
  benchmarks/multilingual-v2/results.jsonl "$metrics" --languages ru kk uz ky en
cmp "$metrics" benchmarks/multilingual-v2/metrics.json
```

**Conversion.** Run the affected converter, check the output manifest and open the generated package with its matching client. For INT4 compare the encoder digest with the [published digest and recipe](benchmarks/multilingual-v2/int4/README.md). Report the model revision, runtime version, OS and sample alongside any compatibility or speed claim.

Edit maintained evaluation code under `benchmarks/tools/`. The scripts alongside versioned predictions in `benchmarks/v4/` and `benchmarks/multilingual-v2/` record how those results were produced. See the [quality dashboard](quality/README.md) for regression cases, measured thresholds and known limits.
