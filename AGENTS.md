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
| Transcribe with Multilingual-v2 | [clients/multi_client.py](clients/multi_client.py) | JSON with `raw` and `e2e`. Select INT8 or INT4 via `--package DIR` and pass `--language kk` or `auto`. |
| Export or quantize | [conversion/](conversion/) | Run `python -m conversion.<module>`. Use a separate output directory. Keep FP32 external tensor files beside their graph. |
| Sample, score or time a run | [benchmarks/tools/](benchmarks/tools/) | Run `python -m benchmarks.tools.<module>`. Write outputs outside the checkout. |

Start with the dependency block for your model in the [README](README.md#download-model-files-and-run-a-client). Each CLI lists its arguments with `--help` after its dependencies are installed. The four clients can also be downloaded and run as standalone files. Keep their sibling weights discoverable through the existing CLI arguments.

## Verify a change

**Navigation and syntax** need only Python. These commands work before downloading weights.

```sh
git ls-files 'clients/*.py' 'conversion/*.py' 'benchmarks/tools/*.py'
make check
```

**Inference.** Run the matching client on a known speech clip, using the pinned dependencies. For offline models use a single utterance between 20 ms and 30 s. The following command uses the v4 CTC package from the quickstart.

```sh
python clients/ctc_transcribe.py clip.wav --package ../gigaam-weights/v4/ctc-e2e
```

Compare formatted text with the saved output for the same clip and runtime. For RNN-T also compare the raw output where a baseline exists. A multilingual test should pass the same language hint as the reference. Use the [platform matrix](benchmarks/compatibility/README.md) to choose the relevant native loader.

**Scoring.** Use the [pinned evaluation environment](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/v0.2.1/conversion/README.md). Recompute all four published INT8 score reports, including the paired bootstrap intervals, from the tracked references and predictions. This check needs no weights or audio and exits with a nonzero status on a mismatch.

```sh
make verify-scores
```

Set `PYTHON=/path/to/python` when invoking Make to select an existing environment. For other predictions, run `python -m benchmarks.tools.score_release --help`.

**Conversion.** Run the affected converter, check the output manifest and open the generated package with its matching client. For INT4 compare the encoder digest with the [published digest and recipe](benchmarks/multilingual-v2/int4/README.md). Report the model revision, runtime version, OS and sample alongside any compatibility or speed claim.

Edit maintained evaluation code under `benchmarks/tools/`. The scripts alongside versioned predictions in `benchmarks/v4/` and `benchmarks/multilingual-v2/` record how those results were produced. See the [quality dashboard](quality/README.md) for regression cases, measured thresholds and known limits.
