# GigaAM ONNX release checks

**Model release v0.2.1, code layout v0.2.2.** 🟢 met the stated target · 🟡 observed limit or partial coverage · 🔴 missed the stated target. Each result applies to the named model, tool version and sample. [Model × runtime matrix](../benchmarks/compatibility/README.md) · [Tested dependencies](../benchmarks/compatibility/README.md#tested-dependency-versions)

## Application scenarios

| Scenario | Target | Observed | Status | Note |
|---|---|---:|:---:|---|
| Native CPU clients, macOS and Linux | Every prescribed smoke case completes | 16/16 per OS, INT8 | 🟢 | 28/30 compared outputs matched verbatim. |
| Fresh code checkout, macOS and Linux | Reproduce all four INT8 score reports | 4/4 byte-identical per OS | 🟢 | `make verify-scores` uses stored predictions, including both bootstrap seeds. |
| Relocated transcription clients | Decode one clip per weight variant | 5/5 per OS | 🟢 | All five Mac transcripts matched their saved predictions. |
| Converter packaging, macOS | Copy the correct standalone client and verify the manifest | 11/11 + 12/12 files | 🟢 | Rebuilt v4 and multilingual offline RNN-T INT8 packages from their FP32 exports. |
| Fresh macOS Python environment | Dependency check and transcription | 2/2 model runs | 🟢 | Python 3.12.11, Sherpa CTC and multilingual INT4. |
| Fresh public INT4 download | All manifest hashes match and Kazakh output reproduces | 9/9 hashes, raw + formatted match | 🟢 | Anonymous Hugging Face download and pinned GitHub client. |
| INT4 multilingual, Linux | Nonempty raw and formatted text | 20/20 | 🟢 | Five clips, each with a language hint and in auto mode. |
| Mac file and native loader contracts | All defined expectations pass | 108/108 | 🟢 | Seven entry points, with another 14 exploratory cases. |
| Streaming v4 | Partial text, packet boundaries, reset and flush | 14/14 API checks | 🟢 | Tested 100, 400 and 640 ms input packets. |
| Loader and duration regressions | Catchable errors and unchanged RU/EN text | 15/15 final checks | 🟢 | Covers missing blank tokens and the 20 ms boundary. |
| Two utterances separated by silence | Recover both utterances | Incomplete in tested cases | 🟡 | Split at long pauses, even within the 30 s limit. |
| English in transcribe-rs 0.3.11 | Decode English and advertise it | Decodes EN, metadata lists RU | 🟡 | Language-filtered applications may hide English. |
| LM Studio 0.4.25+1 stock importer | Catalog recognizes the ONNX file | Model not found | 🟡 | Use an ONNX speech runtime for these weights. |

[Public INT4 download and exact commands](public-linux-int4.json) · [Mac cases, fixtures, transcripts and errors](mac-cpu-runtime.json) · [INT8 Linux and Mac results](../benchmarks/compatibility/README.md) · [INT4 Linux checks](../benchmarks/multilingual-v2/int4/linux.json) · [Recognition quality by language](../benchmarks/README.md)

## Code quality

Four inference clients form the runtime column. The full scope covers 18 maintained Python files and 74 functions, including the converters and benchmark tools. Release copies are counted once. Coverage was collected from 22 entry-point runs and 21 scratch tests on macOS.

| Metric | Target | Runtime | Full scope | Status, full scope | Note |
|---|---:|---:|---:|:---:|---|
| Cyclomatic complexity, max | ≤10 | 10 | 26 | 🔴 | Highest in INT8 encoder conversion. |
| Cognitive complexity, max | ≤15 | 6 | 31 | 🔴 | Highest in FP16 graph promotion. |
| CRAP score, mean | <5 | 2.600 | 5.052 | 🔴 | Uses observed function statement coverage. |
| CRAP score, max | ≤30 | 10.000 | 34.125 | 🔴 | FP16 promotion had 9 of 18 statements covered. |
| Changed executable line coverage | ≥80% | 9/9 | 50/50 | 🟢 | Compared with v0.2.0, including the new converter. |
| Local import cycles | 0 | 0 | 0 | 🟢 | Includes imports inside functions. |
| Generated property examples | All accepted examples pass | — | 130/130 | 🟢 | Frontend, language vector, head order and real 4-bit packing. |
| Selected mutations | Every exercised fault detected | — | 11/12 | 🟡 | Dropping token ID zero survived the selected assertions. |

[Per-file measurements, tool versions and test counts](code-metrics.json) · [Audited code](https://github.com/ai-babai/gigaam-onnx/tree/v0.2.1)
