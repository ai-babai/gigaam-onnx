# Choosing a GigaAM CPU model by size and accuracy

These are the observed Pareto choices when minimizing ONNX weight bytes and formatted word error rate (WER). Compare models on the same task and audio. A point stays on the frontier when every smaller option has more word errors. The sizes include runtime graphs and external tensors. See the [full size and speed table](README.md) for all sixteen combinations.

| Task | Model | Weights | WER on the release sample | When to choose it |
|---|---|---:|---:|---|
| Russian–English, offline | v4 CTC INT4 | 152.4 MB | 4.71% | Smallest download |
| Russian–English, offline | v4 CTC INT8 | 242.5 MB | 4.26% | Fastest measured v4 offline client on M4 Pro |
| Russian–English, offline | Multilingual-v2 INT4 | 519.2 MB | 2.89% | Lowest observed WER on these 32 clips with the language supplied; 3.19% with automatic selection |
| Russian–English, streaming | v4 RNN-T INT4 | 168.9 MB | 6.38% | Smallest streaming download |
| Russian–English, streaming | v4 RNN-T INT8 | 259.0 MB | 5.02% | Faster streaming and fewer word errors on this sample |
| Multilingual, explicit hint | Multilingual-v2 INT4 | 519.2 MB | 4.85% | Smallest multilingual download; INT8 decoded faster on M4 Pro |

The first five rows use the same 658 reference words. The multilingual row uses 1,422 words from five languages. For the latter set, **automatic** language selection changes the trade-off: FP32 had 4.92% WER, INT8 5.34% and INT4 5.49%. The timed INT8/INT4 release comparison measured 16.3× and 12.5× real time on M4 Pro. For source-matching output, FP16 storage preserved all FP32 transcripts in the explicit-hint evaluation. [Both language modes and benchmark setup](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/685d629e8ecc0839938a5b097b6f91caece6c513/evaluation/metrics.json)

## Why INT8 sometimes had fewer errors

| v4 CTC sample | Reference words | FP32 and FP16 | INT8 | INT8 minus FP32 |
|---|---:|---:|---:|---:|
| Release, 16 RU + 16 EN | 658 | 32 errors (4.86%) | 28 errors (4.26%) | −0.61 pp |
| Independent check, 16 more RU + 16 more EN | 571 | 30 errors (5.25%) | 31 errors (5.43%) | +0.18 pp |

The second sample used a seed fixed before inference. It excluded earlier clips and duplicate reference texts. Upstream PyTorch, FP32 ONNX and FP16 storage returned identical text on all 32 new clips. FP16 and FP32 also produced bit-identical CTC scores on all 6,896 new frames.

INT8 perturbs scores before greedy decoding. With identical frontend output on the release sample, the leading CTC token changed on 545 of 8,268 frames. CTC collapses repeated tokens and removes blanks. Some changes improve a word or its spelling, others introduce an error. On one English recording, the scorer splits “M. P.” into two tokens and treats “MP” as one. That recording contributed two of the four net CTC corrections and both net streaming corrections.

Two language-stratified paired bootstrap runs on each CTC sample gave 95% intervals for the WER difference that included zero. These samples support the speed and size trade-off. A consistent INT8 accuracy advantage would require more evidence. [Counts, sample IDs and scoring checks](quantization-checks.json)

## Multilingual INT4 benchmark correction

The released INT4 encoder has SHA-256 `6d6bac76de138ec8af2c76652ddabc9b41c9c515f298a3bf236c54a27e75192a` and uses accuracy level 4. One benchmark in v0.3.0 selected a level-1 development graph. Its 4.92% explicit-hint WER and 5.23× throughput belonged to that graph. A rerun with the published digest gave 69 errors (4.85%) and 10.89× real time in that separate M4 Pro run. Raw text had 64 errors for both graphs. [Corrected model card](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/blob/2dbc6ad0b9e5ccdeb4d9a8e2e6a5507cc212ecff/README.md)
