# Multilingual-v2 INT4 on CPU

The INT4 encoder is an optional lower-memory alternative to the default INT8 encoder. Both packages use the same floating-point decoder, frontend and tokenizers. [Model files](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/v0.2.1/int4) · [Conversion code](https://github.com/ai-babai/gigaam-onnx/blob/v0.2.1/multi_quantize_int4.py)

| Measured on M4 Pro | INT8 | INT4 |
|---|---:|---:|
| Download size | 811.0 MB | 519.4 MB |
| Peak process RSS | 1.81 GiB | 1.36 GiB |
| Warm throughput | 16.34× real time | 12.46× real time |
| Formatted WER, 80 clips | 5.20% | 4.85% |

Each encoder ran in a separate process using ONNX Runtime 1.23.2, three CPU threads and the same 80 FLEURS clips. Wall times after warmup were 51.21 s for INT8 and 67.14 s for INT4 on 836.54 s of speech. The runs include feature extraction and both decoding heads. Disk I/O and model loading are excluded. Peak RSS includes the whole process. The original INT8 release timing used a separate run.

| Language | Clips | INT8 WER | INT4 WER | Errors, INT8 → INT4 |
|---|---:|---:|---:|---:|
| Russian | 16 | 1.81% | 1.44% | 5 → 4 |
| Kazakh | 16 | 4.56% | 3.32% | 11 → 8 |
| Uzbek | 16 | 8.49% | 8.86% | 23 → 24 |
| Kyrgyz | 16 | 6.75% | 7.14% | 17 → 18 |
| English | 16 | 4.72% | 3.94% | 18 → 15 |
| All five | 80 | 5.20% | 4.85% | 74 → 69 |

The overall WER difference was −0.35 percentage points (paired 95% bootstrap interval −0.94 to +0.21). This sample supports the memory saving. Additional speech is needed to establish an accuracy advantage. [Raw and formatted metrics with both bootstrap seeds](metrics.json) · [Per-clip predictions and timings](predictions.jsonl)

The FP32 export contains external weight files. With those files present, run the following command from the code checkout to reproduce the 496,211,213-byte encoder. It contains 238 symmetric 4-bit MatMulNBits operators, block size 32, ORT accuracy level 4.

```sh
python -m conversion.multi_quantize_int4 work/fp/multilingual_v2_rnnt_encoder.onnx int4/multilingual_v2_rnnt_encoder.int4.onnx
```

Encoder SHA-256: `6d6bac76de138ec8af2c76652ddabc9b41c9c515f298a3bf236c54a27e75192a`.

On Linux x86-64 (AMD EPYC 7763), the same five recordings ran in ten language-hint cases. INT4 took 21.78 s versus 14.70 s for INT8 after warmup. All 20 raw and formatted outputs were nonempty. Nineteen matched INT8 exactly. [Linux smoke and timing](linux.json)
