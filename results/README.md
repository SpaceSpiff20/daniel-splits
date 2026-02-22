# Results Logging Format

Standardize experiment outputs for all runs:

- ICL
- LoRA
- P-Tuning v2
- ReFT
(lmk if I am missing anything)

Each experimetn appends one JSON object per line to
results/results.jsonl

My main reasoning is that you can just append to the file from your notebook and you don't need to rewrite the full thing.

---

## Schema (One Line Per Run)

Here is an example, each line contains the following fields:

{
  "timestamp": "ISO-8601 string",
  "model": "qwen3-4b",
  "method": "icl",
  "dataset": "rte",
  "split": "validation | test",
  "N": 16,
  "seed": 0,
  "accuracy": 0.884,
  "tokens_per_sec": 45.2,
  "latency_ms": 1200,
  "prompt_tokens": 1420,
  "peak_vram_gb": 9.73
}

---

## Required Fields

- model
- method
- dataset
- N
- accuracy
