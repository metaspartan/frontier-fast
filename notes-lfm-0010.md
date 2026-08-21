# Group decode KV-cache set_rows stores into one launch
## Attribution
- Model: Cybara (deepseek-v4-pro)
- Track: lfm2.5-2.6b-gguf-r9700-v1
## Summary
Merge adjacent single-row F32->F16 set_rows stores into one launch. Ported from laguna-xs 0007.
## Measured
Local iterate: decode 227.1 vs 226.3, prefill 11890 vs 11705, hash 79335bb6ec56151a exact.