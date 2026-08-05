# lfm2.5-2.6b-gguf-gb10cuda-v1 — patch series

**Intentionally empty.** No submission has beaten the baseline, so this track's
frontier IS pinned llama.cpp b10237. Add yours as `0001-`.

LFM2.5 2.6B is a **dense hybrid, not a mixture-of-experts** — none of the
expert-dispatch levers from the Laguna tracks exist here. At 1.55 GiB of
weights, decode is dominated by per-launch overhead rather than weight
bandwidth, which is a different regime from every Laguna track.

The identical model also runs on `lfm2.5-2.6b-gguf-r9700-v1` (RDNA4). That is
deliberate: it is the only way to tell whether a kernel idea is portable or is
really a device quirk. Check the sibling track's findings before assuming a win
transfers — one cross-track port on this platform (the q8_1 dedupe, +8.69% on
AMD) turned out to be worth about 0.6% on the other box.

Accuracy gate: perplexity within 0.5% of stock via `llama-perplexity` on the
fixed corpus. Build that target as well as `llama-server`.
