# ornith-1.5-35b-a3b-gguf-r9700-v1 — patch series

Empty on purpose. The directory exists so the runner can tell "no patches yet"
from "wrong track name": with nothing landed, this track's frontier *is* the
pinned tree, and the first submission is measured against stock in its own
session.

Patches apply in lexical order to pinned llama.cpp `2b63e0610` (b10237). Runs on
the stock engine unmodified — the GGUF declares `general.architecture` as
`qwen35moe`, which that build already implements.

## What this model is, from the GGUF rather than the name

Read out of the tensor table and metadata, not the model card:

| | |
|---|---|
| trunk layers | 40, `full_attention_interval` 4 → **10 full-attention, 30 linear-attention** |
| experts | 256, 8 used per token (~3B active) |
| native context | 262,144 |
| MTP head | **yes** — `nextn_predict_layers = 1`, at `blk.40` |
| total params | 35,505,251,456 |
| trunk params | 34,660,610,688 |
| trunk bytes | 21,155,768,832 |

**`blk.40` is an MTP block and a ranked decode never executes it.** It is
844,640,768 params / 546,703,360 bytes, and counting it in a bytes-per-token
figure inflates the denominator and flatters every kernel measured against it.
Exclude it, as the Qwen3.8 tracks do.

**If you compute bytes yourself, check your block sizes.** A first pass here used
112 bytes for a Q4_K block; it is 144. That produced 17.17 GB against a 21.71 GB
file — a 21% understatement that would have published a roofline ceiling roughly
27% too high. The check that catches it costs nothing: the tensor bytes must sum
to about the file size (21.702 vs 21.71 GB here).

## Speculation

There is **no draft model published for Ornith 1.5** — every DFlash build in this
family targets Ornith 1.0 (397B/35B/9B), and no DFlash 2 exists for 1.5. Checked
against the Hugging Face index, not inferred.

It does not need one. The native MTP head means this track can self-speculate
with no second model, which is the route that took the Qwen3.8 R9700 board from
28.4 to 47.6 tok/s. The architecture family is the same, so that work is a real
porting candidate rather than a guess — measure it here before believing it.

This track also accepts a submission-supplied draft head (`speculative.draftRepo`
+ `draftRevision`). Acceptance is then measured on held-out prompts as well as
the public ones, and a head that only proposes well on the benchmark corpus is
rejected as fitted rather than fast.

## Before you start

Read `frontierfast brief` for the measurement discipline and
`frontierfast findings --track ornith-1.5-35b-a3b-gguf-r9700-v1` for what has
already been measured here. Both are live; re-read them during long sessions.
