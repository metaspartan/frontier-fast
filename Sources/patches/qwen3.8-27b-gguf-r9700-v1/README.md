# qwen3.8-27b-gguf-r9700-v1 — patch series

**Empty by design.** This track was commissioned on 2026-08-14 and has no landed
wins yet, so its frontier *is* the pinned tree. A candidate here is measured as
vanilla llama.cpp `2b63e0610bbc2be990ae1360d5256efcdc3f9efb` plus your own
patches — you are not building on anyone else's series.

The directory exists so the runner can tell "empty on purpose" from "you pointed
me at the wrong track". Do not delete it.

## The model

Qwen3.8-27B Q4_K_M (`unsloth/Qwen3.8-27B-GGUF`), a **dense** 27B with a hybrid
attention stack: 64 trunk layers, of which **48 are linear-attention and 16 are
full-attention** — one full layer every fourth.

Two things follow from that, and they are what make this track different from the
MoE tracks next to it:

1. **Dense means every weight is read every token.** There is no routing slack.
   Decode is weight-bandwidth bound from the first token, and a patch that wins
   here has to move fewer bytes or spend fewer launches — it cannot win by
   touching less of the model.
2. **The linear-attention layers carry recurrent state**, the way a Mamba stack
   does. On the sibling Nemotron tracks that state turned out to be a large,
   *uncounted* traffic term, and correcting for it moved a published ceiling by
   8.7%. Nobody has censused it here yet.

## Engine

Stock pinned engine — no custom pin. The GGUF declares
`general.architecture = qwen35` and llama.cpp `2b63e0610` already implements
`LLM_ARCH_QWEN35`, so the standard build works unmodified.

## The MTP block

The file declares `block_count = 65` against the config's 64 hidden layers,
because `qwen35.nextn_predict_layers = 1` puts a multi-token-prediction block at
`blk.64` (15 tensors, including `nextn.eh_proj`). llama.cpp loads it but executes
it only when the context type is `LLAMA_CONTEXT_TYPE_MTP`, so during a ranked
decode its **424,699,392 params / 289,527,808 bytes never move**.

The track's published parameter and byte figures are the trunk only, for that
reason. If you are computing bytes-per-token, exclude `blk.64.*` — counting it
inflates your denominator and will make your kernel look better than it is.

## Where to look first

Nobody has profiled this track yet, so the honest answer is "measure before you
patch". Two cheap things that would help everyone:

- A **byte census** from the GGUF tensor table, *including* the linear-attention
  recurrent and conv state, so the roofline stops being weights-only.
- A **dispatch census** per decoded token. On this box a dispatch costs ~2.08 µs,
  so 1% of decode is about 20 launches — that ratio tells you immediately whether
  launch structure or bandwidth is the lever here.
