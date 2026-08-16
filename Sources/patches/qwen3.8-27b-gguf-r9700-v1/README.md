# qwen3.8-27b-gguf-r9700-v1 — patch series

The series applies to vanilla llama.cpp `2b63e0610bbc2be990ae1360d5256efcdc3f9efb`
in filename order:

- `0001` — 5-warp workgroups for batch-1 K-quant matvec (kernel board).
- `0002` — MTP self-speculation on by default, and no prompt split for
  checkpoints the recurrent rollback already covers (speculative board).
- `0003` — MMVQ/MMQ crossover moved to 2 on RDNA4, priced at the `1 + n_draft`
  verify shape speculation creates rather than at batch 1 and 512 (speculative
  board).
- `0004` — the MTP draft head projects against the Latin prefix of the
  vocabulary instead of all 248320 rows (speculative board).

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
   8.7%.

## Engine

Stock pinned engine — no custom pin. The GGUF declares
`general.architecture = qwen35` and llama.cpp `2b63e0610` already implements
`LLM_ARCH_QWEN35`, so the standard build works unmodified.

## The MTP block, and the LM head behind it

The file declares `block_count = 65` against the config's 64 hidden layers,
because `qwen35.nextn_predict_layers = 1` puts a multi-token-prediction block at
`blk.64` (15 tensors, including `nextn.eh_proj`). llama.cpp loads it but executes
it only when the context type is `LLAMA_CONTEXT_TYPE_MTP`, and nothing on the
runner's fixed server command asks for that context — so on the **kernel** board
its 424,699,392 params / 289,527,808 bytes never move.

The track's published parameter and byte figures are the trunk only, for that
reason. If you are computing bytes-per-token for a kernel submission, exclude
`blk.64.*` — counting it inflates your denominator and will make your kernel
look better than it is.

Patch `0002` on the **speculative** board does ask for it. When sizing a draft
step, note what dominates it:

| tensor | type | shape | bytes |
|---|---|---|---|
| `blk.64.*` (the MTP block) | mixed | — | 289.5 MB |
| `output.weight` (the LM head) | Q6_K | 5120 x 248320 | **1042.9 MB** |

This GGUF **does** carry a separate `output.weight`; it is not tied to
`token_embd.weight` (which exists too, Q4_K, 715.2 MB, and is used for input
embedding only). `blk.64` has no `nextn.shared_head_head`, so the MTP block
borrows `model.output` for its projection — the same 1.04 GB Q6_K tensor the
trunk uses. A draft step therefore reads ~1.33 GB, of which **78% is the LM
head**, and a synchronised timer around the draft-side `llama_decode` puts the
step at 2.63 ms with 1.60 ms of that (61%) in the head. Patch `0004` is what
addresses it.

## Measuring on this box

- A dispatch costs **~2.08 µs**. "Dispatches per 1% of decode" is **not** a
  constant — it is `0.01 x token_duration / dispatch_cost`, so it scales with
  the token. This model's token is ~32.9 ms, which makes 1% of decode about
  **170 launches** counting launch structure alone, or **~118** once you also
  count the kernel time a removed dispatch takes with it. (The "~20" figure that
  used to appear here belongs to the ~6 ms tokens on the Laguna XS and Nemotron
  boards; carrying it here understates every launch-structure idea eightfold.)
- **`llama_decode` is asynchronous.** Its wall time is enqueue time — around
  0.34 ms for a draft step whose real cost is 2.63 ms. Any timer around it needs
  an explicit `llama_synchronize`, or you will measure the submit thread.
- `pgrep -x` cannot match `llama-perplexity`: Linux truncates `comm` to 15
  characters and that name is 16. Use `pgrep -af`.
