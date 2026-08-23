Track `ornith-1.5-35b-a3b-gguf-r9700-v1` — MTP self-speculation, depth 1, with the
draft head's projection cut in the class that actually runs it.

**86.36 -> 91.46 tok/s, +5.90%**, speculation on, no second model.

`Sources/runner/ornith-1.5-35b-a3b-gguf-r9700-v1/serving.json` turns it on:

```json
{ "speculative": { "specType": "draft-mtp", "draftMax": 1, "draftMin": 0 } }
```

## The series

| patch | what |
|---|---|
| `0001-mtp-autoenable` | run the GGUF's own NextN head as the drafter; no draft model |
| `0002-mmvq-verify-width` | MMVQ block shape + the MMVQ->MMQ crossover at verify widths |
| `0003-recurrent-rollback-splits` | don't split the prompt when the memory can already roll itself back |
| `0004-mmvq-3warp-smallk` | retunes `0002`'s block to 3 warps and re-enables small-K row blocking on RDNA4 |
| `0005-moe-mtp-head-prefix` | **new** — the draft head projects a 49152-row prefix of the vocabulary |

`0004` is the kernel-board submission's change, carried here so the speculative
arm is measured on the same trunk. `0005` is the lever this submission is really
about.

## `0005`: the head cut, in the class that runs

llama.cpp keeps **two** model classes for this architecture family:
`llama_model_qwen35` (dense, `src/models/qwen35.cpp`) and `llama_model_qwen35moe`
(`src/models/qwen35moe.cpp`), and **each has its own `graph_mtp`**. This GGUF
loads the MoE class, so the dense track's reduced-draft-head patch is dead code
here — and a naive A/B records it as "the idea does not work on this model".

It was censused rather than assumed. Sweeping `GAINZ_MTP_HEAD_K` over
0 / 98304 / 49152 / 32768 / 16384 / 8192 against `qwen35.cpp` left the per-slice
accepted-length vector **bit-identical** (`E = 1.3474` in all six), cumulative
draft time within 0.2% (1392–1396 ms), the greedy text hash identical and decode
flat within 0.23%. That is not a null result, it is an **absent lever**.

Ported into `qwen35moe::graph_mtp` it works, and it is worth **more** here than on
the dense sibling, for a structural reason. One MTP draft step reads:

```
blk.40 attention             15.61 MB
blk.40 routed experts        16.32 MB   (8 of 256 active, 2.04 MB each)
blk.40 shared expert + gate   4.20 MB
nextn.eh_proj                 4.72 MB
model.output (LM head)      417.18 MB   <- 91% of the step
```

The LM head is dense and full-vocabulary (2048 x 248320, Q6_K) while the block
feeding it is sparse. On the dense sibling the head is 78% of the draft step;
here it is 91%, so cutting it is worth proportionally more.

The cut is a plain prefix, so it is a zero-copy view rather than a gather and
candidate index is still token id. The draft only *proposes*: the trunk recomputes
logits at every drafted position and accepts or rejects against its own argmax, so
a proposal the full projection would not have made costs acceptance, never
correctness. Swept at `n_max = 1`, on two trunks — kept apart below because they
are not comparable, and the full-projection reference was only ever measured on
the predecessor:

On **this** trunk (`0004`'s 3-warp + small-K matvec), base stock 86.53:

| K | draft ms | decode | accepted E |
|---|---|---|---|
| 98304 | 1000 | 90.70 | 1.3474 |
| **49152 (shipped)** | **878** | **92.34** | **1.3474** |

On the **predecessor 5-warp trunk**, base stock 86.22:

| K | draft ms | decode | accepted E |
|---|---|---|---|
| full projection | 1396 | 89.33 | 1.3474 |
| 98304 | 1018 | 91.50 | 1.3474 |
| 49152 | 896 | 92.51 | 1.3474 |
| 32768 | 852 | 90.04 | 1.3196 |
| 16384 | 804 | 91.64 | 1.3333 |

49152 is where the curve stops paying: the median accepted length does not move
between full, 98304 and 49152 — the proposals falling outside 49152 rows are not
the ones the median slice was accepting — and below it E starts moving. Both
trunks agree on the ordering and on where E breaks.

## Depth is 1, and that is a property of the trunk

A verify pass over `k` positions on a dense model costs about what one position
costs: the pass streams one fixed weight set and the extra tokens ride along. On a
256-expert A3B trunk each token routes to its **own** 8 experts, so `k` tokens
touch up to `8k` distinct expert tiles per layer and the pass gets genuinely wider.
`llama-bench -p k -n 0 -r 5`:

| positions | 1 | 2 | 3 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| ms/pass | 14.07 | 16.33 | 17.25 | 18.53 | 24.27 | 27.84 | 30.19 |

~1.46 ms per extra verified position against a 14.07 ms pass — **10% of a full
pass per node**, where the dense sibling of this family pays 1–2.7%. The marginal
cost falls again past width 8 (0.89 then 0.59 ms), which is the check that this is
expert fan-out and not fixed launch overhead: 16.32 MB of routed expert weight per
layer at 8-of-256 is 1.23 ms of traffic at this part's achieved bandwidth, against
1.46 ms measured.

Per-position acceptance from the engine's own census is 0.379 / 0.048 / 0.002
against a break-even of ~0.29 accepted tokens per added node. Only the first
drafted position clears it; depths 2 and 3 measure −13.8% and −22.6%.

**No tree work is possible or worth it.** No shape clears ΔE 0.29/node, and
`qwen35moe.cpp` calls only `build_conv_state`, never `build_conv_state_tree`, and
never consults `inp->tree_src` — the MoE class has no tree path at all.

Draftless speculation is a wash: `ngram-simple` n=1/n=2, `ngram-map-k`,
`ngram-mod` all land within 0.15% of no speculation, and `ngram-cache` is −22.4%.

## Measurement

Ranked window replicated locally — 512-token prefill, 128 greedy decode,
`cache_prompt:false`, 2 warmups then the median of runs 100..108, prompts from
`fixtures/gainz-corpus.txt` at 2600 chars split into exactly 20 passages,
`HIP_VISIBLE_DEVICES=0`, `/proc/loadavg` gated under 0.60 before every boot.
Palindromic slot design; both arm orders shown.

**Confirmation round on the exact shipped binary** (2 arms, 4 boots):

| arm | slot-lo | slot-hi | decode | spread | prefill | E | vs stock |
|---|---|---|---|---|---|---|---|
| stock | 86.463 | 86.258 | 86.361 | 0.237% | 2044.07 | — | — |
| **shipped (MTP depth 1 + head cut)** | 92.496 | 90.421 | **91.458** | 2.270% | 2321.81 | 1.3474 | **+5.90%** |

Round selecting the cut width (5 arms, 10 boots):

| arm | slot-lo | slot-hi | decode | spread | vs stock |
|---|---|---|---|---|---|
| stock | 86.808 | 86.250 | 86.529 | 0.644% | — |
| MTP depth 1, K=98304 | 90.739 | 90.667 | 90.703 | 0.080% | +4.82% |
| MTP depth 1, K=49152 | 92.232 | 92.452 | 92.342 | 0.239% | +6.72% |

And a round putting the speculative arm beside the same trunk with speculation
simply switched off (3 arms, 6 boots):

| arm | slot-lo | slot-hi | decode | spread | vs stock |
|---|---|---|---|---|---|
| stock | 86.024 | 86.096 | 86.060 | 0.083% | — |
| speculation off, same trunk | 102.153 | 102.280 | 102.217 | 0.125% | +18.77% |
| MTP depth 1 + head cut | 90.309 | 92.279 | 91.294 | 2.158% | +6.08% |

Note the honest wart: the speculative arm's slot-to-slot spread is ~2.2%, an
order of magnitude wider than any non-speculative arm in the same design
(0.05–0.6%), and it does not shrink with repetition. Acceptance interacts with
which slice lands in which slot, so a speculative median-of-9 is simply a noisier
statistic on this track. Four independent rounds put the arm at 91.29 / 91.44 /
91.46 / 92.34, all comfortably above stock, and the palindromic design means the
two slots of each arm bracket rather than bias the result.

## Speculation is still a net loss against leaving it off

Stated plainly because it is the useful finding: **102.22 with speculation off
against 91.29 with it on.** This submission ranks the speculative board, and the
head cut is a real +6.1% over the same trunk's stock arm, but nobody should read
it as "speculation pays on this track". It does not, and `0004` widens the gap by
making the non-speculative step cheaper without lowering the node price.

## Accuracy gate

`llama-perplexity` on the **exact shipped binary**, three loads per arm, control
re-measured every load:

| load | stock (control) | shipped |
|---|---|---|
| 1 | 5.1589 | 5.1589 |
| 2 | 5.1589 | 5.1589 |
| 3 | 5.1589 | 5.1589 |

Long corpus `-c 32768 --chunks 1`: **7.5415** both. Relative delta **0.000%**.

The MTP block only executes under `LLAMA_CONTEXT_TYPE_MTP`, which
`llama-perplexity` never creates, so `0005` is structurally invisible to this
gate — as is a matvec block-shape change, because a 512-token prefill dispatches
MMQ rather than MMVQ. A clean perplexity here is a necessary check, not evidence
that either kernel is exact.

## Verification

The measured tree is byte-identical to a fresh checkout of the pin (`2b63e0610`)
with these five patches applied, `diff -r -x build`: no differences. Every arm's
`libggml-hip.so` and `libllama.so` were md5-summed before measuring.

## Note for the runner

`specType: "draft-mtp"` with no draft model was previously rejected as "needs a
draft model and this track has none pinned", which is backwards for a GGUF
carrying its own NextN head — the track metadata itself sets
`customDraftHeads: true`. That fix is deployed and this submission depends on it.
