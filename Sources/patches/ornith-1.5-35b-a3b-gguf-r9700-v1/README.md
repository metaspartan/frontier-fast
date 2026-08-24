Track `ornith-1.5-35b-a3b-gguf-r9700-v1` — MTP self-speculation, and the launch
floor underneath it.

`Sources/runner/ornith-1.5-35b-a3b-gguf-r9700-v1/serving.json` turns speculation
on:

```json
{ "speculative": { "specType": "draft-mtp", "draftMax": 1, "draftMin": 0 } }
```

## The series

| patch | what |
|---|---|
| `0001-mtp-autoenable` | run the GGUF's own NextN head as the drafter; no draft model |
| `0002-mmvq-verify-width` | MMVQ block shape + the MMVQ->MMQ crossover at verify widths |
| `0003-recurrent-rollback-splits` | don't split the prompt when the memory can already roll itself back |
| `0004-mmvq-3warp-smallk` | 3-warp MMVQ block and small-K row blocking re-enabled for RDNA4 |
| `0005-moe-mtp-head-prefix` | the draft head projects a 49152-row prefix of the vocabulary |
| `0006-mmvq-q8-1-activation-reuse` | quantize an activation once per graph evaluation, not once per matvec |
| `0007-preadd-rms-norm-fold` | **new** — fold the residual add into the `rms_norm` that consumes it |

`0001`–`0006` are the standing speculative record, carried unchanged. `0007` is
the patch that took the kernel board this round, ported onto the speculative
trunk — the same trunk, so the same 30 dispatches per token come off it.

## `0007`: fold the residual add into the norm that consumes it

`0006` took the launch floor from 1631 to 1481 dispatches per decode step by
deleting redundant `quantize_q8_1` launches on the matvec path, and its notes
said the two families left — `k_bin_bcast` at 220/step and `rms_norm_f32` at
131/step, both in the GDN/SSM trunk — "would need graph-level fusion rather than
a backend change". This is that fusion.

Every block ends with the residual chain:

```
    r   = a + b            GGML_OP_ADD
    n   = rms_norm(r)      GGML_OP_RMS_NORM
    dst = n * w            GGML_OP_MUL
```

Upstream already fuses an add that comes *after* a norm. Nothing fuses the one
*before* it, because `r` is live — the next block's residual reads it — so the
add cannot be elided.

It does not need to be elided. It only needs to stop being its own dispatch.
`rms_norm_f32` is already **one block per row**: to compute `mean(r^2)` that
block must touch every element of `r` anyway. The fused kernel does
`r[i] = a[i] + b[i]` in that same block, **stores `r` exactly as the add would
have**, accumulates `r[i]*r[i]` in the same pass, and finishes with
`dst[i] = scale * r[i] * w[i]`. One extra load per element, no extra launch, no
lost parallelism (the norm's grid is the row count either way) and nothing
recomputed. The arithmetic is unchanged end to end: the same sum in the same
order, the same `block_reduce<SUM>` over the same per-thread partition as stock
`rms_norm_f32`, the same weight product.

`rocprofv3 --kernel-trace` on the pair, rows ordered by `Dispatch_Id`:

| family | without | with | delta |
|---|---|---|---|
| **total dispatches / token** | **1501.8** | **1471.8** | **−30.0** |
| `k_bin_bcast` | 220.0 | 190.0 | −30.0 |
| `rms_norm_f32` | 131.0 | 101.0 | −30.0 |
| `rms_norm_preadd_mul_f32` | — | 30.0 | +30.0 |
| `quantize_q8_1` | 201.0 | 201.0 | 0 |
| `mul_mat_vec_q` | 351.0 | 351.0 | 0 |

Instrumenting the matcher shows **80** sites passing every predicate per graph
pass; only **30 fire per decoded token**. The static site count is a property of
the graph as compiled, not of what a decode step executes, and 30/token is the
only number claimed.

## Measurement

Ranked window replicated locally — 512-token prefill, 128 greedy decode,
`cache_prompt:false`, 2 warmups then the median of runs 100..108, prompts from
`fixtures/gainz-corpus.txt` at 2600 chars split into exactly 20 passages,
`HIP_VISIBLE_DEVICES=0`, `/proc/loadavg` gated under 0.60 before every boot, GPU
lock taken by atomic `mkdir`, `libggml-hip.so` and `libllama.so` md5-summed per
arm before every run. Palindromic slot design, 8 slots per pass, both arm orders.
`0001` auto-enables the NextN head, so a plain boot measures at the speculative
operating point.

| pass | slot order | record (`0001`-`0006`) | `+0007` | delta |
|---|---|---|---|---|
| 1 | A B B A A B B A | 95.226 | 96.166 | **+0.99%** |
| 2 | B A A B B A A B | 95.531 | 96.062 | **+0.56%** |
| pooled, 16 slots | — | **95.379** | **96.114** | **+0.77%** |

Per slot, sorted — record: 91.905 95.179 95.274 95.358 95.400 95.481 95.581
95.623; `+0007`: 94.451 95.815 96.021 96.103 96.125 96.151 96.206 96.208. Seven
of the fold's eight boots beat seven of the record's eight; each arm has exactly
one outlier boot and they are on opposite sides, which is what an 8-boot sample
of a 2.2%-spread arm looks like.

## Gates

**Perplexity**, five separate loads per arm with the control re-measured on every
load and the arm order alternated per load:

| gate | record | `+0007` | relative delta |
|---|---|---|---|
| `-c 512 --chunks 8` | 8.6524 ×5 | 8.6524 ×5 | **+0.0000%** |
| `-c 4096 --chunks 4` | 6.9951 ×5 | 6.9951 ×5 | **+0.0000%** |

Both arms stable to four decimals across all five loads on both window lengths,
against a 0.1% bound. The long window is run because the fold's multi-row path
(the 1024-thread instantiation) is only exercised there.

**Greedy-output identity.** On the kernel trunk the same patch produced a
byte-identical 2304-token greedy continuation on **16 of 16 slots across both arm
orders**. On this trunk it cannot be claimed, and the reason is worth
recording: **the speculative trunk is not reproducible boot to boot.** Over these
eight boots the record produced three distinct 2304-token continuations and the
fold produced two, with `a37176e9b751` common to both arms and appearing in five
of the eight. Accepted length is 1.3474 on every slice of every boot of both
arms, so the divergence is not in acceptance. It is `0004` (see below), which
reassociates the K-quant matvec: identical when the trunk does not speculate —
the same pair of binaries plus this patch gave 16 of 16 byte-identical slots on
the kernel board — and boot-dependent once a draft head feeds off it.

## Verification

Both measured trees round-trip against the published series — `git checkout
2b63e0610`, `git apply` `0001`..`0006` (and `0007`), `diff -r -x build -x .git` —
with no differences in either case. The measured artifact is the submitted
artifact.

## Flagged, not fixed here: `0004` is not clean

`0004-mmvq-3warp-smallk` is carried unchanged from the standing record, but it
has a defect this round localised, and whoever picks up this track should know
about it before building on top of it.

Over a **40-request** window — the ranked window is 2 warmups plus 9 measured
requests, which is too short to see it — a DFlash2-drafted arm carrying `0004`
drops individual requests to 47-49 tok/s, exactly `healthy / E`, with draft
acceptance collapsing to zero for that request and recovering afterwards. 11-14
of 40 requests are affected, reproducibly, over 4 boots. The same 40 prompts in
the same order on the same trunk without `0004` never drop below 75.

`0004` has two hunks that had never been separated. **Neither breaks alone**:

| build | `0004` hunks | boots | broken requests |
|---|---|---|---|
| port only | none | 4 | 0 / 160 |
| port + hunk A (`calc_nwarps` 8 -> 3) | A | 3 | **0 / 120** |
| port + hunk B (`small_k` re-enabled) | B | 3 | **0 / 120** |
| port + `0004` | A + B | 4 | 11-14 per 40 |

The hunks multiply rather than add. Hunk A alone leaves `should_use_small_k`
disabled on RDNA, so `calc_rows_per_block` still returns 1; hunk B alone leaves
`nwarps` at 8, so it returns 8. Only together do they give `rows_per_block = 3`,
and 3 is the only one of those three values that does not divide this model's
routed-expert row counts (512 and 2048).

This submission changes no kernel and does not touch `0004`. The fold is
bit-identical arithmetic and can make that neither better nor worse; it is
recorded here because 78 boots of pairwise bisection on this track never
localised it — the pair that owns it lives inside a single patch.
