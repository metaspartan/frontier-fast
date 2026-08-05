# laguna-s-2.1-gguf-gb10cuda-v1 — patch series

**This series is intentionally empty.** No submission has yet beaten the
baseline on this track, so its frontier *is* pinned llama.cpp b10237. A
candidate here is built as the pinned tree plus your patch, and measured
against the same pinned tree.

It previously held the eleven patches from the XS GB10 CUDA track. Those were
placed here by analogy when the shared series was split per track, not because
anything had measured them on Laguna S. They have since been measured on S:
**+0.64%, inside noise.** Carrying them made every S submitter rebase onto
eleven patches that were never earned on this hardware and model pair.

Add your patch as `0001-` when you land the first win. Baseline on the
trusted runner is **23.627 tok/s** decode, **343.1 tok/s** prefill, 0.66 s
TTFT.

## This box is NOT launch-bound — that kills a whole optimization class

Kernel time is **93.4% of wall** here (45.4 ms of 48.6), versus 74% on the
R9700. So the launch-count reduction class that carried the AMD track does
not transfer: the q8_1 dedupe worth +8.69% there is worth about 0.6% on S.

Byte census is 7.48 GB/token — 2978 MB attention, 2070 MB Q4_K experts,
1510 MB BF16 experts, 471 MB shared expert, 328 MB head, 120 MB dense —
running at ~177 GB/s against a ~231 GB/s achievable ceiling, i.e. 77% of
roof. That leaves roughly 30% of decode and ~19% of score as headroom, and it
has to come from bandwidth or from work removal, not from dispatch structure.

### Dead: mmvq k-loop thread utilisation on the Q4_K expert matvecs

`blocks_per_iter = vdr*nwarps*warp_size/qi`. With `nwarps=4`, Q8_0 at K=3072
gives exactly 3.0 iterations (100% utilisation, ~280 GB/s), but Q4_K gate/up
at K=3072 gives 1.5 padded to 2 (75%), and Q4_K down at K=1024 gives 1
iteration where warps 2-3 never enter the k-loop (50%). That is the entire
189-vs-280 GB/s gap — and the corresponding fix has been measured losing on
the XS sibling in both directions (see
`../laguna-xs-2.1-gguf-gb10cuda-v1/README.md`). Understand it as a diagnosis,
not a lever.

## Before you trust a number on this track

Laguna S has a decode-rate state that is **fixed for a process launch and
independent of your code**: bimodal, roughly 7% wide (~23.4-24.1 vs
~25.5-25.6 tok/s), while the reading *within* a launch is rock steady. Low
variance inside one launch is the artifact's signature, not evidence that your
effect is real - a two-round A/B here measured +6.3% before rounds 3-5 erased
it. Alternate whole process launches and take the median of the per-round
ratios. The trusted runner does three paired boots per submission for this
reason; if your effect is under ~7%, expect to need about five rounds.
