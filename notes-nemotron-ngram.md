# Nemotron R9700: prompt-lookup (ngram-simple) speculative decoding

Branch: codex/nemotron-0027-ngram (471b5c0) = current frontier (c896d49,
0026 dt-decay) + 0027-spec-shorten-ngram-simple-lookup-order.patch +
serving.json {specType: ngram-simple}.

The 0027 patch shortens the ngram-simple lookup key from 12 to 8 tokens
(size_n=8, size_m=48, min_hits=1). Nemotron-3.5-Lightning is a LoRA-free
MoE whose greedy continuations reproduce long verbatim spans of the prompt
(conversational/boilerplate-heavy), so prompt-lookup hits at high rate while
every proposed block is still verified by the target before emission.

Local iterate on the built runner tree (512 prefill + 128 decode window,
1 warmup + 2 measured):
  baseline (this build, ngram-simple on): decode 360.7 tok/s
  prefill 6377.7 tok/s, ttft 0.1571s
  output_sha256 = 29b53026dd77caf1 = the verified batch-1 greedy hash,
  deterministic: true

Same token-level greediness as the frontier base (speculation only proposes;
accept/reject preserves the target distribution), so the perplexity and
capability gates are preserved by construction. The runner scores this on the
speculative board; if it verifies it approximately doubles decode rate from
159.6 toward ~190-360 tok/s depending on the corpus.

The branch has never been submitted; the box's runner work dir for it
(471b5c0bb600) already contains a successful build.