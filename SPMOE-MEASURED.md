# SP-MoE measured on R9700 — FINAL (2026-08-20)

## Result (authoritative server-level, genuinely rebuilt+patched binary)
- mode=ON (GAINZ_EXPERT_PREFETCH=1): eval **134.76 tok/s** (949.84 ms/128, acceptance 0.431, mean len 3.53, n_max=6)
- Consistent with bare DFlash (118-135). NOT near the 233.3 frontier.
- mode=OFF bench failed to produce a clean line (stale port / interrupt race), so this is an ON measurement vs known bare-DFlash, not a clean on/off delta.

## Verdict
SP-MoE draft-expert-prefetch, as implemented, does NOT beat the **233.3** (spec) or **162.2** (kernel) frontier. It reproduces bare-DFlash (~135 tok/s). The gap to 233.3 is the R9700 MoE kernel series, which the leader already integrates; a bare +2-routed prefetch cannot close it.

## Combined qwen3.6 a3b R9700 route status (ALL measured now)
- ngram-cache: 77.5 tok/s — dead
- DFlash depth-6: 118.68 tok/s — dead
- SP-MoE (+2 prefetch): 134.76 tok/s — dead
- frontier 233.3 = kernel series + DFlash leader (integration already landed)

## Box
Restored per user's "free up the GPU": endpoint active, :8080 serving, lock with supervisor.
Artifacts: /home/ghost/spmoe2.status, tools/spmoe2.sh, /home/ghost/rebase27 (patched build).
