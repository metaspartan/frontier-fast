# Hosting a trusted runner

gainz.fast runners are community hardware. If you own a machine that matches
a track on the [roadmap](https://gainz.fast/#roadmap), you can apply to host
its trusted runner — that is how new model/device tracks go live.

## Hardware requirements

| Track | Device | Memory | Engine | Weights |
| --- | --- | --- | --- | --- |
| Laguna XS 2.1 · NVFP4 | NVIDIA (sm_90+, GB10 class) | 32 GB+ VRAM/unified | vLLM 0.25.1 (pinned image) | 21.6 GB |
| Laguna S 2.1 · NVFP4 | NVIDIA (GB10 class) | 120 GB+ unified | vLLM 0.25.1 (pinned image) | 93 GB |
| Laguna XS 2.1 · MLX | Apple Silicon M-series | 36 GB+ unified | MLX (pinned) | 21.6 GB |
| Proposed: Laguna XS · ROCm | AMD RDNA 32 GB (e.g. R 9700) | 32 GB VRAM | vLLM ROCm | 21.6 GB — NVFP4 support unconfirmed |

Plus, for any runner: Docker, a stable network connection, tolerance for the
box being busy ~20 minutes per submission, and a location that will not
thermal-throttle the device (ambient heat is fine — the runner warns on
inter-phase drift and only refuses to measure while actually throttling).

## The trust model (why this is not fully self-serve)

Ranked verdicts move a public leaderboard, so a runner must be trusted:

- Each runner gets its **own revocable credential**, issued by the
  maintainers after a vetting conversation. One credential per box.
- All timing is **paired on the same box** — your machine's absolute speed
  does not matter, only the baseline/candidate ratio it measures.
- Correctness is teacher-forced against the track's pinned goldens, and
  results are HMAC-signed; a runner that reports impossible numbers is
  cross-checked against the calibration band and revoked if it lies.
- The runner code ships as a container image; you never modify it.

## Apply

Open a [runner application](../../issues/new?template=runner-application.yml)
with your hardware details. Applications are reviewed roughly in roadmap-vote
order — vote for your track so it rises.
