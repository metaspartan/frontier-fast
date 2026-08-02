import type { BenchmarkResult, ScoreRecord } from "./types";

export const SCORE_WEIGHTS = { decode: 0.65, prefill: 0.20, ttft: 0.15 } as const;

export function speedup(baseline: number, candidate: number): number {
  return baseline / candidate;
}

export function computeScore(decode: number, prefill: number, ttft: number | null): number {
  if (ttft !== null && ttft > 0) {
    return decode ** SCORE_WEIGHTS.decode * prefill ** SCORE_WEIGHTS.prefill * ttft ** SCORE_WEIGHTS.ttft;
  }
  return decode ** (SCORE_WEIGHTS.decode + SCORE_WEIGHTS.ttft) * prefill ** SCORE_WEIGHTS.prefill;
}

export function score(candidate: BenchmarkResult, baseline: BenchmarkResult, floors = { prefill: 0.95, decode: 0.95, ttft: 0.90 }): ScoreRecord {
  const prefillSpeedup = speedup(baseline.prefill.secondsPerToken, candidate.prefill.secondsPerToken);
  const decodeSpeedup = speedup(baseline.decode.secondsPerToken, candidate.decode.secondsPerToken);
  const hasTtft = candidate.ttft && baseline.ttft && candidate.ttft.seconds > 0 && baseline.ttft.seconds > 0;
  const ttftSpeedup = hasTtft ? speedup(baseline.ttft!.seconds, candidate.ttft!.seconds) : null;

  const ttftFloorOk = ttftSpeedup === null || ttftSpeedup >= (floors.ttft ?? 0.90);
  const eligible = candidate.correctness.passed
    && candidate.contractId === baseline.contractId
    && prefillSpeedup >= floors.prefill
    && decodeSpeedup >= floors.decode
    && ttftFloorOk;

  const value = eligible ? computeScore(decodeSpeedup, prefillSpeedup, ttftSpeedup) : null;

  return {
    schemaVersion: 1,
    contractId: candidate.contractId,
    candidate: candidate.candidate,
    score: value,
    prefillSpeedup,
    decodeSpeedup,
    ttftSpeedup,
    deltaPercent: value === null ? null : (value - 1) * 100,
    eligible,
    createdAt: new Date().toISOString(),
    source: "paired-baseline",
    prefillTokensPerSecond: candidate.prefill.tokensPerSecond,
    decodeTokensPerSecond: candidate.decode.tokensPerSecond,
    ttftSeconds: candidate.ttft?.seconds,
  };
}
