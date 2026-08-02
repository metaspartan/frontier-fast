import { expect, test } from "bun:test";
import { score, computeScore, SCORE_WEIGHTS } from "../Sources/scoring";
import type { BenchmarkResult } from "../Sources/types";

const r = (candidate: string, s: number, correct = true, ttft?: number): BenchmarkResult => ({
  schemaVersion: 1, contractId: "c", candidate, timestamp: new Date().toISOString(),
  correctness: { passed: correct, mismatches: correct ? 0 : 1 },
  prefill: { tokens: 1, secondsPerToken: s, tokensPerSecond: 1 / s },
  decode: { tokens: 1, secondsPerToken: s, tokensPerSecond: 1 / s },
  ttft: ttft !== undefined ? { seconds: ttft } : undefined,
  telemetry: { gpuName: "test" }, eligible: correct,
});

test("three-component TTFT score", () => {
  const result = score(r("fast", 0.5, true, 0.5), r("base", 1, true, 1));
  expect(result.eligible).toBe(true);
  expect(result.ttftSpeedup).toBe(2);
  expect(result.score).toBeCloseTo(computeScore(2, 2, 2));
});

test("two-component fallback when no TTFT", () => {
  const result = score(r("fast", 0.5), r("base", 1));
  expect(result.ttftSpeedup).toBeNull();
  expect(result.score).toBeCloseTo(computeScore(2, 2, null));
});

test("correctness failure rejects", () => {
  expect(score(r("bad", 0.5, false), r("base", 1)).score).toBeNull();
});

test("weights sum to 1", () => {
  expect(SCORE_WEIGHTS.decode + SCORE_WEIGHTS.prefill + SCORE_WEIGHTS.ttft).toBeCloseTo(1);
});
