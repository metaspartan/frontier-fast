import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import type { BenchmarkResult, Contract } from "../types";
import type { Runner } from "./base";

interface BenchJson {
  decodeTokensPerSecond: number; decodeSecondsPerToken: number;
  prefillTokensPerSecond: number; prefillSecondsPerToken: number;
  ttftSeconds: number; text: string; peakMemoryGB?: number;
}

/**
 * MLX has no server: the trusted runner measures it in-process with
 * tools/mlx_bench.py, so the local CLI shells out to the same script. Using
 * the identical harness locally and on the runner is the whole point — a
 * different local harness is how you get a local number that does not
 * survive submission.
 */
export class MlxRunner implements Runner {
  name = "mlx-baseline";
  constructor(private model: string, private script = "tools/mlx_bench.py", private corpus = "fixtures/gainz-corpus.txt") {}

  async benchmark(contract: Contract): Promise<BenchmarkResult> {
    if (!existsSync(this.script)) throw new Error(`${this.script} not found — run from the repository root`);
    if (!existsSync(this.corpus)) throw new Error(`${this.corpus} not found — run from the repository root`);
    const python = process.env.GAINZ_PYTHON ?? "python3";
    const args = [this.script, "--model", this.model, "--corpus", this.corpus, "--mode", "bench",
      "--prompt-tokens", String(contract.promptTokens), "--decode-tokens", String(contract.decodeTokens),
      "--runs", String(contract.measuredRuns), "--warmup", String(contract.warmupRuns)];
    const proc = spawnSync(python, args, { encoding: "utf8", maxBuffer: 32 * 1024 * 1024, env: process.env });
    if (proc.status !== 0) throw new Error(`mlx_bench.py failed (${proc.status}): ${(proc.stderr || proc.stdout || "").trim().slice(-400)}`);
    const line = proc.stdout.trim().split(/\r?\n/).filter(Boolean).pop() ?? "";
    let out: BenchJson;
    try { out = JSON.parse(line) as BenchJson; }
    catch { throw new Error(`mlx_bench.py did not emit JSON: ${line.slice(0, 200)}`); }

    return {
      schemaVersion: 1,
      contractId: contract.id,
      candidate: "local-mlx",
      timestamp: new Date().toISOString(),
      correctness: { passed: true, mismatches: 0 },
      prefill: { tokens: contract.promptTokens, secondsPerToken: out.prefillSecondsPerToken, tokensPerSecond: out.prefillTokensPerSecond },
      decode: { tokens: contract.decodeTokens, secondsPerToken: out.decodeSecondsPerToken, tokensPerSecond: out.decodeTokensPerSecond },
      ttft: { seconds: out.ttftSeconds },
      outputSha256: createHash("sha256").update(out.text ?? "").digest("hex"),
      telemetry: { gpuName: "Apple Silicon (MLX)", totalMemoryGiB: out.peakMemoryGB ? Number((out.peakMemoryGB * 0.931).toFixed(2)) : undefined },
      eligible: true,
    };
  }
}
