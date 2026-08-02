import { createHash } from "node:crypto";
import type { BenchmarkResult, Contract } from "../types";

export interface Runner {
  name: string;
  benchmark(contract: Contract): Promise<BenchmarkResult>;
}

const CORRECTNESS_PROMPT = "List the prime numbers below fifty, then explain in two sentences why there are infinitely many primes. ";

interface TimedCompletion {
  seconds: number;
  promptTokens: number;
  completionTokens: number;
  text: string;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Baseline runner — participants optimize the serving stack this measures.
 *
 * Measurement is wall-clock and cache-cold:
 * - TTFT is the wall time of a max_tokens=1 completion over a fresh prompt.
 * - Decode rate comes from a full decode-window completion minus that TTFT.
 * - Every timing prompt carries a unique prefix so vLLM prefix caching can
 *   never turn a repeated prompt into a fake prefill speedup.
 * - A fixed greedy correctness probe runs twice; if the engine cannot
 *   reproduce its own output, the result is marked ineligible. Serve with
 *   VLLM_BATCH_INVARIANT=1 for deterministic greedy decoding.
 */
export class VllmRunner implements Runner {
  name = "vllm-baseline";
  private baseUrl: string;
  private model: string;
  private apiKey?: string;

  constructor(baseUrl = "http://127.0.0.1:8000/v1", model = "poolside/Laguna-XS-2.1-NVFP4", apiKey?: string) {
    this.baseUrl = baseUrl;
    this.model = model;
    this.apiKey = apiKey ?? process.env.VLLM_API_KEY;
  }

  private async timedCompletion(prompt: string, maxTokens: number): Promise<TimedCompletion> {
    const headers: Record<string, string> = { "content-type": "application/json" };
    if (this.apiKey) headers.authorization = `Bearer ${this.apiKey}`;
    const started = performance.now();
    const response = await fetch(`${this.baseUrl}/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model: this.model, prompt, max_tokens: maxTokens, temperature: 0, stream: false }),
      signal: AbortSignal.timeout(180_000),
    });
    const seconds = (performance.now() - started) / 1000;
    if (!response.ok) throw new Error(`vLLM request failed: ${response.status} ${(await response.text()).slice(0, 200)}`);
    const body = await response.json() as { usage?: { prompt_tokens?: number; completion_tokens?: number }; choices?: { text: string }[] };
    if (!body.usage?.prompt_tokens || !body.usage?.completion_tokens) throw new Error("vLLM did not return usage stats");
    return { seconds, promptTokens: body.usage.prompt_tokens, completionTokens: body.usage.completion_tokens, text: body.choices?.[0]?.text ?? "" };
  }

  async benchmark(contract: Contract): Promise<BenchmarkResult> {
    const basePrompt = "The quick brown fox jumps over the lazy dog. ".repeat(Math.ceil(contract.promptTokens / 10));
    const session = crypto.randomUUID().slice(0, 8);

    const probe = await this.timedCompletion(CORRECTNESS_PROMPT, contract.decodeTokens);
    const probeRepeat = await this.timedCompletion(CORRECTNESS_PROMPT, contract.decodeTokens);
    const selfConsistent = probe.text === probeRepeat.text;
    const outputSha256 = createHash("sha256").update(probe.text).digest("hex");

    for (let i = 0; i < Math.max(contract.warmupRuns, 1); i++) {
      await this.timedCompletion(`warmup ${session} ${i}: ${basePrompt}`, 8);
    }

    const ttfts: number[] = [];
    const decodeSecondsPerToken: number[] = [];
    let promptTokens = contract.promptTokens;
    let completionTokens = contract.decodeTokens;
    for (let i = 0; i < Math.max(contract.measuredRuns, 1); i++) {
      const first = await this.timedCompletion(`ttft ${session} ${i}: ${basePrompt}`, 1);
      const full = await this.timedCompletion(`full ${session} ${i}: ${basePrompt}`, contract.decodeTokens);
      promptTokens = full.promptTokens;
      completionTokens = full.completionTokens;
      ttfts.push(first.seconds);
      decodeSecondsPerToken.push(Math.max(full.seconds - first.seconds, 1e-6) / Math.max(full.completionTokens - 1, 1));
    }

    const ttftSeconds = median(ttfts);
    const decodePerToken = median(decodeSecondsPerToken);
    const prefillPerToken = ttftSeconds / Math.max(promptTokens, 1);

    return {
      schemaVersion: 1,
      contractId: contract.id,
      candidate: this.name,
      timestamp: new Date().toISOString(),
      correctness: { passed: selfConsistent, mismatches: selfConsistent ? 0 : 1 },
      prefill: { tokens: promptTokens, secondsPerToken: prefillPerToken, tokensPerSecond: 1 / prefillPerToken },
      decode: { tokens: completionTokens, secondsPerToken: decodePerToken, tokensPerSecond: 1 / decodePerToken },
      ttft: { seconds: ttftSeconds },
      outputSha256,
      telemetry: { gpuName: "GB10" },
      eligible: selfConsistent,
      failure: selfConsistent ? undefined : "engine output is nondeterministic for identical greedy input; serve with VLLM_BATCH_INVARIANT=1",
    };
  }
}
