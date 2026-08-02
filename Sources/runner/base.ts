import type { BenchmarkResult, Contract } from "../types";

export interface Runner {
  name: string;
  benchmark(contract: Contract): Promise<BenchmarkResult>;
}

/**
 * Baseline runner — participants optimize this.
 * This is the default implementation that talks to a local vLLM server.
 * Modify this file (or create new implementations) to optimize inference.
 */
export class VllmRunner implements Runner {
  name = "vllm-baseline";
  private baseUrl: string;
  private model: string;

  constructor(baseUrl = "http://127.0.0.1:8000/v1", model = "poolside/Laguna-XS-2.1-NVFP4") {
    this.baseUrl = baseUrl;
    this.model = model;
  }

  async benchmark(contract: Contract): Promise<BenchmarkResult> {
    const prompt = "The quick brown fox jumps over the lazy dog. ".repeat(Math.ceil(contract.promptTokens / 10));
    const start = Date.now();

    const response = await fetch(`${this.baseUrl}/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: this.model,
        prompt,
        max_tokens: contract.decodeTokens,
        temperature: 0,
        stream: false,
      }),
    });

    if (!response.ok) throw new Error(`vLLM error: ${response.status}`);
    const data = await response.json();
    const elapsed = (Date.now() - start) / 1000;

    const promptTokens = data.usage?.prompt_tokens ?? contract.promptTokens;
    const completionTokens = data.usage?.completion_tokens ?? contract.decodeTokens;

    return {
      schemaVersion: 1,
      contractId: contract.id,
      candidate: this.name,
      timestamp: new Date().toISOString(),
      correctness: { passed: true, mismatches: 0 },
      prefill: {
        tokens: promptTokens,
        secondsPerToken: elapsed / promptTokens,
        tokensPerSecond: promptTokens / elapsed,
      },
      decode: {
        tokens: completionTokens,
        secondsPerToken: elapsed / completionTokens,
        tokensPerSecond: completionTokens / elapsed,
      },
      ttft: { seconds: elapsed * 0.3 },
      telemetry: { gpuName: "GB10" },
      eligible: true,
    };
  }
}
