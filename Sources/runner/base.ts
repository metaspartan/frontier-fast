import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import type { BenchmarkResult, Contract } from "../types";

/**
 * The prompts the trusted runner actually measures on.
 *
 * This harness used to time `"The quick brown fox jumps over the lazy dog. "`
 * repeated to length, and that is fine for a kernel change and catastrophic for
 * a speculative one: repeated filler is trivially predictable, so a draft head
 * accepts nearly everything and the local number is fiction. Measured on
 * qwen3.8-27b/R9700 — DFlash2 on filler reads **acceptance 0.956, 118.5 tok/s
 * against MTP's 77.4, a fake +53%**; the same configuration on this corpus
 * reads acceptance 0.74 and decode parity, and the ranked runner then scored it
 * BELOW the incumbent. An agent iterating against filler is measuring the
 * prompt, not the engine.
 *
 * `fixtures/gainz-corpus.txt` is byte-identical to the passages the coordinator
 * slices for ranked runs, so a local number computed here can predict a ranked
 * verdict. Slices rotate by run index exactly as the ranked runner's do.
 */
const CORPUS_PATH = process.env.GAINZ_CORPUS ?? "fixtures/gainz-corpus.txt";
/** 512 ranked tokens is 2600 chars of this corpus — the runner's own default. */
const CHARS_PER_PROMPT_TOKEN = 2600 / 512;

let passagesCache: string[] | null = null;
function corpusPassages(): string[] {
  if (passagesCache) return passagesCache;
  try {
    passagesCache = readFileSync(CORPUS_PATH, "utf8").split("\n\n").map((p) => p.trim()).filter(Boolean);
  } catch {
    passagesCache = [];
  }
  if (!passagesCache.length) {
    // Loud, because the alternative is a number that cannot predict anything.
    console.warn(`WARNING: ${CORPUS_PATH} is missing or empty. Falling back to filler text — speculative measurements taken this way are MEANINGLESS (filler inflates draft acceptance toward 1.0) and will not predict a ranked verdict.`);
  }
  return passagesCache;
}

/** Deterministic per index, byte-identical across processes and machines. */
function corpusPrompt(runIndex: number, targetChars: number): string {
  const passages = corpusPassages();
  if (!passages.length) return "The quick brown fox jumps over the lazy dog. ".repeat(Math.ceil(targetChars / 44));
  const parts: string[] = [];
  let length = 0;
  for (let i = 0; length < targetChars; i++) {
    const passage = passages[(runIndex + i) % passages.length];
    parts.push(passage);
    length += passage.length + 2;
  }
  return parts.join("\n\n").slice(0, targetChars);
}

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

/** Telemetry label from the contract's machine id. This was hardcoded to
 * "GB10", which mislabelled every R9700 result. */
function deviceLabel(machine: string): string {
  if (machine.includes("r9700")) return "Radeon AI PRO R9700";
  if (machine.includes("apple")) return "Apple Silicon";
  if (machine.includes("gb10")) return "GB10";
  return machine;
}

/**
 * OpenAI-completions runner — drives vLLM or llama-server. Participants
 * optimize the serving stack this measures.
 *
 * Measurement is wall-clock and cache-cold:
 * - TTFT is the wall time of a max_tokens=1 completion over a fresh prompt.
 * - Decode rate comes from a full decode-window completion minus that TTFT.
 * - Prefill is a two-point slope: the same cold request at a long and a short
 *   prompt, differenced, so the fixed per-request cost cancels and what is
 *   left is the marginal prompt-processing rate. This used to be
 *   `ttft / promptTokens`, which made prefillSpeedup identical to
 *   ttftSpeedup and rested 0.35 of the score on one number; the trusted
 *   runner and tools/mlx_bench.py both use the slope, so the local harness
 *   has to as well or local prefill cannot predict a ranked verdict.
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
    const targetChars = Math.round(contract.promptTokens * CHARS_PER_PROMPT_TOKEN);
    // Short arm of the prefill slope. Same request shape, ~1/8 the prompt.
    const shortTokens = Math.max(32, Math.floor(contract.promptTokens / 8));
    const shortChars = Math.round(shortTokens * CHARS_PER_PROMPT_TOKEN);
    const session = crypto.randomUUID().slice(0, 8);

    const probe = await this.timedCompletion(CORRECTNESS_PROMPT, contract.decodeTokens);
    const probeRepeat = await this.timedCompletion(CORRECTNESS_PROMPT, contract.decodeTokens);
    const selfConsistent = probe.text === probeRepeat.text;
    const outputSha256 = createHash("sha256").update(probe.text).digest("hex");

    for (let i = 0; i < Math.max(contract.warmupRuns, 1); i++) {
      await this.timedCompletion(`warmup ${session} ${i}: ${corpusPrompt(i, targetChars)}`, 8);
    }

    const ttfts: number[] = [];
    const shortTtfts: number[] = [];
    const decodeSecondsPerToken: number[] = [];
    let promptTokens = contract.promptTokens;
    let shortPromptTokens = shortTokens;
    let completionTokens = contract.decodeTokens;
    for (let i = 0; i < Math.max(contract.measuredRuns, 1); i++) {
      // One slice per run index, rotated as the ranked runner rotates it. The
      // ttft and full arms must see the SAME text, or their difference is not
      // a decode time.
      const slice = corpusPrompt(100 + i, targetChars);
      const first = await this.timedCompletion(`ttft ${session} ${i}: ${slice}`, 1);
      const short = await this.timedCompletion(`short ${session} ${i}: ${corpusPrompt(100 + i, shortChars)}`, 1);
      const full = await this.timedCompletion(`full ${session} ${i}: ${slice}`, contract.decodeTokens);
      promptTokens = full.promptTokens;
      shortPromptTokens = short.promptTokens;
      completionTokens = full.completionTokens;
      ttfts.push(first.seconds);
      shortTtfts.push(short.seconds);
      decodeSecondsPerToken.push(Math.max(full.seconds - first.seconds, 1e-6) / Math.max(full.completionTokens - 1, 1));
    }

    const ttftSeconds = median(ttfts);
    const decodePerToken = median(decodeSecondsPerToken);

    // Two-point slope; fall back to the TTFT-derived rate only if the two
    // points are too close to difference meaningfully.
    const deltaTokens = promptTokens - shortPromptTokens;
    const deltaSeconds = ttftSeconds - median(shortTtfts);
    const prefillPerToken = deltaTokens > 0 && deltaSeconds > 1e-4
      ? deltaSeconds / deltaTokens
      : ttftSeconds / Math.max(promptTokens, 1);

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
      telemetry: { gpuName: deviceLabel(contract.machine) },
      eligible: selfConsistent,
      failure: selfConsistent ? undefined : "engine output is nondeterministic for identical greedy input; serve with VLLM_BATCH_INVARIANT=1",
    };
  }
}
