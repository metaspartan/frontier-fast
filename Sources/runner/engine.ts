import type { BenchmarkResult, Contract } from "../types";
import type { Runner } from "./base";
import { VllmRunner } from "./base";
import { MlxRunner } from "./mlx";

export type Engine = "vllm" | "llamacpp" | "mlx";

/** Which engine actually serves a track. The CLI used to assume vLLM for
 * everything, which meant an MLX track was measured by POSTing to an OpenAI
 * endpoint that does not exist on that box. */
export function engineFor(track: Contract): Engine {
  if (track.quantization.startsWith("gguf")) return "llamacpp";
  if (track.quantization.startsWith("mlx")) return "mlx";
  return "vllm";
}

const DEFAULT_URL: Record<Exclude<Engine, "mlx">, string> = {
  // llama-server's own default port, not vLLM's — getting this wrong is the
  // difference between a benchmark and a connection refused.
  vllm: "http://127.0.0.1:8000/v1",
  llamacpp: "http://127.0.0.1:8080/v1",
};

export function runnerFor(track: Contract): Runner {
  const engine = engineFor(track);
  if (engine === "mlx") return new MlxRunner(track.model);
  const url = process.env.GAINZ_BASE_URL ?? process.env.VLLM_BASE_URL ?? DEFAULT_URL[engine];
  const runner = new VllmRunner(url, track.model);
  runner.name = engine === "llamacpp" ? "llamacpp-baseline" : "vllm-baseline";
  return runner;
}

export function serverHint(track: Contract): string {
  switch (engineFor(track)) {
    case "mlx":
      return "MLX runs in-process — no server needed. Requires mlx-lm installed (pip install mlx-lm) and tools/mlx_bench.py present.";
    case "llamacpp":
      return "Make sure llama-server is running (GAINZ_BASE_URL, default http://127.0.0.1:8080/v1), e.g.\n  ./build/bin/llama-server -m <your gguf> -ngl 99 -c 8192 --parallel 1";
    default:
      return "Make sure a vLLM server is running (GAINZ_BASE_URL or VLLM_BASE_URL, VLLM_API_KEY).\nDeterministic serving requires VLLM_BATCH_INVARIANT=1.";
  }
}

export type { BenchmarkResult };
