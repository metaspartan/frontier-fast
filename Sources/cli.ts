import { TRACKS } from "./contracts";
import { VllmRunner } from "./runner";
import { score } from "./scoring";
import type { Contract } from "./types";

const args = process.argv.slice(2);
const command = args[0] ?? "help";

function getTrack(): Contract {
  const trackFlag = args.indexOf("--track");
  const trackId = trackFlag >= 0 ? args[trackFlag + 1] : "laguna-xs-2.1-nvfp4-gb10-v1";
  const track = TRACKS[trackId];
  if (!track) { console.error(`Unknown track: ${trackId}`); process.exit(1); }
  return track;
}

if (command === "benchmark") {
  const track = getTrack();
  const mode = args[1] ?? "--local-iterate";
  console.log(`Track: ${track.id}`);
  console.log(`Model: ${track.model} (${track.quantization})`);
  console.log(`Device: ${track.machine}`);
  console.log(`Window: ${track.promptTokens}-token prefill + ${track.decodeTokens}-token decode`);
  console.log("");

  const vllmUrl = process.env.VLLM_BASE_URL ?? "http://127.0.0.1:8000/v1";
  const runner = new VllmRunner(vllmUrl, track.model);

  console.log("Running baseline...");
  runner.benchmark(track).then(async (baseline) => {
    console.log("Baseline:", JSON.stringify({ prefill_tps: baseline.prefill.tokensPerSecond.toFixed(1), decode_tps: baseline.decode.tokensPerSecond.toFixed(1), ttft: baseline.ttft?.seconds.toFixed(3) }));

    console.log("\nRunning candidate (same config for local estimate)...");
    const candidate = await runner.benchmark(track);
    const result = score(candidate, baseline);

    console.log("Score:", JSON.stringify({
      score: result.score?.toFixed(4),
      decode_speedup: result.decodeSpeedup?.toFixed(4),
      prefill_speedup: result.prefillSpeedup?.toFixed(4),
      ttft_speedup: result.ttftSpeedup?.toFixed(4),
      eligible: result.eligible,
    }));

    if (!result.eligible) {
      console.log("\nNOTE: Local timing is directional only. Official results come from the trusted runner.");
    }
  }).catch((err) => {
    console.error("Benchmark failed:", err.message);
    console.error("\nMake sure vLLM is running at the configured URL.");
    console.error("Set VLLM_BASE_URL to point to your vLLM server.");
    process.exit(1);
  });
} else {
  console.log("Usage: bun run Sources/cli.ts benchmark --track <id>");
  console.log("Tracks:", Object.keys(TRACKS).join(", "));
}
