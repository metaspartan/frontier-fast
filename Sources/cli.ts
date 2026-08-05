import { readFileSync } from "node:fs";
import { TRACKS } from "./contracts";
import { runnerFor, engineFor, serverHint } from "./runner";
import { score } from "./scoring";
import type { Contract } from "./types";

const args = process.argv.slice(2);
const command = args[0] ?? "help";

function getTrack(): Contract {
  const trackFlag = args.indexOf("--track");
  const trackId = trackFlag >= 0 ? args[trackFlag + 1] : (process.env.GAINZ_TRACK ?? "laguna-xs-2.1-nvfp4-gb10-v1");
  const track = TRACKS[trackId];
  if (!track) { console.error(`Unknown track: ${trackId}`); process.exit(1); }
  return track;
}

function readGolden(trackId: string): { goldenSha256?: string | null } | undefined {
  try { return JSON.parse(readFileSync(`correctness_prompts/${trackId.replace(/-gb10-v\d+$/, "")}/public_prompt.json`, "utf8")); }
  catch { return undefined; }
}

if (command === "benchmark" || command === "run") {
  const baseTrack = getTrack();
  const mode = args.includes("--local-submit") ? "--local-submit" : args.includes("--baseline") ? "--baseline" : "--local-iterate";
  // Local iterate keeps the loop fast; local submit uses the full contract window.
  const track: Contract = mode === "--local-iterate" ? { ...baseTrack, warmupRuns: 1, measuredRuns: 2 } : baseTrack;

  console.log(`Track: ${track.id}`);
  console.log(`Model: ${track.model} (${track.quantization})`);
  console.log(`Device: ${track.machine}`);
  console.log(`Window: ${track.promptTokens}-token prefill + ${track.decodeTokens}-token decode · ${track.warmupRuns} warmup / ${track.measuredRuns} measured`);
  console.log("");

  const runner = runnerFor(track);
  console.log(`Engine: ${engineFor(track)} (${runner.name})`);

  const main = async () => {
    console.log("Measuring baseline...");
    const baseline = await runner.benchmark(track);
    console.log("Baseline:", JSON.stringify({
      prefill_tps: baseline.prefill.tokensPerSecond.toFixed(1),
      decode_tps: baseline.decode.tokensPerSecond.toFixed(1),
      ttft_s: baseline.ttft?.seconds.toFixed(4),
      output_sha256: baseline.outputSha256?.slice(0, 16),
      deterministic: baseline.correctness.passed,
    }));

    const golden = readGolden(track.id);
    if (golden?.goldenSha256 && baseline.outputSha256) {
      const match = golden.goldenSha256 === baseline.outputSha256;
      console.log(match
        ? "Public drift tripwire: output matches the committed GB10 golden."
        : "Public drift tripwire: output DIFFERS from the committed GB10 golden. Local hardware/kernels may reorder near-tie argmaxes; the ranked GB10 result is authoritative.");
    }

    if (mode === "--baseline") return;

    console.log("\nMeasuring candidate (current working tree serving config)...");
    const candidate = await runner.benchmark(track);
    if (baseline.outputSha256 && candidate.outputSha256 && baseline.outputSha256 !== candidate.outputSha256) {
      candidate.correctness = { passed: false, mismatches: candidate.correctness.mismatches + 1 };
    }
    const result = score(candidate, baseline);

    console.log("Score:", JSON.stringify({
      score: result.score?.toFixed(4) ?? null,
      decode_speedup: result.decodeSpeedup?.toFixed(4),
      prefill_speedup: result.prefillSpeedup?.toFixed(4),
      ttft_speedup: result.ttftSpeedup?.toFixed(4),
      eligible: result.eligible,
    }));
    if (!result.eligible && candidate.failure) console.log(`Ineligible: ${candidate.failure}`);
    console.log("\nLocal timing is directional only. Official results come from the trusted GB10 runner.");
  };

  main().catch((err) => {
    console.error("Benchmark failed:", err.message);
    console.error("\n" + serverHint(track));
    process.exit(1);
  });
} else {
  console.log("Usage: bun run Sources/cli.ts run --track <id> [--local-iterate|--local-submit|--baseline]");
  console.log("       (\"benchmark\" is an alias for \"run\"; GAINZ_TRACK also sets the track)");
  console.log("Tracks:", Object.keys(TRACKS).join(", "));
}
