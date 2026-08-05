import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { homedir } from "node:os";
import { join } from "node:path";
import { TRACKS } from "./contracts";
import { runnerFor, engineFor, serverHint } from "./runner";
import { score } from "./scoring";
import type { Contract } from "./types";

const args = process.argv.slice(2);
const command = args[0] ?? "help";

const API = process.env.GAINZFAST_URL ?? "https://gainz.fast";
const REPO_URL = "https://github.com/metaspartan/gainz-fast.git";

function flag(name: string): string | undefined {
  const at = args.indexOf(`--${name}`);
  return at >= 0 ? args[at + 1] : undefined;
}

function getTrack(): Contract {
  const trackId = flag("track") ?? process.env.GAINZ_TRACK ?? "laguna-xs-2.1-nvfp4-gb10-v1";
  const track = TRACKS[trackId];
  if (!track) {
    console.error(`Unknown track: ${trackId}`);
    console.error(`Known tracks: ${Object.keys(TRACKS).join(", ")}`);
    process.exit(1);
  }
  return track;
}

function readGolden(trackId: string): { goldenSha256?: string | null } | undefined {
  try { return JSON.parse(readFileSync(`correctness_prompts/${trackId.replace(/-gb10-v\d+$/, "")}/public_prompt.json`, "utf8")); }
  catch { return undefined; }
}

/** Submit-only token. Never hardcoded and never written to the repository:
 * the environment first, then the location the hosted gainzfast CLI uses. */
function readToken(): string | undefined {
  if (process.env.GAINZ_TOKEN) return process.env.GAINZ_TOKEN.trim();
  const configHome = process.env.XDG_CONFIG_HOME ?? join(homedir(), ".config");
  try { return readFileSync(join(configHome, "gainzfast", "token"), "utf8").trim() || undefined; }
  catch { return undefined; }
}

function usage(): void {
  console.log(`gainzfast — gainz.fast challenge CLI (in-repo entry point)

  bun run Sources/cli.ts tracks
  bun run Sources/cli.ts clone [--track <id>]
  bun run Sources/cli.ts setup
  bun run Sources/cli.ts run [--track <id>] [--local-iterate|--local-submit|--baseline]
  bun run Sources/cli.ts submit --name "<change summary>" [--track <id>]
                                [--notes <text>] [--pr <url>] [--agent <name>]
                                [--dry-run]

"benchmark" is an alias for "run". GAINZ_TRACK sets the default track.
submit reads a submit-only token from GAINZ_TOKEN, falling back to
~/.config/gainzfast/token. Never commit that token.

A submission claims a physical GPU for ~22 minutes and cannot be recalled.
Pass --dry-run to print the payload without sending it.

The hosted CLI (curl -fsSL ${API}/install.sh | sh) offers the same
commands plus login/whoami/status; setup and run are thin wrappers around
./setup.sh and ./benchmark.sh, which you can always call directly.

Tracks: ${Object.keys(TRACKS).join(", ")}`);
}

function shell(cmd: string, cmdArgs: string[]): never {
  const proc = spawnSync(cmd, cmdArgs, { stdio: "inherit", shell: process.platform === "win32" });
  process.exit(proc.status ?? 1);
}

async function submit(): Promise<void> {
  const track = getTrack();
  const name = flag("name");
  if (!name) {
    console.error('submit requires --name "<change summary>" — it is the leaderboard row everyone reads,');
    console.error("and a name that describes the model rather than the change is rejected.");
    process.exit(1);
  }
  const token = readToken();
  if (!token) {
    console.error("No token. Set GAINZ_TOKEN=<gz_token>, or run: gainzfast login <gz_token>");
    console.error("Mint one on https://gainz.fast while signed in with GitHub. Never commit it.");
    process.exit(1);
  }

  const git = (a: string[]) => spawnSync("git", a, { encoding: "utf8" }).stdout?.trim() ?? "";
  const commitSha = git(["rev-parse", "HEAD"]);
  const origin = git(["remote", "get-url", "origin"])
    .replace(/^git@github\.com:/, "https://github.com/")
    .replace(/\.git$/, "");

  const body: Record<string, unknown> = {
    id: `${name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}-${Math.floor(Date.now() / 1000)}`,
    displayName: name,
    family: track.family,
    model: track.model,
    revision: track.revision,
    quantization: track.quantization,
    machine: track.machine,
    promptTokens: track.promptTokens,
    decodeTokens: track.decodeTokens,
    warmupRuns: track.warmupRuns,
    measuredRuns: track.measuredRuns,
  };
  const notes = flag("notes"); if (notes) body.notes = notes;
  const pr = flag("pr"); if (pr) body.pullRequestUrl = pr;
  const agent = flag("agent") ?? process.env.GAINZFAST_AGENT; if (agent) body.agentName = agent;
  if (commitSha) body.commitSha = commitSha;
  if (origin.startsWith("https://github.com/")) body.repositoryUrl = origin;

  // A submission claims a physical GPU for ~22 minutes and cannot be
  // recalled — only the trusted runner may transition its state. Use
  // --dry-run to inspect the payload without spending a slot.
  if (args.includes("--dry-run")) {
    console.log("--dry-run: not sending. Payload that WOULD be POSTed:");
    console.log(JSON.stringify(body, null, 2));
    console.log(`\nPOST ${API}/api/submissions  (token: ${token ? "present" : "missing"})`);
    return;
  }

  const response = await fetch(`${API}/api/submissions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
  });
  const result = await response.json() as { id?: string; status?: string; error?: string };
  if (!response.ok || result.error) {
    console.error(`Submission rejected (${response.status}): ${result.error ?? "unknown error"}`);
    process.exit(1);
  }
  console.log(`Submitted ${result.id} (${result.status}) on ${track.id}.`);
  console.log("The trusted runner will verify it; watch https://gainz.fast or /api/submissions/mine.");
}

if (command === "help" || command === "--help" || command === "-h") {
  usage();
} else if (command === "tracks") {
  for (const t of Object.values(TRACKS)) {
    console.log(`${t.id.padEnd(32)} ${engineFor(t).padEnd(9)} ${t.machine.padEnd(22)} ${t.model}`);
  }
} else if (command === "clone") {
  const track = flag("track");
  console.log(`Cloning ${REPO_URL}${track ? ` (track: ${track})` : ""}`);
  console.log("This repository ships agent hooks (.claude, .cursor, .codex, .opencode).");
  console.log("If your harness loads repository hooks at startup, relaunch it from gainz-fast/.");
  shell("git", ["clone", REPO_URL]);
} else if (command === "setup") {
  shell(process.platform === "win32" ? "bash" : "./setup.sh", process.platform === "win32" ? ["./setup.sh"] : []);
} else if (command === "submit") {
  submit().catch((err) => { console.error("Submit failed:", err.message); process.exit(1); });
} else if (command === "benchmark" || command === "run") {
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
        ? "Public drift tripwire: output matches the committed golden."
        : "Public drift tripwire: output DIFFERS from the committed golden. Local hardware/kernels may reorder near-tie argmaxes; the ranked result on this track's pinned machine is authoritative.");
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
    console.log(`\nLocal timing is directional only. Official results come from the ${track.machine} trusted runner.`);
  };

  main().catch((err) => {
    console.error("Benchmark failed:", err.message);
    console.error("\n" + serverHint(track));
    process.exit(1);
  });
} else {
  console.error(`Unknown command: ${command}\n`);
  usage();
  process.exit(1);
}
