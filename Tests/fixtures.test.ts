import { expect, test } from "bun:test";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { TRACKS } from "../Sources/contracts";
import { engineFor } from "../Sources/runner/engine";

const fixtureDir = (trackId: string) => `correctness_prompts/${trackId.replace(/-gb10-v\d+$/, "")}/public_prompt.json`;

test("every track has a public correctness fixture", () => {
  for (const track of Object.values(TRACKS)) {
    const fixture = JSON.parse(readFileSync(fixtureDir(track.id), "utf8"));
    expect(fixture.trackId).toBe(track.id);
    expect(typeof fixture.prompt).toBe("string");
    expect(fixture.prompt.length).toBeGreaterThan(10);
    expect(fixture.temperature).toBe(0);
    expect(fixture.maxTokens).toBe(track.decodeTokens);
    if (fixture.goldenSha256 !== null) {
      expect(fixture.goldenSha256).toMatch(/^[a-f0-9]{64}$/);
    }
  }
});

test("benchmark.json editable paths cover the participant surface", () => {
  const manifest = JSON.parse(readFileSync("benchmark.json", "utf8"));
  expect(manifest.editablePaths).toContain("Sources/runner/");
  expect(manifest.scoring.formula).toBe("decode_speedup^0.65 * prefill_speedup^0.20 * ttft_speedup^0.15");
  for (const trackId of manifest.tracks) expect(TRACKS[trackId]).toBeDefined();
});

test("benchmark.json and contracts.ts register the same eight tracks", () => {
  const manifest = JSON.parse(readFileSync("benchmark.json", "utf8"));
  expect(new Set(manifest.tracks)).toEqual(new Set(Object.keys(TRACKS)));
  expect(Object.keys(TRACKS)).toHaveLength(8);
  expect(TRACKS[manifest.defaultTrack]).toBeDefined();
});

// The modifiable-surface script once allowed none of the patch directories,
// which silently rejected every llama.cpp, MLX and vLLM-source submission.
// Anything benchmark.json calls editable must survive that script.
test("enforce-modifiable-surface.sh accepts every editable path", () => {
  const manifest = JSON.parse(readFileSync("benchmark.json", "utf8"));
  const script = readFileSync("tools/enforce-modifiable-surface.sh", "utf8");
  const accepted = script
    .split("\n")
    .filter((line) => /Sources\/.*\)\s*echo "OK:/.test(line))
    .join("|");
  for (const path of manifest.editablePaths as string[]) {
    const prefix = path.replace(/\*+$/, "").replace(/\/$/, "");
    expect(accepted).toContain(`${prefix}/*`);
  }
});

test("every track resolves to a real engine and a real patch surface", () => {
  for (const track of Object.values(TRACKS)) {
    const engine = engineFor(track);
    expect(["vllm", "llamacpp", "mlx"]).toContain(engine);
    if (engine === "vllm") {
      // The two vLLM tracks share one series; there is no per-track directory.
      expect(existsSync(`Sources/patches/${track.id}`)).toBe(false);
      expect(existsSync("Sources/vllm-patches/README.md")).toBe(true);
    } else {
      // llama.cpp and MLX tracks are per-track, and each must document itself
      // so an incoming agent can read what has already been tried.
      expect(existsSync(`Sources/patches/${track.id}/README.md`)).toBe(true);
    }
  }
});

test("no patch series directory shadows a track that does not exist", () => {
  for (const dir of readdirSync("Sources/patches", { withFileTypes: true })) {
    if (dir.isDirectory()) expect(TRACKS[dir.name]).toBeDefined();
  }
});
