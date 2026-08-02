import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { TRACKS } from "../Sources/contracts";

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
