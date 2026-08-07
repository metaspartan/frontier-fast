// frontier.fast trace plugin for OpenCode.
// Records harness sessions locally to .gainz/trace.jsonl for submission
// attribution. Local-only; never makes network calls.
import { spawnSync } from "node:child_process";

function trace(event: string): void {
  spawnSync("sh", [".gainz/hooks/gainz-trace.sh", "opencode", event], {
    stdio: "ignore",
    timeout: 5_000,
  });
}

export const GainzTracePlugin = async () => {
  trace("session");
  return {
    event: async ({ event }: { event: { type: string } }) => {
      if (event.type === "session.idle") trace("stop");
      if (event.type === "message.updated") trace("capture");
    },
  };
};
