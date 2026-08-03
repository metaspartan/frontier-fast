#!/usr/bin/env bash
# gainz.fast pre-flight: validate a candidate LOCALLY before spending a
# trusted-runner slot. Boots your Sources/runner/serving.json (and
# Sources/kernels when "kernels": true) as a second engine ALONGSIDE the
# pinned baseline, then runs the same teacher-forced correctness check the
# ranked runner uses. Correctness is exact here; timing still comes from the
# trusted runner (this engine shares the GPU, so local timing is indicative).
#
#   BASE_URL=http://127.0.0.1:8001/v1 API_KEY=... ./tools/preflight.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8001/v1}"
API_KEY="${API_KEY:-}"
MODEL="${MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.1}"
HF_CACHE="${HF_CACHE:-/mnt/data4tb/cache/huggingface}"
PORT="${PREFLIGHT_PORT:-8004}"
NAME="gainz-preflight-$$"
PROMPT="List the prime numbers below fifty, then explain in two sentences why there are infinitely many primes. "

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; docker rm -f "${CONTROL:-}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> reading Sources/runner/serving.json"
OVERRIDES=$(python3 -c "import json;print(json.dumps(json.load(open('Sources/runner/serving.json')).get('overrides',{})))")
echo "    $OVERRIDES"

FLAGS=$(python3 - "$OVERRIDES" <<'PY'
import json, sys
o = json.loads(sys.argv[1]); f = []
if "maxNumSeqs" in o: f += ["--max-num-seqs", str(o["maxNumSeqs"])]
if "maxNumBatchedTokens" in o: f += ["--max-num-batched-tokens", str(o["maxNumBatchedTokens"])]
if o.get("enforceEager"): f += ["--enforce-eager"]
if "compilationLevel" in o: f += ["-O%d" % o["compilationLevel"]]
if "speculative" in o:
    s = o["speculative"]; spec = {"num_speculative_tokens": s["numSpeculativeTokens"]}
    if s.get("method") == "ngram":
        spec["method"] = "ngram"
        if "promptLookupMax" in s: spec["prompt_lookup_max"] = s["promptLookupMax"]
    else: spec["model"] = s["model"]
    f += ["--speculative-config", "'" + json.dumps(spec) + "'"]
print(" ".join(f))
PY
)

echo "==> capturing the baseline golden completion"
GOLDEN=$(curl -s -m 300 -H "Authorization: Bearer $API_KEY" -H "content-type: application/json" \
  -d "$(python3 -c "import json,sys;print(json.dumps({'model':'$MODEL','prompt':sys.argv[1],'max_tokens':128,'temperature':0}))" "$PROMPT")" \
  "$BASE_URL/completions" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['text'])")
[ -n "$GOLDEN" ] || { echo "FAIL: could not reach the baseline engine at $BASE_URL"; exit 1; }
echo "    golden captured ($(printf %s "$GOLDEN" | wc -c) chars)"

KERNELS=$(python3 -c "import json;print('1' if json.load(open('Sources/runner/serving.json')).get('overrides',{}).get('kernels') else '')")
echo "==> booting CANDIDATE (your patch) on :$PORT"
boot_engine "$NAME" "$FLAGS" "$KERNELS" || exit 1
echo "    candidate ready"

echo "==> teacher-forced correctness (same method as the ranked runner)"
python3 - "$PORT" "$API_KEY" "$MODEL" "$PROMPT" "$GOLDEN" <<'PY'
import json, sys, urllib.request
port, key, model, prompt, golden = sys.argv[1:6]
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "authorization": f"Bearer {key}"})
    return json.load(urllib.request.urlopen(req, timeout=300))
start = post({"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0})["usage"]["prompt_tokens"]
pl = post({"model": model, "prompt": prompt + golden, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})["choices"][0]["prompt_logprobs"]
checked = mism = 0; first = None
for i, slot in enumerate(pl):
    if i < start or not slot: continue
    checked += 1
    if any((e.get("rank") or 1) > 1 for e in slot.values()):
        mism += 1; first = first or checked
print(f"    checked_steps={checked} mismatches={mism}" + (f" first_at={first}" if first else ""))
print("PASS: bit-exact under teacher forcing — safe to submit." if mism == 0
      else f"FAIL: {mism}/{checked} tokens diverge from the control — the ranked runner will reject this.")
sys.exit(0 if mism == 0 else 2)
PY
