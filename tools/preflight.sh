#!/usr/bin/env bash
# gainz.fast pre-flight — validate a candidate LOCALLY before spending a
# trusted-runner slot.
#
# Method: boot a CONTROL (no patch) and your CANDIDATE sequentially with
# byte-identical flags and the same co-residency, then compare them using the
# same teacher-forced check the ranked runner uses.
#
# Do NOT compare against the ranked baseline on :8001. That engine does not
# share the GPU, so its KV-cache size differs from a co-resident engine, and
# the memory difference alone produces false mismatches in both directions
# (measured: ~3/128 tokens against a provable no-op).
#
#   set -a; . ~/gainz-runner/.env; set +a
#   API_KEY=$DGX_VLLM_API_KEY ./tools/preflight.sh
set -euo pipefail

API_KEY="${API_KEY:-}"
MODEL="${MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.1}"
HF_CACHE="${HF_CACHE:-/mnt/data4tb/cache/huggingface}"
PORT="${PREFLIGHT_PORT:-8004}"
UTIL="${PREFLIGHT_UTIL:-0.40}"
CONTROL="gainz-preflight-control-$$"
CANDIDATE="gainz-preflight-candidate-$$"
PROMPT="List the prime numbers below fifty, then explain in two sentences why there are infinitely many primes. "

cleanup() { docker rm -f "$CONTROL" "$CANDIDATE" >/dev/null 2>&1 || true; }
trap cleanup EXIT

[ -n "$API_KEY" ] || { echo "FAIL: set API_KEY (see ~/gainz-runner/.env)"; exit 1; }
[ -f Sources/runner/serving.json ] || { echo "FAIL: run from the challenge repository root"; exit 1; }

OVERRIDES=$(python3 -c "import json;print(json.dumps(json.load(open('Sources/runner/serving.json')).get('overrides',{})))")
KERNELS=$(python3 -c "import json,sys;print('1' if json.loads(sys.argv[1]).get('kernels') else '')" "$OVERRIDES")
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
echo "==> overrides: $OVERRIDES"
[ -n "$FLAGS" ] && echo "    engine flags: $FLAGS"

boot() {  # boot <container-name> <with_kernels>
  name="$1"; with_kernels="$2"
  if [ -n "$with_kernels" ]; then
    docker run -d --name "$name" --gpus all --ipc=host --shm-size=16g -e VLLM_BATCH_INVARIANT=1 \
      -v "$HF_CACHE":/root/.cache/huggingface -v "$PWD":/submission:ro -e HF_HUB_OFFLINE=1 \
      -p "$PORT":8000 --entrypoint bash "$IMAGE" \
      -c "pip install --no-deps --quiet /submission/Sources/kernels && exec vllm serve $MODEL --served-model-name $MODEL --trust-remote-code --host 0.0.0.0 --port 8000 --api-key $API_KEY --max-model-len 8192 --gpu-memory-utilization $UTIL --num-gpu-blocks-override 40000 --max-num-seqs 4 $FLAGS" >/dev/null
  else
    eval docker run -d --name "$name" --gpus all --ipc=host --shm-size=16g -e VLLM_BATCH_INVARIANT=1 \
      -v "$HF_CACHE":/root/.cache/huggingface -p "$PORT":8000 "$IMAGE" \
      --model "$MODEL" --served-model-name "$MODEL" --trust-remote-code --host 0.0.0.0 --port 8000 \
      --api-key "$API_KEY" --max-model-len 8192 --gpu-memory-utilization "$UTIL" --num-gpu-blocks-override 40000 --max-num-seqs 4 $FLAGS >/dev/null
  fi
  for _ in $(seq 1 180); do
    if curl -sf -m 4 -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then return 0; fi
    if [ "$(docker inspect "$name" --format '{{.State.Status}}' 2>/dev/null)" = "exited" ]; then
      echo "FAIL: $name exited during boot:"
      docker logs "$name" --tail 8 2>&1 | grep -iE "error|usage:" | tail -3
      return 1
    fi
    sleep 10
  done
  echo "FAIL: $name never became ready"
  return 1
}

golden_from() {
  curl -s -m 300 -H "Authorization: Bearer $API_KEY" -H "content-type: application/json" \
    -d "$(python3 -c "import json,sys;print(json.dumps({'model':sys.argv[1],'prompt':sys.argv[2],'max_tokens':128,'temperature':0}))" "$MODEL" "$PROMPT")" \
    "http://127.0.0.1:$PORT/v1/completions" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['text'])"
}

echo "==> booting CONTROL (no patch, util=$UTIL)"
boot "$CONTROL" "" || exit 1
GOLDEN=$(golden_from)
docker rm -f "$CONTROL" >/dev/null 2>&1
[ -n "$GOLDEN" ] || { echo "FAIL: control produced no output"; exit 1; }
echo "    control golden captured ($(printf %s "$GOLDEN" | wc -c) chars)"

echo "==> booting CANDIDATE (your patch, identical config)"
boot "$CANDIDATE" "$KERNELS" || exit 1

echo "==> teacher-forced comparison (candidate vs control)"
python3 - "$PORT" "$API_KEY" "$MODEL" "$PROMPT" "$GOLDEN" <<'PY'
import json, sys, urllib.request
port, key, model, prompt, golden = sys.argv[1:6]
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "authorization": f"Bearer {key}"})
    return json.load(urllib.request.urlopen(req, timeout=300))
start = post({"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0})["usage"]["prompt_tokens"]
pl = post({"model": model, "prompt": prompt + golden, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})["choices"][0]["prompt_logprobs"]
checked = mism = 0
first = None
for i, slot in enumerate(pl):
    if i < start or not slot:
        continue
    checked += 1
    if any((e.get("rank") or 1) > 1 for e in slot.values()):
        mism += 1
        first = first or checked
print(f"    checked_steps={checked} mismatches={mism}" + (f" first_at={first}" if first else ""))
if mism == 0:
    print("PASS: bit-exact against an identically-configured control — safe to submit.")
else:
    print(f"FAIL: {mism}/{checked} tokens diverge from the control.")
    print("      Control and candidate differ only by your patch, so this is a real")
    print("      numerical change. Preserve the floating-point reduction order.")
sys.exit(0 if mism == 0 else 2)
PY
