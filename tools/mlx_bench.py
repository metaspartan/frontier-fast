"""MLX measurement harness for gainz.fast Apple Silicon tracks.

Mirrors what the llama.cpp and vLLM runners measure so scores mean the same
thing across engines: a 512-token prefill, 128 greedy decode steps, TTFT from a
one-token completion, and perplexity over the fixed corpus as the accuracy gate.

Two modes:
  bench  -> {decodeTokensPerSecond, prefillTokensPerSecond, ttftSeconds, text}
  ppl    -> {perplexity}

Prefill uses the same two-point slope the vLLM runner uses: the same cold
request at a long and a short prompt, differenced, so the fixed per-request
cost cancels and what is left is the marginal prompt-processing rate. Deriving
prefill from TTFT alone would make prefillSpeedup identical to ttftSpeedup and
spend 0.35 of the score on one number.
"""
import argparse
import json
import math
import statistics
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


def timed_generate(model, tok, prompt_ids, max_tokens):
    """Greedy generation with an explicit prefill/decode split.

    mlx_lm's generate() helper hides the split, and we need TTFT separately, so
    the loop is written out: one forward pass over the prompt (prefill), then
    argmax steps (decode).
    """
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    ids = mx.array(prompt_ids)[None]

    t0 = time.perf_counter()
    logits = model(ids, cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    ttft = time.perf_counter() - t0

    out = [token.item()]
    t1 = time.perf_counter()
    for _ in range(max_tokens - 1):
        logits = model(token[None], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        out.append(token.item())
    decode_seconds = time.perf_counter() - t1

    steps = max(len(out) - 1, 1)
    return {
        "ttft": ttft,
        "decodeSecondsPerToken": decode_seconds / steps,
        "text": tok.decode(out),
    }


def corpus_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def bench(model, tok, corpus, prompt_tokens, decode_tokens, runs):
    """Long and short prompts sliced from the same corpus, so the only
    difference between the two points is length."""
    full = tok.encode(corpus)
    if len(full) < prompt_tokens * 2:
        full = full * (prompt_tokens * 2 // max(len(full), 1) + 2)
    short_tokens = max(32, prompt_tokens // 8)

    ttfts, shorts, decodes, text = [], [], [], ""
    for i in range(runs):
        # Rotate the slice per run so no two runs are the identical prompt.
        offset = (i * 97) % max(len(full) - prompt_tokens - 1, 1)
        long_ids = full[offset:offset + prompt_tokens]
        short_ids = full[offset:offset + short_tokens]

        long_run = timed_generate(model, tok, long_ids, decode_tokens)
        short_run = timed_generate(model, tok, short_ids, 1)

        ttfts.append(long_run["ttft"])
        shorts.append(short_run["ttft"])
        decodes.append(long_run["decodeSecondsPerToken"])
        if i == 0:
            text = long_run["text"]

    ttft = statistics.median(ttfts)
    short_ttft = statistics.median(shorts)
    delta_tokens = prompt_tokens - short_tokens
    delta_seconds = ttft - short_ttft
    if delta_tokens > 0 and delta_seconds > 1e-4:
        prefill_per_token = delta_seconds / delta_tokens
    else:
        prefill_per_token = ttft / max(prompt_tokens, 1)
        print("prefill slope unusable; fell back to the TTFT-derived rate", file=sys.stderr)

    decode_per_token = statistics.median(decodes)
    return {
        "decodeTokensPerSecond": 1.0 / decode_per_token,
        "decodeSecondsPerToken": decode_per_token,
        "prefillTokensPerSecond": 1.0 / prefill_per_token,
        "prefillSecondsPerToken": prefill_per_token,
        "ttftSeconds": ttft,
        "text": text,
    }


def perplexity(model, tok, corpus, window=512, chunks=8):
    """Cascade-immune accuracy metric, same role as llama-perplexity on the
    llama.cpp tracks: a numeric change that shifts the distribution shows up
    here even when greedy text happens to survive."""
    ids = tok.encode(corpus)
    total_nll, total_tokens = 0.0, 0
    for c in range(chunks):
        start = c * window
        chunk = ids[start:start + window + 1]
        if len(chunk) < 2:
            break
        inputs = mx.array(chunk[:-1])[None]
        targets = mx.array(chunk[1:])[None]
        logits = model(inputs).astype(mx.float32)
        nll = nn.losses.cross_entropy(logits, targets, reduction="sum")
        mx.eval(nll)
        total_nll += float(nll.item())
        total_tokens += targets.size
    if not total_tokens:
        raise SystemExit("corpus too short for a perplexity window")
    return {"perplexity": math.exp(total_nll / total_tokens), "tokens": total_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path or HF repo id")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--mode", choices=["bench", "ppl"], default="bench")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    model, tok = load(args.model)
    text = corpus_text(args.corpus)

    if args.mode == "ppl":
        print(json.dumps(perplexity(model, tok, text)))
        return

    warm = tok.encode(text)[:args.prompt_tokens]
    for _ in range(args.warmup):
        timed_generate(model, tok, warm, 8)

    result = bench(model, tok, text, args.prompt_tokens, args.decode_tokens, args.runs)
    result["peakMemoryGB"] = round(mx.get_peak_memory() / 1e9, 3)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
