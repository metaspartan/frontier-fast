"""MLX measurement harness for gainz.fast Apple Silicon tracks.

Mirrors what the llama.cpp and vLLM runners measure so scores mean the same
thing across engines: a 512-token prefill, 128 greedy decode steps, TTFT, and
perplexity over the fixed corpus as the accuracy gate.

Decode and prefill rates come from `mlx_lm.stream_generate` — the real
generation path — not from a hand-rolled loop.

That distinction is the whole point. An earlier version of this file drove the
model directly:

    logits = model(token[None], cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token); out.append(token.item())

which is greedy and correct in its output, but measures the wrong machine.
`generate_step` pipelines with `mx.async_eval` and evaluates one step ahead,
so the GPU is never idle waiting for the CPU; forcing `mx.eval` + `.item()`
every step serialises that and makes CPU synchronisation, not the kernels,
a large part of per-token time. It also skips the periodic `mx.clear_cache()`
and the `prefill_step_size` chunking that real generation performs. The result
is a number that is reproducible but not the number anyone cares about: an
optimization that helps the pipelined path can be invisible here, and a
"win" measured here need not be a win in real generation.

`GenerationResponse` reports `prompt_tps` (computed by the engine as
prompt.size / prompt_time) and `generation_tps` directly, so prefill is a
genuine independent measurement rather than something derived from TTFT —
deriving it from TTFT would make prefillSpeedup identical to ttftSpeedup and
spend 0.35 of the score on one number.

Two modes:
  bench  -> {decodeTokensPerSecond, prefillTokensPerSecond, ttftSeconds, text}
  ppl    -> {perplexity}
"""
import argparse
import json
import math
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate


def corpus_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def one_run(model, tok, prompt_ids, decode_tokens):
    """One cache-cold generation through the real path.

    The default sampler is argmax, so this is greedy — the same decoding the
    correctness gate assumes — without bypassing the generation machinery.
    """
    started = time.perf_counter()
    ttft = None
    last = None
    pieces = []
    for response in stream_generate(model, tok, prompt_ids, max_tokens=decode_tokens):
        if ttft is None:
            ttft = time.perf_counter() - started
        pieces.append(response.text)
        last = response
    if last is None:
        raise SystemExit("stream_generate yielded nothing")
    return {
        "ttft": ttft,
        "decodeTokensPerSecond": last.generation_tps,
        "prefillTokensPerSecond": last.prompt_tps,
        # GenerationResponse.peak_memory is ALREADY in GB. Dividing by 1e9
        # again floored every reading to 0.0 — a field that is always zero is
        # worse than no field, because it reads as "this model uses no memory".
        "peakMemoryGB": last.peak_memory if last.peak_memory else mx.get_peak_memory() / 1e9,
        "text": "".join(pieces),
    }


def bench(model, tok, corpus, prompt_tokens, decode_tokens, runs):
    full = tok.encode(corpus)
    if len(full) < prompt_tokens * 2:
        full = full * (prompt_tokens * 2 // max(len(full), 1) + 2)

    ttfts, decodes, prefills, peaks, text = [], [], [], [], ""
    for i in range(runs):
        # Rotate the slice per run so no two runs are the identical prompt and
        # nothing can be served from a warm prefix.
        offset = (i * 97) % max(len(full) - prompt_tokens - 1, 1)
        prompt_ids = full[offset:offset + prompt_tokens]
        result = one_run(model, tok, prompt_ids, decode_tokens)
        ttfts.append(result["ttft"])
        decodes.append(result["decodeTokensPerSecond"])
        prefills.append(result["prefillTokensPerSecond"])
        peaks.append(result["peakMemoryGB"])
        if i == 0:
            text = result["text"]

    decode_tps = statistics.median(decodes)
    prefill_tps = statistics.median(prefills)
    return {
        "decodeTokensPerSecond": decode_tps,
        "decodeSecondsPerToken": 1.0 / decode_tps,
        "prefillTokensPerSecond": prefill_tps,
        "prefillSecondsPerToken": 1.0 / prefill_tps,
        "ttftSeconds": statistics.median(ttfts),
        "peakMemoryGB": round(max(peaks), 3),
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
        one_run(model, tok, warm, 8)

    print(json.dumps(bench(model, tok, text, args.prompt_tokens, args.decode_tokens, args.runs)))


if __name__ == "__main__":
    main()
