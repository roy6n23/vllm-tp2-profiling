"""Tokenizer and detokenizer CPU cost on the box (spec 5, row 5).

Prompts are built like `vllm bench serve --dataset-name random`: random token ids that are
not special ids, decoded to text, then re-encoded and adjusted until the text encodes to
exactly input_len - 1 tokens (the tokenizer adds BOS, so a request has input_len prompt
tokens). Per request it measures encode time, one full decode of output_len ids, and
incremental detokenization per token. The incremental path uses
tokenizers.decoders.DecodeStream primed with the prompt ids, as vLLM's
FastIncrementalDetokenizer does (vllm/v1/engine/detokenizer.py:184), and falls back to
decoding growing prefixes. `--synthetic` writes fixed plausible numbers for dry runs.

    python -m tpprof.tokbench --model-dir D --n 200 --input-len 1024 --output-len 256 --out F [--synthetic]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from tpprof import stats

KEYS = ("model_dir", "n", "input_len", "output_len", "prompt_tokens", "token_mismatch_requests", "synthetic",
        "detok_method", "encode_ms_p50", "encode_ms_p90", "decode_ms_p50", "decode_ms_p90",
        "detok_us_per_token_p50", "detok_us_per_token_p90", "tokenizer",
        "t_wall_start", "t_mono_start", "t_wall_end", "t_mono_end")
SPECIAL_TOKENS_ADDED = 1       # Llama 3.1 prepends BOS
MAX_RETRY = 10                 # as in vllm/benchmarks/datasets/datasets.py gen_prompt_decode_to_target_len
WARMUP = 5
SYNTHETIC = {"encode_ms_p50": 1.2, "encode_ms_p90": 1.5, "decode_ms_p50": 0.15, "decode_ms_p90": 0.2,
             "detok_us_per_token_p50": 4.0, "detok_us_per_token_p90": 5.0}


def _allowed_ids(tok) -> np.ndarray:
    special = set(tok.all_special_ids)
    return np.array([i for i in range(tok.vocab_size) if i not in special], dtype=np.int64)


def build_prompts(tok, n: int, input_len: int, rng: np.random.Generator) -> tuple[list[tuple[str, list[int]]], int]:
    """n (text, ids) prompts whose text re-encodes (without special tokens) to input_len - 1 ids.

    Returns the prompts and how many still missed the target after MAX_RETRY adjustments.
    """
    target = input_len - SPECIAL_TOKENS_ADDED
    allowed = _allowed_ids(tok)
    prompts, mismatched = [], 0
    for _ in range(n):
        ids = rng.choice(allowed, size=target).tolist()
        for attempt in range(MAX_RETRY + 1):
            text = tok.decode(ids)
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) == target or attempt == MAX_RETRY:
                break
            if len(ids) < target:
                ids = ids + rng.choice(allowed, size=target - len(ids)).tolist()
            else:
                ids = ids[:target]
        mismatched += len(ids) != target
        prompts.append((text, ids))
    return prompts, mismatched


def _decode_stream_cls():
    try:
        from tokenizers.decoders import DecodeStream
    except (ImportError, AttributeError):
        return None
    return DecodeStream


def _incremental(tok, decode_stream, prompt_ids: list[int], out_ids: list[int]) -> tuple[float, str]:
    """Seconds spent detokenizing out_ids one token at a time, and the method used."""
    if decode_stream is None:
        t0 = time.perf_counter()
        for i in range(1, len(out_ids) + 1):
            tok.decode(out_ids[:i], skip_special_tokens=True)
        return time.perf_counter() - t0, "prefix_decode"
    try:
        stream, method = decode_stream(ids=prompt_ids, skip_special_tokens=True), "DecodeStream(ids=prompt)"
    except TypeError:          # tokenizers < 0.22 has no prefill argument
        stream, method = decode_stream(skip_special_tokens=True), "DecodeStream"
    backend = getattr(tok, "backend_tokenizer", tok)
    t0 = time.perf_counter()
    for t in out_ids:
        stream.step(backend, t)
    return time.perf_counter() - t0, method


def measure(tok, n: int, input_len: int, output_len: int, seed: int = 0, decode_stream=None) -> dict:
    """p50/p90 of encode ms/request, full decode ms/request and incremental detokenization µs/token."""
    rng = np.random.default_rng(seed)
    prompts, _ = build_prompts(tok, WARMUP + n, input_len, rng)
    allowed = _allowed_ids(tok)
    enc, dec, inc, method = [], [], [], None
    for k, (text, _) in enumerate(prompts):
        t0 = time.perf_counter()
        prompt_ids = tok.encode(text)                 # with BOS, as the server tokenizes a prompt
        t1 = time.perf_counter()
        out_ids = rng.choice(allowed, size=output_len).tolist()
        t2 = time.perf_counter()
        tok.decode(out_ids, skip_special_tokens=True)
        t3 = time.perf_counter()
        inc_s, method = _incremental(tok, decode_stream, prompt_ids, out_ids)
        if k >= WARMUP:
            enc.append((t1 - t0) * 1e3)
            dec.append((t3 - t2) * 1e3)
            inc.append(inc_s / output_len * 1e6)
    target = input_len - SPECIAL_TOKENS_ADDED
    return {"prompt_tokens": target,
            "token_mismatch_requests": sum(len(ids) != target for _, ids in prompts[WARMUP:]),
            "detok_method": method,
            "encode_ms_p50": stats.percentile(enc, 50), "encode_ms_p90": stats.percentile(enc, 90),
            "decode_ms_p50": stats.percentile(dec, 50), "decode_ms_p90": stats.percentile(dec, 90),
            "detok_us_per_token_p50": stats.percentile(inc, 50), "detok_us_per_token_p90": stats.percentile(inc, 90)}


def _load_tokenizer(model_dir: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(f"tokbench needs transformers (it runs on the box, or use --synthetic): {e}") from e
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def _positive(minimum: int):
    def parse(s: str) -> int:
        v = int(s)
        if v < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}, got {v}")
        return v
    return parse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tpprof.tokbench", description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n", type=_positive(1), default=200)
    ap.add_argument("--input-len", type=_positive(SPECIAL_TOKENS_ADDED + 1), default=1024)
    ap.add_argument("--output-len", type=_positive(1), default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="write fixed plausible numbers; needs no transformers")
    args = ap.parse_args(argv)

    t_wall, t_mono = time.time(), time.monotonic()
    if args.synthetic:
        result = {"prompt_tokens": args.input_len - SPECIAL_TOKENS_ADDED, "token_mismatch_requests": 0,
                  "detok_method": "synthetic", **SYNTHETIC}
        info = None
    else:
        tok = _load_tokenizer(args.model_dir)
        result = measure(tok, args.n, args.input_len, args.output_len, decode_stream=_decode_stream_cls())
        import tokenizers
        import transformers
        info = {"class": type(tok).__name__, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__}
    out = {"model_dir": args.model_dir, "n": args.n, "input_len": args.input_len, "output_len": args.output_len,
           **result, "synthetic": args.synthetic, "tokenizer": info,
           "t_wall_start": t_wall, "t_mono_start": t_mono, "t_wall_end": time.time(), "t_mono_end": time.monotonic()}
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({k: out[k] for k in KEYS}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
