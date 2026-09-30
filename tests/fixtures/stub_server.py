"""p2spike stub: minimal OpenAI-compatible server for driving the real `vllm bench serve` client.

Endpoints: GET /health, GET /v1/models, GET /metrics, POST /tokenize, POST /detokenize,
POST /v1/completions (SSE stream). Unrouted requests are logged too (then 404).
Every request (method, path, headers, JSON body) is appended to $STUB_LOG as one JSON line.
If $STUB_TOKENIZER is set, prompt_tokens in the usage chunk is counted with that HF tokenizer
(add_special_tokens=True, i.e. with BOS, like the real server); otherwise it is omitted.
"""

import asyncio
import json
import os
import time
import uuid

from aiohttp import web

MODEL = os.environ.get("STUB_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LOG = os.environ.get("STUB_LOG", "/out/client_requests.jsonl")
DELAY = float(os.environ.get("STUB_TOKEN_DELAY_S", "0.005"))
TOK = None
if os.environ.get("STUB_TOKENIZER"):
    from transformers import AutoTokenizer

    TOK = AutoTokenizer.from_pretrained(os.environ["STUB_TOKENIZER"])


def log(request, body):
    rec = {"t": time.time(), "method": request.method, "path": request.path_qs,
           "headers": dict(request.headers), "body": body}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


async def health(request):
    log(request, None)
    return web.Response(status=200)


async def models(request):
    log(request, None)
    return web.json_response({"object": "list", "data": [
        {"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "vllm",
         "root": MODEL, "parent": None, "max_model_len": 8192}]})


async def metrics(request):
    log(request, None)
    return web.Response(text="# HELP vllm:num_requests_running stub\n"
                             "# TYPE vllm:num_requests_running gauge\n"
                             f'vllm:num_requests_running{{model_name="{MODEL}"}} 0.0\n',
                        content_type="text/plain")


async def tokenize(request):
    body = await request.json()
    log(request, body)
    if TOK is None:
        return web.json_response({"error": "no tokenizer"}, status=404)
    ids = TOK(body["prompt"], add_special_tokens=body.get("add_special_tokens", True))["input_ids"]
    return web.json_response({"count": len(ids), "max_model_len": 8192, "tokens": ids,
                              "token_strs": None})


async def detokenize(request):
    body = await request.json()
    log(request, body)
    return web.json_response({"prompt": TOK.decode(body["tokens"])})


@web.middleware
async def log_unrouted(request, handler):
    try:
        return await handler(request)
    except web.HTTPNotFound:
        log(request, {"_unrouted": True})
        raise


async def completions(request):
    body = await request.json()
    log(request, body)
    n = int(body.get("max_tokens") or 16)
    rid, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)

    def chunk(choices, usage=None):
        return ("data: " + json.dumps({"id": rid, "object": "text_completion", "created": created,
                                       "model": body.get("model"), "choices": choices,
                                       "usage": usage}) + "\n\n").encode()

    for i in range(n):
        await asyncio.sleep(DELAY)
        fin = "length" if i == n - 1 else None
        await resp.write(chunk([{"index": 0, "text": " tok", "logprobs": None,
                                 "finish_reason": fin, "stop_reason": None}]))
    usage = {"completion_tokens": n}
    if TOK is not None:
        usage["prompt_tokens"] = len(TOK(body["prompt"])["input_ids"])
        usage["total_tokens"] = usage["prompt_tokens"] + n
    if (body.get("stream_options") or {}).get("include_usage"):
        await resp.write(chunk([], usage))
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


app = web.Application(middlewares=[log_unrouted])
app.router.add_get("/health", health)
app.router.add_get("/v1/models", models)
app.router.add_get("/metrics", metrics)
app.router.add_post("/v1/completions", completions)
app.router.add_post("/tokenize", tokenize)
app.router.add_post("/detokenize", detokenize)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.environ.get("STUB_PORT", "8000")))
