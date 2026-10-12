"""Greedy-parity probe: 12 fixed prompts (128-1792 tokens), temperature 0, 128 output tokens.
Sent once sequentially (batch 1) and once all at once (continuous batching); saves token ids."""
import asyncio, json, sys, time
import aiohttp
from transformers import AutoTokenizer

model_path, out = sys.argv[1], sys.argv[2]
URL = "http://127.0.0.1:8000/v1/completions"
tok = AutoTokenizer.from_pretrained(model_path)
paras = [
    "Explain how attention uses cached keys and values to generate the next token. ",
    "A paged KV cache stores each sequence in fixed-size blocks indexed by a block table. ",
    "Continuous batching admits new requests between decode steps, so batch shape changes every iteration. ",
    "Summarize the trade-offs between latency and throughput when serving large language models. ",
]
lengths = [128, 256, 384, 512, 768, 1024, 1152, 1280, 1408, 1536, 1664, 1792]
prompts = []
for i, n in enumerate(lengths):
    base = tok.encode("".join(paras[(i + j) % 4] for j in range(4)), add_special_tokens=False)
    prompts.append((base * (n // len(base) + 1))[:n])


async def one(session, ids):
    body = {"model": "qwen3-8b-awq", "prompt": ids, "max_tokens": 128, "temperature": 0, "ignore_eos": True,
            "return_token_ids": True}
    async with session.post(URL, json=body) as r:
        j = await r.json()
    c = j["choices"][0]
    return {"text": c["text"], "token_ids": c.get("token_ids")}


async def main():
    res = {"prompt_lengths": lengths}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        t0 = time.time()
        res["sequential"] = [await one(s, p) for p in prompts]
        res["sequential_s"] = time.time() - t0
        t0 = time.time()
        res["concurrent"] = list(await asyncio.gather(*(one(s, p) for p in prompts)))
        res["concurrent_s"] = time.time() - t0
    json.dump(res, open(out, "w"))
    print("parity saved", out, "token_ids" if res["sequential"][0]["token_ids"] else "text-only")

asyncio.run(main())
