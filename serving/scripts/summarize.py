"""Summarize the paired serving blocks. Each block ran FLASH_ATTN, shipped TRITON_ATTN and guarded tile64 back to back
on one H100 (order alternates between blocks). Deltas are computed within a block, then averaged over blocks.
usage: python3 summarize.py OUT.json BLOCK_RESULTS_DIR...   (each dir holds <block>-<variant>/c*.json)"""
import glob, json, os, statistics, sys
out_path, dirs = sys.argv[1], sys.argv[2:]
V = ["flash", "shipped", "guarded"]
LEVELS = ["1", "4", "8", "16", "max"]
M = [("output_throughput", "out tok/s", True), ("median_ttft_ms", "TTFT p50", False), ("p99_ttft_ms", "TTFT p99", False),
     ("median_tpot_ms", "TPOT p50", False), ("p99_tpot_ms", "TPOT p99", False),
     ("median_itl_ms", "ITL p50", False), ("p99_itl_ms", "ITL p99", False)]

blocks = {}
for d in dirs:
    for vd in glob.glob(f"{d}/*-*/"):
        tag = os.path.basename(vd.rstrip("/")); b, v = tag.rsplit("-", 1)
        blocks.setdefault(b, {})[v] = vd
blocks = {b: x for b, x in sorted(blocks.items()) if "shipped" in x and "guarded" in x}  # flash may be missing (preempted)

def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

def gain(new, base, higher_better):  # + = new is better
    return (new / base - 1) * 100 if higher_better else (base / new - 1) * 100

res = {"blocks": list(blocks), "levels": {}}
for c in LEVELS:
    L = res["levels"][c] = {"per_block": {}, "mean": {}}
    for b, vds in blocks.items():
        runs = {v: load(f"{vds[v]}/c{c}.json") if v in vds else None for v in V}
        if not (runs["shipped"] and runs["guarded"]):
            continue
        row = L["per_block"][b] = {"num_prompts": runs["guarded"]["completed"],
                                   "failed": {v: runs[v].get("failed", 0) for v in V if runs[v]}}
        for k, _, hb in M:
            row[k] = {v: runs[v][k] for v in V if runs[v]}
            row[k]["guarded_vs_shipped_pct"] = gain(runs["guarded"][k], runs["shipped"][k], hb)
            if runs["flash"]:
                row[k]["guarded_vs_flash_pct"] = gain(runs["guarded"][k], runs["flash"][k], hb)
                row[k]["shipped_vs_flash_pct"] = gain(runs["shipped"][k], runs["flash"][k], hb)
    for k, _, _ in M:
        rows = [r[k] for r in L["per_block"].values()]
        if rows:
            keys = sorted({x for r in rows for x in r})
            L["mean"][k] = {x: statistics.mean(r[x] for r in rows if x in r) for x in keys}
            L["mean"][k]["n_blocks"] = {x: sum(x in r for r in rows) for x in keys}
            for x in ("guarded_vs_shipped_pct", "guarded_vs_flash_pct"):
                vals = [r[x] for r in rows if x in r]
                L["mean"][k]["range_" + x] = [min(vals), max(vals)]

# greedy parity (12 fixed prompts x 128 tokens)
def toks(vd, mode):
    p = load(f"{vd}/parity.json")
    return [o["token_ids"] for o in p[mode]] if p else None
par = res["parity"] = {}
for b, vds in blocks.items():
    for mode in ("sequential", "concurrent"):
        t = {v: toks(vds[v], mode) if v in vds else None for v in V}
        for a, z in (("shipped", "guarded"), ("flash", "guarded"), ("flash", "shipped")):
            if t[a] and t[z]:
                par[f"{b}:{mode}:{a}~{z}"] = sum(x == y for x, y in zip(t[a], t[z]))
bl = list(blocks)
for v in V:
    for mode in ("sequential", "concurrent"):
        ts = [toks(blocks[b][v], mode) for b in bl if v in blocks[b]]
        if len(ts) > 1:
            par[f"across_blocks:{mode}:{v}"] = min(sum(x == y for x, y in zip(ts[0], t)) for t in ts[1:])

# launch telemetry (captured CUDA graphs)
tel = res["telemetry"] = {}
for b, vds in blocks.items():
    for v in ("shipped", "guarded"):
        agg = {}
        for p in glob.glob(f"{vds[v]}/telemetry/*.json"):
            for k, n in json.load(open(p))["counts"].items():
                if k.startswith("capture|3d=1"):
                    agg[k] = agg.get(k, 0) + n
        tel[f"{b}-{v}"] = sorted(agg)
json.dump(res, open(out_path, "w"), indent=1)

print(f"blocks: {', '.join(blocks)} (n per comparison in summary json)")
print("| conc | metric | FLASH_ATTN | Triton shipped | Triton guarded | guarded vs shipped | guarded vs FLASH |")
print("|---|---|---|---|---|---|---|")
for c in LEVELS:
    for k, name, hb in M:
        m = res["levels"][c]["mean"].get(k)
        if not m:
            continue
        f = (lambda x: f"{x:.0f}") if hb else (lambda x: f"{x:.2f}")
        r1, r2 = m["range_guarded_vs_shipped_pct"], m["range_guarded_vs_flash_pct"]
        print(f"| {c} | {name} | {f(m['flash'])} | {f(m['shipped'])} | {f(m['guarded'])} | "
              f"{m['guarded_vs_shipped_pct']:+.1f}% [{r1[0]:+.1f},{r1[1]:+.1f}] | {m['guarded_vs_flash_pct']:+.1f}% [{r2[0]:+.1f},{r2[1]:+.1f}] |")
print("\nparity (identical of 12):", json.dumps(par))
print("telemetry:", json.dumps(tel, indent=0))
