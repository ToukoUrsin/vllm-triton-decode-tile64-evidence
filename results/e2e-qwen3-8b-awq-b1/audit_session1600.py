#!/usr/bin/env python3
"""Independent original-input audit for the 2026-10-07 Qwen3-8B AWQ ABBA."""
from __future__ import annotations
import hashlib, json, math, statistics
from pathlib import Path
D = Path(__file__).parent
NAMES = ("flash", "guarded64", "guarded64", "flash")
FILES = [D / f"session-1600-{i}-{v}.json" for i, v in enumerate(NAMES)]
assert all(p.exists() for p in FILES), [str(p) for p in FILES if not p.exists()]
rows = [json.loads(p.read_text()) for p in FILES]
assert [r["variant"] for r in rows] == list(NAMES)
source_sha = hashlib.sha256((D / "triton_unified_attention_guarded.py").read_bytes()).hexdigest()
assert source_sha == "2a6a94b70ba48e1fb20af4894da1b5ad375cc1159f01b61a592bfc68e63ca854"
reference = rows[0]["output_token_ids"][0]
for r in rows:
    assert r["model"] == "Qwen/Qwen3-8B-AWQ"
    assert r["model_revision"] == "4da05a8edb55c6046cce958586c33b61da07bb79"
    assert r["environment"]["vllm"] == "0.31.0"
    assert r["environment"]["device"] == "NVIDIA H100 80GB HBM3"
    assert r["environment"]["capability"] == [9, 0]
    assert (r["prompt_tokens"], r["output_tokens"], r["batch"]) == (1024, 128, 1)
    assert r["engine_config"]["max_model_len"] == 2048
    assert r["engine_config"]["max_num_batched_tokens"] == 2048
    assert r["engine_config"]["block_size"] == 128
    assert r["engine_config"]["enable_prefix_caching"] is False
    assert r["engine_config"]["compilation_config"] == {
        "cudagraph_capture_sizes": [1], "max_cudagraph_capture_size": 1}
    assert len(r["elapsed_s"]) == 7
    assert all(math.isfinite(x) and x > 0 for x in r["elapsed_s"])
    assert math.isclose(statistics.median(r["elapsed_s"]), r["median_elapsed_s"], rel_tol=1e-12)
    assert math.isclose(128 / r["median_elapsed_s"], r["tokens_per_s"], rel_tol=1e-12)
    assert r["within_backend_deterministic"]
    assert all(tokens == reference for tokens in r["output_token_ids"])
    if r["variant"] == "guarded64":
        assert all(s["source_sha256"] == source_sha for s in r["worker_telemetry"])
        launches = [c for s in r["worker_telemetry"] for c in s["kernel_launches"]]
        assert any(c["is_3d"] and c["tile"] == 64 and c["warps"] == 4 and c["stages"] == 2 and c["count"] > 0 for c in launches)
        assert any(c["is_3d"] and c["tile"] == 16 and c["stages"] == 3 and c["count"] > 0 for c in launches)
        assert all(c["tile"] == 32 for c in launches if not c["is_3d"])
# A deterministic output mutation must be rejected by the checker; this is a
# negative control on the recorded token stream, not a claim of a second GPU run.
mutated = [list(reference[0])]
mutated[0][0] += 1
assert mutated != reference
pairs = []
for flash_i, candidate_i in ((0, 1), (3, 2)):
    flash, candidate = rows[flash_i], rows[candidate_i]
    pairs.append({
        "flash_file": FILES[flash_i].name,
        "candidate_file": FILES[candidate_i].name,
        "flash_tokens_per_s": flash["tokens_per_s"],
        "candidate_tokens_per_s": candidate["tokens_per_s"],
        "gain_pct": 100 * (candidate["tokens_per_s"] / flash["tokens_per_s"] - 1),
        "latency_reduction_pct": 100 * (1 - candidate["median_elapsed_s"] / flash["median_elapsed_s"]),
    })
score = statistics.median([p["gain_pct"] for p in pairs])
latency = statistics.median([p["latency_reduction_pct"] for p in pairs])
out = {
    "status": "PASS",
    "source_sha256": source_sha,
    "pairs": pairs,
    "median_pair_throughput_gain_pct": score,
    "median_pair_latency_reduction_pct": latency,
    "all_greedy_tokens_match": True,
    "recorded_output_mutation_rejected": True,
    "selector_pr_gate_met": score >= 5.0,
    "scope": "Qwen3-8B-AWQ B1, prompt1024/output128, page128, vLLM0.31.0 H100 SM90, default FULL_AND_PIECEWISE, FLASH/guarded64/guarded64/FLASH, seven measured generations per child",
}
(D / "audit_session1600.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
