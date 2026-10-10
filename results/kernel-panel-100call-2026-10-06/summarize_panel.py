"""Require a complete six-geometry panel and recompute the scores."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from verify_result import verify

SHAPES = [("qwen3-8b", 32, 8), ("qwen3-14b", 40, 8), ("qwen3-32b", 64, 8),
          ("qwen3-30b-a3b", 32, 4), ("qwen3-1p7b", 16, 8), ("qwen2p5-7b", 28, 4)]
ap = argparse.ArgumentParser()
ap.add_argument("--suffix", default="tuned64")
ap.add_argument("--out", type=Path, default=Path("panel-summary.json"))
a = ap.parse_args()
root = Path(__file__).resolve().parent
rows = []
for label, qheads, kvheads in SHAPES:
    row = verify(root / (label + "-" + a.suffix + ".json"))
    shape = row["shape"]
    assert shape["q_heads"] == qheads and shape["kv_heads"] == kvheads
    assert shape["batch"] == 1 and shape["seq_len"] == 1024 and shape["head_dim"] == 128
    times = row["medians_us"]
    rows.append({"label": label, "q_heads": qheads, "kv_heads": kvheads,
                 "graph_calls": row["graph_calls"], "file": Path(row["file"]).name,
                 "medians_us": times,
                 "gain_vs_flash_pct": 100 * (1 - times["tuned64"] / times["flash"]),
                 "gain_vs_triton16_pct": 100 * (1 - times["tuned64"] / times["triton16"]),
                 "gain_vs_tile64_pct": 100 * (1 - times["tuned64"] / times["triton64"]),
                 "max_abs_error": row["max_abs_error"], "negative_control_error": row["negative_control_error"]})
assert len({row["graph_calls"] for row in rows}) == 1
payload = {"rows": rows,
           "median_gain_vs_flash_pct": statistics.median(row["gain_vs_flash_pct"] for row in rows),
           "median_gain_vs_triton16_pct": statistics.median(row["gain_vs_triton16_pct"] for row in rows),
           "median_gain_vs_tile64_pct": statistics.median(row["gain_vs_tile64_pct"] for row in rows),
           "all_original_input_checks_pass": True, "all_negative_controls_rejected": True,
           "scope": "B1/context1024 six attention geometries, deterministic random tensors, H100 SM90, bf16",
           "charter_acceptance_met": False}
a.out.write_text(json.dumps(payload, indent=2)+"\n")
print(json.dumps(payload, indent=2))
