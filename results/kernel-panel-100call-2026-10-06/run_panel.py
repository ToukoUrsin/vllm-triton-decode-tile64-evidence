"""Bounded six-geometry original-input panel for the selected stage2 launch."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

GEOMETRIES = [
    ("qwen3-8b", 32, 8), ("qwen3-14b", 40, 8), ("qwen3-32b", 64, 8),
    ("qwen3-30b-a3b", 32, 4), ("qwen3-1p7b", 16, 8), ("qwen2p5-7b", 28, 4),
]
root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--calls", type=int, default=100)
ap.add_argument("--suffix", default="tuned64")
a = ap.parse_args()
for name, q_heads, kv_heads in GEOMETRIES:
    subprocess.run([
        sys.executable, "-u", str(root / "segment_launch_control.py"),
        "--batch", "1", "--seq-len", "1024", "--q-heads", str(q_heads),
        "--kv-heads", str(kv_heads), "--head-dim", "128", "--label", name,
        "--calls", str(a.calls),
        "--variants", "flash:0:0:0:0,triton16:0:0:0:0,triton64:64:0:0:0,tuned64:64:0:4:2",
        "--out", str(root / (name + "-" + a.suffix + ".json")),
    ], check=True, timeout=300)
