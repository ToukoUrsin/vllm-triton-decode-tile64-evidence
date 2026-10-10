#!/usr/bin/env bash
set -euo pipefail
python3 compare_triton_attn_tile64.py --batch 1 --seq-len 1024 --q-heads 32 --kv-heads 8 --head-dim 128 --label Qwen3-8B-B1-S1024-TILE64 --out qwen3-8b.json
python3 compare_triton_attn_tile64.py --batch 1 --seq-len 1024 --q-heads 40 --kv-heads 8 --head-dim 128 --label Qwen3-14B-B1-S1024-TILE64 --out qwen3-14b.json
python3 compare_triton_attn_tile64.py --batch 1 --seq-len 1024 --q-heads 64 --kv-heads 8 --head-dim 128 --label Qwen3-32B-B1-S1024-TILE64 --out qwen3-32b.json
python3 compare_triton_attn_tile64.py --batch 1 --seq-len 1024 --q-heads 32 --kv-heads 4 --head-dim 128 --label Qwen3-30B-A3B-B1-S1024-TILE64 --out qwen3-30b-a3b.json
python3 compare_triton_attn_tile64.py --batch 1 --seq-len 1024 --q-heads 16 --kv-heads 8 --head-dim 128 --label Qwen3-1.7B-B1-S1024-TILE64 --out qwen3-1p7b.json
python3 compare_triton_attn_tile64.py --batch 1 --seq-len 1024 --q-heads 28 --kv-heads 4 --head-dim 128 --label Qwen2.5-7B-B1-S1024-TILE64 --out qwen2p5-7b.json
