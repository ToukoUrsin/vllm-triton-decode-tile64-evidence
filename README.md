# vLLM Triton decode attention on H100: tile64 evidence and a page-table bounds fix

Evidence packet for two vLLM findings in `vllm/v1/attention/ops/triton_unified_attention.py` (TRITON_ATTN backend):

1. **Performance (kernel-level, end-to-end small).** For short-context bf16 GQA decode on H100, launching the
   Triton unified-attention decode kernel with 64-token KV tiles instead of the shipped 16 is faster than
   FLASH_ATTN at kernel level, but the end-to-end gain in a real serving run is only **+2.61%** throughput.
2. **Bug fix.** The compact page-table load in the kernel reads past the end of the block table when the KV tile is
   wider than a page (for example the 32-token prefill tile with 16-token pages). Compute Sanitizer reports 30
   invalid global reads on current vLLM main; a one-hunk mask removes them.

All numbers below come from the raw JSON in `results/`. Nothing here is externally reproduced or merged.

## Hardware and versions

- NVIDIA H100 80GB HBM3 (SM90), Daytona cloud sandbox, driver 595.91.07
- Docker image `vllm/vllm-openai:v0.31.0` (latest vLLM release as of 2026-10-10): vLLM 0.31.0, torch 2.13.0+cu130,
  CUDA 13.0, Triton 3.7.1, FlashInfer 0.7.0.post1
- Kernel inputs: bf16, head dim 128, deterministic random tensors (seed 20261006), public Qwen attention geometries
  (Q/KV heads 32/8, 40/8, 64/8, 32/4, 16/8, 28/4), batch 1, context 1024, 128-token pages
- Numerical gate on every timed variant: finite output, max abs error <= 0.03 vs an independent fp32 dense GQA
  reference (observed 0.001), and a +1 output corruption must be rejected

## 1. Kernel-level results (CUDA-graph replay, 40 CUDA-event samples, medians)

| Measurement | Date | tile64 vs FLASH_ATTN | tile64 vs shipped Triton tile16 |
|---|---|---|---|
| Single call per replay, 6 shapes, median (per-shape range) | 2026-10-06 | **22.08%** (11.78 to 26.99%) | n/a |
| Single call per replay, 6 shapes, fresh allocation | 2026-10-10 | **9.92%** (8.50 to 13.73%) | n/a |
| 100 calls per replay, Qwen3-8B shape (32/8), same input | 2026-10-06 | 32.47% (29.48% reversed order) | 19.90% (20.41% reversed) |
| 100 calls per replay, Qwen3-8B shape, fresh allocation | 2026-10-10 | 33.42% | 20.62% |
| 100 calls per replay, 6 shapes, guarded tile64 + 4 warps + 2 stages | 2026-10-06 | 37.22% | 25.32% |

Single-call replays are about 11-16 us and dominated by launch overhead; they moved between allocations
(22.08% then 9.92%). The 100-call replays amortize launch cost and were stable across allocations. Report both.

`tile64` in the panel is vLLM's own kernel with `_get_tile_size` returning 64 for bf16 non-sliding-window decode
(stock warps/stages); see `bench/kernel-panel/compare_triton_attn_tile64.py`. The guarded variant used end to end
is `kernel/triton_unified_attention_guarded_v0.31.0.py` (diff: `kernel/guarded-tile64-and-pagemask-vs-v0.31.0.patch`):
tile64, 4 warps, 2 stages, only for SM90 bf16 D128 3D causal decode with 1-4 sequences, context 1024-2048,
pages 16/128, the six measured head tuples, and no optional features; everything else keeps the shipped launch.

## 2. End-to-end result (the honest number)

`Qwen/Qwen3-8B-AWQ` (revision `4da05a8edb55c6046cce958586c33b61da07bb79`), batch 1, prompt 1024 / output 128,
block size 128, default compilation (FULL_AND_PIECEWISE CUDA graphs), prefix caching off, 7 measured generations
per child, order FLASH_ATTN / guarded tile64 / guarded tile64 / FLASH_ATTN on one H100:

- Paired throughput gains **+2.70%** and **+2.52%**, median **+2.61%** (203.2 to 208.6 tok/s in the first pair)
- Exact greedy-token parity across all children; worker telemetry confirms tile64 / 4 warps / 2 stages in decode
- Audit: `results/e2e-qwen3-8b-awq-b1/audit_session1600.py` (recomputes medians, checks source hash, tokens,
  telemetry, and rejects a mutated token)

Other measured serving scopes, all below 5%: Qwen3-8B BF16 batch 2 / output 512 **+1.46%**; widening the guard to
batch 8 (AWQ) **-1.34%**. Attention decode is a small share of a batch-1 8B decode step, so a ~10-35% kernel gain
becomes a low single-digit serving gain.

## 3. Page-table bounds fix

The tile loop loads `block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE` for every lane before any
mask. When `TILE_SIZE > BLOCK_SIZE`, lanes in the last tile can index past `cdiv(seq_len, BLOCK_SIZE)` and read
beyond the row. Patch (`kernel/page-table-mask-vs-main-ba77c4c130.patch`):

```python
physical_block_idx = tl.load(
    block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE,
    mask=(seq_offset // BLOCK_SIZE) < cdiv_fn(seq_len, BLOCK_SIZE),
    other=0,
).to(tl.int64)
```

Compute Sanitizer 2025.3 memcheck, B1, context 1025, 16-token pages, compact block table, caching allocator off:

| Source | Path | Invalid global reads |
|---|---|---|
| vLLM main `ba77c4c130` (2026-10-08), unpatched | 2D (prefill tile 32) | **30** (all at the page-table load, line 430) |
| same, patched | 2D, B1 and B4 | 0 |
| same, unpatched / patched | 3D decode (tile 16) | 0 / 0 |
| vLLM 0.31.0, unpatched | 2D | 30 |

Masked lanes are already excluded from the KV loads by `tile_mask`, so outputs do not change. The masked load is
also in the guarded source used for the Qwen3-8B end-to-end runs above (exact token parity). Overhead in an incomplete-page boundary
panel (six shapes, context 1025, permuted pages, 100-call graphs): 0.50% median.

## Reproduce (one H100, `vllm/vllm-openai:v0.31.0`)

```sh
# kernel panel (single call) and 100-call control
cd bench/kernel-panel && bash run_panel.sh && python3 three_way.py --out three-way.json
# sanitizer (downloads NVIDIA's standalone compute-sanitizer .deb; put the vLLM file to test next to the probe
# as main_triton_unified_attention.py and the patched one as main_pagemasked_triton_unified_attention.py)
cd bench/sanitizer && python3 run_memcheck_main.py
# end to end (needs the model download)
cd bench/e2e && VLLM_ENABLE_V1_MULTIPROCESSING=0 python3 -u engine_compare.py --model Qwen/Qwen3-8B-AWQ \
  --revision 4da05a8edb55c6046cce958586c33b61da07bb79 --default-compile --warmup 2 --repeats 7 --rounds 2 \
  --prompt-tokens 1024 --output-tokens 128 --batch 1 --block-size 128
```

## Layout

- `kernel/` guarded source and patches
- `bench/` kernel panel, end-to-end, and sanitizer scripts
- `serving/` continuous-batching serving evaluation (ITL, TTFT, throughput at concurrency 1-16 and max; six paired H100 blocks, 2026-10-12)
- `results/` raw JSON and logs: `kernel-panel-2026-10-06`, `kernel-panel-100call-2026-10-06`, `rerun-2026-10-10`,
  `e2e-qwen3-8b-awq-b1`, `e2e-other-scopes`, `sanitizer-v0.31.0`

## Limitations

- Kernel panels use equal-length sequential pages (ragged/permuted pages measured only in the boundary panel).
- Maintainer test `tests/kernels/attention/test_triton_unified_attention.py::test_triton_unified_attn` from vLLM main
  `ba77c4c130`, run against the patched main kernel on H100: **768 passed** (`results/rerun-2026-10-10/pytest-patched.log`).
  The rest of the vLLM test suite was not run.
- End-to-end covers Qwen3-8B on one H100 only; no multi-GPU or high-batch serving claim.

Code derived from vLLM is Apache-2.0 (see `LICENSE`). Prepared by Touko Ursin with AI assistance.
