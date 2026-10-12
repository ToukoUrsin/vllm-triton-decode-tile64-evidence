# Serving evaluation (continuous batching, mixed context lengths)

Answers the question on vllm-project/vllm#61029: how does the guarded tile64 decode launch behave under a realistic
serving workload (ITL and throughput), not just batch-1 offline generation?

## Setup

- `vllm/vllm-openai:v0.31.0` (vLLM 0.31.0, torch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.7.0.post1), one H100 80GB HBM3
  per block (Daytona spot sandboxes; drivers 580.126 / 595.71 / 595.91, see each block's `environment.txt`)
- `vllm serve Qwen/Qwen3-8B-AWQ` (revision `4da05a8edb55c6046cce958586c33b61da07bb79`) `--dtype bfloat16 --max-model-len 2048`,
  everything else default (FULL_AND_PIECEWISE CUDA graphs, block size 16, prefix caching on, default capture sizes)
- Workload: `vllm bench serve --dataset-name random --random-input-len 900 --random-output-len 192 --random-range-ratio 0.85
  --ignore-eos --seed 0`: input 139-1661 tokens, output 29-355 tokens, so contexts in one batch range ~170-2000 tokens.
  Closed loop at `--max-concurrency` 1 / 4 / 8 / 16 (64 / 160 / 96 / 160 prompts) and unbounded (320 prompts at once)
- Variants, one server lifetime each, run back to back on the same machine (a "block"):
  `FLASH_ATTN`; shipped `TRITON_ATTN` (tile16); guarded tile64 `TRITON_ATTN`
  (`../kernel/triton_unified_attention_guarded_v0.31.0.py`, which also carries the page-table mask)
- Blocks: `b1` (guarded, shipped, flash), `b2` (flash, shipped, guarded), `b3` (guarded, shipped; flash lost to spot
  preemption), `a1` (flash, shipped, guarded; shorter c=1/c=4 runs of 24/64 prompts). All deltas are computed within a
  block and then averaged; brackets give the per-block range
- Every server start: 48-prompt warmup, then a greedy-parity probe (12 fixed prompts of 128-1792 tokens, 128 output
  tokens, temperature 0, sent sequentially and all at once; token ids saved)
- Launch telemetry (`scripts/instrument.py`) records the tile/warps/stages of every 3D decode launch captured into a
  CUDA graph

## Results (mean over blocks; + = guarded better)

| conc | metric | FLASH_ATTN | Triton shipped | Triton guarded | guarded vs shipped | guarded vs FLASH |
|---|---|---|---|---|---|---|
| 1 | out tok/s | 205 | 205 | 209 | +1.9% [-1.2, +5.0] | +2.4% [+0.5, +4.0] |
| 1 | ITL p50 / p99 ms | 4.71 / 5.03 | 4.70 / 5.03 | 4.60 / 4.90 | +2.2% / +2.8% | +2.8% / +2.7% |
| 1 | TTFT p50 ms | 34.0 | 37.7 | 38.4 | -1.6% | -11.8% |
| 4 | out tok/s | 727 | 710 | 721 | +1.6% [-0.9, +3.9] | -0.6% [-1.4, +0.2] |
| 4 | ITL p50 / p99 ms | 4.94 / 14.4 | 5.01 / 15.9 | 4.91 / 15.8 | +1.9% / +0.6% | +1.0% / -7.7% |
| 4 | TTFT p50 ms | 40.9 | 45.1 | 44.7 | +0.9% | -9.4% |
| 8 | out tok/s | 1240 | 1207 | 1214 | +0.6% [-0.4, +2.3] | -2.4% [-3.1, -1.8] |
| 8 | ITL p50 / p99 ms | 5.32 / 25.3 | 5.33 / 27.0 | 5.30 / 26.6 | +0.4% / +1.7% | +0.1% / -6.4% |
| 8 | TTFT p50 ms | 43.1 | 49.2 | 48.1 | +2.3% | -11.3% |
| 16 | out tok/s | 2362 | 2319 | 2332 | +0.6% [-0.6, +2.5] | -3.6% [-5.6, -2.1] |
| 16 | ITL p50 / p99 ms | 5.76 / 16.6 | 5.87 / 16.7 | 5.84 / 16.9 | +0.4% / -0.7% | -1.8% / -19.0% |
| 16 | TTFT p50 ms | 29.1 | 31.2 | 30.5 | +2.9% | -10.9% |
| max | out tok/s | 4477 | 4114 | 4121 | +0.2% [-0.0, +0.4] | -8.0% [-8.3, -7.9] |
| max | ITL p50 / p99 ms | 21.8 / 430 | 22.6 / 472 | 22.6 / 475 | +0.3% / -0.7% | -3.5% / -9.8% |
| max | TTFT p50 ms | 4204 | 4809 | 4670 | +2.9% | -9.8% |

Full table (TTFT/TPOT/ITL p50/p99 for every level): `results/summary.md`; per-block numbers: `results/summary.json`.
FLASH_ATTN columns average the blocks that have it (a1, b1, b2).

Run-to-run drift between machines/server lifetimes is about +-2.5% (for example b2's shipped run was slow at every
concurrency, b3's was fast). Using c=16 from the same server lifetime as a within-block control (the guard is off there,
see below), the guarded gain over shipped is **+1.4% throughput / +1.9% ITL p50 at c=1** (per block -0.6 to +2.5%) and
**+1.0% / +1.5% at c=4** (-0.4 to +1.7%).

## What the guard actually does under serving

- Under FULL CUDA graphs the decode launch is fixed at capture time, and vLLM captures with
  `max_seq_len = max_model_len` (`gpu_model_runner.py`). So the guard's `1024 <= max_seqlen_k <= 2048` condition is
  really a `max_model_len` condition: with `--max-model-len 2048` every captured decode graph of 1, 2 or 4 sequences uses
  tile64 (telemetry: `capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2`, `seqs=4 ... stages=3`) regardless of the
  real context of the requests; graphs of 8 and 16 keep tile16. With `max_model_len` > 2048 it never fires.
- Plain `vllm serve Qwen/Qwen3-8B-AWQ` runs AWQ in fp16, so the bf16-only guard is inert unless `--dtype bfloat16`
  is passed (`results/fp16-default-guard-inert/`, telemetry shows tile16 for every graph).
- Mixed prefill+decode steps run attention eagerly through the 2D kernel; the guard does not touch them.

## Greedy parity

Guarded vs shipped Triton: identical token ids on 11/12, 12/12, 12/12 prompts (b1, b2, b3) in both the sequential and the
concurrent probe. The single divergence (b1, token 44-48) is the same kind of divergence seen between two runs of the
*same* backend on different machines (FLASH_ATTN b1 vs b2: 11/12 sequential, 9/12 concurrent). FLASH_ATTN vs Triton:
9-10/12.

## Conclusion

The guarded tile64 launch does not regress anywhere in this workload, but its serving gain is small: about +1-2% ITL
and throughput at 1-4 concurrent requests, nothing at 8+ (where it is not active), and of the same size as the
run-to-run drift. Above 4 concurrent requests the Triton backend trails FLASH_ATTN on this model (-2 to -8%
throughput, ~10% higher TTFT), independent of this change. Because the context-length bound only sees `max_model_len`
under CUDA graphs, a hardcoded shape guard keyed on `max_seqlen_k` is the wrong design; this does not justify a
hardcoded guard or a tuning table.

## Reproduce

`scripts/setup.sh` (model download, instrumented kernel copies), then `scripts/run_block.sh b1 "guarded shipped flash"`
inside the image; `python3 scripts/summarize.py summary.json results/a1 results/b1 results/b2 results/b3`.
`a1` used `scripts/run_all.sh` with c=1/c=4 at 24/64 prompts. Infrastructure notes: one Verda H100 was rejected for
uncorrectable HBM ECC errors (no data used); four Daytona spot sandboxes were preempted (only completed server lifetimes
are included).

Prepared by Touko Ursin with AI assistance.
