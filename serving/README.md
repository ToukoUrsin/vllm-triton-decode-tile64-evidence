# Serving evaluation (continuous batching, mixed context lengths)

Answers the question on vllm-project/vllm#61029: how does the guarded tile64 decode launch behave under a realistic
serving workload (ITL and throughput), not just batch-1 offline generation?

`results/` is the clean run (2026-10-12 18:10-18:35 UTC): six complete blocks. `first-run-results/` is the earlier
run from the same day, which was cut up by spot preemptions (blocks of mixed size, one block without FLASH_ATTN). It
points the same way but is noisier; the numbers below come from `results/` only.

## Setup

- `vllm/vllm-openai:v0.31.0` (vLLM 0.31.0, torch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.7.0.post1), one H100 80GB HBM3
  per block (six Daytona sandboxes, drivers 580.105 / 580.159 / 580.178 / 595.71; see `results/runs/<block>/`)
- `vllm serve Qwen/Qwen3-8B-AWQ` (revision `4da05a8edb55c6046cce958586c33b61da07bb79`) `--dtype bfloat16 --max-model-len 2048`,
  everything else at the defaults (FULL_AND_PIECEWISE CUDA graphs, block size 16, prefix caching on, default capture sizes)
- Workload: `vllm bench serve --dataset-name random --random-input-len 900 --random-output-len 192 --random-range-ratio 0.85
  --ignore-eos --seed 0`. Inputs are 139-1661 tokens and outputs 29-355 tokens, so contexts in one batch range from about
  170 to 2000 tokens. Closed loop at `--max-concurrency` 1 / 4 / 8 / 16 (64 / 160 / 96 / 160 prompts), then all 320
  prompts at once
- Variants, one server lifetime each, run back to back on the same H100 (a "block"): `FLASH_ATTN`; shipped
  `TRITON_ATTN` (tile16); guarded tile64 `TRITON_ATTN` (`../kernel/triton_unified_attention_guarded_v0.31.0.py`, which
  also carries the page-table mask)
- Six blocks `p1`-`p6`, one per order of the three variants, so each variant runs in each position twice. Deltas are
  computed within a block and then averaged; brackets give the per-block min and max
- Every server start: 48-prompt warmup, then a greedy-parity probe (12 fixed prompts of 128-1792 tokens, 128 output
  tokens, temperature 0, sent sequentially and all at once; token ids saved)
- Launch telemetry (`scripts/instrument.py`) records tile/warps/stages of every 3D decode launch captured into a CUDA
  graph; the guarded server refuses to start the benchmark unless its batch-1 graph uses tile64

## Results (mean of 6 blocks; + = guarded better)

| conc | metric | FLASH_ATTN | Triton shipped | Triton guarded | guarded vs shipped | guarded vs FLASH |
|---|---|---|---|---|---|---|
| 1 | out tok/s | 201 | 205 | 211 | **+3.0%** [+1.2, +4.7] | **+4.9%** [+2.8, +6.6] |
| 1 | ITL p50 / p99 ms | 4.81 / 6.70 | 4.71 / 6.53 | 4.57 / 6.50 | +3.2% / +0.5% | +5.3% / +3.2% |
| 1 | TTFT p50 ms | 38.1 | 40.6 | 40.8 | -0.7% | -6.7% |
| 4 | out tok/s | 719 | 710 | 730 | **+2.8%** [+1.6, +4.7] | **+1.6%** [+0.1, +3.7] |
| 4 | ITL p50 / p99 ms | 5.01 / 15.3 | 4.99 / 17.5 | 4.84 / 17.8 | +3.0% / -1.3% | +3.4% / -12.1% |
| 4 | TTFT p50 ms | 42.9 | 48.4 | 46.9 | +2.9% | -8.7% |
| 8 | out tok/s | 1225 | 1212 | 1220 | +0.6% [-0.3, +3.1] | -0.4% [-2.0, +1.5] |
| 8 | ITL p50 / p99 ms | 5.37 / 24.9 | 5.28 / 27.7 | 5.26 / 27.0 | +0.4% / +3.1% | +2.2% / -7.6% |
| 16 | out tok/s | 2512 | 2453 | 2466 | +0.5% [-0.4, +2.7] | -1.9% [-2.8, -0.7] |
| 16 | ITL p50 / p99 ms | 5.76 / 9.3 | 5.80 / 11.0 | 5.77 / 11.3 | +0.5% / -2.7% | -0.1% / -17.1% |
| max | out tok/s | 4447 | 4097 | 4094 | -0.1% [-0.4, +0.5] | -7.9% [-8.2, -7.5] |
| max | ITL p50 / p99 ms | 21.9 / 426 | 22.9 / 470 | 22.8 / 475 | +0.4% / -1.1% | -4.0% / -10.2% |

Full table (TTFT/TPOT/ITL p50/p99 for every level): `results/summary.md`; per-block numbers: `results/summary.json`.

At 1 and 4 concurrent requests the guarded launch has higher throughput than shipped Triton in all 6 blocks, and
higher than FLASH_ATTN in all 6 blocks. At 8+ it is not active (see below) and the difference is noise around zero; one block
(`p6`) has +3% at c=8/16 where both Triton servers run identical kernels, which shows the size of the per-server noise.

## What the guard actually does under serving

- Under FULL CUDA graphs the decode launch is fixed at capture time, and vLLM captures with
  `max_seq_len = max_model_len` (`gpu_model_runner.py`). So the guard's `1024 <= max_seqlen_k <= 2048` condition is
  really a `max_model_len` condition: with `--max-model-len 2048` every captured decode graph of 1, 2 or 4 sequences uses
  tile64 (telemetry: `capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2`, `seqs=4 ... stages=3`) regardless of the
  real context of the requests; graphs of 8 and 16 keep tile16. With `max_model_len` > 2048 it never fires.
- Plain `vllm serve Qwen/Qwen3-8B-AWQ` runs AWQ in fp16, so the bf16-only guard does nothing unless `--dtype bfloat16`
  is passed (`first-run-results/fp16-default-guard-inert/`: telemetry shows tile16 for every graph).
- Mixed prefill+decode steps run attention eagerly through the 2D kernel; the guard does not touch them.

## Greedy parity

Guarded vs shipped Triton on the same machine: identical token ids on 141 of 144 prompt runs (12 prompts x 2 probes x
6 blocks). The same backend run on two different machines differs at least as often (FLASH_ATTN concurrent probe: 8-12
of 12 identical across machine pairs, mean 9.5). FLASH_ATTN vs Triton on the same machine: 8-12 of 12.

## Conclusion

The guarded tile64 launch gives a consistent ~3% throughput and ITL gain over shipped Triton at 1-4 concurrent
requests (6/6 blocks), and puts TRITON_ATTN ahead of FLASH_ATTN there; it is neutral at 8+ and never regresses. The
gain is real but small, and the current guard keys on the wrong quantity: under CUDA graphs `max_seqlen_k` is
`max_model_len`, not the request length. A version worth upstreaming would key on what is known at capture time
(SM90, bf16, head dim 128, <= 4 decode sequences, measured head geometries) and would need a measurement with
`max_model_len` above 2048 before dropping the context bound. Otherwise it should stay out of tree.

## Reproduce

Inside the image, from `scripts/`: `bash block.sh p1 "flash shipped guarded"` (runs `setup.sh`, then `run_block.sh`;
resumable: finished variants are skipped). Summary: `python3 summarize.py summary.json ../results`.

Prepared by Touko Ursin with AI assistance.
