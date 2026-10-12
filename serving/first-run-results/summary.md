blocks: a1, b1, b2, b3 (n per comparison in summary json)
| conc | metric | FLASH_ATTN | Triton shipped | Triton guarded | guarded vs shipped | guarded vs FLASH |
|---|---|---|---|---|---|---|
| 1 | out tok/s | 205 | 205 | 209 | +1.9% [-1.2,+5.0] | +2.4% [+0.5,+4.0] |
| 1 | TTFT p50 | 33.96 | 37.72 | 38.37 | -1.6% [-4.3,+4.2] | -11.8% [-13.9,-9.9] |
| 1 | TTFT p99 | 51.60 | 60.95 | 61.70 | -1.2% [-3.3,+2.8] | -16.3% [-16.5,-16.0] |
| 1 | TPOT p50 | 4.73 | 4.71 | 4.61 | +2.1% [-1.1,+5.2] | +2.8% [+1.2,+4.1] |
| 1 | TPOT p99 | 4.86 | 4.85 | 4.70 | +3.3% [-0.3,+6.3] | +3.9% [+1.2,+7.0] |
| 1 | ITL p50 | 4.71 | 4.70 | 4.60 | +2.2% [-1.0,+5.3] | +2.8% [+0.8,+4.3] |
| 1 | ITL p99 | 5.03 | 5.03 | 4.90 | +2.8% [-1.4,+5.6] | +2.7% [+0.9,+3.9] |
| 4 | out tok/s | 727 | 710 | 721 | +1.6% [-0.9,+3.9] | -0.6% [-1.4,+0.2] |
| 4 | TTFT p50 | 40.90 | 45.05 | 44.65 | +0.9% [+0.2,+1.8] | -9.4% [-10.0,-8.6] |
| 4 | TTFT p99 | 116.52 | 128.38 | 128.22 | +0.1% [-1.9,+3.5] | -9.5% [-10.6,-7.8] |
| 4 | TPOT p50 | 5.22 | 5.34 | 5.25 | +1.6% [-1.2,+4.1] | -0.3% [-1.4,+0.5] |
| 4 | TPOT p99 | 5.76 | 6.03 | 5.93 | +1.8% [-2.4,+5.2] | -2.1% [-3.2,-1.4] |
| 4 | ITL p50 | 4.94 | 5.01 | 4.91 | +1.9% [-0.9,+4.4] | +1.0% [-0.1,+1.9] |
| 4 | ITL p99 | 14.39 | 15.88 | 15.79 | +0.6% [-1.7,+1.9] | -7.7% [-12.6,-4.1] |
| 8 | out tok/s | 1240 | 1207 | 1214 | +0.6% [-0.4,+2.3] | -2.4% [-3.1,-1.8] |
| 8 | TTFT p50 | 43.06 | 49.22 | 48.13 | +2.3% [+0.1,+4.8] | -11.3% [-13.1,-9.7] |
| 8 | TTFT p99 | 142.79 | 157.41 | 158.22 | -0.5% [-1.8,+1.5] | -10.4% [-10.9,-9.7] |
| 8 | TPOT p50 | 5.96 | 6.07 | 6.05 | +0.3% [-0.6,+1.9] | -1.9% [-2.7,-1.0] |
| 8 | TPOT p99 | 7.10 | 7.40 | 7.36 | +0.5% [-0.6,+2.4] | -4.0% [-5.8,-2.3] |
| 8 | ITL p50 | 5.32 | 5.33 | 5.30 | +0.4% [-0.9,+2.5] | +0.1% [-1.0,+1.0] |
| 8 | ITL p99 | 25.34 | 27.04 | 26.57 | +1.7% [+0.3,+3.9] | -6.4% [-8.4,-4.8] |
| 16 | out tok/s | 2362 | 2319 | 2332 | +0.6% [-0.6,+2.5] | -3.6% [-5.6,-2.1] |
| 16 | TTFT p50 | 29.07 | 31.15 | 30.48 | +2.9% [-9.3,+20.3] | -10.9% [-13.4,-7.1] |
| 16 | TTFT p99 | 159.62 | 158.67 | 160.98 | -3.4% [-10.0,+1.2] | -18.0% [-24.5,-13.8] |
| 16 | TPOT p50 | 6.29 | 6.41 | 6.38 | +0.5% [-0.6,+2.4] | -3.7% [-5.6,-2.3] |
| 16 | TPOT p99 | 6.82 | 6.93 | 6.86 | +0.9% [-0.6,+2.1] | -4.2% [-6.5,-2.4] |
| 16 | ITL p50 | 5.76 | 5.87 | 5.84 | +0.4% [-0.7,+2.2] | -1.8% [-2.9,-0.5] |
| 16 | ITL p99 | 16.58 | 16.72 | 16.94 | -0.7% [-2.4,+2.8] | -19.0% [-23.5,-10.7] |
| max | out tok/s | 4477 | 4114 | 4121 | +0.2% [-0.0,+0.4] | -8.0% [-8.3,-7.9] |
| max | TTFT p50 | 4203.96 | 4808.57 | 4670.49 | +2.9% [+0.4,+5.8] | -9.8% [-11.7,-8.7] |
| max | TTFT p99 | 8192.25 | 9080.03 | 9064.30 | +0.2% [-0.0,+0.4] | -9.8% [-10.3,-9.0] |
| max | TPOT p50 | 39.02 | 42.08 | 42.34 | -0.6% [-1.5,-0.0] | -8.0% [-8.5,-7.3] |
| max | TPOT p99 | 162.44 | 178.55 | 181.47 | -1.6% [-3.1,+1.1] | -10.3% [-14.1,-7.3] |
| max | ITL p50 | 21.77 | 22.63 | 22.56 | +0.3% [-0.6,+1.4] | -3.5% [-4.0,-2.6] |
| max | ITL p99 | 429.68 | 471.58 | 474.83 | -0.7% [-1.7,+0.3] | -9.8% [-10.9,-9.0] |

parity (identical of 12): {"a1:sequential:shipped~guarded": 12, "a1:sequential:flash~guarded": 9, "a1:sequential:flash~shipped": 9, "a1:concurrent:shipped~guarded": 12, "a1:concurrent:flash~guarded": 10, "a1:concurrent:flash~shipped": 10, "b1:sequential:shipped~guarded": 11, "b1:sequential:flash~guarded": 10, "b1:sequential:flash~shipped": 9, "b1:concurrent:shipped~guarded": 11, "b1:concurrent:flash~guarded": 9, "b1:concurrent:flash~shipped": 10, "b2:sequential:shipped~guarded": 12, "b2:sequential:flash~guarded": 9, "b2:sequential:flash~shipped": 9, "b2:concurrent:shipped~guarded": 12, "b2:concurrent:flash~guarded": 10, "b2:concurrent:flash~shipped": 10, "b3:sequential:shipped~guarded": 12, "b3:concurrent:shipped~guarded": 12, "across_blocks:sequential:flash": 11, "across_blocks:concurrent:flash": 9, "across_blocks:sequential:shipped": 11, "across_blocks:concurrent:shipped": 10, "across_blocks:sequential:guarded": 11, "across_blocks:concurrent:guarded": 11}
telemetry: {
"a1-shipped": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=2|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=4|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"a1-guarded": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=2|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=4|maxk<=2048|tile=64|warps=4|stages=3",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"b1-shipped": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=2|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=4|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"b1-guarded": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=2|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=4|maxk<=2048|tile=64|warps=4|stages=3",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"b2-shipped": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=2|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=4|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"b2-guarded": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=2|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=4|maxk<=2048|tile=64|warps=4|stages=3",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"b3-shipped": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=2|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=4|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"b3-guarded": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=2|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=4|maxk<=2048|tile=64|warps=4|stages=3",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
]
}
