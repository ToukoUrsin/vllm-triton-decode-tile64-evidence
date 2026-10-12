blocks: p1, p2, p3, p4, p5, p6 (n per comparison in summary json)
| conc | metric | FLASH_ATTN | Triton shipped | Triton guarded | guarded vs shipped | guarded vs FLASH |
|---|---|---|---|---|---|---|
| 1 | out tok/s | 201 | 205 | 211 | +3.0% [+1.2,+4.7] | +4.9% [+2.8,+6.6] |
| 1 | TTFT p50 | 38.13 | 40.62 | 40.78 | -0.7% [-8.9,+5.2] | -6.7% [-15.4,-2.3] |
| 1 | TTFT p99 | 59.00 | 66.87 | 67.33 | -0.3% [-5.0,+5.2] | -12.7% [-16.1,-8.7] |
| 1 | TPOT p50 | 4.82 | 4.72 | 4.57 | +3.3% [+1.5,+5.2] | +5.4% [+3.4,+7.0] |
| 1 | TPOT p99 | 4.90 | 4.85 | 4.69 | +3.5% [+0.4,+6.9] | +4.6% [+1.3,+7.7] |
| 1 | ITL p50 | 4.81 | 4.71 | 4.57 | +3.2% [+1.6,+5.1] | +5.3% [+3.4,+7.1] |
| 1 | ITL p99 | 6.70 | 6.53 | 6.50 | +0.5% [-5.5,+3.2] | +3.2% [-1.8,+6.9] |
| 4 | out tok/s | 719 | 710 | 730 | +2.8% [+1.6,+4.7] | +1.6% [+0.1,+3.7] |
| 4 | TTFT p50 | 42.90 | 48.35 | 46.85 | +2.9% [-2.6,+9.8] | -8.7% [-14.3,-2.2] |
| 4 | TTFT p99 | 122.17 | 136.47 | 135.41 | +0.6% [-4.4,+6.9] | -10.3% [-14.5,+4.3] |
| 4 | TPOT p50 | 5.30 | 5.33 | 5.19 | +2.7% [+1.6,+4.1] | +2.1% [+1.0,+4.2] |
| 4 | TPOT p99 | 5.94 | 6.06 | 5.91 | +2.5% [+0.4,+6.1] | +0.5% [-1.0,+3.0] |
| 4 | ITL p50 | 5.01 | 4.99 | 4.84 | +3.0% [+1.9,+4.7] | +3.4% [+2.0,+5.8] |
| 4 | ITL p99 | 15.34 | 17.47 | 17.75 | -1.3% [-8.5,+5.5] | -12.1% [-24.0,+3.3] |
| 8 | out tok/s | 1225 | 1212 | 1220 | +0.6% [-0.3,+3.1] | -0.4% [-2.0,+1.5] |
| 8 | TTFT p50 | 47.72 | 53.26 | 51.07 | +3.8% [-0.7,+13.3] | -6.9% [-16.0,-0.3] |
| 8 | TTFT p99 | 152.55 | 164.95 | 160.40 | +2.6% [-0.2,+11.2] | -5.3% [-9.5,+10.6] |
| 8 | TPOT p50 | 6.00 | 6.02 | 6.00 | +0.4% [-0.7,+2.4] | +0.1% [-1.1,+2.0] |
| 8 | TPOT p99 | 7.19 | 7.35 | 7.32 | +0.4% [-0.8,+2.5] | -1.9% [-2.4,-1.0] |
| 8 | ITL p50 | 5.37 | 5.28 | 5.26 | +0.4% [-0.6,+2.7] | +2.2% [+0.5,+4.3] |
| 8 | ITL p99 | 24.88 | 27.72 | 26.95 | +3.1% [-3.7,+11.2] | -7.6% [-19.9,+1.7] |
| 16 | out tok/s | 2512 | 2453 | 2466 | +0.5% [-0.4,+2.7] | -1.9% [-2.8,-0.7] |
| 16 | TTFT p50 | 26.29 | 27.08 | 27.65 | -3.7% [-17.3,+7.7] | -4.6% [-17.3,+17.6] |
| 16 | TTFT p99 | 81.03 | 82.21 | 77.46 | +4.2% [-23.2,+25.6] | +0.3% [-23.4,+36.7] |
| 16 | TPOT p50 | 5.79 | 5.95 | 5.92 | +0.4% [-0.4,+2.2] | -2.2% [-2.9,-0.8] |
| 16 | TPOT p99 | 5.91 | 6.10 | 6.09 | +0.2% [-0.6,+0.9] | -2.9% [-5.4,-1.2] |
| 16 | ITL p50 | 5.76 | 5.80 | 5.77 | +0.5% [-0.1,+2.5] | -0.1% [-1.5,+2.1] |
| 16 | ITL p99 | 9.31 | 11.00 | 11.27 | -2.7% [-10.4,+0.6] | -17.1% [-28.8,-3.7] |
| max | out tok/s | 4447 | 4097 | 4094 | -0.1% [-0.4,+0.5] | -7.9% [-8.2,-7.5] |
| max | TTFT p50 | 4321.66 | 4880.25 | 4878.94 | +0.1% [-5.4,+7.0] | -11.3% [-16.1,-6.8] |
| max | TTFT p99 | 8262.34 | 9131.65 | 9137.84 | -0.1% [-0.8,+0.6] | -9.6% [-10.3,-8.9] |
| max | TPOT p50 | 39.27 | 42.36 | 42.31 | +0.1% [-1.3,+1.8] | -7.2% [-8.1,-5.5] |
| max | TPOT p99 | 159.76 | 183.09 | 182.91 | +0.1% [-1.0,+1.4] | -12.6% [-26.4,-8.6] |
| max | ITL p50 | 21.85 | 22.86 | 22.76 | +0.4% [-1.1,+1.9] | -4.0% [-5.9,-2.9] |
| max | ITL p99 | 426.05 | 469.64 | 474.63 | -1.1% [-3.1,-0.1] | -10.2% [-11.2,-8.9] |

parity (identical of 12): {"p1:sequential:shipped~guarded": 12, "p1:sequential:flash~guarded": 9, "p1:sequential:flash~shipped": 9, "p1:concurrent:shipped~guarded": 12, "p1:concurrent:flash~guarded": 10, "p1:concurrent:flash~shipped": 10, "p2:sequential:shipped~guarded": 12, "p2:sequential:flash~guarded": 9, "p2:sequential:flash~shipped": 9, "p2:concurrent:shipped~guarded": 12, "p2:concurrent:flash~guarded": 11, "p2:concurrent:flash~shipped": 11, "p3:sequential:shipped~guarded": 12, "p3:sequential:flash~guarded": 9, "p3:sequential:flash~shipped": 9, "p3:concurrent:shipped~guarded": 12, "p3:concurrent:flash~guarded": 10, "p3:concurrent:flash~shipped": 10, "p4:sequential:shipped~guarded": 12, "p4:sequential:flash~guarded": 9, "p4:sequential:flash~shipped": 9, "p4:concurrent:shipped~guarded": 12, "p4:concurrent:flash~guarded": 8, "p4:concurrent:flash~shipped": 8, "p5:sequential:shipped~guarded": 11, "p5:sequential:flash~guarded": 10, "p5:sequential:flash~shipped": 9, "p5:concurrent:shipped~guarded": 11, "p5:concurrent:flash~guarded": 9, "p5:concurrent:flash~shipped": 10, "p6:sequential:shipped~guarded": 12, "p6:sequential:flash~guarded": 9, "p6:sequential:flash~shipped": 9, "p6:concurrent:shipped~guarded": 11, "p6:concurrent:flash~guarded": 12, "p6:concurrent:flash~shipped": 11, "across_blocks:sequential:flash": 11, "across_blocks:concurrent:flash": 9, "across_blocks:sequential:shipped": 11, "across_blocks:concurrent:shipped": 10, "across_blocks:sequential:guarded": 11, "across_blocks:concurrent:guarded": 11}
telemetry: {
"p1-shipped": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=2|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=4|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"p1-guarded": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=2|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=4|maxk<=2048|tile=64|warps=4|stages=3",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"p2-shipped": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=2|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=4|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"p2-guarded": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=2|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=4|maxk<=2048|tile=64|warps=4|stages=3",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"p3-shipped": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=2|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=4|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"p3-guarded": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=2|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=4|maxk<=2048|tile=64|warps=4|stages=3",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"p4-shipped": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=2|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=4|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"p4-guarded": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=2|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=4|maxk<=2048|tile=64|warps=4|stages=3",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"p5-shipped": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=2|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=4|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"p5-guarded": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=2|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=4|maxk<=2048|tile=64|warps=4|stages=3",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"p6-shipped": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=2|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=4|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
],
"p6-guarded": [
"capture|3d=1|seqs=16|maxk<=2048|tile=16|warps=None|stages=None",
"capture|3d=1|seqs=1|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=2|maxk<=2048|tile=64|warps=4|stages=2",
"capture|3d=1|seqs=4|maxk<=2048|tile=64|warps=4|stages=3",
"capture|3d=1|seqs=8|maxk<=2048|tile=16|warps=None|stages=None"
]
}
