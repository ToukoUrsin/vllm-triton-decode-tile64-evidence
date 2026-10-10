"""Recompute all newly admitted tile64 timing and recorded numerical gates."""
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).parent
PANEL=['qwen3-8b.json','qwen3-14b.json','qwen3-32b.json','qwen3-30b-a3b.json','qwen3-1p7b.json','qwen2p5-7b.json']
EXTRA=['tile64-b1-s1024-reverse.json','tile64-b1-s2048.json','tile64-b2-s2048.json','three-way.json','three-way-reverse.json']
rows=[]
for name in PANEL+EXTRA:
 x=json.loads((ROOT/name).read_text())
 assert x['environment']['vllm']=='0.31.0' and x['environment']['capability']==[9,0]
 med={}
 for backend,timing in x['timing_us'].items():
  vals=timing['samples']
  assert len(vals)==40 and all(math.isfinite(v) and v>0 for v in vals)
  med[backend]=statistics.median(vals)
  assert math.isclose(med[backend],timing['median'],abs_tol=1e-8)
  key=backend.removesuffix('_graph'); check=x['checks'][key]
  assert check['finite'] and math.isfinite(check['max_abs_error'])
  assert check['max_abs_error']<=check['tolerance']==0.03 and check['pass']
 n=x['checks']['negative_control']
 assert n['finite'] and n['max_abs_error']>n['tolerance']==0.03 and not n['pass']
 tk='triton64_graph' if 'triton64_graph' in med else 'triton_graph'
 row={'file':name,'shape':x['shape'],'flash_us':med['flash_graph'],'tile64_us':med[tk],
      'gain_vs_flash_pct':100*(1-med[tk]/med['flash_graph']),
      'graph_calls':x['timing_us'][tk]['graph_calls'],'all_numerical_gates_pass':True}
 if 'triton16_graph' in med:
  row.update(triton16_us=med['triton16_graph'],gain_vs_triton16_pct=100*(1-med[tk]/med['triton16_graph']))
 rows.append(row)
panel=rows[:6]
assert len({(r['shape']['q_heads'],r['shape']['kv_heads']) for r in panel})==6
summary={'single_call_panel':panel,'median_paired_gain_pct':statistics.median(r['gain_vs_flash_pct'] for r in panel),
         'median_flash_us':statistics.median(r['flash_us'] for r in panel),
         'median_tile64_us':statistics.median(r['tile64_us'] for r in panel),
         'other_controls':rows[6:],'all_recorded_numerical_gates_pass':True,
         'scope':'Recorded GPU evidence audit; no new GPU execution; not external reproduction or engine throughput'}
(ROOT/'verified-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
