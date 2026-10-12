"""Add launch telemetry (env ATTN_TELEMETRY_DIR) to a triton_unified_attention.py copy.

Records, per process, how often each decode/prefill launch shape is taken, split into CUDA-graph capture
(launch frozen into a FULL graph and replayed later) and eager calls. Off when the env var is unset.
"""
import sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
anchor = "    launch_kwargs: dict[str, int] = {}\n"
assert text.count(anchor) == 1, "anchor not unique"
text = text.replace(anchor, "    _attn_telemetry(use_3d, num_seqs, max_seqlen_k, tile_size, launch_num_warps, launch_num_stages)\n" + anchor)
text += '''

import json as _tel_json
import os as _tel_os

_TEL_COUNTS: dict[str, int] = {}
_TEL_CALLS = [0]


def _attn_telemetry(use_3d, num_seqs, max_seqlen_k, tile, warps, stages):
    out_dir = _tel_os.environ.get("ATTN_TELEMETRY_DIR")
    if not out_dir:
        return
    capturing = torch.cuda.is_current_stream_capturing()
    kmax = int(max_seqlen_k) if capturing else (int(max_seqlen_k) + 255) // 256 * 256
    key = (f"{'capture' if capturing else 'eager'}|3d={int(bool(use_3d))}|seqs={int(num_seqs)}|"
           f"maxk<={kmax}|tile={tile}|warps={warps}|stages={stages}")
    new = key not in _TEL_COUNTS
    _TEL_COUNTS[key] = _TEL_COUNTS.get(key, 0) + 1
    _TEL_CALLS[0] += 1
    if new or capturing or _TEL_CALLS[0] % 500 == 0:
        path = _tel_os.path.join(out_dir, f"telemetry-{_tel_os.getpid()}.json")
        with open(path + ".tmp", "w") as f:
            _tel_json.dump({"calls": _TEL_CALLS[0], "counts": _TEL_COUNTS}, f, indent=1, sort_keys=True)
        _tel_os.replace(path + ".tmp", path)
'''
open(dst, "w").write(text)
