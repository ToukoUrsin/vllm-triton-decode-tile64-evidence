#!/bin/bash
# Download the model (outside the synced dir) and build instrumented shipped + guarded kernel copies. Idempotent.
set -ex
W=$PWD; mkdir -p $W/kernels $W/results
python3 - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-8B-AWQ", revision="4da05a8edb55c6046cce958586c33b61da07bb79",
                        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"], local_dir="/root/model", max_workers=8))
PY
SITE=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
[ -f kernels/shipped_v0.31.0.py ] || cp $SITE/v1/attention/ops/triton_unified_attention.py kernels/shipped_v0.31.0.py
python3 instrument.py kernels/shipped_v0.31.0.py kernels/shipped_tel.py
python3 instrument.py triton_unified_attention_guarded_v0.31.0.py kernels/guarded_tel.py
{ python3 -c "import vllm, torch, triton; print(vllm.__version__, torch.__version__, triton.__version__)"; nvidia-smi;
  nvidia-smi --query-gpu=uuid,ecc.errors.uncorrected.aggregate.total --format=csv; } > results/environment-$FLEET_RUN_ATTEMPT.txt
echo SETUP_OK
