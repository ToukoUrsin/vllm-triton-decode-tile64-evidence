#!/bin/bash
# Download model at the pinned revision, build instrumented shipped + guarded kernel files.
set -ex
W=/root/w; mkdir -p $W/kernels $W/results
cd $W
python3 - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-8B-AWQ", revision="4da05a8edb55c6046cce958586c33b61da07bb79",
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"], local_dir="/root/w/model", max_workers=8)
print(p)
PY
SITE=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
cp $SITE/v1/attention/ops/triton_unified_attention.py kernels/shipped_v0.31.0.py
diff -u kernels/shipped_v0.31.0.py triton_unified_attention_guarded_v0.31.0.py > kernels/guarded-vs-shipped.diff || true
python3 instrument.py kernels/shipped_v0.31.0.py kernels/shipped_tel.py
python3 instrument.py triton_unified_attention_guarded_v0.31.0.py kernels/guarded_tel.py
python3 -c "import vllm, torch, triton; print(vllm.__version__, torch.__version__, triton.__version__)" > results/environment.txt
nvidia-smi >> results/environment.txt
pip list 2>/dev/null | grep -iE "^(vllm|torch|triton|flashinfer|transformers) " >> results/environment.txt
vllm bench serve --help 2>&1 | grep -E "random-range-ratio|save-detailed|max-concurrency|percentile" | head
vllm serve --help=all 2>&1 | grep -E "attention-backend" | head -3
echo SETUP_OK
