"""Exact-size paged decode input for CUDA memcheck and a dense reference."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import torch
from bench_vllm_default import dense_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("main", "main_pagemasked"), default="main")
    ap.add_argument("--length", type=int, default=1025)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--force-2d", action="store_true")
    ap.add_argument("--out", default="probe.json")
    a = ap.parse_args()
    source = Path(a.source+"_triton_unified_attention.py")

    spec = importlib.util.spec_from_file_location("probe_attention", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_tile = module._get_tile_size
    original_kernel = module.kernel_unified_attention
    telemetry = {}
    class Proxy:
        def __getitem__(self, grid):
            launch = original_kernel[grid]
            def run(*args, **kwargs):
                kernel = launch(*args, **kwargs)
                telemetry.update(grid=list(grid), tile=kwargs["TILE_SIZE"],
                                 is_3d=kwargs["IS_3D"], stages=kernel.metadata.num_stages,
                                 warps=kernel.metadata.num_warps)
                return kernel
            return run
    module.kernel_unified_attention = Proxy()
    torch.manual_seed(20261006)
    torch.set_num_threads(1)
    hq, hk, dim, block = 32, 8, 128, 16
    pages = math.ceil(a.length/block)
    nblocks = a.batch*pages + 4
    lengths = [a.length] * a.batch
    q = torch.randn(a.batch, hq, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(a.batch, hk, a.length, dim, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    kc = torch.full((nblocks, block, hk, dim), 100., device="cuda", dtype=q.dtype)
    vc = torch.full_like(kc, 100.)
    for b in range(a.batch):
        for p in range(pages):
            lo, hi = p*block, min((p+1)*block, a.length)
            kc[b*pages+p, :hi-lo] = k[b, :, lo:hi].transpose(0, 1)
            vc[b*pages+p, :hi-lo] = v[b, :, lo:hi].transpose(0, 1)
    # NO_CUDA_MEMORY_CACHING forces a cudaMalloc request for exactly table bytes.
    table = torch.arange(a.batch*pages, dtype=torch.int32, device="cuda").reshape(a.batch, pages)
    seqs = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cuq = torch.arange(a.batch+1, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    partial = torch.full((a.batch,hq,16,dim), float("nan"), dtype=torch.float32, device="cuda")
    maxima = torch.full((a.batch,hq,16), float("nan"), dtype=torch.float32, device="cuda")
    sums = torch.full_like(maxima, float("nan"))
    print(json.dumps({"phase":"before_kernel", "args":vars(a),
                      "table_bytes":table.numel()*table.element_size(),
                      "allocator_caching_disabled":os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")}),flush=True)
    module.unified_attention(q=q,k=kc,v=vc,out=out,cu_seqlens_q=cuq,max_seqlen_q=1,
        seqused_k=seqs,max_seqlen_k=a.length,softmax_scale=1/math.sqrt(dim),causal=True,
        window_size=(-1,-1),block_table=table,softcap=0.,q_descale=None,k_descale=None,v_descale=None,
        seq_threshold_3D=(0 if a.force_2d else a.batch),num_par_softmax_segments=16,softmax_segm_output=partial,
        softmax_segm_max=maxima,softmax_segm_expsum=sums)
    torch.cuda.synchronize()
    ref = dense_reference(q,k,v)
    torch.testing.assert_close(out,ref,atol=0.015,rtol=0.01)
    corrupt=out.clone(); corrupt[0,0,0]+=1
    try:
        torch.testing.assert_close(corrupt,ref,atol=0.015,rtol=0.01)
    except AssertionError:
        negative=True
    else:
        raise AssertionError("corruption admitted")
    result={"args":vars(a),"table_bytes":table.numel()*table.element_size(),
            "page_width":pages,"max_abs_error":float((out.float()-ref.float()).abs().max()),
            "finite":bool(torch.isfinite(out).all()),"numerical_pass":True,
            "negative_control_rejected":negative,"telemetry":telemetry,
            "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "native_no_tile_or_stage_override":True,
            "no_cuda_memory_caching":os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
            "environment":{"torch":torch.__version__,"cuda":torch.version.cuda,
                "device":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability()}}
    Path(a.out).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"phase":"complete",**result}),flush=True)


if __name__=="__main__":
    main()
