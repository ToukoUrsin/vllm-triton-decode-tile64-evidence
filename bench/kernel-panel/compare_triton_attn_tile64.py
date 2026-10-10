from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path
from types import SimpleNamespace
import torch
import vllm
from bench_vllm_default import Shape, build_config, dense_reference


def graph_time(fn, warmup=20, samples=40):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    vals=[]
    for _ in range(samples):
        st,en=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        st.record(); g.replay(); en.record(); en.synchronize()
        vals.append(st.elapsed_time(en)*1000.0)
    return {'median':statistics.median(vals),'min':min(vals),'max':max(vals),'samples':vals,'graph_calls':1}


def check(out, ref, tol=0.03):
    finite=bool(torch.isfinite(out).all().item())
    err=float((out.float()-ref.float()).abs().max().item()) if finite else None
    return {'finite':finite,'max_abs_error':err,'tolerance':tol,'pass':finite and err <= tol}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--batch',type=int,default=1); ap.add_argument('--seq-len',type=int,default=1024)
    ap.add_argument('--block-size',type=int,default=128); ap.add_argument('--warmup',type=int,default=20)
    ap.add_argument('--q-heads',type=int,default=32); ap.add_argument('--kv-heads',type=int,default=8); ap.add_argument('--head-dim',type=int,default=128); ap.add_argument('--label',default='unknown')
    ap.add_argument('--samples',type=int,default=40); ap.add_argument('--reverse',action='store_true'); ap.add_argument('--out',default='triton-attn-pivot.json')
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    shape=Shape(a.batch,a.seq_len,a.q_heads,a.kv_heads,a.head_dim)
    torch.manual_seed(20261006)
    q=torch.randn((shape.batch,shape.q_heads,1,shape.head_dim),device='cuda',dtype=torch.bfloat16)
    k=torch.randn((shape.batch,shape.kv_heads,shape.seq_len,shape.head_dim),device='cuda',dtype=q.dtype)
    v=torch.randn_like(k)
    pages=(shape.seq_len+a.block_size-1)//a.block_size
    cache=torch.zeros((shape.batch*pages+2,shape.kv_heads,a.block_size,2*shape.head_dim),device='cuda',dtype=q.dtype)
    for b in range(shape.batch):
        for p in range(pages):
            lo,hi=p*a.block_size,min((p+1)*a.block_size,shape.seq_len)
            cache[b*pages+p,:,:(hi-lo),:shape.head_dim]=k[b,:,lo:hi]
            cache[b*pages+p,:,:(hi-lo),shape.head_dim:]=v[b,:,lo:hi]
    table=torch.arange(shape.batch*pages,device='cuda',dtype=torch.int32).view(shape.batch,pages)
    ref=dense_reference(q.squeeze(2),k,v)
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    config=build_config(shape)
    config.cache_config.block_size=a.block_size
    config.cache_config.num_gpu_blocks=cache.shape[0]
    common=CommonAttentionMetadata(
       query_start_loc=torch.arange(shape.batch+1,dtype=torch.int32,device='cuda'),
       query_start_loc_cpu=torch.arange(shape.batch+1,dtype=torch.int32),
       seq_lens=torch.full((shape.batch,),shape.seq_len,dtype=torch.int32,device='cuda'),
       seq_lens_cpu_upper_bound=torch.full((shape.batch,),shape.seq_len,dtype=torch.int32),
       num_reqs=shape.batch,num_actual_tokens=shape.batch,max_query_len=1,max_seq_len=shape.seq_len,
       block_table_tensor=table,slot_mapping=(torch.arange(shape.batch,dtype=torch.int64,device='cuda')*shape.seq_len+shape.seq_len-1),causal=True)
    layer=SimpleNamespace(_q_scale=torch.ones(1,device='cuda'),_k_scale=torch.ones(1,device='cuda'),_v_scale=torch.ones(1,device='cuda'))
    qflat=q.squeeze(2); knew=k[:,:,-1].contiguous(); vnew=v[:,:,-1].contiguous()
    # Worker2 variant: force vLLM Triton unified-attention decode KV tiles
    # from the shipped default 16 tokens to 64.  This is a runtime launch
    # configuration change only; the FLASH_ATTN comparator remains untouched.
    import vllm.v1.attention.ops.triton_unified_attention as tua
    _default_tile_selector = tua._get_tile_size
    def _tile64(head_size, sliding_window, element_size, is_prefill):
        if not is_prefill and element_size >= 2 and sliding_window <= 0:
            return 64
        return _default_tile_selector(head_size, sliding_window, element_size, is_prefill)
    tua._get_tile_size = _tile64

    outputs={}; runners={}; checks={}; timing={}
    backend_order=[('flash',AttentionBackendEnum.FLASH_ATTN),('triton',AttentionBackendEnum.TRITON_ATTN)]
    if a.reverse: backend_order=list(reversed(backend_order))
    for name,backend in backend_order:
        spec=FullAttentionSpec(block_size=a.block_size,num_kv_heads=shape.kv_heads,head_size=shape.head_dim,dtype=q.dtype)
        with set_current_vllm_config(config):
            builder=backend.get_class().get_builder_cls()(spec,[f'pivot_{name}'],config,q.device)
            metadata=builder.build(common_prefix_len=0,common_attn_metadata=common)
            impl=backend.get_class().get_impl_cls()(num_heads=shape.q_heads,head_size=shape.head_dim,scale=1/math.sqrt(shape.head_dim),num_kv_heads=shape.kv_heads,alibi_slopes=None,sliding_window=None,kv_cache_dtype='auto')
        # Each implementation gets its own cache because update semantics and graph buffers must not alias.
        c=cache.clone(); out=torch.empty_like(qflat)
        impl.do_kv_cache_update(layer,knew,vnew,c,metadata.slot_mapping)
        def fn(impl=impl,c=c,metadata=metadata,out=out):
            return impl.forward(layer,qflat,knew,vnew,c,metadata,output=out)
        fn(); torch.cuda.synchronize()
        outputs[name]=out
        checks[name]=check(out,ref)
        runners[name]=fn
        if checks[name]['pass']: timing[name+'_graph']=graph_time(fn,a.warmup,a.samples)
    corrupted=ref.clone(); corrupted[0,0,0] += 1.0
    checks['negative_control']=check(corrupted,ref)
    if checks['negative_control']['pass']: raise AssertionError('negative control unexpectedly passed')
    if not all(checks[x]['pass'] for x in ('flash','triton')): raise AssertionError(checks)
    payload={'environment':{'torch':torch.__version__,'vllm':vllm.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability()},'shape':{**shape.__dict__,'block_size':a.block_size,'dtype':str(q.dtype),'label':a.label},'timing_scope':'CUDA graph replay; CUDA-event elapsed microseconds; one backend call per replay', 'backend_order': ['triton','flash'] if a.reverse else ['flash','triton'],'checks':checks,'timing_us':timing,'speedup_flash_over_triton':timing['flash_graph']['median']/timing['triton_graph']['median'],'triton_delta_vs_flash_pct':(timing['triton_graph']['median']/timing['flash_graph']['median']-1)*100}
    Path(a.out).write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
