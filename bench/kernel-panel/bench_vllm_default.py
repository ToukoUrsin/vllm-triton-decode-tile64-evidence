"""One-shape vLLM V1 FlashAttention decode baseline on a CUDA GPU.

The benchmark uses vLLM's own metadata builder and ``FLASH_ATTN`` implementation
(the default on Hopper), with a small paged KV cache. Correctness is checked
against an independent float32 dense GQA reference before CUDA-event timing.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch


@dataclass
class Shape:
    batch: int
    seq_len: int
    q_heads: int
    kv_heads: int
    head_dim: int


def dense_reference(q, k, v):
    """Independent one-token GQA reference; q [B,Hq,D], kv [B,Hk,L,D]."""
    b, hq, d = q.shape
    _, hk, _, _ = k.shape
    assert hq % hk == 0
    kr = k.repeat_interleave(hq // hk, dim=1)
    vr = v.repeat_interleave(hq // hk, dim=1)
    scores = torch.matmul(q.float().unsqueeze(2), kr.float().transpose(-1, -2))
    p = torch.softmax(scores / math.sqrt(d), dim=-1)
    return torch.matmul(p, vr.float()).squeeze(2).to(q.dtype)


def local_model_config(shape: Shape):
    """Build a tiny local Llama config, avoiding model-weight downloads."""
    p = Path(tempfile.mkdtemp(prefix="vllm-attn-config-"))
    cfg = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": shape.q_heads * shape.head_dim,
        "intermediate_size": shape.q_heads * shape.head_dim * 4,
        "num_hidden_layers": 1, "num_attention_heads": shape.q_heads,
        "num_key_value_heads": shape.kv_heads,
        "max_position_embeddings": max(shape.seq_len + 8, 2048),
        "vocab_size": 32000, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
        "torch_dtype": "bfloat16",
    }
    (p / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return p


def build_config(shape: Shape):
    from vllm.config import (
        CacheConfig, CompilationConfig, DeviceConfig, LoadConfig, ModelConfig,
        ParallelConfig, SchedulerConfig, VllmConfig,
    )
    model_dir = local_model_config(shape)
    mc = ModelConfig(model=str(model_dir), tokenizer=str(model_dir), dtype=torch.bfloat16,
                     max_model_len=shape.seq_len + 8, trust_remote_code=False)
    # Minimal hooks used by vLLM's own standalone backend tests.
    mc.get_num_layers = lambda: 1
    mc.get_sliding_window_for_layer = lambda i: None
    mc.get_logits_soft_cap_for_layer = lambda i: 0.0
    mc.get_sm_scale_for_layer = lambda i: 1.0 / math.sqrt(shape.head_dim)
    cc = CacheConfig(block_size=shape.seq_len, cache_dtype="auto")
    cc.num_gpu_blocks = shape.batch + 2
    cc.num_cpu_blocks = 0
    cc.kv_cache_layout = "LBNHC"
    pc = ParallelConfig(tensor_parallel_size=1)
    sc = SchedulerConfig(max_num_seqs=shape.batch, max_num_batched_tokens=(shape.seq_len + 8) * shape.batch,
                         enable_chunked_prefill=False, max_model_len=mc.max_model_len,
                         is_encoder_decoder=mc.is_encoder_decoder)
    return VllmConfig(model_config=mc, cache_config=cc, parallel_config=pc,
                      scheduler_config=sc, device_config=DeviceConfig(), load_config=LoadConfig(),
                      compilation_config=CompilationConfig())


def common_metadata(shape: Shape, device):
    from vllm.v1.attention.backend import CommonAttentionMetadata
    n = shape.batch
    return CommonAttentionMetadata(
        query_start_loc=torch.arange(n + 1, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.arange(n + 1, dtype=torch.int32),
        seq_lens=torch.full((n,), shape.seq_len, dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.full((n,), shape.seq_len, dtype=torch.int32),
        num_reqs=n, num_actual_tokens=n, max_query_len=1, max_seq_len=shape.seq_len,
        block_table_tensor=torch.arange(n, dtype=torch.int32, device=device).view(n, 1),
        slot_mapping=(torch.arange(n, dtype=torch.int64, device=device) * shape.seq_len
                      + shape.seq_len - 1),
        causal=True,
    )


def make_cache(shape: Shape, k_context, v_context, device):
    # Logical LBNHC view: [num_blocks, KV heads, block tokens, K/V channels].
    cache = torch.zeros((shape.batch + 2, shape.kv_heads, shape.seq_len, 2 * shape.head_dim),
                        dtype=k_context.dtype, device=device)
    context_len = shape.seq_len - 1
    for b in range(shape.batch):
        cache[b, :, :context_len, :shape.head_dim] = k_context[b]
        cache[b, :, :context_len, shape.head_dim:] = v_context[b]
    return cache


def run_once(shape: Shape, q, k_full, v_full, *, warmup: int, iters: int):
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    config = build_config(shape)
    device = q.device
    common = common_metadata(shape, device)
    k_context, v_context = k_full[:, :, :-1], v_full[:, :, :-1]
    key_new, value_new = k_full[:, :, -1], v_full[:, :, -1]
    cache = make_cache(shape, k_context, v_context, device)
    backend_cls = AttentionBackendEnum.FLASH_ATTN.get_class()
    builder_cls, impl_cls = backend_cls.get_builder_cls(), backend_cls.get_impl_cls()
    spec = FullAttentionSpec(block_size=shape.seq_len, num_kv_heads=shape.kv_heads,
                             head_size=shape.head_dim, dtype=q.dtype)
    with set_current_vllm_config(config):
        builder = builder_cls(spec, ["worker_baseline"], config, device)
        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
        impl = impl_cls(num_heads=shape.q_heads, head_size=shape.head_dim,
                        scale=1.0 / math.sqrt(shape.head_dim), num_kv_heads=shape.kv_heads,
                        alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto")
    layer = SimpleNamespace(_q_scale=torch.ones(1, device=device),
                            _k_scale=torch.ones(1, device=device),
                            _v_scale=torch.ones(1, device=device))
    q_flat = q.reshape(shape.batch, shape.q_heads, shape.head_dim)
    k_new = key_new.reshape(shape.batch, shape.kv_heads, shape.head_dim)
    v_new = value_new.reshape(shape.batch, shape.kv_heads, shape.head_dim)
    output = torch.empty_like(q_flat)
    impl.do_kv_cache_update(layer, k_new, v_new, cache, metadata.slot_mapping)
    impl.forward(layer, q_flat, k_new, v_new, cache, metadata, output=output)
    torch.cuda.synchronize()
    ref = dense_reference(q_flat, k_full, v_full)
    max_abs_err = float((output.float() - ref.float()).abs().max().item())
    rms_err = float(torch.sqrt(torch.mean((output.float() - ref.float()) ** 2)).item())
    for _ in range(warmup):
        impl.forward(layer, q_flat, k_new, v_new, cache, metadata, output=output)
    torch.cuda.synchronize()
    times_us = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); impl.forward(layer, q_flat, k_new, v_new, cache, metadata, output=output); end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)
    return {"backend": backend_cls.get_name(), "flash_attn_impl": type(impl).__name__,
            "max_abs_err": max_abs_err, "rms_err": rms_err, "tolerance": 3e-2,
            "correctness_pass": max_abs_err <= 3e-2,
            "timing_us": {"median": statistics.median(times_us), "samples": sorted(times_us)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--q-heads", type=int, default=32); ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128); ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30); ap.add_argument("--out", default="baseline.json")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    import vllm
    info = {"torch": torch.__version__, "vllm": getattr(vllm, "__version__", "unknown"),
            "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability()}
    if args.probe: print(json.dumps(info, indent=2, default=str)); return
    shape = Shape(args.batch, args.seq_len, args.q_heads, args.kv_heads, args.head_dim)
    torch.manual_seed(20261006)
    q = torch.randn((shape.batch, shape.q_heads, 1, shape.head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((shape.batch, shape.kv_heads, shape.seq_len, shape.head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    result = {**info, "shape": {**shape.__dict__, "dtype": str(q.dtype)},
              **run_once(shape, q, k, v, warmup=args.warmup, iters=args.iters)}
    if not result["correctness_pass"]: raise AssertionError(f"correctness failed: {result['max_abs_err']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__": main()
