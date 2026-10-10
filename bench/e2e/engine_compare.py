"""Bounded real-weight vLLM comparison with worker-side tile64 telemetry."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def emit(**payload):
    print(json.dumps(payload), flush=True)


def child(a):
    os.environ['ATTN_EXPERIMENT_VARIANT'] = a.variant
    emit(event='child_import', variant=a.variant)
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    backend = 'FLASH_ATTN' if a.variant == 'flash' else 'TRITON_ATTN'
    compile_cfg = {'cudagraph_capture_sizes':[a.batch], 'max_cudagraph_capture_size':a.batch}
    if not a.default_compile:
        compile_cfg.update(mode=0, cudagraph_mode='FULL_DECODE_ONLY')
    emit(event='engine_start', variant=a.variant, model=a.model_path, compilation=compile_cfg)
    t0 = time.perf_counter()
    llm = LLM(model=a.model_path, dtype='bfloat16', tensor_parallel_size=1,
              max_model_len=2048, max_num_batched_tokens=2048, max_num_seqs=a.batch,
              gpu_memory_utilization=0.20, block_size=a.block_size, enable_prefix_caching=False,
              seed=20261006, attention_config={'backend':backend},
              compilation_config=compile_cfg, worker_cls='measured_worker.MeasuredWorker')
    init_s = time.perf_counter()-t0
    emit(event='engine_ready', variant=a.variant, init_s=init_s)
    tokenizer = llm.get_tokenizer()
    base = tokenizer.encode('Explain how attention uses cached keys and values to generate the next token. ', add_special_tokens=False)
    ids = (base*(a.prompt_tokens//len(base)+1))[:a.prompt_tokens]
    req = [{'prompt_token_ids':ids.copy()} for _ in range(a.batch)]
    params = SamplingParams(temperature=0, max_tokens=a.output_tokens,
                            min_tokens=a.output_tokens, ignore_eos=True)
    for rep in range(a.warmup):
        llm.generate(req, params, use_tqdm=False)
        emit(event='warmup_done', variant=a.variant, rep=rep)
    times, tokens = [], []
    for rep in range(a.repeats):
        t0=time.perf_counter()
        output=llm.generate(req, params, use_tqdm=False)
        dt=time.perf_counter()-t0
        ids_out=[list(o.outputs[0].token_ids) for o in output]
        assert len(ids_out)==a.batch and all(len(t)==a.output_tokens for t in ids_out)
        times.append(dt); tokens.append(ids_out)
        emit(event='measurement', variant=a.variant, rep=rep, elapsed_s=dt, tokens_per_s=a.batch*a.output_tokens/dt)
    stats=llm.collective_rpc('attention_experiment_stats')
    if a.variant == 'guarded64':
        assert any(c['is_3d'] and c['tile']==64 and c['warps']==4 and c['stages']==(3 if a.batch==4 and a.block_size==16 else 2) for st in stats for c in st['kernel_launches']), stats
    median=statistics.median(times)
    result={'variant':a.variant, 'backend':backend, 'model':a.model, 'model_path':a.model_path,
            'model_revision':a.revision, 'initialization_s':init_s,
            'prompt_token_ids':ids, 'prompt_tokens':len(ids), 'output_tokens':a.output_tokens,
            'elapsed_s':times, 'median_elapsed_s':median, 'tokens_per_s':a.batch*a.output_tokens/median, 'batch':a.batch,
            'warmup':a.warmup, 'repeats':a.repeats, 'output_token_ids':tokens, 'within_backend_deterministic':all(x==tokens[0] for x in tokens),
            'worker_telemetry':stats, 'engine_config':{'max_model_len':2048,'max_num_batched_tokens':2048,
             'max_num_seqs':a.batch,'block_size':a.block_size,'enable_prefix_caching':False,'compilation_config':compile_cfg},
            'environment':{'torch':torch.__version__,'vllm':vllm.__version__, 'cuda':torch.version.cuda,
                           'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability()},
            'scope':'offline vLLM engine generate wall time including prefill, decode, scheduling and output collection; public bf16 weights'}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    emit(event='child_complete', variant=a.variant, out=a.out, tokens_per_s=result['tokens_per_s'])


def parent(a):
    emit(event='download_start', model=a.model)
    from huggingface_hub import HfApi, snapshot_download
    revision=HfApi().model_info(a.model).sha
    model_path=snapshot_download(a.model, revision=revision,
       allow_patterns=['*.json','*.safetensors','*.txt','*.model'], max_workers=2)
    emit(event='download_complete', model_path=model_path, revision=revision)
    order=['flash','guarded64','guarded64','flash'] if a.rounds==2 else ['flash','guarded64']
    completed=[]
    for index,variant in enumerate(order):
        outfile=f'{a.tag}{index}-{variant}.json'
        cmd=[sys.executable,'-u',__file__,'--variant',variant,'--model',a.model,'--revision',revision,
             '--model-path',model_path,'--out',outfile,'--repeats',str(a.repeats),
             '--prompt-tokens',str(a.prompt_tokens),'--output-tokens',str(a.output_tokens),'--warmup',str(a.warmup),
             '--batch',str(a.batch),'--block-size',str(a.block_size)]
        if a.default_compile: cmd.append('--default-compile')
        emit(event='launch_child', command=cmd)
        subprocess.run(cmd, check=True, timeout=900)
        completed.append(outfile)
        Path(a.tag+'completed.json').write_text(json.dumps({'files':completed,'order':order,'model':a.model,'revision':revision},indent=2)+'\n')
    emit(event='all_complete', files=completed)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['flash','guarded64','triton16'])
    p.add_argument('--model',default='Qwen/Qwen3-8B-AWQ')
    p.add_argument('--revision',default='unknown'); p.add_argument('--model-path')
    p.add_argument('--repeats',type=int,default=7); p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--warmup',type=int,default=2); p.add_argument('--tag',default='')
    p.add_argument('--prompt-tokens',type=int,default=1024); p.add_argument('--output-tokens',type=int,default=128)
    p.add_argument('--batch',type=int,default=1); p.add_argument('--block-size',type=int,default=128)
    p.add_argument('--default-compile',action='store_true'); p.add_argument('--out',default='engine-result.json')
    a=p.parse_args()
    if a.variant: child(a)
    else: parent(a)
