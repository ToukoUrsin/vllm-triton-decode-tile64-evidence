"""Install and record the exact guarded source inside the model worker."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from vllm.v1.worker.gpu_worker import Worker


class MeasuredWorker(Worker):
    def init_device(self):
        super().init_device()
        self._variant=os.environ['ATTN_EXPERIMENT_VARIANT']
        self._tile_counts=Counter(); self._launch_counts=Counter(); self._source_sha=None
        if self._variant=='guarded64':
            source=Path(__file__).with_name('triton_unified_attention_guarded.py')
            self._source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
            name='vllm.v1.attention.ops.lane_guarded_attention'
            spec=importlib.util.spec_from_file_location(name,source)
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
            original=module.kernel_unified_attention
            counts=self._launch_counts
            class Recorder:
                def __getitem__(self,grid):
                    launcher=original[grid]
                    def run(*args,**kwargs):
                        compiled=launcher(*args,**kwargs)
                        counts[(kwargs['IS_3D'],kwargs['TILE_SIZE'],compiled.metadata.num_warps,
                                compiled.metadata.num_stages)]+=1
                        return compiled
                    return run
            module.kernel_unified_attention=Recorder()
            import vllm.v1.attention.backends.triton_attn as backend
            backend.unified_attention=module.unified_attention
        print(json.dumps({'event':'native_patch_ready','pid':os.getpid(),'variant':self._variant,
                          'source_sha256':self._source_sha}),flush=True)

    def attention_experiment_stats(self):
        return {'pid':os.getpid(),'variant':self._variant,'source_sha256':self._source_sha,
                'calls':[], 'kernel_launches':[{'is_3d':k[0],'tile':k[1],'warps':k[2],'stages':k[3],'count':v}
                                               for k,v in self._launch_counts.items()],
                'compilation_config':str(self.vllm_config.compilation_config),
                'attention_config':str(self.vllm_config.attention_config)}
