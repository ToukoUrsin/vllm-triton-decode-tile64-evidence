"""Install no system packages; unpack NVIDIA's standalone tool in GPU scratch."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request

URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-sanitizer-13-0_13.0.48-1_amd64.deb"
scratch=Path("/root/rel-sanitizer")
scratch.mkdir(exist_ok=True)
deb=scratch/"tool.deb"
urllib.request.urlretrieve(URL,deb)
sha=hashlib.sha256(deb.read_bytes()).hexdigest()
subprocess.run(["dpkg-deb","--extract",str(deb),str(scratch)],check=True)
tool=next(scratch.rglob("compute-sanitizer/compute-sanitizer"))
version=subprocess.run([str(tool),"--version"],capture_output=True,text=True,check=True).stdout
cases=[("main2d-s1025","main",16,1025,1,True,True),
       ("main_pagemasked2d-s1025","main_pagemasked",16,1025,1,False,True),
       ("main3d-s1025","main",16,1025,1,None,False),
       ("main_pagemasked3d-s1025","main_pagemasked",16,1025,1,False,False),
       ("main_pagemasked2d-b4-s1025","main_pagemasked",16,1025,4,False,True),
       ("main_pagemasked3d-b4-s1025","main_pagemasked",16,1025,4,False,False)]
results=[]
env=dict(os.environ,PYTORCH_NO_CUDA_MEMORY_CACHING="1",OMP_NUM_THREADS="1")
for label,source,tile,length,batch,expected_error,force_2d in cases:
    command=[str(tool),"--tool","memcheck","--error-exitcode","77",
             "--kernel-name","kns=kernel_unified_attention","--launch-timeout","120",
             "python3","-u","memory_probe.py","--source",source,"--tile",str(tile),
             "--length",str(length),"--batch",str(batch),"--out",label+".json"]
    if force_2d:
        command.extend(["--force-2d"])
    started=time.time()
    run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
    log=run.stdout+run.stderr
    Path(label+".log").write_text(log)
    summaries=re.findall(r"ERROR SUMMARY: (\d+) errors",log)
    assert summaries,(label,run.returncode,log[-2000:])
    errors=int(summaries[-1])
    invalid_read="Invalid __global__ read" in log
    if expected_error is None:
        passed=True  # observational case: record only
    else:
        passed=((errors>0 and invalid_read and run.returncode!=0) if expected_error else
                (errors==0 and run.returncode==0 and Path(label+".json").exists()))
    result=dict(label=label,source=source,tile=tile,length=length,batch=batch,
                expected_memory_error=expected_error,force_2d=force_2d,returncode=run.returncode,
                sanitizer_error_count=errors,invalid_global_read=invalid_read,
                expected_result_pass=passed,elapsed_seconds=time.time()-started,
                command=command,log=label+".log")
    results.append(result)
    Path("memcheck-summary.json").write_text(json.dumps(dict(tool_version=version,
         download_url=URL,download_sha256=sha,run_id=os.getenv("FLEET_RUN_ID"),results=results,complete=len(results)==len(cases)),indent=2)+"\n")
    print(json.dumps(result),flush=True)
    assert passed,(label,log[-4000:])
print("all original-input sanitizer cases and negative control complete",flush=True)
