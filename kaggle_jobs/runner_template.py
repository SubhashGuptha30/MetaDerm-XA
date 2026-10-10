"""
Kaggle kernel entry point (generated per job by kaggle_jobs/launch.py — do not edit the copy).

Copies the code dataset to /kaggle/working/code, then runs JOBS on all visible GPUs in
parallel (one run per GPU at a time): `python -m src.train` followed by `python -m src.probes`.
Stops launching new runs after TIME_BUDGET_H hours so outputs are saved before Kaggle's 12 h limit.
Outputs: /kaggle/working/runs_v2/** and /kaggle/working/logs/*.log
"""

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

JOBS = __JOBS__            # list of [config_path, [overrides...]]
TIME_BUDGET_H = __BUDGET__
WORKERS_PER_RUN = __NUM_WORKERS__

T0 = time.time()
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


EXTRACT = Path("/kaggle/temp/extracted")


def find(marker: str) -> Path:
    for root in (INPUT, EXTRACT):
        if root.exists():
            for p in root.rglob(marker):
                return p.parent
    # fallback: datasets uploaded as un-extracted archives
    import zipfile
    for z in INPUT.rglob("*.zip"):
        dest = EXTRACT / z.stem
        if not dest.exists():
            zipfile.ZipFile(z).extractall(dest)
    for p in EXTRACT.rglob(marker):
        return p.parent
    raise FileNotFoundError(f"{marker} not found under {INPUT}: {list(INPUT.rglob('*'))[:50]}")


code_src = find("METADERM_CODE")
image_root = find("ISIC_0000000.jpg").parent  # .../<root>/isic2019_train/ISIC_0000000.jpg
code = WORK / "code"
if not code.exists():
    shutil.copytree(code_src, code)
os.chdir(code)
(WORK / "logs").mkdir(exist_ok=True)

env = dict(os.environ, METADERM_IMAGE_ROOT=str(image_root), METADERM_PRETRAINED_DIR=str(code / "pretrained"),
           PYTHONUNBUFFERED="1")
print("code:", code_src, "| images:", image_root, flush=True)
subprocess.run([sys.executable, "-c", "import torch,timm,torchvision;print(torch.__version__,timm.__version__,"
                "torchvision.__version__,torch.cuda.device_count())"], env=env)

import torch  # noqa: E402

n_gpu = max(1, torch.cuda.device_count())
q = queue.Queue()
for j in JOBS:
    q.put(j)
status = []


def worker(gpu: int):
    while True:
        try:
            cfg, sets = q.get_nowait()
        except queue.Empty:
            return
        if (time.time() - T0) / 3600 > TIME_BUDGET_H:
            status.append(("SKIPPED(time)", cfg, sets))
            continue
        name = Path(cfg).stem + "_" + "_".join(s.replace("=", "") for s in sets)
        e = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
        overrides = sets + [f"output_root={WORK / 'runs_v2'}", f"num_workers={WORKERS_PER_RUN}"]
        with open(WORK / "logs" / f"{name}.log", "w") as log:
            r = subprocess.run([sys.executable, "-m", "src.train", "--config", cfg, "--set", *overrides],
                               env=e, stdout=log, stderr=subprocess.STDOUT)
            ok = r.returncode == 0
            if ok and "probes=false" not in sets:
                from src.config import load_config, run_dir  # noqa: E402
                rd = run_dir(load_config(cfg, overrides))
                r = subprocess.run([sys.executable, "-m", "src.probes", "--run_dir", str(rd)],
                                   env=e, stdout=log, stderr=subprocess.STDOUT)
                ok = r.returncode == 0
        status.append(("OK" if ok else "FAILED", cfg, sets))
        print(f"[{(time.time() - T0) / 3600:.2f}h] gpu{gpu} {status[-1]}", flush=True)


sys.path.insert(0, str(code))
threads = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpu)]
for t in threads:
    t.start()
for t in threads:
    t.join()

(WORK / "job_status.txt").write_text("\n".join(f"{s}\t{c}\t{' '.join(o)}" for s, c, o in status))
print("\n".join(map(str, status)))
shutil.rmtree(code, ignore_errors=True)  # don't ship the code copy back as output
