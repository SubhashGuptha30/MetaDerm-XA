"""
Laptop-side control of Kaggle GPU jobs (all via the official `kaggle` CLI).

    python kaggle_jobs/launch.py code   -m "message"     # stage + upload/version the code dataset
    python kaggle_jobs/launch.py push   <jobfile>        # create kernel from kaggle_jobs/jobs/<jobfile>.txt and run it
    python kaggle_jobs/launch.py status <jobfile>
    python kaggle_jobs/launch.py pull   <jobfile>        # download outputs, merge into runs_v2/

Job files: one run per line — `<config> [key=value ...]`, '#' comments allowed.
Kaggle username is read from kaggle_jobs/kaggle_user.txt (one line).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "kaggle_jobs" / "jobs"
KERNELS_DIR = ROOT / "kaggle_jobs" / "kernels"
STAGE = ROOT / "data" / "kaggle_upload" / "metaderm-code"
CODE_ITEMS = ["src", "configs/v2", "splits/v2", "pretrained"]


def user() -> str:
    return (ROOT / "kaggle_jobs" / "kaggle_user.txt").read_text().strip()


def kaggle(*args, check=True):
    exe = Path(sys.executable).parent / ("kaggle.exe" if sys.platform == "win32" else "kaggle")
    cmd = [str(exe) if exe.exists() else "kaggle", *args]
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def cmd_code(msg: str):
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for item in CODE_ITEMS:
        src = ROOT / item
        dst = STAGE / item
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (STAGE / "METADERM_CODE").write_text("marker")
    meta = {"title": "metaderm-code", "id": f"{user()}/metaderm-code", "licenses": [{"name": "other"}]}
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    r = kaggle("datasets", "version", "-p", str(STAGE), "-m", msg, "--dir-mode", "zip", check=False)
    if r.returncode != 0:  # first time
        kaggle("datasets", "create", "-p", str(STAGE), "--dir-mode", "zip")


def parse_jobs(name: str):
    jobs = []
    for line in (JOBS_DIR / f"{name}.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            cfg, *sets = line.split()
            jobs.append([cfg, sets])
    return jobs


def cmd_push(name: str, budget_h: float, workers: int, accelerator: str):
    jobs = parse_jobs(name)
    kdir = KERNELS_DIR / name
    kdir.mkdir(parents=True, exist_ok=True)
    tpl = (ROOT / "kaggle_jobs" / "runner_template.py").read_text()
    code = (tpl.replace("__JOBS__", json.dumps(jobs))
               .replace("__BUDGET__", str(budget_h))
               .replace("__NUM_WORKERS__", str(workers)))
    (kdir / "runner.py").write_text(code)
    slug = f"metaderm-{name}".replace("_", "-").lower()
    meta = {
        "id": f"{user()}/{slug}",
        "title": slug,
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": accelerator,
        "dataset_sources": [f"{user()}/metaderm-code", f"{user()}/metaderm-images"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(jobs)} runs -> {meta['id']} ({accelerator}, budget {budget_h} h)")
    kaggle("kernels", "push", "-p", str(kdir))


def cmd_status(name: str):
    kaggle("kernels", "status", f"{user()}/metaderm-{name}".replace("_", "-").lower())


def cmd_pull(name: str):
    dest = ROOT / "kaggle_outputs" / name
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user()}/metaderm-{name}".replace("_", "-").lower(), "-p", str(dest))
    src = dest / "runs_v2"
    if src.exists():
        n = 0
        for f in src.rglob("*"):
            if f.is_file():
                t = ROOT / "runs_v2" / f.relative_to(src)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t)
                n += 1
        print(f"merged {n} files into runs_v2/")
    st = dest / "job_status.txt"
    print(st.read_text() if st.exists() else "no job_status.txt (job still running or crashed — see logs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["code", "push", "status", "pull"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("-m", "--message", default="update")
    ap.add_argument("--budget_h", type=float, default=10.5)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers per run")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4", help="Kaggle machine_shape")
    a = ap.parse_args()
    {"code": lambda: cmd_code(a.message),
     "push": lambda: cmd_push(a.name, a.budget_h, a.workers, a.accelerator),
     "status": lambda: cmd_status(a.name),
     "pull": lambda: cmd_pull(a.name)}[a.action]()
