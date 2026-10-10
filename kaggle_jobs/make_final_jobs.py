"""Write the stage-2 job lists from the stage-1 learning-rate selection.

    .venv/Scripts/python kaggle_jobs/make_final_jobs.py     # -> kaggle_jobs/jobs/final_a.txt, final_b.txt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.report import sec_lr  # noqa: E402

SEEDS = [42, 123, 456, 789, 2024]
MAIN = ["baseline", "baseline_wide", "concat_fusion", "cross_attention", "film", "field_token"]
ABLATIONS = ["field_token_k1", "field_token_k2", "field_token_k3"]  # reuse the field-token lr

best = sec_lr()
missing = [m for m in MAIN if m not in best]
assert not missing, f"no tuning result for {missing}"
lines = [f"configs/v2/{m}.yaml seed={s} lr={best[m]:g}" for s in SEEDS for m in MAIN]
lines += [f"configs/v2/{m}.yaml seed={s} lr={best['field_token']:g}" for s in SEEDS for m in ABLATIONS]
half = (len(lines) + 1) // 2
for name, chunk in (("final_a", lines[:half]), ("final_b", lines[half:])):
    (ROOT / "kaggle_jobs" / "jobs" / f"{name}.txt").write_text(
        f"# Stage 2 ({len(chunk)} runs). lr per model from stage-1 validation AUROC: {best}\n" + "\n".join(chunk) + "\n")
    print(name, len(chunk), "runs")
