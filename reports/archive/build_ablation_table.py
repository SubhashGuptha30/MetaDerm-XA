"""
Ablation table aggregator — Phase 6/8.
Reads all *_metrics.json files from reports/results/ and compiles the ablation table.

Run after all 4 models have been evaluated:
    python reports/build_ablation_table.py
"""

import json
import time
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("reports/results")
MODEL_ORDER = ["baseline", "concat_fusion", "cross_attention", "quantum_fusion"]
MODEL_LABELS = {
    "baseline": "Image-Only Baseline",
    "concat_fusion": "Naive Concatenation Fusion",
    "cross_attention": "Cross-Attention Fusion (Ours)",
    "quantum_fusion": "Cross-Attn + Quantum Layer",
}
METRIC_COLS = ["auroc", "recall", "precision", "f1", "auroc_ci", "recall_ci"]


def load_metrics(model_name: str) -> dict:
    path = RESULTS_DIR / f"{model_name}_metrics.json"
    if not path.exists():
        return {m: "—" for m in METRIC_COLS}
    with open(path) as f:
        return json.load(f)


def build_ablation_table() -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        m = load_metrics(model_name)
        rows.append({
            "Model": MODEL_LABELS.get(model_name, model_name),
            "AUROC": f"{m.get('auroc', '—'):.4f}" if isinstance(m.get('auroc'), float) else m.get('auroc', '—'),
            "Recall": f"{m.get('recall', '—'):.4f}" if isinstance(m.get('recall'), float) else m.get('recall', '—'),
            "Precision": f"{m.get('precision', '—'):.4f}" if isinstance(m.get('precision'), float) else m.get('precision', '—'),
            "F1": f"{m.get('f1', '—'):.4f}" if isinstance(m.get('f1'), float) else m.get('f1', '—'),
            "AUROC 95% CI": m.get("auroc_ci", "—"),
            "Recall 95% CI": m.get("recall_ci", "—"),
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = build_ablation_table()
    print("\n=== Ablation Table ===")
    print(df.to_string(index=False))

    out_csv = RESULTS_DIR / "ablation_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

    # Also write markdown table for the report
    md_table = df.to_markdown(index=False)
    with open(RESULTS_DIR / "ablation_table.md", "w") as f:
        f.write("# Ablation Results\n\n")
        f.write(f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write(md_table)
    print(f"Markdown saved to {RESULTS_DIR}/ablation_table.md")
