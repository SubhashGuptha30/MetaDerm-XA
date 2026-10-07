"""
Subgroup / Fairness Analysis
=============================
Stratified performance evaluation across sex, age group, and anatomical site.
Uses existing predictions — no GPU needed.

Usage: python scripts/subgroup_analysis.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / 'reports' / 'isic2019_internal'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed_isic2019'
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'audit_analyses'
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = ['baseline', 'concat_fusion', 'cross_attention', 'field_token', 'film']
LABELS = {'baseline': 'Baseline', 'concat_fusion': 'Concat', 'cross_attention': 'Cross-Attn',
          'field_token': 'Field-Token', 'film': 'FiLM'}


def compute_metrics(y_true, y_prob, threshold):
    if len(np.unique(y_true)) < 2 or len(y_true) < 10:
        return {'auroc': float('nan'), 'auprc': float('nan'), 'recall': float('nan'),
                'specificity': float('nan'), 'n': len(y_true), 'n_pos': int(y_true.sum())}
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    preds = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    
    return {
        'auroc': auroc, 'auprc': auprc,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'n': len(y_true), 'n_pos': int(y_true.sum()),
    }


def main():
    # Load test metadata
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    
    # Create subgroups
    test_df['age_group'] = pd.cut(
        test_df['age_approx'], bins=[0, 30, 50, 70, 100],
        labels=['0-30', '31-50', '51-70', '71+'], right=True
    )
    test_df['sex_group'] = test_df['sex'].fillna('unknown').str.lower()
    test_df['site_group'] = test_df['anatom_site_general_challenge'].fillna('unknown').str.lower()
    
    print("=" * 70)
    print("SUBGROUP / FAIRNESS ANALYSIS")
    print("=" * 70)
    
    all_results = []
    
    for model_name in MODELS:
        labels = np.load(REPORT_DIR / f'{model_name}_isic2019_labels.npy')
        probs = np.load(REPORT_DIR / f'{model_name}_isic2019_probs.npy')
        metrics_json = json.loads((REPORT_DIR / f'{model_name}_isic2019_metrics.json').read_text())
        threshold = metrics_json.get('threshold', 0.5)
        
        label = LABELS[model_name]
        
        # Overall
        overall = compute_metrics(labels, probs, threshold)
        overall.update({'model': label, 'group_type': 'Overall', 'group': 'All'})
        all_results.append(overall)
        
        # By sex
        for sex_val in ['male', 'female', 'unknown']:
            mask = test_df['sex_group'].values == sex_val
            if mask.sum() == 0:
                continue
            m = compute_metrics(labels[mask], probs[mask], threshold)
            m.update({'model': label, 'group_type': 'Sex', 'group': sex_val.title()})
            all_results.append(m)
        
        # By age group
        for age_val in ['0-30', '31-50', '51-70', '71+']:
            mask = test_df['age_group'].values == age_val
            if mask.sum() == 0:
                continue
            m = compute_metrics(labels[mask], probs[mask], threshold)
            m.update({'model': label, 'group_type': 'Age', 'group': age_val})
            all_results.append(m)
        
        # By site
        for site_val in test_df['site_group'].unique():
            mask = test_df['site_group'].values == site_val
            if mask.sum() < 20:
                continue
            m = compute_metrics(labels[mask], probs[mask], threshold)
            m.update({'model': label, 'group_type': 'Site', 'group': site_val.title()})
            all_results.append(m)
    
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / 'subgroup_analysis.csv', index=False)
    
    # Print summary tables
    for group_type in ['Sex', 'Age', 'Site']:
        print(f"\n{'='*70}")
        print(f"SUBGROUP: {group_type}")
        print(f"{'='*70}")
        
        subset = df[df['group_type'] == group_type]
        pivot_auroc = subset.pivot_table(index='group', columns='model', values='auroc')
        pivot_n = subset.pivot_table(index='group', columns='model', values='n')
        pivot_npos = subset.pivot_table(index='group', columns='model', values='n_pos')
        
        print(f"\nAUROC:")
        for group in pivot_auroc.index:
            row = pivot_auroc.loc[group]
            n = int(pivot_n.loc[group].iloc[0])
            npos = int(pivot_npos.loc[group].iloc[0])
            vals = '  '.join([f"{LABELS.get(m,m)}: {v:.3f}" for m, v in row.items()])
            print(f"  {group} (n={n}, pos={npos}): {vals}")
        
        # Max disparity
        for model_label in ['Baseline', 'Concat', 'Cross-Attn', 'Field-Token', 'FiLM']:
            model_subset = subset[subset['model'] == model_label]
            aurocs = model_subset['auroc'].dropna()
            if len(aurocs) > 1:
                disparity = aurocs.max() - aurocs.min()
                if disparity > 0.05:
                    print(f"  ⚠ {model_label} max AUROC disparity: {disparity:.3f}")
    
    # LaTeX table for sex subgroup (most important for the paper)
    print(f"\n{'='*70}")
    print("LATEX: Sex Subgroup Table")
    print(f"{'='*70}")
    
    sex_df = df[df['group_type'] == 'Sex']
    print(r"""
\begin{table}[htbp]
\centering
\caption{Subgroup performance by biological sex on the ISIC 2019 test set. AUROC and recall at the model-specific Youden threshold. $n$: subgroup size; $n_+$: melanoma cases.}
\label{tab:subgroup_sex}
\small
\begin{tabular}{llcccc}
\toprule
Model & Sex & $n$ & $n_+$ & AUROC & Recall \\
\midrule""")
    
    for model_label in ['Baseline', 'Concat', 'Cross-Attn', 'Field-Token', 'FiLM']:
        for sex in ['Male', 'Female', 'Unknown']:
            row = sex_df[(sex_df['model'] == model_label) & (sex_df['group'] == sex)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            suffix = r'$^\dagger$' if model_label == 'Field-Token' else ''
            auroc_str = f"{r['auroc']:.3f}" if not np.isnan(r['auroc']) else '---'
            recall_str = f"{r['recall']:.3f}" if not np.isnan(r['recall']) else '---'
            print(f"{model_label}{suffix} & {sex} & {int(r['n'])} & {int(r['n_pos'])} & {auroc_str} & {recall_str} \\\\")
        print(r"\addlinespace")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}""")
    
    print(f"\nResults saved to {OUTPUT_DIR / 'subgroup_analysis.csv'}")


if __name__ == '__main__':
    main()
