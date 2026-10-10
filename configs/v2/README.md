# v2 experimental protocol

1. **Splits** (`splits/v2/`, built by `python -m src.data.preprocess_v2`): ISIC 2019 training set split
   by `lesion_id` into train / val / calib (≈71/14/14). ISIC 2019 official test = `test`; ISIC 2020
   training set = `external`; PAD-UFES-20 = `pad`. Missing age imputed with the *train* median.
   Near-duplicates (`scripts/near_duplicates.py` → `splits/v2/duplicate_pairs.csv`) are merged into one
   group for splitting; evaluation images that duplicate a development image are flagged `dup_in_dev`
   and excluded by `analysis/collect.py`.
2. **Roles.** train → fit weights. val → early stopping and learning-rate selection.
   calib → decision thresholds and post-hoc calibration. test / external → reported only.
3. **Stage 1 – LR selection** (`kaggle_jobs/jobs/tune_lr*.txt`): lr ∈ {5e-5, 1.5e-4, 4.5e-4}, seed 42,
   per main model, extended by ×3 steps (1.35e-3, 4.05e-3) while the best value sat at the grid edge;
   the lr with the highest best-epoch val AUROC is chosen, and an lr that diverged in any attempt is
   ineligible (`analysis.report.sec_lr`). Result: FiLM 1.35e-3, all others 4.5e-4. Only val predictions exist.
   The K-ablations reuse the field-token lr. Everything else is fixed in `base.yaml`.
4. **Stage 2 – final runs**: every model × seeds {42, 123, 456, 789, 2024}, then `src.probes`.
5. **Analysis**: `python -m analysis.report` (CPU) → `reports/v2/*.csv`, `paper/generated/*.tex`.
