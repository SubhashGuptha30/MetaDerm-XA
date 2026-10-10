# v2 experimental protocol

1. **Splits** (`splits/v2/`, built by `python -m src.data.preprocess_v2`): ISIC 2019 training set split
   by `lesion_id` into train / val / calib (≈71/14/14). ISIC 2019 official test = `test`; ISIC 2020
   training set = `external`. Missing age imputed with the *train* median.
2. **Roles.** train → fit weights. val → early stopping and learning-rate selection.
   calib → decision thresholds and post-hoc calibration. test / external → reported only.
3. **Stage 1 – LR selection** (`kaggle_jobs/jobs/tune_lr.txt`): lr ∈ {5e-5, 1.5e-4, 4.5e-4}, seed 42,
   per main model; the lr with the highest best-epoch val AUROC is chosen. Only val predictions exist.
   The K-ablations reuse the field-token lr. Everything else is fixed in `base.yaml`.
4. **Stage 2 – final runs**: every model × seeds {42, 123, 456, 789, 2024}, then `src.probes`.
5. **Analysis**: `python -m analysis.report` (CPU) → `reports/v2/*.csv`, `paper/generated/*.tex`.
