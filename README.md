# MetaDerm-XA

Melanoma detection from dermoscopic images plus patient metadata (age, sex, anatomical site), evaluated
under dataset shift. The repository compares five ways of adding metadata to the same EfficientNet-B2
image model, trains every model with five seeds, and tests on three datasets: the ISIC 2019 test set
(same archive), ISIC 2020 (external dermoscopy) and PAD-UFES-20 (external smartphone photographs).

> Research code. Not a medical device and not for diagnostic use.

## Models

| Name | What metadata does |
|---|---|
| `baseline` | nothing (image only) |
| `baseline_wide` | nothing; head widened to match the fusion models' parameter count |
| `concat_fusion` | encoded metadata concatenated with pooled image features |
| `film` | metadata predicts per-channel scale/shift of the image feature map |
| `cross_attention` | one metadata token attends over the 7x7 image tokens |
| `field_token` | one token per field (age, sex, site, missingness) attends over image tokens |
| `field_token_k1/k2/k3` | ablations with 1-3 metadata tokens |

All models share the backbone, image size (224), augmentation, optimiser and early-stopping rule; only the
learning rate is tuned per model, on the validation split.

## Results (AUROC, mean ± SD over 5 seeds, duplicates removed)

| Model | ISIC 2019 test | ISIC 2020 | PAD-UFES-20 |
|---|---|---|---|
| Metadata only (logistic regression) | 0.613 | 0.650 | 0.321 |
| Image only | 0.888 ± 0.007 | 0.759 ± 0.012 | 0.700 ± 0.055 |
| Image only (wide) | 0.884 ± 0.007 | 0.749 ± 0.013 | 0.657 ± 0.029 |
| Concatenation | 0.888 ± 0.012 | 0.741 ± 0.023 | 0.740 ± 0.055 |
| Cross-attention (1 token) | 0.889 ± 0.005 | 0.767 ± 0.024 | 0.698 ± 0.082 |
| FiLM | 0.893 ± 0.006 | 0.751 ± 0.015 | 0.660 ± 0.040 |
| Field-token | 0.884 ± 0.005 | 0.727 ± 0.023 | 0.639 ± 0.110 |

No fusion model differs significantly from the image-only model after Holm correction. Full tables
(calibration, operating points, metadata probes, attention faithfulness, subgroups, threshold transfer)
are in [`reports/v2/`](reports/v2).

## Repository layout

```
src/data/preprocess_v2.py   build frozen splits (lesion-grouped, duplicate-aware) -> splits/v2/
src/data/                   dataset, metadata encoder, torchvision-v2 transforms, dataloaders
src/models/                 backbone + the fusion models (factory.build_model)
src/train.py                train one model/seed -> runs_v2/<model>/seed<k>/
src/probes.py               withheld/shuffled metadata, image-fixed sweeps, attention masking
src/xai/                    Grad-CAM and counterfactual sweeps for single images
analysis/                   statistics (DeLong, bootstrap, ECE, recalibration), tables, figures
configs/v2/                 base.yaml + one file per model; protocol in configs/v2/README.md
kaggle_jobs/                run training on Kaggle GPUs from a laptop
scripts/                    near-duplicate screen, image resizing, pretrained-weight download
app/streamlit_app.py        demo: prediction, Grad-CAM and metadata sweeps for an uploaded image
tests/                      pytest suite
legacy/v1/                  first-version preprocessing, kept for reference only
```

## Setup

```bash
pip install torch torchvision          # pick the CUDA/CPU build for your machine
pip install -r requirements.txt
python scripts/fetch_pretrained.py     # ImageNet EfficientNet-B2 weights -> pretrained/
```

Data (not included; download from the providers and accept their licences):

```
data/raw/isic2019/{train,test}/   images/ + ISIC_2019_{Training,Test}_{GroundTruth,Metadata}.csv
data/raw/isic2020/                images/ + train.csv
data/raw/PAD-UFES-20/             metadata.csv + images/unz/imgs_part_*/
```

## Reproducing

```bash
python -m src.data.preprocess_v2                 # splits/v2/*.csv (already committed)
python scripts/prepare_kaggle_images.py          # resized copies -> data/kaggle_upload/metaderm-images/
python -m src.train --config configs/v2/field_token.yaml --set seed=42 lr=4.5e-4
python -m src.probes --run_dir runs_v2/field_token/seed42
python -m analysis.report                        # reports/v2/*.csv
python -m analysis.figures
```

The learning rates used for the final runs (selected on validation) are 1.35e-3 for FiLM and 4.5e-4 for every
other model; `kaggle_jobs/jobs/final_{a,b}.txt` lists every final run with its exact overrides.

Training on Kaggle instead of a local GPU (needs a configured Kaggle API token):

```bash
python kaggle_jobs/launch.py code -m "update"    # upload src/, configs/, splits/ as a private dataset
python kaggle_jobs/launch.py push final_a        # run the jobs listed in kaggle_jobs/jobs/final_a.txt
python kaggle_jobs/launch.py status final_a
python kaggle_jobs/launch.py pull final_a        # merge outputs into runs_v2/
```

Demo app and single-image explanations (after training or pulling runs):

```bash
streamlit run app/streamlit_app.py
python -m src.xai.explain --checkpoint runs_v2/cross_attention/seed42/model.pt --n_cases 10
```

## What changed from the first version

The first version selected hyperparameters on ISIC 2020 while also calling it the external test set, fitted
temperature scaling on the test set, reported the best of five seeds, applied class re-weighting twice
(sampler and loss weight), imputed missing age with each split's own median, and did not check for duplicate
images across splits and datasets. The v2 pipeline uses a separate calibration split, selects learning rates
on validation only, imputes with the training median, screens for near-duplicates (68 confirmed pairs,
excluded from evaluation), reports all five seeds with seed-aware statistics, and trains without
re-weighting. The v1 numbers should not be used.

## Datasets and licences

- ISIC 2019: <https://challenge.isic-archive.com/data/#2019> (CC BY-NC)
- ISIC 2020: <https://doi.org/10.34970/2020-DS01> (CC BY-NC 4.0)
- PAD-UFES-20: <https://doi.org/10.17632/zr7vgbcyr2.1> (CC BY 4.0)
