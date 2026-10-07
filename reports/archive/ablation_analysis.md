# Ablation Analysis
## Phase 6 — Quantum-Inspired Layer as Architectural Variant

*Fill in real metric values from `reports/results/*_metrics.json` after training all 4 models.*

---

## Results Summary

*(Populated after training — see `reports/results/ablation_table.md` for exact numbers)*

| Model | AUROC | Recall | Precision | F1 | Notes |
|---|---|---|---|---|---|
| Image-only Baseline | — | — | — | — | EfficientNet-B2, no metadata |
| Naive Concatenation Fusion | — | — | — | — | Flat concat of image+meta vectors |
| **Cross-Attention Fusion** | — | — | — | — | **Core contribution** |
| Cross-Attn + Quantum Layer | — | — | — | — | Only classifier head differs |

---

## Interpretation Framework

### If Cross-Attention > Baseline and Concat:
The structured, token-level conditioning via cross-attention adds measurable value over simply appending 
metadata to a pooled feature vector. This validates the architectural choice and supports the thesis 
claim that *how* modalities are fused matters, not just *whether* they are fused.

### If Quantum Layer ≈ Cross-Attention:
A null result is scientifically valid and defensible. The honest conclusion is:  
> "The quantum-inspired layer does not significantly improve over the standard classifier head under 
> these conditions. Its parameter count is identical, training stability is comparable, and the 
> amplitude-phase decomposition did not provide measurable benefit for this task. We include it as a 
> rigorous ablation to avoid overclaiming."

This is preferable to cherry-picking results or overstating the quantum layer's contribution.

### If Quantum Layer > Cross-Attention:
Report with caution — verify the improvement holds across multiple seeds. Run a DeLong test 
(`src/utils/metrics.py: delong_test`) and report p-value. If p < 0.05, the difference is 
statistically meaningful.

---

## Panel Defense Notes

**Anticipated question:** *"Why use a quantum-inspired layer? Does it actually help?"*

**Answer template:**
> "We evaluated the quantum-inspired layer as one architectural variant in a controlled ablation, 
> keeping all other components identical. The results show [it improved AUROC by X% / it performed 
> equivalently]. We treat this as an empirical finding rather than a theoretical claim — the layer's 
> amplitude-phase structure introduces a different inductive bias compared to standard linear layers, 
> but whether this translates to better performance is data-dependent. The cross-attention fusion 
> mechanism, not the quantum layer, is our primary architectural contribution."

---

## Reproducibility

All four models trained with:
- Seed: `42` (fixed via `src/utils/seed.py`)  
- Same patient-level split (`data/processed/splits.csv`)  
- Same YAML config structure (`configs/*.yaml`)  
- Same backbone: EfficientNet-B2 (pretrained ImageNet)  
- Same optimizer: AdamW, CosineAnnealingLR  
- Same imbalance handling: `pos_weight=55`, WeightedRandomSampler  

Only `model_type` differs between configs — this ensures a fair, single-variable comparison.
