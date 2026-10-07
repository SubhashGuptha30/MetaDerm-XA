"""
Task 3: Attention Faithfulness Validation.

Perturbation-based test: if the cross-attention weights are meaningful,
zeroing out high-attention spatial regions should reduce P(melanoma) more
than zeroing out random regions.

Usage:
    python src/xai/attention_faithfulness.py \
        --config configs/isic2019/field_token.yaml \
        --checkpoint checkpoints/field_token_isic2019_best.pth
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataloader import get_dataloaders
from src.train import build_model
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def run_faithfulness_test(model, dataloader, device, n_samples=500, top_k_pct=0.2):
    """
    For each sample:
    1. Get attention weights and baseline probability
    2. Mask the top-k% most-attended spatial positions → get prob_high
    3. Mask random-k% spatial positions → get prob_random
    4. If attention is faithful: prob_drop_high > prob_drop_random
    """
    model.eval()
    
    results = {
        'base_probs': [], 'high_attn_probs': [], 'random_probs': [],
        'high_attn_drop': [], 'random_drop': [],
    }
    
    n_processed = 0
    k = int(49 * top_k_pct)  # Number of spatial positions to mask
    
    np.random.seed(42)
    
    for batch in tqdm(dataloader, desc="Faithfulness test"):
        if n_processed >= n_samples:
            break
            
        images, meta_cont, site_idx, labels, _ = batch
        
        # Only melanoma-positive cases
        mel_mask = labels == 1
        if mel_mask.sum() == 0:
            continue
            
        images = images[mel_mask].to(device)
        meta_cont = meta_cont[mel_mask].to(device)
        site_idx = site_idx[mel_mask].to(device)
        
        remaining = n_samples - n_processed
        if images.shape[0] > remaining:
            images = images[:remaining]
            meta_cont = meta_cont[:remaining]
            site_idx = site_idx[:remaining]
        
        B = images.shape[0]
        
        with torch.no_grad():
            # 1. Full forward pass to get baseline probs
            logits = model(images, meta_cont, site_idx)
            base_probs = torch.sigmoid(logits)  # (B,)
            
            # 2. Get attention weights from the stored property
            attn_weights = model.cross_attn_weights  # (B, H, N=49, K=4)
            # Average over heads, sum over metadata tokens → per-position importance
            attn_importance = attn_weights.mean(dim=1).sum(dim=-1)  # (B, 49)
            
            # 3. Get intermediate representations for masked forward passes
            spatial_tokens = model.vision_backbone(images)  # (B, 49, 256)
            meta_tokens = model.meta_encoder(meta_cont, site_idx)  # (B, 4, 256)
        
        for b in range(B):
            if n_processed >= n_samples:
                break
            
            imp = attn_importance[b].cpu().numpy()
            
            # Top-k most attended positions
            top_k_idx = np.argsort(imp)[-k:]
            # Random-k positions
            random_idx = np.random.choice(49, size=k, replace=False)
            
            with torch.no_grad():
                # Masked forward — high attention positions zeroed
                masked_high = spatial_tokens[b:b+1].clone()
                masked_high[0, top_k_idx] = 0.0
                fused_high = model.cross_attention(masked_high, meta_tokens[b:b+1])
                logit_high = model.head(fused_high)
                prob_high = torch.sigmoid(logit_high.squeeze()).item()
                
                # Masked forward — random positions zeroed
                masked_rand = spatial_tokens[b:b+1].clone()
                masked_rand[0, random_idx] = 0.0
                fused_rand = model.cross_attention(masked_rand, meta_tokens[b:b+1])
                logit_rand = model.head(fused_rand)
                prob_rand = torch.sigmoid(logit_rand.squeeze()).item()
            
            base_p = base_probs[b].item()
            
            results['base_probs'].append(base_p)
            results['high_attn_probs'].append(prob_high)
            results['random_probs'].append(prob_rand)
            results['high_attn_drop'].append(base_p - prob_high)
            results['random_drop'].append(base_p - prob_rand)
            
            n_processed += 1
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--top_k_pct", type=float, default=0.2)
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    log.info(f"Loaded checkpoint: {args.checkpoint}")
    
    # Get test dataloader
    loaders = get_dataloaders(cfg)
    test_loader = loaders['test']
    
    # Run faithfulness test
    results = run_faithfulness_test(
        model, test_loader, device, 
        n_samples=args.n_samples, top_k_pct=args.top_k_pct
    )
    
    # Analyze results
    high_drops = np.array(results['high_attn_drop'])
    rand_drops = np.array(results['random_drop'])
    
    mean_high_drop = high_drops.mean()
    mean_rand_drop = rand_drops.mean()
    
    # Paired comparison: how often does masking high-attention hurt more?
    faithful_pct = (high_drops > rand_drops).mean() * 100
    
    # Wilcoxon signed-rank test
    from scipy.stats import wilcoxon
    try:
        stat, p_value = wilcoxon(high_drops, rand_drops, alternative='greater')
    except ValueError:
        stat, p_value = 0, 1.0
    
    # Effect size (Cohen's d for paired data)
    diff = high_drops - rand_drops
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
    
    k = int(49 * args.top_k_pct)
    
    log.info("=" * 60)
    log.info("ATTENTION FAITHFULNESS RESULTS")
    log.info("=" * 60)
    log.info(f"Samples tested: {len(high_drops)}")
    log.info(f"Top-{int(args.top_k_pct*100)}% masking ({k} of 49 positions)")
    log.info(f"Mean prob drop (high-attention masking): {mean_high_drop:.4f}")
    log.info(f"Mean prob drop (random masking):         {mean_rand_drop:.4f}")
    log.info(f"Ratio: {mean_high_drop/mean_rand_drop:.2f}x" if mean_rand_drop != 0 else "Ratio: inf")
    log.info(f"Faithful direction: {faithful_pct:.1f}% of cases")
    log.info(f"Wilcoxon p-value: {p_value:.6f}")
    log.info(f"Cohen's d: {cohens_d:.3f}")
    
    # Save results
    out_dir = Path(cfg.get("reports_dir", "reports")) / "attention_faithfulness"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    k = int(49 * args.top_k_pct)
    summary = {
        'n_samples': len(high_drops),
        'top_k_pct': args.top_k_pct,
        'k_positions': k,
        'mean_base_prob': float(np.mean(results['base_probs'])),
        'mean_high_attn_drop': float(mean_high_drop),
        'mean_random_drop': float(mean_rand_drop),
        'drop_ratio': float(mean_high_drop / mean_rand_drop) if mean_rand_drop != 0 else float('inf'),
        'faithful_pct': float(faithful_pct),
        'wilcoxon_stat': float(stat),
        'wilcoxon_p': float(p_value),
        'cohens_d': float(cohens_d),
    }
    
    with open(out_dir / 'faithfulness_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save raw data for potential reanalysis
    np.savez(
        out_dir / 'faithfulness_raw.npz',
        base_probs=np.array(results['base_probs']),
        high_attn_probs=np.array(results['high_attn_probs']),
        random_probs=np.array(results['random_probs']),
    )
    
    log.info(f"✓ Saved to {out_dir}")


if __name__ == "__main__":
    main()
