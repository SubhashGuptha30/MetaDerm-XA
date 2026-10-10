"""Download ImageNet weights once (locally) so Kaggle jobs need no internet.

    python scripts/fetch_pretrained.py            # -> pretrained/efficientnet_b2.pth
"""

import sys
from pathlib import Path

import timm
import torch

name = sys.argv[1] if len(sys.argv) > 1 else "efficientnet_b2"
out = Path("pretrained")
out.mkdir(exist_ok=True)
model = timm.create_model(name, pretrained=True)
torch.save(model.state_dict(), out / f"{name}.pth")
print(f"saved {out / name}.pth  ({model.pretrained_cfg.get('hf_hub_id', model.pretrained_cfg.get('tag'))})")
