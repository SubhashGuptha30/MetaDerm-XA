"""YAML configs with `_base_` inheritance and `key=value` command-line overrides."""

from pathlib import Path

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _coerce(v):
    """PyYAML (YAML 1.1) reads '1e-4' / '1.5e-4' as strings; turn numeric strings into floats."""
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    if isinstance(v, str):
        try:
            return float(v) if any(c in v.lower() for c in ".e") else v
        except ValueError:
            return v
    return v


def load_config(path, overrides=()) -> dict:
    return _coerce(_load(path, overrides))


def _load(path, overrides=()) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "_base_" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("_base_")), cfg)
    for item in overrides:
        k, v = item.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def run_dir(cfg: dict) -> Path:
    return Path(cfg.get("output_root", "runs_v2")) / cfg["model_name"] / f"seed{cfg['seed']}"
