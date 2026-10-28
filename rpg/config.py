"""Config loading: YAML with `extends:` deep-merge."""
from pathlib import Path

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if parent := cfg.pop("extends", None):
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def model_cfg(cfg: dict, name: str) -> dict:
    return {"name": name, **cfg["models"][name]}


def out_dir(cfg: dict, model_name: str | None = None) -> Path:
    p = Path(cfg["out_dir"])
    if model_name:
        p = p / model_name
    p.mkdir(parents=True, exist_ok=True)
    return p
