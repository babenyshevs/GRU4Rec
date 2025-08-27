"""High-level helpers for configuration-driven workflows.

This module reads YAML files, builds models, splits data and triggers
evaluation to streamline typical GRU4Rec experiments.
"""

import yaml
from .gru4rec import GRU4Rec
from .data_loader import train_valid_test_split
from .evaluation import batch_eval


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> GRU4Rec:
    """Instantiate :class:`GRU4Rec` from ``cfg``."""
    model_cfg = cfg.get("model", {})
    return GRU4Rec(**model_cfg)


def split_data(df, cfg: dict):
    """Split a DataFrame into train/validation/test sets.

    Parameters are taken from both ``cfg['data']`` for column names and
    ``cfg['data_split']`` for the split fractions.
    """
    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("data_split", {})
    return train_valid_test_split(df, **split_cfg, **data_cfg)


def evaluate(gru: GRU4Rec, test_df, cfg: dict):
    """Evaluate ``gru`` on ``test_df`` using ``cfg`` parameters."""
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})
    return batch_eval(gru, test_df, **eval_cfg, **data_cfg)


def save_model(gru: GRU4Rec, cfg: dict) -> None:
    """Save ``gru`` to the path specified in ``cfg['paths']['model_save']``."""
    path = cfg.get("paths", {}).get("model_save")
    if path:
        gru.savemodel(path)


def load_model(cfg: dict) -> GRU4Rec:
    """Load a model from ``cfg['paths']['model_load']`` if provided."""
    paths = cfg.get("paths", {})
    load_path = paths.get("model_load")
    if not load_path:
        raise ValueError("No model_load path specified in configuration")
    device = cfg.get("model", {}).get("device", "cpu")
    return GRU4Rec.loadmodel(load_path, device=device)
