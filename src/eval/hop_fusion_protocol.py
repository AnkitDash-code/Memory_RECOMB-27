"""Shared protocol helpers for Hop-Fusion selection and generalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DEFAULT_SELECTION_PATH = Path(__file__).resolve().parents[2] / "configs" / "hop_fusion_selection.json"
DEFAULT_LOCK_PATH = Path(__file__).resolve().parents[2] / "configs" / "hop_fusion_dlpfc.json"


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_locked_hop_fusion_config(path=DEFAULT_LOCK_PATH):
    """Load only a config produced by the leakage-safe DLPFC selector."""
    config = load_json(path)
    if config.get("status") != "locked":
        raise RuntimeError(
            f"{path} is not a locked DLPFC-selected config; run "
            "cross_validate_hop_fusion.py before evaluating generalization"
        )
    required = ("physical_radius_um", "reference_max_hops", "fusion_hidden_dim", "fusion_depth")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"locked Hop-Fusion config is missing {missing}")
    return config


def contiguous_spatial_block_masks(coords, n_rows=2, n_cols=2):
    """Return contiguous quantile-grid train/validation masks.

    Quantile cut points keep blocks non-empty for rectangular tissue sections,
    unlike a fixed coordinate midpoint that can leave one quadrant empty.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must have shape (n_observations, >=2)")
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must be positive")

    row_bins = np.minimum(
        np.searchsorted(np.quantile(coords[:, 0], np.linspace(0, 1, n_rows + 1)[1:-1]), coords[:, 0]),
        n_rows - 1,
    )
    col_bins = np.minimum(
        np.searchsorted(np.quantile(coords[:, 1], np.linspace(0, 1, n_cols + 1)[1:-1]), coords[:, 1]),
        n_cols - 1,
    )
    block_ids = row_bins * n_cols + col_bins
    masks = {}
    for block_id in range(n_rows * n_cols):
        validation = block_ids == block_id
        if not validation.any() or validation.all():
            continue
        masks[f"block_{block_id}"] = {
            "train": ~validation,
            "validation": validation,
        }
    if not masks:
        raise ValueError("could not construct a non-empty contiguous spatial holdout")
    return masks
