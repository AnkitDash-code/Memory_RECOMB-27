"""Standardized evaluation protocol for every architecture in the Master Rerun Plan.

One identical, leakage-safe protocol for DLPFC and breast cancer, so every
number in the final table means the same thing. Exports:

* DLPFC: SELECTION_SLICES (3), REPORT_SLICES (8), TUNING_EXCLUDED ("151673")
* Breast cancer: SELECTION_BLOCKS (2), REPORT_BLOCKS (4) via
  `breast_cancer_spatial_blocks`
* run_standard_protocol(arch_name, train_fn, hp_grid=None, ...) — the single
  entry point that:
    1. If hp_grid is non-empty: run hyperparameter selection on SELECTION slices/blocks
       and pick best config
    2. If hp_grid is empty or no tunable param: use the shared baseline config
    3. Evaluate final config on REPORT slices/blocks, 5 seeds each
    4. Write incrementally to JSON (per-fit, not only at end)
    5. Read back and verify the output file is well-formed before returning

Mandatory logging discipline (enforced here, not optional):
* Output JSON is written to disk after EACH individual fit completes.
  If a background job dies at fit 30 of 40, the partial result survives.
* After the full run, the script reloads its own output file from disk and
  asserts non-empty, correct schema, and all expected keys present.

File naming:
  outputs/logs/{architecture_name}_dlpfc_results.json
  outputs/logs/{architecture_name}_breast_cancer_results.json
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.data.load_breast_cancer import N_REGIONS as BC_N_REGIONS
from src.data.load_breast_cancer import load_breast_cancer
from src.data.load_dlpfc import ALL_DLPFC_SAMPLES, load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.eval.breast_cancer_spatial_blocks import (
    REPORT_BLOCKS as BC_REPORT_BLOCKS,
    SELECTION_BLOCKS as BC_SELECTION_BLOCKS,
    load_breast_cancer_blocks,
    spot_mask_for_blocks,
)
from src.eval.clustering import cluster_embedding, consensus_cluster

LOGS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "logs"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"

TUNING_EXCLUDED = "151673"
_DLPFC_SUBJECT1 = ["151507", "151508", "151509", "151510"]
_DLPFC_SUBJECT2 = ["151669", "151670", "151671", "151672"]
_DLPFC_SUBJECT3 = ["151673", "151674", "151675", "151676"]

SELECTION_SLICES = [
    _DLPFC_SUBJECT1[0],
    _DLPFC_SUBJECT2[0],
    _DLPFC_SUBJECT3[1],
]
assert TUNING_EXCLUDED not in SELECTION_SLICES, "151673 must never be used for selection or report as-is"

REPORT_SLICES = [
    s for s in ALL_DLPFC_SAMPLES
    if s not in SELECTION_SLICES and s != TUNING_EXCLUDED
]
assert len(SELECTION_SLICES) == 3
assert len(REPORT_SLICES) == 8
assert len(set(SELECTION_SLICES) & set(REPORT_SLICES)) == 0
assert TUNING_EXCLUDED not in REPORT_SLICES

SELECTION_SEEDS = [0, 1]
REPORT_SEEDS = [0, 1, 2, 3, 4]


@dataclass
class FitResult:
    dataset: str
    unit_id: str
    seed: int
    hp_config: dict
    ari_per_seed: float | None = None
    units_used: dict = field(default_factory=dict)
    training_history: list[dict] = field(default_factory=list)
    error: str | None = None


@dataclass
class UnitReport:
    unit_id: str
    n_spots: int
    n_classes: int
    per_seed_aris: list[float]
    mean: float
    std: float
    consensus_ari: float
    hp_config: dict


@dataclass
class ProtocolOutput:
    architecture: str
    dataset: str
    selection: dict = field(default_factory=dict)
    per_unit: list[UnitReport] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    hp_config_used: dict = field(default_factory=dict)


def _ari(truth: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    return float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))


def _safe_mean(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _safe_std(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.std(values)) if values else None


def _incremental_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _output_path(arch_name: str, dataset: str) -> Path:
    return LOGS_DIR / f"{arch_name}_{dataset}_results.json"


def _validate_output(path: Path, dataset: str) -> dict:
    if not path.exists():
        raise RuntimeError(f"Output file missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ["architecture", "dataset", "per_unit", "summary"]
    for k in required:
        if k not in data:
            raise RuntimeError(f"Output file {path} missing key: {k}")
    if not data["per_unit"]:
        raise RuntimeError(f"Output file {path} has empty per_unit list")
    expected_n = {"dlpfc": len(REPORT_SLICES), "breast_cancer": len(BC_REPORT_BLOCKS)}
    if len(data["per_unit"]) != expected_n.get(dataset, len(data["per_unit"])):
        raise RuntimeError(
            f"Output file {path} has {len(data['per_unit'])} units, "
            f"expected {expected_n.get(dataset)} for dataset={dataset}"
        )
    return data


def _iter_hp_configs(hp_grid: dict | None) -> list[dict]:
    if not hp_grid:
        return [{}]
    keys = sorted(hp_grid.keys())
    combos = list(itertools.product(*[hp_grid[k] for k in keys]))
    return [dict(zip(keys, vals)) for vals in combos]


def _load_and_prep_dlpfc(sample: str):
    raw = load_dlpfc_slice(sample)
    adata = preprocess_hvg(raw.copy())
    truth = adata.obs["ground_truth_layer"]
    mask = truth.notna().to_numpy()
    n_classes = int(truth.nunique())
    coords = adata.obsm["spatial"]
    return adata, truth.to_numpy(), mask, n_classes, coords


def _load_and_prep_breast_cancer():
    raw = load_breast_cancer()
    adata = preprocess_hvg(raw.copy(), platform="visium")
    truth = adata.obs["ground_truth_region"].to_numpy()
    block_id = load_breast_cancer_blocks(adata)
    coords = adata.obsm["spatial"]
    return adata, truth, block_id, BC_N_REGIONS, coords


def _evaluate_unit_dlpfc(
    arch_name: str,
    train_fn: Callable,
    sample: str,
    seeds: list[int],
    hp_config: dict,
    device,
    out_path: Path,
    running_state: dict,
    fit_tag: str = "report",
):
    adata, truth, mask, n_classes, coords = _load_and_prep_dlpfc(sample)
    labels_by_seed = []
    aris_by_seed = []
    for seed in seeds:
        try:
            _, trained, history = train_fn(
                adata.copy(), seed=seed, device=device, verbose=False, **hp_config
            )
            obsm_key = _guess_obsm_key(trained, arch_name)
            embedding = trained.obsm[obsm_key]
            labels = cluster_embedding(embedding, n_classes, coords=coords, refine=True)
            ari = _ari(truth, labels, mask)
        except Exception as exc:
            labels = None
            ari = None
            history = []
            fit = FitResult(
                dataset="dlpfc",
                unit_id=sample,
                seed=seed,
                hp_config=hp_config,
                error=f"{type(exc).__name__}: {exc}",
            )
            running_state.setdefault("failed_fits", []).append(asdict(fit))
            _incremental_write(out_path, running_state)
            raise

        labels_by_seed.append(labels)
        aris_by_seed.append(ari)
        fit = FitResult(
            dataset="dlpfc",
            unit_id=sample,
            seed=seed,
            hp_config=hp_config,
            ari_per_seed=ari,
            training_history=history,
        )
        running_state.setdefault(f"fits_{fit_tag}", []).append(asdict(fit))
        _incremental_write(out_path, running_state)
        time.sleep(3)

    consensus_labels = consensus_cluster(labels_by_seed, n_classes)
    consensus_ari = _ari(truth, consensus_labels, mask)
    return UnitReport(
        unit_id=sample,
        n_spots=int(adata.n_obs),
        n_classes=n_classes,
        per_seed_aris=aris_by_seed,
        mean=_safe_mean(aris_by_seed),
        std=_safe_std(aris_by_seed),
        consensus_ari=consensus_ari,
        hp_config=hp_config,
    )


def _evaluate_unit_breast_cancer(
    arch_name: str,
    train_fn: Callable,
    mask_blocks: tuple,
    seeds: list[int],
    hp_config: dict,
    device,
    out_path: Path,
    running_state: dict,
    fit_tag: str = "report",
):
    adata, truth, block_id, n_classes, coords = _load_and_prep_breast_cancer()
    unit_mask = spot_mask_for_blocks(block_id, mask_blocks)
    unit_name = "blocks_" + "_".join(str(b) for b in mask_blocks)

    labels_by_seed = []
    aris_by_seed = []
    for seed in seeds:
        try:
            _, trained, history = train_fn(
                adata.copy(), seed=seed, device=device, verbose=False, **hp_config
            )
            obsm_key = _guess_obsm_key(trained, arch_name)
            embedding_full = trained.obsm[obsm_key]
            labels_full = cluster_embedding(embedding_full, n_classes, coords=coords, refine=True)
            labels = np.asarray(labels_full)
            ari = _ari(truth, labels, unit_mask)
        except Exception as exc:
            labels = None
            ari = None
            history = []
            fit = FitResult(
                dataset="breast_cancer",
                unit_id=unit_name,
                seed=seed,
                hp_config=hp_config,
                error=f"{type(exc).__name__}: {exc}",
            )
            running_state.setdefault("failed_fits", []).append(asdict(fit))
            _incremental_write(out_path, running_state)
            raise

        labels_by_seed.append(labels)
        aris_by_seed.append(ari)
        fit = FitResult(
            dataset="breast_cancer",
            unit_id=unit_name,
            seed=seed,
            hp_config=hp_config,
            ari_per_seed=ari,
            training_history=history,
        )
        running_state.setdefault(f"fits_{fit_tag}", []).append(asdict(fit))
        _incremental_write(out_path, running_state)
        time.sleep(3)

    consensus_labels = consensus_cluster(labels_by_seed, n_classes)
    consensus_ari = _ari(truth, consensus_labels, unit_mask)
    return UnitReport(
        unit_id=unit_name,
        n_spots=int(unit_mask.sum()),
        n_classes=n_classes,
        per_seed_aris=aris_by_seed,
        mean=_safe_mean(aris_by_seed),
        std=_safe_std(aris_by_seed),
        consensus_ari=consensus_ari,
        hp_config=hp_config,
    )


def _guess_obsm_key(trained_adata, arch_name: str) -> str:
    candidates = [
        f"X_{arch_name}",
        "X_spatial_address",
        "X_ldcm",
        "X_ppr",
        "X_agap",
        "X_hma",
        "X_gmsm",
        "X_msap",
        "X_baap",
        "X_zism",
    ]
    for key in candidates:
        if key in trained_adata.obsm:
            return key
    for key in trained_adata.obsm.keys():
        if key.startswith("X_"):
            return key
    raise KeyError(f"No X_* embedding found in adata.obsm for arch={arch_name}. Keys: {list(trained_adata.obsm.keys())}")


def _selection_score(unit_reports: list[UnitReport]) -> float:
    return float(np.mean([r.consensus_ari for r in unit_reports]))


def run_standard_protocol(
    arch_name: str,
    train_fn: Callable,
    dataset: str,
    hp_grid: dict | None = None,
    skip_selection: bool = False,
    default_hp: dict | None = None,
    device=None,
) -> Path:
    """Run the Master Rerun Plan standard protocol for one architecture × dataset.

    Parameters
    ----------
    arch_name : str
        Short architecture name, used for file naming (e.g. "baseline", "ldcm").
    train_fn : callable
        Signature: train_fn(adata_copy, seed=.., device=.., verbose=.., **hp) -> (model, trained_adata, history_list).
        The trained_adata must have an embedding in obsm under X_{arch_name} or a
        recognized fallback key.
    dataset : {"dlpfc", "breast_cancer"}
    hp_grid : dict[str, list] or None
        Hyperparameter grid for selection step. None or empty => no tunable param,
        skip selection, go straight to report with default_hp.
    skip_selection : bool
        If True, bypass selection entirely even if hp_grid is non-empty; uses
        default_hp directly. Useful for reruns where best hp is known.
    default_hp : dict or None
        Fallback config used when skipping selection or if hp_grid is empty.
        Shared baseline values (memory_slots=16, n_hops=4, lambda_usage=0.02,
        expression_weighted=True) are the correct defaults here.
    device : torch.device or None

    Returns
    -------
    output_path : Path
        Location of the verified JSON output file.
    """
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset.lower()
    if dataset not in ("dlpfc", "breast_cancer"):
        raise ValueError(f"Unknown dataset: {dataset}")

    out_path = _output_path(arch_name, dataset)
    state: dict[str, Any] = {"architecture": arch_name, "dataset": dataset}
    _incremental_write(out_path, state)

    hp_configs = _iter_hp_configs(hp_grid)
    best_hp = dict(default_hp or {})

    if hp_grid and not skip_selection:
        selection_reports_by_cfg: list[tuple[dict, list[UnitReport]]] = []
        state["selection"] = {"hp_grid": hp_grid, "candidates": []}
        for cfg in hp_configs:
            cfg_reports = []
            if dataset == "dlpfc":
                for sample in SELECTION_SLICES:
                    rep = _evaluate_unit_dlpfc(
                        arch_name, train_fn, sample, SELECTION_SEEDS, cfg,
                        device, out_path, state, fit_tag="selection",
                    )
                    cfg_reports.append(rep)
            else:
                for blk in BC_SELECTION_BLOCKS:
                    rep = _evaluate_unit_breast_cancer(
                        arch_name, train_fn, (blk,), SELECTION_SEEDS, cfg,
                        device, out_path, state, fit_tag="selection",
                    )
                    cfg_reports.append(rep)
            score = _selection_score(cfg_reports)
            state["selection"]["candidates"].append({
                "hp_config": cfg,
                "score": score,
                "units": [asdict(r) for r in cfg_reports],
            })
            _incremental_write(out_path, state)
            selection_reports_by_cfg.append((cfg, cfg_reports))
        best_pair = max(selection_reports_by_cfg, key=lambda x: _selection_score(x[1]))
        best_hp = dict(best_pair[0])
        state["selection"]["best_hp"] = best_hp
        state["selection"]["best_score"] = _selection_score(best_pair[1])
        _incremental_write(out_path, state)
    elif default_hp:
        best_hp = dict(default_hp)

    state["hp_config_used"] = best_hp
    _incremental_write(out_path, state)

    report_units = []
    if dataset == "dlpfc":
        for sample in REPORT_SLICES:
            rep = _evaluate_unit_dlpfc(
                arch_name, train_fn, sample, REPORT_SEEDS, best_hp,
                device, out_path, state, fit_tag="report",
            )
            report_units.append(rep)
            state["per_unit"] = [asdict(r) for r in report_units]
            state["summary"] = _build_summary(report_units)
            _incremental_write(out_path, state)
    else:
        for blk in BC_REPORT_BLOCKS:
            rep = _evaluate_unit_breast_cancer(
                arch_name, train_fn, (blk,), REPORT_SEEDS, best_hp,
                device, out_path, state, fit_tag="report",
            )
            report_units.append(rep)
            state["per_unit"] = [asdict(r) for r in report_units]
            state["summary"] = _build_summary(report_units)
            _incremental_write(out_path, state)

    _validate_output(out_path, dataset)
    return out_path


def _build_summary(report_units: list[UnitReport]) -> dict:
    per_seed_means = [r.mean for r in report_units if r.mean is not None]
    consensus_aris = [r.consensus_ari for r in report_units]
    return {
        "n_units": len(report_units),
        "per_seed_mean": float(np.mean(per_seed_means)) if per_seed_means else None,
        "per_seed_std": float(np.std(per_seed_means)) if per_seed_means else None,
        "consensus_mean": float(np.mean(consensus_aris)),
        "consensus_std": float(np.std(consensus_aris)),
    }
