"""Master Rerun Plan — All Proposed Architectures, One Standard Protocol.

Priority order (per plan):
  1. LDCM    (full, both datasets) — highest priority, genuinely untested live idea
  2. PPR     (full, both datasets, with alpha tuning on selection)
  3. AGAP    (full, both datasets)
  4. HMA     (full, both datasets)
  5. GMSM    (full, both datasets)
  6. MSAP    (BC only under block protocol; DLPFC only if BC looks promising)
  7. BAAP    (BC only under block protocol)
  8. BASELINE (BC only under block protocol; DLPFC result already trustworthy and is the
                cross-validated default — only the BC number is pre-block and needs rerun)
  9. ZISM    (BC only for closure if time; DLPFC result 0.235 is conclusive)

Usage:
  # Run everything in order (hours of compute on laptop GPU):
  uv run python -m src.eval.run_master_rerun

  # Run only a specific architecture for debug:
  uv run python -m src.eval.run_master_rerun --only ldcm

  # Run only breast-cancer reruns (baseline + msap + baap + zism + ldcm+ppr BC halves):
  uv run python -m src.eval.run_master_rerun --dataset breast_cancer

After each architecture completes, the harness re-generates the master table and
figures so partial results are always visible, not waiting on the slowest run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.eval.build_master_table import main as build_master_table
from src.eval.generate_standard_figures import (
    generate_all_for_arch,
    generate_master_comparison,
)
from src.eval.standard_protocol import run_standard_protocol
from src.models.train_agap_model import train_agap_model
from src.models.train_baap_model import train_baap_model
from src.models.train_gmsm_model import train_gmsm_model
from src.models.train_hma_model import train_hma_model
from src.models.train_ldcm_model import train_ldcm_model
from src.models.train_msap_model import train_msap_model
from src.models.train_ppr_model import train_ppr_model
from src.models.train_spatial_address import train_spatial_address_model
from src.models.train_zism_model import train_zism_model

SHARED_BASELINE_HP = {
    "memory_slots": 16,
    "memory_dim": 128,
    "n_hops": 4,
    "lambda_usage": 0.02,
    "expression_weighted": True,
}

ARCH_SPECS = [
    {
        "name": "ldcm",
        "train_fn": train_ldcm_model,
        "dlpfc": True,
        "breast_cancer": True,
        "hp_grid": {
            "lambda_contrastive": [0.01, 0.1, 0.2],
            "temperature": [0.5, 1.0],
        },
        "default_hp": dict(SHARED_BASELINE_HP, lambda_contrastive=0.1, temperature=1.0),
    },
    {
        "name": "ppr",
        "train_fn": train_ppr_model,
        "dlpfc": True,
        "breast_cancer": True,
        "hp_grid": {"alpha": [0.05, 0.1, 0.2, 0.3, 0.5]},
        "default_hp": dict(SHARED_BASELINE_HP, alpha=0.2),
    },
    {
        "name": "agap",
        "train_fn": train_agap_model,
        "dlpfc": True,
        "breast_cancer": True,
        "hp_grid": {},
        "default_hp": dict(SHARED_BASELINE_HP),
    },
    {
        "name": "hma",
        "train_fn": train_hma_model,
        "dlpfc": True,
        "breast_cancer": True,
        "hp_grid": {"temperature": [0.5, 1.0]},
        "default_hp": dict(SHARED_BASELINE_HP),
    },
    {
        "name": "gmsm",
        "train_fn": train_gmsm_model,
        "dlpfc": True,
        "breast_cancer": True,
        "hp_grid": {},
        "default_hp": dict(SHARED_BASELINE_HP),
    },
    {
        "name": "msap",
        "train_fn": train_msap_model,
        "dlpfc": False,
        "breast_cancer": True,
        "hp_grid": {"max_hops": [4, 6], "lambda_usage": [0.02, 0.05]},
        "default_hp": {
            "memory_slots": 16,
            "memory_dim": 128,
            "max_hops": 6,
            "lambda_usage": 0.02,
            "expression_weighted": True,
        },
    },
    {
        "name": "baap",
        "train_fn": train_baap_model,
        "dlpfc": False,
        "breast_cancer": True,
        "hp_grid": {"max_hops": [4, 6]},
        "default_hp": {
            "memory_slots": 16,
            "memory_dim": 128,
            "max_hops": 6,
            "lambda_usage": 0.02,
            "expression_weighted": True,
        },
    },
    {
        "name": "baseline",
        "train_fn": train_spatial_address_model,
        "dlpfc": False,
        "breast_cancer": True,
        "hp_grid": {},
        "default_hp": dict(SHARED_BASELINE_HP),
    },
    {
        "name": "zism",
        "train_fn": train_zism_model,
        "dlpfc": False,
        "breast_cancer": True,
        "hp_grid": {"zero_inflated": [True, False]},
        "default_hp": dict(SHARED_BASELINE_HP),
    },
]


def _run_single(spec: dict, dataset: str, device) -> Path | None:
    if not spec[f"{dataset.replace('breast_cancer', 'breast_cancer')}"]:
        # the dict keys are "dlpfc" / "breast_cancer" booleans
        pass
    if dataset == "dlpfc" and not spec["dlpfc"]:
        return None
    if dataset == "breast_cancer" and not spec["breast_cancer"]:
        return None
    print(f"\n\n========== {spec['name']} / {dataset.upper()} ==========", flush=True)
    try:
        return run_standard_protocol(
            arch_name=spec["name"],
            train_fn=spec["train_fn"],
            dataset=dataset,
            hp_grid=spec.get("hp_grid") or None,
            default_hp=spec["default_hp"],
            device=device,
        )
    except Exception as exc:
        print(f"[ERROR] {spec['name']}/{dataset} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this architecture name (e.g. ldcm, ppr)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["dlpfc", "breast_cancer"],
                        help="Run only this dataset across all applicable architectures")
    parser.add_argument("--skip-dlpfc", action="store_true")
    parser.add_argument("--skip-breast-cancer", action="store_true")
    parser.add_argument("--skip-table", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Master Rerun — using device: {device}", flush=True)

    specs = ARCH_SPECS
    if args.only:
        specs = [s for s in specs if s["name"] == args.only]
        if not specs:
            raise SystemExit(f"Unknown --only={args.only!r}. Known: {[s['name'] for s in ARCH_SPECS]}")

    datasets = []
    if not args.skip_dlpfc and (args.dataset is None or args.dataset == "dlpfc"):
        datasets.append("dlpfc")
    if not args.skip_breast_cancer and (args.dataset is None or args.dataset == "breast_cancer"):
        datasets.append("breast_cancer")
    if not datasets:
        raise SystemExit("No datasets to run (both skipped).")

    any_done = False
    for spec in specs:
        for ds in datasets:
            out = _run_single(spec, ds, device)
            if out is not None:
                any_done = True
                print(f"DONE: {spec['name']}/{ds} -> {out.name}")
                if not args.skip_table:
                    try:
                        build_master_table()
                    except Exception as exc:
                        print(f"[warn] build_master_table failed: {exc}")
                if not args.skip_figures:
                    try:
                        for p in generate_all_for_arch(spec["name"]):
                            print(f"  figure: {p.name}")
                    except Exception as exc:
                        print(f"[warn] figure gen for {spec['name']} failed: {exc}")

    if any_done and not args.skip_figures:
        try:
            p = generate_master_comparison()
            print(f"master comparison -> {p.name}")
        except Exception as exc:
            print(f"[warn] master comparison figure failed: {exc}")

    if not args.skip_table:
        try:
            build_master_table()
            print("master_results_table.md regenerated")
        except Exception as exc:
            print(f"[warn] final build_master_table failed: {exc}")

    print("\n=== Master Rerun complete ===")


if __name__ == "__main__":
    main()
