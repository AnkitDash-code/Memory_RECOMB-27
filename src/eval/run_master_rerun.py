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
import json
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
from src.models.train_heterogeneity_gated_model import train_heterogeneity_gated_model
from src.models.train_hma_model import train_hma_model
from src.models.train_ldcm_model import train_ldcm_model
from src.models.train_loss_free_gated_model import train_loss_free_gated_model
from src.models.train_msap_model import train_msap_model
from src.models.train_ppr_model import train_ppr_model
from src.models.train_simvq_model import train_simvq_model
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
        # train_agap_model has no expression_weighted param (it always builds
        # its own edge-index graph via connectivities_to_edge_index) -- unlike
        # every other architecture here, so it can't inherit SHARED_BASELINE_HP
        # wholesale. Confirmed via signature inspection of all 8 train_fns
        # before this was caught the hard way (TypeError, no wasted GPU time).
        "default_hp": {k: v for k, v in SHARED_BASELINE_HP.items() if k != "expression_weighted"},
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
        # DLPFC was False here on the premise that run_dlpfc_multislice.py's
        # 11-slice held-out result (0.5621 consensus) already covers baseline
        # on DLPFC. It does, but not under the SAME slice split standard_
        # protocol.py uses for other architectures here (a nested selection/
        # report split, not "exclude 151673 only") -- so AGAP's DLPFC number
        # (or any other architecture's) had nothing directly comparable.
        # Flipped to True so baseline gets a real number under the identical
        # split; the 11-slice number remains the headline, this is for
        # apples-to-apples comparison against other architectures only.
        "dlpfc": True,
        "breast_cancer": True,
        "hp_grid": {},
        "default_hp": dict(SHARED_BASELINE_HP),
    },
    {
        # Stage 1 of the domain-scale-vs-propagation-depth follow-up plan
        # (see PROGRESS.md, session dated 2026-08-27/28): gate propagation
        # depth by a FIXED, externally-precomputed heterogeneity statistic
        # (ClustSIGNAL-style) instead of any learned gate -- every learned
        # gate tried so far (adaptive_hops, entropy_gated_propagation)
        # collapsed toward the easiest-to-reconstruct behavior. No hp_grid:
        # deliberately uses the same n_hops=4/reference default as baseline,
        # since the whole point is a drop-in replacement for the propagation
        # step, not a new hyperparameter to tune.
        "name": "heterogeneity_gated",
        "train_fn": train_heterogeneity_gated_model,
        "dlpfc": True,
        "breast_cancer": True,
        "hp_grid": {},
        "default_hp": dict(SHARED_BASELINE_HP),
    },
    {
        # Stage 2: SimVQ-style codebook reparameterization (see
        # src/models/simvq_layer.py docstring for the calibration check --
        # this project's dense softmax addressing already shows no dead-
        # codebook-entry problem in every baseline fit logged tonight, so
        # this is a correctly-calibrated null test, not an expected win).
        "name": "simvq",
        "train_fn": train_simvq_model,
        "dlpfc": True,
        "breast_cancer": True,
        "hp_grid": {},
        "default_hp": dict(SHARED_BASELINE_HP),
    },
    {
        # Stage 3: adaptive per-spot depth gate, balanced via loss-free bias
        # updates (DeepSeek arXiv 2408.15664) instead of an auxiliary loss --
        # only reached because Stage 1's fixed monotone map failed its DLPFC
        # threshold. See loss_free_gated_layer.py for why this differs from
        # Phase D's already-rejected adaptive_hops (that one's balancing
        # signal, when present at all, went through backprop; this one never
        # does).
        "name": "loss_free_gated",
        "train_fn": train_loss_free_gated_model,
        "dlpfc": True,
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
                # Read back the file we just supposedly wrote and verify it's
                # real before declaring success -- this project has lost
                # results before (LDCM, ZISM's breast-cancer run) to trusting
                # an in-memory "it finished" without checking the artifact on
                # disk actually exists and is well-formed.
                if not out.exists():
                    raise RuntimeError(f"{spec['name']}/{ds} reported done but {out} does not exist")
                verify_payload = json.loads(out.read_text(encoding="utf-8"))
                if not verify_payload or "summary" not in verify_payload:
                    raise RuntimeError(
                        f"{spec['name']}/{ds} wrote {out} but it is empty or missing 'summary': "
                        f"keys={list(verify_payload.keys())}"
                    )
                any_done = True
                print(f"DONE: {spec['name']}/{ds} -> {out.name} (verified on disk, summary={verify_payload['summary']})")
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
