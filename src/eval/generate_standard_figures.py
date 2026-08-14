"""Standardized figure generation for all architectures.

For every architecture, produces the same set of figures so they're comparable:
  1. Address entropy by hop (line chart, mean ± std across seeds)
     – same diagnostic used for Hop-Fusion's sharpening fix. Flags collapse or
       diffuseness at a glance.
  2. Per-slice / per-block ARI bar chart with error bars
     – DLPFC (8 report slices) + breast cancer (4 report blocks) as two panels
       of one figure per architecture.
  3. Training curve (reconstruction loss + any arch-specific loss term vs. epoch)
     – one seed is enough, sanity/convergence check.

Additionally, after all reruns complete:
  4. Master comparison figure: grouped bar chart, all architectures ×
     {DLPFC consensus, BC consensus}, baseline + GraphST as reference lines.

All figures are saved to outputs/figures/{architecture_name}_*.png and
generated programmatically from saved JSON logs — never hand-assembled, so
no transcription risk between the log and the figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"
LOGS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "logs"

REPORT_SLICES = [
    "151508", "151509", "151510",
    "151670", "151671", "151672",
    "151674", "151675", "151676",
][:8]


def _load_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _get_cmap(n):
    try:
        cmap = plt.colormaps["tab10"]
        return cmap.resampled(n)
    except (AttributeError, KeyError):
        from matplotlib import cm
        return cm.get_cmap("tab10", n)


def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_training_curve_from_history(histories: list[list[dict]], arch_name: str, dataset: str) -> Path:
    """Plot reconstruction loss + arch-specific loss vs epoch (1 seed sufficient)."""
    if not histories:
        raise ValueError("No training histories provided")
    h = histories[0]
    epochs = [row.get("epoch", i) for i, row in enumerate(h)]
    fig, ax = plt.subplots(figsize=(7, 4))
    keys_blacklist = {"epoch", "max_entropy", "median_row_entropy", "usage_entropy",
                      "local_usage_entropy", "global_usage_entropy", "local_row_entropy",
                      "global_row_entropy", "n_slots_used", "local_slots_used", "global_slots_used",
                      "key_cosine_similarity", "mean_gate", "alpha", "attractor_beta", "total_loss"}
    plotted = False
    for key in sorted(h[0].keys()):
        if key in keys_blacklist:
            continue
        values = [row[key] for row in h]
        if not all(isinstance(v, (int, float)) for v in values):
            continue
        label = key
        if key == "recon_loss":
            ax.plot(epochs, values, label="reconstruction loss", linewidth=2, color="black")
        else:
            ax.plot(epochs, values, label=label, alpha=0.75)
        plotted = True
    if not plotted:
        ax.plot(epochs, [row.get("total_loss", row.get("recon_loss", 0)) for row in h], label="total loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{arch_name} — training curve ({dataset}, seed 0)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, f"{arch_name}_{dataset}_training_curve.png")


def plot_per_unit_ari(per_units_dlpfc: list[dict] | None, per_units_bc: list[dict] | None, arch_name: str) -> Path:
    """Per-slice / per-block ARI bar chart with error bars (DLPFC + BC side-by-side)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cmap = _get_cmap(2)

    if per_units_dlpfc:
        ids = [u["unit_id"] for u in per_units_dlpfc]
        means = [u.get("mean") or 0.0 for u in per_units_dlpfc]
        stds = [u.get("std") or 0.0 for u in per_units_dlpfc]
        consensus = [u.get("consensus_ari") for u in per_units_dlpfc]
        x = np.arange(len(ids))
        axes[0].bar(x, means, yerr=stds, color=cmap(0), alpha=0.75,
                    capsize=4, label="per-seed mean ± std")
        axes[0].plot(x, consensus, "o", color="red", markersize=6, label="consensus ARI")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
        axes[0].set_title(f"{arch_name} — DLPFC (8 report slices)")
        axes[0].set_ylabel("ARI vs. ground truth")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, axis="y", alpha=0.3)

    if per_units_bc:
        ids = [u["unit_id"] for u in per_units_bc]
        means = [u.get("mean") or 0.0 for u in per_units_bc]
        stds = [u.get("std") or 0.0 for u in per_units_bc]
        consensus = [u.get("consensus_ari") for u in per_units_bc]
        x = np.arange(len(ids))
        axes[1].bar(x, means, yerr=stds, color=cmap(1), alpha=0.75,
                    capsize=4, label="per-seed mean ± std")
        axes[1].plot(x, consensus, "o", color="red", markersize=6, label="consensus ARI")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
        axes[1].set_title(f"{arch_name} — Breast Cancer (4 report blocks)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{arch_name} — per-unit ARI breakdown", fontsize=13)
    fig.tight_layout()
    return _save(fig, f"{arch_name}_per_unit_ari.png")


def plot_entropy_by_hop(fits_report: list[dict], arch_name: str, dataset: str) -> Path:
    """Address entropy by hop — line chart mean ± std across report seeds.

    Uses `last_hop_weights` diagnostics from MSAP/BAAP/AGAP/GMSM where available;
    otherwise falls back to reconstructing per-row attention entropy at each
    stored history tick from the recorded n_slots_used/usage_entropy proxies.
    """
    if not fits_report:
        raise ValueError("No fits provided")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    have_explicit_hop_weights = any(
        f.get("training_history") and any("mean_gate" in step or "local_usage_entropy" in step
                                          or "global_usage_entropy" in step
                                          for step in f["training_history"])
        for f in fits_report if isinstance(f, dict)
    )

    if have_explicit_hop_weights:
        use_keys = set()
        for f in fits_report:
            for step in (f.get("training_history") or []):
                for k in ("local_usage_entropy", "global_usage_entropy", "usage_entropy"):
                    if k in step:
                        use_keys.add(k)
        histories = [f["training_history"] for f in fits_report if f.get("training_history")]
        epochs = [step["epoch"] for step in histories[0]]
        for key in sorted(use_keys):
            vals = np.array([[s.get(key, np.nan) for s in h] for h in histories])
            mean = np.nanmean(vals, axis=0)
            std = np.nanstd(vals, axis=0)
            ax.plot(epochs, mean, label=key, linewidth=1.5)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("epoch")
        ax.set_ylabel("usage entropy (nats)")
        ax.set_title(f"{arch_name} — slot usage entropy by stream ({dataset})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        histories = [f["training_history"] for f in fits_report if f.get("training_history")]
        if histories:
            epochs = [step.get("epoch", i) for i, step in enumerate(histories[0])]
            ue = np.array([[s.get("usage_entropy", np.nan) for s in h] for h in histories])
            mean = np.nanmean(ue, axis=0)
            std = np.nanstd(ue, axis=0)
            ax.plot(epochs, mean, linewidth=1.8, color="C0", label="usage entropy (mean ± std)")
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color="C0")
            re = np.array([[s.get("median_row_entropy", np.nan) for s in h] for h in histories])
            mean2 = np.nanmean(re, axis=0)
            std2 = np.nanstd(re, axis=0)
            ax.plot(epochs, mean2, linewidth=1.4, color="C1", linestyle="--", label="median per-row entropy (mean ± std)")
            ax.fill_between(epochs, mean2 - std2, mean2 + std2, alpha=0.15, color="C1")
            max_e = histories[0][0].get("max_entropy")
            if max_e:
                ax.axhline(max_e, color="gray", linestyle=":", alpha=0.5, label=f"max entropy (slots) = {max_e:.2f}")
            ax.set_xlabel("epoch")
            ax.set_ylabel("entropy (nats)")
            ax.set_title(f"{arch_name} — address entropy ({dataset})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    return _save(fig, f"{arch_name}_{dataset}_entropy_by_hop.png")


def plot_master_comparison(archs_dlpfc: dict[str, dict], archs_bc: dict[str, dict]) -> Path:
    """Master grouped bar chart: all architectures × {DLPFC, BC} consensus.

    Reference lines (horizontal) for baseline/GraphST where their legacy values
    are available from the existing dlpfc_multislice_results.json and
    breast_cancer_results.json logs.
    """
    ref_dlpfc = None
    ref_bc = None
    ref_graphst_dlpfc = None
    ref_graphst_bc = None
    legacy_dlpfc = _load_log(LOGS_DIR / "dlpfc_multislice_results.json")
    legacy_bc = _load_log(LOGS_DIR / "breast_cancer_results.json")
    if legacy_dlpfc:
        h11 = legacy_dlpfc.get("summary", {}).get("held_out_11_slices", {})
        ref_dlpfc = (h11.get("ours_consensus") or {}).get("mean")
        ref_graphst_dlpfc = (h11.get("graphst_consensus") or {}).get("mean")
    if legacy_bc:
        ref_bc = legacy_bc.get("ours", {}).get("consensus")
        ref_graphst_bc = legacy_bc.get("graphst", {}).get("consensus")

    all_names = sorted(set(archs_dlpfc.keys()) | set(archs_bc.keys()))
    x = np.arange(len(all_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(all_names)), 5.5))

    vals_dlpfc = [archs_dlpfc.get(n, {}).get("consensus_mean") for n in all_names]
    vals_bc = [archs_bc.get(n, {}).get("consensus_mean") for n in all_names]
    std_dlpfc = [archs_dlpfc.get(n, {}).get("consensus_std", 0.0) or 0.0 for n in all_names]
    std_bc = [archs_bc.get(n, {}).get("consensus_std", 0.0) or 0.0 for n in all_names]

    dlpfc_mask = [v is not None for v in vals_dlpfc]
    bc_mask = [v is not None for v in vals_bc]
    vals_dlpfc_plot = [(v if v is not None else 0.0) for v in vals_dlpfc]
    vals_bc_plot = [(v if v is not None else 0.0) for v in vals_bc]

    ax.bar(x - width / 2, vals_dlpfc_plot, width, yerr=[(s if m else 0.0) for s, m in zip(std_dlpfc, dlpfc_mask)],
           label="DLPFC (8 report slices, consensus)", color="#1f77b4", alpha=0.8, capsize=4,
           error_kw={"alpha": 0.6})
    ax.bar(x + width / 2, vals_bc_plot, width, yerr=[(s if m else 0.0) for s, m in zip(std_bc, bc_mask)],
           label="Breast Cancer (4 report blocks, consensus)", color="#ff7f0e", alpha=0.8, capsize=4,
           error_kw={"alpha": 0.6})

    for i, (m, v) in enumerate(zip(dlpfc_mask, vals_dlpfc)):
        if not m:
            ax.text(i - width / 2, 0.01, "N/R", ha="center", fontsize=7, color="gray")
    for i, (m, v) in enumerate(zip(bc_mask, vals_bc)):
        if not m:
            ax.text(i + width / 2, 0.01, "N/R", ha="center", fontsize=7, color="gray")

    if ref_graphst_dlpfc is not None:
        ax.axhspan(ref_graphst_dlpfc - 0.003, ref_graphst_dlpfc + 0.003,
                   color="#2ca02c", alpha=0.15)
        ax.axhline(ref_graphst_dlpfc, color="#2ca02c", linestyle="--", linewidth=1.5,
                   label=f"GraphST (legacy DLPFC): {ref_graphst_dlpfc:.3f}")
    if ref_graphst_bc is not None:
        ax.axhline(ref_graphst_bc, color="#d62728", linestyle="-.", linewidth=1.2,
                   label=f"GraphST (legacy BC): {ref_graphst_bc:.3f}")
    if ref_dlpfc is not None:
        ax.axhline(ref_dlpfc, color="#1f77b4", linestyle=":", linewidth=1, alpha=0.5,
                   label=f"Baseline legacy DLPFC: {ref_dlpfc:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("consensus ARI")
    ax.set_title("Master comparison: all architectures × {DLPFC, Breast Cancer}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ymax = max([v for v in vals_dlpfc + vals_bc if v is not None] + [ref_graphst_dlpfc or 0, ref_graphst_bc or 0, ref_dlpfc or 0])
    ax.set_ylim(0, min(1.0, max(0.7, 1.1 * ymax)))
    fig.tight_layout()
    return _save(fig, "master_comparison.png")


def generate_all_for_arch(arch_name: str) -> list[Path]:
    """Generate entropy-by-hop, per-unit ARI, and training-curve figures for one
    architecture from the standard-protocol log files, if they exist. Returns list
    of saved figure paths."""
    out_paths = []

    for dataset in ("dlpfc", "breast_cancer"):
        log = _load_log(LOGS_DIR / f"{arch_name}_{dataset}_results.json")
        if log is None:
            continue
        fits_report = log.get("fits_report") or []
        if fits_report:
            try:
                out_paths.append(plot_entropy_by_hop(fits_report, arch_name, dataset))
            except Exception as exc:
                print(f"[warn] entropy figure for {arch_name}/{dataset} failed: {exc}")
        histories = [f.get("training_history") or [] for f in fits_report[:1]]
        if histories and histories[0]:
            try:
                out_paths.append(plot_training_curve_from_history(histories, arch_name, dataset))
            except Exception as exc:
                print(f"[warn] training curve figure for {arch_name}/{dataset} failed: {exc}")

    log_dlpfc = _load_log(LOGS_DIR / f"{arch_name}_dlpfc_results.json")
    log_bc = _load_log(LOGS_DIR / f"{arch_name}_breast_cancer_results.json")
    if log_dlpfc or log_bc:
        try:
            out_paths.append(plot_per_unit_ari(
                log_dlpfc.get("per_unit") if log_dlpfc else None,
                log_bc.get("per_unit") if log_bc else None,
                arch_name,
            ))
        except Exception as exc:
            print(f"[warn] per-unit ARI figure for {arch_name} failed: {exc}")

    return out_paths


def generate_master_comparison() -> Path:
    archs_dlpfc: dict[str, dict] = {}
    archs_bc: dict[str, dict] = {}
    for path in sorted(LOGS_DIR.glob("*_dlpfc_results.json")):
        name = path.stem[: -len("_dlpfc_results")]
        data = _load_log(path)
        if data:
            archs_dlpfc[name] = data.get("summary") or {}
    for path in sorted(LOGS_DIR.glob("*_breast_cancer_results.json")):
        name = path.stem[: -len("_breast_cancer_results")]
        data = _load_log(path)
        if data:
            archs_bc[name] = data.get("summary") or {}
    return plot_master_comparison(archs_dlpfc, archs_bc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="*", default=None,
                        help="Specific architecture names to generate for; default = all discovered + master")
    parser.add_argument("--skip-master", action="store_true")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    discovered = set()
    if args.archs:
        for name in args.archs:
            paths = generate_all_for_arch(name)
            for p in paths:
                print(f"  wrote {p.name}")
                discovered.add(name)
    else:
        for path in sorted(LOGS_DIR.glob("*_dlpfc_results.json")):
            name = path.stem[: -len("_dlpfc_results")]
            discovered.add(name)
        for path in sorted(LOGS_DIR.glob("*_breast_cancer_results.json")):
            name = path.stem[: -len("_breast_cancer_results")]
            discovered.add(name)
        for name in sorted(discovered):
            print(f"== {name} ==")
            paths = generate_all_for_arch(name)
            for p in paths:
                print(f"  wrote {p.name}")

    if not args.skip_master:
        p = generate_master_comparison()
        print(f"wrote master comparison: {p.name}")
