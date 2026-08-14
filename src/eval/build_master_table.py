"""Build the master results table from JSON log files.

Reads every:
    outputs/logs/{arch}_dlpfc_results.json
    outputs/logs/{arch}_breast_cancer_results.json

Produces:
    outputs/logs/master_results_table.md

Replaces manual number-copying into a table: every entry traces to a real
file. Nothing in the final table is a chat-log claim.

Usage:
    uv run python -m src.eval.build_master_table
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LOGS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "logs"
OUTPUT_PATH = LOGS_DIR / "master_results_table.md"

KNOWN_ARCHS = [
    "baseline",
    "ldcm",
    "ppr",
    "agap",
    "hma",
    "gmsm",
    "msap",
    "baap",
    "zism",
]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _summarize_dataset(data: dict | None, dataset_tag: str) -> dict:
    """Extract a per-method summary from a standard_protocol output JSON.

    Returns dict with keys:
        {dataset_tag}_per_seed_mean, {dataset_tag}_per_seed_std,
        {dataset_tag}_consensus_mean, {dataset_tag}_consensus_std,
        {dataset_tag}_n_units, {dataset_tag}_hp
    """
    empty = {
        f"{dataset_tag}_per_seed_mean": None,
        f"{dataset_tag}_per_seed_std": None,
        f"{dataset_tag}_consensus_mean": None,
        f"{dataset_tag}_consensus_std": None,
        f"{dataset_tag}_n_units": None,
        f"{dataset_tag}_hp": None,
    }
    if data is None:
        return empty
    summary = data.get("summary") or {}
    hp = data.get("hp_config_used") or data.get("selection", {}).get("best_hp") or {}
    return {
        f"{dataset_tag}_per_seed_mean": summary.get("per_seed_mean"),
        f"{dataset_tag}_per_seed_std": summary.get("per_seed_std"),
        f"{dataset_tag}_consensus_mean": summary.get("consensus_mean"),
        f"{dataset_tag}_consensus_std": summary.get("consensus_std"),
        f"{dataset_tag}_n_units": summary.get("n_units"),
        f"{dataset_tag}_hp": hp,
    }


def _fmt(v, sd=None, precision=4):
    if v is None:
        return "—"
    if sd is None or np.isnan(sd):
        return f"{v:.{precision}f}"
    return f"{v:.{precision}f} ± {sd:.{precision}f}"


def _hp_summary(hp: dict | None) -> str:
    if not hp:
        return "baseline defaults"
    parts = []
    for k, v in sorted(hp.items()):
        if isinstance(v, float):
            parts.append(f"{k}={v:.3g}")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _collect_baseline_refs():
    """Extract baseline + GraphST reference numbers from existing logs where the
    master-protocol outputs haven't been generated yet. This keeps the table
    useful even before all reruns complete."""
    refs = {}
    dlpfc_legacy = LOGS_DIR / "dlpfc_multislice_results.json"
    if dlpfc_legacy.exists():
        data = _load_json(dlpfc_legacy)
        if data:
            s = data.get("summary", {}).get("held_out_11_slices", {})
            ours = s.get("ours", {})
            ours_c = s.get("ours_consensus", {})
            g = s.get("graphst", {})
            g_c = s.get("graphst_consensus", {})
            refs.setdefault("ours_legacy", {})["dlpfc"] = (ours, ours_c)
            refs.setdefault("graphst", {})["dlpfc"] = (g, g_c)
    bc_legacy = LOGS_DIR / "breast_cancer_results.json"
    if bc_legacy.exists():
        data = _load_json(bc_legacy)
        if data:
            ours = data.get("ours", {})
            g = data.get("graphst", {})
            refs.setdefault("ours_legacy", {})["breast_cancer"] = (
                {"mean": ours.get("mean"), "std": ours.get("std")},
                {"consensus": ours.get("consensus")},
            )
            refs.setdefault("graphst", {})["breast_cancer"] = (
                {"mean": g.get("mean"), "std": g.get("std")},
                {"consensus": g.get("consensus")},
            )
    return refs


def build_table() -> str:
    rows = []
    refs = _collect_baseline_refs()
    discovered = set()

    for arch in KNOWN_ARCHS:
        d_dlpfc = _load_json(LOGS_DIR / f"{arch}_dlpfc_results.json")
        d_bc = _load_json(LOGS_DIR / f"{arch}_breast_cancer_results.json")
        if d_dlpfc is not None or d_bc is not None:
            discovered.add(arch)
        info = {}
        info.update(_summarize_dataset(d_dlpfc, "dlpfc"))
        info.update(_summarize_dataset(d_bc, "bc"))
        rows.append((arch, info))

    for path in sorted(LOGS_DIR.glob("*_dlpfc_results.json")):
        name = path.stem[: -len("_dlpfc_results")]
        if name in KNOWN_ARCHS:
            continue
        bc_path = LOGS_DIR / f"{name}_breast_cancer_results.json"
        info = {}
        info.update(_summarize_dataset(_load_json(path), "dlpfc"))
        info.update(_summarize_dataset(_load_json(bc_path), "bc"))
        rows.append((name, info))
        discovered.add(name)

    lines = []
    lines.append("# Master Results Table")
    lines.append("")
    lines.append("_Auto-generated by `src/eval/build_master_table.py`._")
    lines.append("_Every number below traces to a `outputs/logs/{arch}_{dataset}_results.json` file;")
    lines.append("rerun the builder script to refresh after new runs complete._")
    lines.append("")

    lines.append("## Headline: consensus ARI (primary metric, per-unit mean ± std)")
    lines.append("")
    header = "| Architecture | DLPFC — 8 report slices, consensus | Breast Cancer — 4 report blocks, consensus | Notes |"
    sep = "|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    def row_line(name, dlpfc_mean, dlpfc_std, bc_mean, bc_std, note):
        return f"| {name} | {_fmt(dlpfc_mean, dlpfc_std)} | {_fmt(bc_mean, bc_std)} | {note} |"

    graphst = refs.get("graphst", {})
    if "dlpfc" in graphst:
        g_dlpfc_mean = graphst["dlpfc"][1].get("mean") or graphst["dlpfc"][1].get("consensus")
        g_dlpfc_std = graphst["dlpfc"][1].get("std")
        g_bc_mean = graphst["breast_cancer"][1].get("mean") or graphst["breast_cancer"][1].get("consensus")
        g_bc_std = graphst["breast_cancer"][1].get("std")
        lines.append(row_line("GraphST (reference, legacy)", g_dlpfc_mean, g_dlpfc_std, g_bc_mean, g_bc_std, "Legacy run, logged pre-block protocol"))

    ours_legacy = refs.get("ours_legacy", {})
    if "ours_legacy" in refs and "baseline" not in discovered:
        d = ours_legacy.get("dlpfc")
        b = ours_legacy.get("breast_cancer")
        d_m = d[1].get("mean") or d[1].get("consensus") if d else None
        d_s = d[1].get("std") if d else None
        b_m = b[1].get("mean") or b[1].get("consensus") if b else None
        b_s = b[1].get("std") if b else None
        lines.append(row_line("Ours baseline (legacy run)", d_m, d_s, b_m, b_s, "Pre-block protocol; use `baseline` row once rerun done"))

    for arch, info in rows:
        note = []
        hp_dlpfc = info.get("dlpfc_hp")
        hp_bc = info.get("bc_hp")
        if hp_dlpfc and hp_dlpfc == hp_bc:
            note.append(f"HP: {_hp_summary(hp_dlpfc)}")
        else:
            if hp_dlpfc:
                note.append(f"DLPFC HP: {_hp_summary(hp_dlpfc)}")
            if hp_bc:
                note.append(f"BC HP: {_hp_summary(hp_bc)}")
        lines.append(row_line(
            arch,
            info.get("dlpfc_consensus_mean"),
            info.get("dlpfc_consensus_std"),
            info.get("bc_consensus_mean"),
            info.get("bc_consensus_std"),
            " | ".join(note) if note else "",
        ))

    lines.append("")
    lines.append("## Per-seed mean ARI (secondary metric, per-unit mean ± std)")
    lines.append("")
    header2 = "| Architecture | DLPFC — 8 report slices, per-seed mean | Breast Cancer — 4 report blocks, per-seed mean |"
    sep2 = "|---|---|---|"
    lines.append(header2)
    lines.append(sep2)
    if "graphst" in refs:
        g_dlpfc = graphst.get("dlpfc")
        g_bc = graphst.get("breast_cancer")
        lines.append(
            "| GraphST (reference, legacy) | "
            f"{_fmt(g_dlpfc[0].get('mean'), g_dlpfc[0].get('std')) if g_dlpfc else '—'} | "
            f"{_fmt(g_bc[0].get('mean'), g_bc[0].get('std')) if g_bc else '—'} |"
        )
    if "ours_legacy" in refs and "baseline" not in discovered:
        d = ours_legacy.get("dlpfc")
        b = ours_legacy.get("breast_cancer")
        lines.append(
            "| Ours baseline (legacy run) | "
            f"{_fmt(d[0].get('mean'), d[0].get('std')) if d else '—'} | "
            f"{_fmt(b[0].get('mean'), b[0].get('std')) if b else '—'} |"
        )
    for arch, info in rows:
        lines.append(
            f"| {arch} | "
            f"{_fmt(info.get('dlpfc_per_seed_mean'), info.get('dlpfc_per_seed_std'))} | "
            f"{_fmt(info.get('bc_per_seed_mean'), info.get('bc_per_seed_std'))} |"
        )

    lines.append("")
    lines.append("## Per-unit detail (consensus ARI, each slice/block)")
    lines.append("")
    for arch, info in rows:
        lines.append(f"### {arch}")
        lines.append("")
        per_unit_dlpfc = []
        per_unit_bc = []
        d_dlpfc = _load_json(LOGS_DIR / f"{arch}_dlpfc_results.json")
        if d_dlpfc:
            per_unit_dlpfc = d_dlpfc.get("per_unit", [])
        d_bc = _load_json(LOGS_DIR / f"{arch}_breast_cancer_results.json")
        if d_bc:
            per_unit_bc = d_bc.get("per_unit", [])

        if per_unit_dlpfc:
            lines.append("**DLPFC report slices:**")
            lines.append("")
            lines.append("| Slice | n_spots | n_classes | Consensus ARI | Per-seed mean ± std |")
            lines.append("|---|---:|---:|---:|---|")
            for u in per_unit_dlpfc:
                lines.append(
                    f"| {u.get('unit_id','?')} | {u.get('n_spots','?')} | "
                    f"{u.get('n_classes','?')} | {_fmt(u.get('consensus_ari'), precision=4)} | "
                    f"{_fmt(u.get('mean'), u.get('std'))} |"
                )
            lines.append("")
        if per_unit_bc:
            lines.append("**Breast Cancer report blocks:**")
            lines.append("")
            lines.append("| Block(s) | n_spots | n_classes | Consensus ARI | Per-seed mean ± std |")
            lines.append("|---|---:|---:|---:|---|")
            for u in per_unit_bc:
                lines.append(
                    f"| {u.get('unit_id','?')} | {u.get('n_spots','?')} | "
                    f"{u.get('n_classes','?')} | {_fmt(u.get('consensus_ari'), precision=4)} | "
                    f"{_fmt(u.get('mean'), u.get('std'))} |"
                )
            lines.append("")

    lines.append("## Raw JSON sources")
    lines.append("")
    lines.append("All rows above come from these files (re-run individual architecture harnesses to update):")
    lines.append("")
    for path in sorted(LOGS_DIR.glob("*_dlpfc_results.json")):
        lines.append(f"- `{path.name}` + `{path.stem[:-len('_dlpfc_results')]}_breast_cancer_results.json`")
    return "\n".join(lines) + "\n"


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    md = build_table()
    OUTPUT_PATH.write_text(md, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    nonempty = [l for l in md.splitlines() if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("|") and not l.strip().startswith("_") and "---" not in l]
    print(f"Content lines: {len(nonempty)}")


if __name__ == "__main__":
    main()
