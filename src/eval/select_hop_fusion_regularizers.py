"""Leakage-safe joint selection of Hop-Fusion regularizers on DLPFC.

The new concat-fusion path needs both marginal slot-usage balancing and an
explicit per-spot sharpness incentive.  Those objectives counteract each
other, so their weights are selected as a joint grid rather than in separate
one-dimensional sweeps.

Only the three configured validation slices select the pair.  DLPFC 151673 is
excluded, and the eight configured true-holdout slices remain unread until a
pair has been selected.  Results are checkpointed atomically after every fit
and a compatible invocation resumes missing work automatically.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.physical_scale import get_average_edge_length_um
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.eval.hop_fusion_protocol import DEFAULT_SELECTION_PATH, load_json
from src.models.train_hop_fusion import train_hop_fusion_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "outputs" / "logs" / "hop_fusion_regularizer_selection.json"
DEFAULT_REGULARIZER_LOCK_PATH = ROOT / "configs" / "hop_fusion_regularizers.json"


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _gpu_telemetry() -> dict:
    """Return best-effort GPU temperature and memory telemetry.

    ``nvidia-smi`` is intentionally optional so the runner still works on a
    CPU-only host.  PyTorch peak allocation is recorded separately per fit.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_gpu = result.stdout.strip().splitlines()[0]
        temperature, used, total = [int(value.strip()) for value in first_gpu.split(",")]
        return {
            "temperature_c": temperature,
            "memory_used_mib": used,
            "memory_total_mib": total,
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return {"temperature_c": None, "memory_used_mib": None, "memory_total_mib": None}


def _fit_id(phase: str, lambda_usage: float, lambda_sharpen: float, sample: str, seed: int) -> str:
    return (
        f"{phase}|usage={lambda_usage:.8g}|sharpen={lambda_sharpen:.8g}|"
        f"slice={sample}|seed={seed}"
    )


def _score_fit(adata, config: dict, sample: str, seed: int, device: torch.device, epochs: int) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    telemetry_before = _gpu_telemetry()
    started = time.perf_counter()
    model = trained = None
    try:
        model, trained, history = train_hop_fusion_model(
            adata.copy(),
            platform="visium",
            physical_radius_um=config["physical_radius_um"],
            memory_slots=config["memory_slots"],
            memory_dim=config["memory_dim"],
            hidden_dim=config["hidden_dim"],
            fusion_hidden_dim=config["fusion_hidden_dim"],
            fusion_depth=config["fusion_depth"],
            temperature=config.get("temperature", 1.0),
            attention_fn=config["attention_fn"],
            lambda_usage=config["lambda_usage"],
            lambda_sharpen=config["lambda_sharpen"],
            lambda_spatial_coherence=config["lambda_spatial_coherence"],
            expression_weighted=config["expression_weighted"],
            seed=seed,
            epochs=epochs,
            device=device,
            verbose=False,
        )
        truth = trained.obs["ground_truth_layer"]
        valid = truth.notna().to_numpy()
        n_layers = int(truth.nunique())
        labels = cluster_embedding(
            trained.obsm["X_hop_fusion"], n_layers,
            coords=trained.obsm["spatial"], refine=True,
        )
        peak_mib = (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        )
        return {
            "sample": sample,
            "seed": int(seed),
            "ari": float(adjusted_rand_score(truth.to_numpy()[valid], np.asarray(labels)[valid])),
            "n_spots": int(trained.n_obs),
            "final_history": history[-1],
            "physical_metadata": trained.uns["hop_fusion"],
            "duration_s": float(time.perf_counter() - started),
            "peak_torch_allocated_mib": peak_mib,
            "gpu_before": telemetry_before,
            "gpu_after": _gpu_telemetry(),
        }
    finally:
        del model, trained
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _aggregate(records: list[dict]) -> dict:
    by_slice: dict[str, list[float]] = {}
    for record in records:
        by_slice.setdefault(record["sample"], []).append(record["ari"])
    per_slice = {
        sample: {
            "mean_ari": float(np.mean(values)),
            "std_ari": float(np.std(values)),
            "n_seeds": len(values),
        }
        for sample, values in sorted(by_slice.items())
    }
    slice_means = [row["mean_ari"] for row in per_slice.values()]
    all_scores = [record["ari"] for record in records]
    return {
        "mean_of_slice_means": float(np.mean(slice_means)),
        "std_of_slice_means": float(np.std(slice_means)),
        "mean_over_all_fits": float(np.mean(all_scores)),
        "std_over_all_fits": float(np.std(all_scores)),
        "per_slice": per_slice,
    }


def _grid_summary(fits: dict[str, dict], lambda_usage_values: list[float], lambda_sharpen_values: list[float]) -> dict:
    summary = {}
    for lambda_sharpen in lambda_sharpen_values:
        for lambda_usage in lambda_usage_values:
            records = [
                record
                for record in fits.values()
                if record["phase"] == "validation_grid"
                and record["lambda_usage"] == lambda_usage
                and record["lambda_sharpen"] == lambda_sharpen
            ]
            summary[f"usage={lambda_usage:.8g}|sharpen={lambda_sharpen:.8g}"] = _aggregate(records)
    return summary


def _protocol(selection: dict, reference_edge_length_um: float, seeds: list[int], epochs: int) -> dict:
    candidates = selection["candidates"]
    fixed = selection["fixed_model"]
    architecture = selection["regularizer_grid_fixed_architecture"]
    return {
        "selection_config": str(DEFAULT_SELECTION_PATH),
        "selection_slices": selection["cv_validation_slices"],
        "true_holdout_slices": selection["true_holdout_slices"],
        "tuning_slice_excluded": selection["tuning_slice_excluded"],
        "seeds": [int(seed) for seed in seeds],
        "epochs": int(epochs),
        "reference_edge_length_um": float(reference_edge_length_um),
        "fixed_model": fixed,
        "fixed_architecture": architecture,
        "lambda_usage_values": [float(value) for value in candidates["lambda_usage"]],
        "lambda_sharpen_values": [float(value) for value in candidates["lambda_sharpen"]],
    }


def _base_config(protocol: dict, lambda_usage: float, lambda_sharpen: float) -> dict:
    config = {
        **protocol["fixed_model"],
        **protocol["fixed_architecture"],
        "lambda_usage": float(lambda_usage),
        "lambda_sharpen": float(lambda_sharpen),
        "physical_radius_um": float(
            protocol["fixed_architecture"]["reference_max_hops"]
            * protocol["reference_edge_length_um"]
        ),
    }
    return config


def _load_or_create_output(output_path: Path, protocol: dict) -> dict:
    if output_path.exists():
        value = json.loads(output_path.read_text(encoding="utf-8"))
        if value.get("protocol") != protocol:
            raise ValueError(
                f"existing output {output_path} was created with a different protocol; "
                "choose a new --output path rather than mixing runs"
            )
        return value
    return {
        "status": "in_progress",
        "protocol": protocol,
        "fits": {},
        "selection": None,
        "true_holdout": None,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_regularizer_lock(protocol: dict, selected: dict, output_path: Path) -> None:
    lock = {
        "status": "regularizers_locked_pending_architecture_selection",
        "reference_platform": "visium",
        "lambda_usage": selected["lambda_usage"],
        "lambda_sharpen": selected["lambda_sharpen"],
        "selection_output": str(output_path),
        "selection_slices": protocol["selection_slices"],
        "true_holdout_slices": protocol["true_holdout_slices"],
        "tuning_slice_excluded": protocol["tuning_slice_excluded"],
        "fixed_architecture_used_for_regularizer_selection": protocol["fixed_architecture"],
        "notes": "Only the two regularizers are locked here. The full Hop-Fusion architecture remains pending DLPFC selection, and this file must not be used for downstream generalization.",
    }
    _atomic_write_json(DEFAULT_REGULARIZER_LOCK_PATH, lock)


def _run_fit(
    output: dict,
    output_path: Path,
    cache: dict,
    phase: str,
    lambda_usage: float,
    lambda_sharpen: float,
    sample: str,
    seed: int,
    protocol: dict,
    device: torch.device,
    epochs: int,
) -> dict | None:
    identifier = _fit_id(phase, lambda_usage, lambda_sharpen, sample, seed)
    if identifier in output["fits"]:
        return None
    record = _score_fit(
        cache[sample], _base_config(protocol, lambda_usage, lambda_sharpen),
        sample, seed, device, epochs,
    )
    record.update({
        "phase": phase,
        "lambda_usage": float(lambda_usage),
        "lambda_sharpen": float(lambda_sharpen),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    output["fits"][identifier] = record
    output["updated_at_utc"] = record["completed_at_utc"]
    _atomic_write_json(output_path, output)
    temperature = record["gpu_after"]["temperature_c"]
    print(
        f"{phase} usage={lambda_usage:.3g} sharpen={lambda_sharpen:.3g} "
        f"slice={sample} seed={seed}: ARI={record['ari']:.4f} "
        f"duration={record['duration_s']:.1f}s temperature={temperature}C",
        flush=True,
    )
    return record


def _pause_for_thermal_ceiling(
    output: dict,
    output_path: Path,
    record: dict | None,
    ceiling_c: int | None,
) -> bool:
    """Persist a safe pause after a completed fit reaches the GPU ceiling."""
    if record is None or ceiling_c is None:
        return False
    temperature = record["gpu_after"]["temperature_c"]
    if temperature is None or temperature < ceiling_c:
        return False
    output["status"] = "paused_for_thermal_ceiling"
    output["thermal_pause"] = {
        "ceiling_c": int(ceiling_c),
        "observed_temperature_c": int(temperature),
        "after_fit": {
            "phase": record["phase"],
            "sample": record["sample"],
            "seed": record["seed"],
            "lambda_usage": record["lambda_usage"],
            "lambda_sharpen": record["lambda_sharpen"],
        },
        "paused_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output["updated_at_utc"] = output["thermal_pause"]["paused_at_utc"]
    _atomic_write_json(output_path, output)
    print(
        f"Thermal ceiling {ceiling_c}C reached after a checkpointed fit "
        f"({temperature}C). Exiting safely; rerun after cooling to resume.",
        flush=True,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--max-fits", type=int,
        help="Run at most this many missing fits, useful for smoke checks; rerun without it to resume.",
    )
    parser.add_argument(
        "--skip-true-holdout", action="store_true",
        help="Select the regularizer pair only; do not run the eight untouched holdout slices.",
    )
    parser.add_argument(
        "--thermal-ceiling-c", type=int,
        help="Safely stop after a checkpointed fit at or above this GPU temperature; rerun to resume.",
    )
    args = parser.parse_args()
    if args.max_fits is not None and args.max_fits < 1:
        parser.error("--max-fits must be positive")
    if args.thermal_ceiling_c is not None and args.thermal_ceiling_c < 1:
        parser.error("--thermal-ceiling-c must be positive")

    selection = load_json(DEFAULT_SELECTION_PATH)
    validation_slices = selection["cv_validation_slices"]
    true_holdout_slices = selection["true_holdout_slices"]
    excluded = selection["tuning_slice_excluded"]
    if excluded in validation_slices or excluded in true_holdout_slices:
        raise ValueError("the historic tuning slice must be excluded from validation and true holdout")
    if set(validation_slices) & set(true_holdout_slices):
        raise ValueError("validation and true holdout DLPFC slices must be disjoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; initial_gpu={_gpu_telemetry()}", flush=True)
    validation_cache = {
        sample: preprocess_hvg(load_dlpfc_slice(sample), platform="visium")
        for sample in validation_slices
    }
    reference_edge_length_um = float(np.median([
        get_average_edge_length_um(adata, "visium") for adata in validation_cache.values()
    ]))
    protocol = _protocol(selection, reference_edge_length_um, args.seeds, args.epochs)
    output_path = args.output.resolve()
    output = _load_or_create_output(output_path, protocol)
    remaining_budget = args.max_fits

    for lambda_sharpen in protocol["lambda_sharpen_values"]:
        for lambda_usage in protocol["lambda_usage_values"]:
            for sample in validation_slices:
                for seed in protocol["seeds"]:
                    identifier = _fit_id("validation_grid", lambda_usage, lambda_sharpen, sample, seed)
                    if identifier in output["fits"]:
                        continue
                    if remaining_budget == 0:
                        print(f"checkpointed partial run to {output_path}", flush=True)
                        return
                    record = _run_fit(
                        output, output_path, validation_cache, "validation_grid",
                        lambda_usage, lambda_sharpen, sample, seed,
                        protocol, device, args.epochs,
                    )
                    if _pause_for_thermal_ceiling(
                        output, output_path, record, args.thermal_ceiling_c
                    ):
                        return
                    if remaining_budget is not None:
                        remaining_budget -= 1

    grid = _grid_summary(
        output["fits"], protocol["lambda_usage_values"], protocol["lambda_sharpen_values"]
    )
    selected_key = max(
        grid,
        key=lambda key: (grid[key]["mean_of_slice_means"], -grid[key]["std_of_slice_means"]),
    )
    selected_usage, selected_sharpen = [
        float(part.split("=")[1]) for part in selected_key.split("|")
    ]
    output["selection"] = {
        "lambda_usage": selected_usage,
        "lambda_sharpen": selected_sharpen,
        "grid_summary": grid,
        "selection_metric": "mean_of_slice_means",
    }
    _write_regularizer_lock(protocol, output["selection"], output_path)
    _atomic_write_json(output_path, output)
    print(
        f"selected usage={selected_usage:.3g} sharpen={selected_sharpen:.3g} "
        f"on validation slices; regularizer lock written to {DEFAULT_REGULARIZER_LOCK_PATH}",
        flush=True,
    )

    if args.skip_true_holdout:
        output["status"] = "regularizers_selected_true_holdout_not_run"
        output["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(output_path, output)
        return

    holdout_cache = {
        sample: preprocess_hvg(load_dlpfc_slice(sample), platform="visium")
        for sample in true_holdout_slices
    }
    for sample in true_holdout_slices:
        for seed in protocol["seeds"]:
            identifier = _fit_id("true_holdout", selected_usage, selected_sharpen, sample, seed)
            if identifier in output["fits"]:
                continue
            if remaining_budget == 0:
                print(f"checkpointed partial run to {output_path}", flush=True)
                return
            record = _run_fit(
                output, output_path, holdout_cache, "true_holdout",
                selected_usage, selected_sharpen, sample, seed,
                protocol, device, args.epochs,
            )
            if _pause_for_thermal_ceiling(
                output, output_path, record, args.thermal_ceiling_c
            ):
                return
            if remaining_budget is not None:
                remaining_budget -= 1

    true_holdout_records = [
        record for record in output["fits"].values() if record["phase"] == "true_holdout"
    ]
    output["true_holdout"] = _aggregate(true_holdout_records)
    output["status"] = "complete_regularizer_selection_and_true_holdout"
    output["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(output_path, output)
    print("Completed regularizer selection and untouched DLPFC holdout evaluation", flush=True)


if __name__ == "__main__":
    main()
