"""Sprint-friendly driver for the enhanced-model smoke test.

Runs ONE slice's worth of run_enhanced_smoke_test.run_slice() and merges the
result into outputs/logs/enhanced_smoke_test_results.json incrementally, so
each invocation is a complete, self-contained unit -- no process is ever left
mid-flight between sprints, and re-running an already-completed slice just
overwrites its own entry.

Usage: python -m src.eval.run_enhanced_smoke_sprint <slice_id>
"""

import json
import sys
from pathlib import Path

import numpy as np

from src.eval.run_enhanced_smoke_test import (
    EPOCHS,
    N_DOMAIN_SLOTS,
    N_STATE_SLOTS,
    SEEDS,
    SLICES,
    run_slice,
)

OUT_FILE = Path("outputs/logs/enhanced_smoke_test_results.json")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in SLICES:
        print(f"Usage: python -m src.eval.run_enhanced_smoke_sprint <slice_id>")
        print(f"  slice_id must be one of {SLICES}")
        sys.exit(1)

    sample_id = sys.argv[1]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        data = json.loads(OUT_FILE.read_text(encoding="utf-8"))
    else:
        data = {
            "config": {
                "slices": SLICES,
                "seeds": SEEDS,
                "epochs": EPOCHS,
                "n_domain_slots": N_DOMAIN_SLOTS,
                "n_state_slots": N_STATE_SLOTS,
            },
            "per_slice": {},
            "aggregate": {},
        }

    print(f"=== Sprint: slice {sample_id} ===")
    data["per_slice"][sample_id] = run_slice(sample_id)

    # Recompute aggregate from whatever slices are done so far.
    done_slices = [s for s in SLICES if s in data["per_slice"]]
    if done_slices:
        variant_names = list(next(iter(data["per_slice"].values()))["variants"].keys())
        agg = {}
        for v in variant_names:
            slice_means = [data["per_slice"][s]["variants"][v]["mean_ari"] for s in done_slices]
            agg[v] = {
                "per_slice_mean_ari": dict(zip(done_slices, [round(x, 4) for x in slice_means])),
                "grand_mean_ari": float(np.mean(slice_means)),
                "grand_std_ari": float(np.std(slice_means)),
                "n_slices_so_far": len(done_slices),
            }
        data["aggregate"] = agg

        print(f"\n=== Aggregate so far ({len(done_slices)}/{len(SLICES)} slices: {done_slices}) ===")
        for v, a in agg.items():
            print(f"  {v:30s}  mean={a['grand_mean_ari']:.4f}  per_slice={a['per_slice_mean_ari']}")

    OUT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nSaved (slice {sample_id} merged) -> {OUT_FILE}")
    remaining = [s for s in SLICES if s not in data["per_slice"]]
    if remaining:
        print(f"Remaining slices: {remaining}")
    else:
        print("All slices done.")


if __name__ == "__main__":
    main()
