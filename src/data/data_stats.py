from pathlib import Path

from src.data.load_geo import load_slideseqv2
from src.data.load_visium import load_visium_crop, load_visium_full

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "data_stats.txt"


def sparsity(adata):
    return 1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1])


def main():
    datasets = {
        "visium_crop": load_visium_crop(),
        "visium_full": load_visium_full(),
        "slideseqv2": load_slideseqv2(),
    }

    lines = []
    for name, adata in datasets.items():
        line = (
            f"{name}: shape={adata.shape}, dtype={adata.X.dtype}, "
            f"sparsity={sparsity(adata):.4f}"
        )
        print(line)
        lines.append(line)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
