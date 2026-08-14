"""Visual verification of breast cancer spatial blocks partition.

This script generates a visual check that the spatial blocks form reasonable
partitions of the tissue - same discipline as the H&E patch-alignment check.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans

from src.data.load_breast_cancer import load_breast_cancer
from src.eval.breast_cancer_spatial_blocks import (
    load_or_build_blocks, SELECTION_BLOCKS, REPORT_BLOCKS, OUTPUT_PATH
)


def plot_blocks_visualization(adata, block_id):
    """Generate spatial visualization of blocks."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    coords = adata.obsm["spatial"]
    
    # Left: Full tissue with blocks colored
    scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=block_id, cmap='tab10', s=10, alpha=0.7)
    axes[0].set_title("Breast Cancer Spatial Blocks Partition", fontsize=14)
    axes[0].set_xlabel("Spatial X")
    axes[0].set_ylabel("Spatial Y")
    plt.colorbar(scatter, ax=axes[0], label="Block ID")
    
    # Right: Selection vs Report blocks
    block_roles = np.array(['SELECTION' if b in SELECTION_BLOCKS else 'REPORT' for b in block_id])
    role_colors = {'SELECTION': 'red', 'REPORT': 'blue'}
    colors = [role_colors[role] for role in block_roles]
    
    for role in ['SELECTION', 'REPORT']:
        mask = block_roles == role
        axes[1].scatter(coords[mask, 0], coords[mask, 1], c=role_colors[role], 
                       label=f'{role} ({mask.sum()} spots)', s=10, alpha=0.7)
    
    axes[1].set_title("Selection vs Report Blocks", fontsize=14)
    axes[1].set_xlabel("Spatial X")
    axes[1].set_ylabel("Spatial Y")
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "breast_cancer_blocks_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved verification plot to {output_path}")
    
    plt.close()
    return output_path


if __name__ == "__main__":
    print("Loading breast cancer data and blocks...")
    adata = load_breast_cancer()
    block_id = load_or_build_blocks()
    
    print("Generating visualization...")
    plot_blocks_visualization(adata, block_id)
    
    print("\n" + "="*50)
    print("Spatial blocks verification complete")
    print("="*50)
    print(f"Blocks file: {OUTPUT_PATH}")
    print("Please visually inspect the plot to ensure blocks form reasonable spatial partitions")