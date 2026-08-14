"""Breast cancer spatial blocks partition - mandatory prerequisite for rerun plan.

This implements the mandatory fix for the breast cancer leakage problem:
partition the single tissue sample into spatial blocks that play the same role
as DLPFC slices do. This prevents architecture-search-on-the-test-set.

CRITICAL: Generate ONCE, save to disk, and reuse identically for every architecture.
Never regenerate — a different partition per architecture would itself be a leakage vector.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans

from src.data.load_breast_cancer import load_breast_cancer

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "breast_cancer_blocks.npy"
N_BLOCKS = 6
SEED = 0  # Fixed seed for reproducibility

# Block assignment: 2 for selection, 4 for report
SELECTION_BLOCKS = {0, 1}  # First 2 blocks for hyperparameter selection
REPORT_BLOCKS = {2, 3, 4, 5}  # Remaining 4 blocks for final reporting


def build_fixed_spatial_blocks(adata, n_blocks=N_BLOCKS, seed=SEED):
    """Deterministic, one-time partition of breast cancer tissue into spatial blocks.
    
    Uses K-means clustering on spatial coordinates to create contiguous-ish blocks.
    The partition is deterministic due to fixed random seed.
    
    Args:
        adata: AnnData object with spatial coordinates in obsm['spatial']
        n_blocks: Number of spatial blocks to create (default: 6)
        seed: Random seed for reproducibility (default: 0)
    
    Returns:
        block_id: Array of block assignments for each spot, shape (n_spots,)
    """
    coords = adata.obsm["spatial"]
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    block_id = km.fit_predict(coords)
    return block_id


def load_or_build_blocks(force_rebuild=False):
    """Load existing blocks from disk, or build and save them if they don't exist.
    
    Args:
        force_rebuild: If True, rebuild even if file exists (DANGEROUS - only for debugging)
    
    Returns:
        block_id: Array of block assignments
    """
    if OUTPUT_PATH.exists() and not force_rebuild:
        print(f"Loading existing breast cancer blocks from {OUTPUT_PATH}")
        block_id = np.load(OUTPUT_PATH)
        print(f"Loaded {len(block_id)} spot assignments")
        return block_id
    
    print("Building new breast cancer spatial blocks (one-time operation)")
    adata = load_breast_cancer()
    block_id = build_fixed_spatial_blocks(adata)
    
    # Save to disk
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, block_id)
    print(f"Saved breast cancer blocks to {OUTPUT_PATH}")
    
    # Print block statistics
    unique, counts = np.unique(block_id, return_counts=True)
    print("Block sizes:")
    for block, count in zip(unique, counts):
        role = "SELECTION" if block in SELECTION_BLOCKS else "REPORT"
        print(f"  Block {block}: {count} spots ({role})")
    
    return block_id


def load_breast_cancer_blocks(adata=None, force_rebuild=False):
    """Alias for load_or_build_blocks, accepting an optional adata argument for compatibility."""
    return load_or_build_blocks(force_rebuild=force_rebuild)


def spot_mask_for_blocks(block_id, blocks):
    """Get boolean mask indicating which spots are in the specified blocks."""
    return np.isin(block_id, list(blocks))


def get_selection_mask(block_id):
    """Get boolean mask for selection blocks (hyperparameter tuning)."""
    mask = np.isin(block_id, list(SELECTION_BLOCKS))
    print(f"Selection mask: {mask.sum()} spots in {len(SELECTION_BLOCKS)} blocks")
    return mask


def get_report_mask(block_id):
    """Get boolean mask for report blocks (final evaluation)."""
    mask = np.isin(block_id, list(REPORT_BLOCKS))
    print(f"Report mask: {mask.sum()} spots in {len(REPORT_BLOCKS)} blocks")
    return mask


if __name__ == "__main__":
    # Run the one-time block generation
    block_id = load_or_build_blocks()
    print("\n" + "="*50)
    print("Breast cancer spatial blocks partition complete")
    print("="*50)
    print(f"Total spots: {len(block_id)}")
    print(f"Selection blocks: {SELECTION_BLOCKS}")
    print(f"Report blocks: {REPORT_BLOCKS}")
    print(f"\nOutput saved to: {OUTPUT_PATH}")
    print("\nIMPORTANT: This file should be committed to git.")
    print("All subsequent architecture runs must load this exact partition.")