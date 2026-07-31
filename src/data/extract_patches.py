"""Extract fixed-size histology patches centered on each Visium spot.

Part of the dual-modality (expression + morphology) memory-addressing plan:
before any image feature can address a memory slot, it needs a per-spot image
crop. This module only does that -- no encoder, no model -- so the extraction
itself can be verified in isolation.

Coordinate handling, the one place a silent, severe bug can hide: Visium spot
pixel coordinates in `adata.obsm['spatial']` are in FULL-RESOLUTION image
pixel space, but the cached image in `adata.uns['spatial'][lib]['images']`
is a downscaled ('hires' or 'lowres') copy. The correct scale factor
(`tissue_hires_scalef` / `tissue_lowres_scalef`) must be applied before
indexing into that image, or every patch is silently misaligned without any
error being raised.
"""

from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "cache"


def _library_key(adata):
    keys = list(adata.uns["spatial"].keys())
    if len(keys) != 1:
        raise ValueError(f"Expected exactly one spatial library key, found {keys}")
    return keys[0]


def extract_spot_patches(adata, patch_size=64, resolution="hires", library_key=None):
    """Return (n_spots, patch_size, patch_size, 3) float32 array, one patch per spot,
    in the same row order as `adata.obs_names` / `adata.obsm['spatial']`.

    Patches that would extend past the image border are edge-padded (via
    `np.pad`) rather than silently clipped to a smaller size -- every patch
    must have identical shape for batched encoder inference, and clipping
    would quietly shrink patches for boundary spots instead of raising.
    """
    library_key = library_key or _library_key(adata)
    spatial_entry = adata.uns["spatial"][library_key]
    img = spatial_entry["images"][resolution]
    scale = spatial_entry["scalefactors"][f"tissue_{resolution}_scalef"]

    coords = adata.obsm["spatial"] * scale
    half = patch_size // 2
    h, w = img.shape[0], img.shape[1]

    # Pad once so every patch can be a simple, uniform-size slice -- avoids
    # per-spot conditional clipping logic that would be easy to get subtly wrong.
    padded = np.pad(img, ((half, half), (half, half), (0, 0)), mode="edge")

    patches = np.empty((len(coords), patch_size, patch_size, img.shape[2]), dtype=img.dtype)
    for i, (x, y) in enumerate(coords):
        cx, cy = int(round(x)) + half, int(round(y)) + half
        patches[i] = padded[cy - half:cy + half, cx - half:cx + half]

    assert patches.shape[0] == adata.n_obs, (
        f"Patch count {patches.shape[0]} != n_obs {adata.n_obs} -- "
        "coordinate/spot mismatch, do not trust this cache."
    )
    return patches.astype(np.float32)


def extract_and_cache_patches(adata, slice_id, patch_size=64, resolution="hires",
                                library_key=None, cache_dir=CACHE_DIR, force=False):
    """extract_spot_patches, cached to outputs/cache/patches_{slice_id}.npy."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"patches_{slice_id}.npy"
    if path.exists() and not force:
        patches = np.load(path)
        assert patches.shape[0] == adata.n_obs, (
            f"Cached patch count {patches.shape[0]} != n_obs {adata.n_obs} for {slice_id} "
            "-- stale cache from a different slice/version, delete and re-extract."
        )
        return patches

    patches = extract_spot_patches(
        adata, patch_size=patch_size, resolution=resolution, library_key=library_key
    )
    np.save(path, patches)
    return patches
