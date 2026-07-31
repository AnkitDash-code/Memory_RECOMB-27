import numpy as np
import pytest

from src.data.extract_patches import extract_spot_patches
from src.data.load_visium import load_visium_crop


def test_extract_spot_patches_shape_and_count():
    adata = load_visium_crop()
    patches = extract_spot_patches(adata, patch_size=64)

    assert patches.shape == (adata.n_obs, 64, 64, 3)
    assert patches.dtype == np.float32


def test_extract_spot_patches_center_pixel_is_finite_and_in_image_range():
    adata = load_visium_crop()
    patches = extract_spot_patches(adata, patch_size=32)

    assert not np.isnan(patches).any()
    # Squidpy Visium images are stored normalized to [0, 1].
    assert patches.min() >= 0.0
    assert patches.max() <= 1.0 + 1e-5


def test_extract_spot_patches_different_spots_give_different_patches():
    """A coordinate-scaling bug would silently produce identical or
    nonsensical patches for every spot -- catch that directly."""
    adata = load_visium_crop()
    patches = extract_spot_patches(adata, patch_size=32)

    # Compare a handful of spots pairwise; real tissue patches should differ.
    n_distinct = len({patches[i].tobytes() for i in range(0, min(10, len(patches)))})
    assert n_distinct > 1


def test_extract_spot_patches_boundary_spot_is_edge_padded_not_cropped():
    """A spot near the image border must still produce a full-size patch
    (edge-padded), not a truncated one -- shape must stay uniform for
    batched encoder inference."""
    adata = load_visium_crop()
    # Force one spot's coordinate to the extreme corner of the image.
    lib = list(adata.uns["spatial"].keys())[0]
    img = adata.uns["spatial"][lib]["images"]["hires"]
    scale = adata.uns["spatial"][lib]["scalefactors"]["tissue_hires_scalef"]
    adata.obsm["spatial"] = adata.obsm["spatial"].copy()
    adata.obsm["spatial"][0] = np.array([0, 0]) / scale  # maps to pixel (0, 0)

    patches = extract_spot_patches(adata, patch_size=64)

    assert patches[0].shape == (64, 64, 3)
    assert not np.isnan(patches[0]).any()


def test_extract_spot_patches_raises_on_ambiguous_library_key():
    adata = load_visium_crop()
    lib = list(adata.uns["spatial"].keys())[0]
    adata.uns["spatial"]["second_library"] = adata.uns["spatial"][lib]

    with pytest.raises(ValueError):
        extract_spot_patches(adata, patch_size=32)
