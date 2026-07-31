"""Frozen morphology feature extractor for Visium histology patches.

ResNet18 (ImageNet-pretrained) over DINO/DINOv2 to start: smaller, faster,
fewer moving parts to debug first. A ViT-based self-supervised encoder is a
legitimate upgrade later *if* this cheap version already shows signal in the
Section 2 diagnostic -- not before.

Frozen, not fine-tuned: features are precomputed once per slice and cached,
never touching the GPU during model training. This keeps VRAM cheap and
avoids a small (~4000-spot) dataset overfitting a fine-tuned CNN.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

CACHE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "cache"

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_frozen_encoder(device=None):
    """512-dim frozen ResNet18 feature extractor (final fc replaced with Identity)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def encode_patches(patches, encoder=None, device=None, batch_size=256):
    """patches: (n_spots, H, W, 3) float32 in [0, 1] (as produced by
    extract_spot_patches). Returns (n_spots, 512) float32 numpy array.

    ImageNet normalization is applied here (mean/std), matching what
    ResNet18 was pretrained on -- patches are otherwise used as-is, no
    resizing beyond whatever patch_size was extracted at (64x64 works
    directly with a fully-convolutional ResNet18 backbone + global pool).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder or get_frozen_encoder(device=device)

    x = torch.from_numpy(patches).permute(0, 3, 1, 2).float()  # (N, 3, H, W)
    mean = _IMAGENET_MEAN.to(device)
    std = _IMAGENET_STD.to(device)

    features = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = x[start:start + batch_size].to(device)
            batch = (batch - mean) / std
            out = encoder(batch)
            features.append(out.cpu().numpy())
    return np.concatenate(features, axis=0).astype(np.float32)


def encode_and_cache_features(patches, slice_id, encoder=None, device=None,
                                cache_dir=CACHE_DIR, force=False):
    """encode_patches, cached to outputs/cache/img_features_{slice_id}.npy."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"img_features_{slice_id}.npy"
    if path.exists() and not force:
        features = np.load(path)
        assert features.shape[0] == patches.shape[0], (
            f"Cached feature count {features.shape[0]} != patch count {patches.shape[0]} "
            f"for {slice_id} -- stale cache, delete and re-extract."
        )
        return features

    features = encode_patches(patches, encoder=encoder, device=device)
    np.save(path, features)
    return features
