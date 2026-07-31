import numpy as np
import pytest
import torch

from src.models.image_encoder import encode_patches, get_dinov2_encoder, get_frozen_encoder


def test_frozen_encoder_has_no_trainable_parameters():
    encoder = get_frozen_encoder(device=torch.device("cpu"))
    assert all(not p.requires_grad for p in encoder.parameters())


def test_encode_patches_output_shape():
    patches = np.random.rand(10, 64, 64, 3).astype(np.float32)
    device = torch.device("cpu")
    encoder = get_frozen_encoder(device=device)

    features = encode_patches(patches, encoder=encoder, device=device)

    assert features.shape == (10, 512)
    assert not np.isnan(features).any()


def test_encode_patches_deterministic_for_frozen_eval_mode():
    """Same input through the same frozen encoder must give identical
    features every time -- no dropout/batchnorm-in-train-mode nondeterminism."""
    patches = np.random.rand(4, 64, 64, 3).astype(np.float32)
    device = torch.device("cpu")
    encoder = get_frozen_encoder(device=device)

    features_a = encode_patches(patches, encoder=encoder, device=device)
    features_b = encode_patches(patches, encoder=encoder, device=device)

    assert np.allclose(features_a, features_b)


def test_encode_patches_distinguishes_different_images():
    blank = np.zeros((2, 64, 64, 3), dtype=np.float32)
    noisy = np.ones((2, 64, 64, 3), dtype=np.float32)
    patches = np.concatenate([blank, noisy], axis=0)
    device = torch.device("cpu")
    encoder = get_frozen_encoder(device=device)

    features = encode_patches(patches, encoder=encoder, device=device)

    assert not np.allclose(features[0], features[2])


def _dinov2_or_skip(device):
    try:
        return get_dinov2_encoder(device=device)
    except Exception as e:  # network unavailable in this environment/CI
        pytest.skip(f"DINOv2 hub download unavailable: {e}")


def test_dinov2_encoder_has_no_trainable_parameters():
    device = torch.device("cpu")
    encoder = _dinov2_or_skip(device)
    assert all(not p.requires_grad for p in encoder.parameters())


def test_encode_patches_with_dinov2_resize_output_shape():
    patches = np.random.rand(4, 64, 64, 3).astype(np.float32)
    device = torch.device("cpu")
    encoder = _dinov2_or_skip(device)

    features = encode_patches(patches, encoder=encoder, device=device, resize_to=224)

    assert features.shape == (4, 384)
    assert not np.isnan(features).any()
