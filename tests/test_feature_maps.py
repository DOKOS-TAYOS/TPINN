import torch

from tnpinn.tn.feature_maps import build_feature_map


def test_fourier_sites_shape_dtype_device_and_grad() -> None:
    coords = torch.rand(5, 2, dtype=torch.float64, requires_grad=True)
    fmap = build_feature_map(
        {
            "kind": "fourier_sites",
            "site_dim": 2,
            "n_sites_per_coord": {"x": 3, "y": 2},
            "frequency_scale": 3.141592653589793,
        },
        coordinates=("x", "y"),
    )

    features = fmap(coords)
    assert features.shape == (5, 5, 2)
    assert features.dtype == coords.dtype
    assert features.device == coords.device
    assert not torch.allclose(features[0], features[1])

    features.sum().backward()
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()


def test_polynomial_and_rbf_sites_have_expected_shape() -> None:
    coords = torch.rand(4, 2, dtype=torch.float32, requires_grad=True)
    for kind in ("polynomial_sites", "rbf_sites"):
        fmap = build_feature_map(
            {"kind": kind, "site_dim": 3, "n_sites_per_coord": {"x": 2, "y": 2}},
            coordinates=("x", "y"),
        )
        assert fmap(coords).shape == (4, 4, 3)
