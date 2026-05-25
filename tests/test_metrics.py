import math

import pytest
import torch

from tnpinn.pinn.metrics import relative_l2


def test_relative_l2_is_zero_for_identical_prediction_and_reference() -> None:
    reference = torch.linspace(0.1, 1.0, 16, dtype=torch.float64).reshape(-1, 1)

    assert relative_l2(reference, reference) < 1.0e-10


def test_relative_l2_is_one_for_zero_prediction_and_nonzero_reference() -> None:
    reference = torch.linspace(0.1, 1.0, 16, dtype=torch.float64).reshape(-1, 1)
    prediction = torch.zeros_like(reference)

    assert relative_l2(prediction, reference) == pytest.approx(1.0, abs=1.0e-10)


def test_relative_l2_is_nan_without_reference() -> None:
    prediction = torch.ones(8, 1, dtype=torch.float64)

    assert math.isnan(relative_l2(prediction, None))
