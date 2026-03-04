"""Tests for core.forecasting.base"""
import numpy as np
from decision_intelligence.core.forecasting.base import ForecastResult


def test_uncertainty_is_none_without_quantiles():
    fr = ForecastResult(point=np.array([1.0]))
    assert fr.uncertainty is None


def test_uncertainty_computed_from_quantiles():
    fr = ForecastResult(
        point=np.array([1.0]),
        quantiles={"0.05": np.array([0.5]), "0.95": np.array([1.5])},
    )
    assert fr.uncertainty is not None
    assert float(fr.uncertainty[0]) == 1.0
