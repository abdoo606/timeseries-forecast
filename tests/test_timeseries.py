from __future__ import annotations

import pytest

from timeseries.forecast import (
    backtest,
    decompose,
    exponential_moving_average,
    forecast,
    seasonal_indices,
    simple_moving_average,
)

SERIES = [10, 12, 14, 16, 18, 20]
SEASONAL = [10, 20, 10, 20, 10, 20, 10, 20]


def test_sma_window():
    assert simple_moving_average([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]


def test_sma_invalid_window():
    with pytest.raises(ValueError):
        simple_moving_average([1, 2], 0)


def test_ema_starts_at_first_value():
    result = exponential_moving_average([5, 5, 5], 0.5)
    assert result[0] == 5 and all(v == pytest.approx(5) for v in result)


def test_ema_reacts_to_change():
    result = exponential_moving_average([0, 0, 0, 10], 0.5)
    assert result[-1] == pytest.approx(5.0)


def test_ema_validates_alpha():
    for alpha in (0, 1.5):
        with pytest.raises(ValueError):
            exponential_moving_average([1], alpha)


def test_ema_of_empty_series():
    assert exponential_moving_average([], 0.5) == []


def test_seasonal_indices_detect_the_pattern():
    indices = seasonal_indices(SEASONAL, 2)
    assert indices[1] > indices[0]


def test_seasonal_indices_default_to_one_without_enough_data():
    assert seasonal_indices([1, 2, 3], 4) == [1.0] * 4


def test_forecast_continues_the_trend():
    result = forecast(SERIES, 3)
    assert len(result) == 3
    assert all(b > a for a, b in zip(result, result[1:]))


def test_forecast_validation():
    with pytest.raises(ValueError):
        forecast([], 1)
    with pytest.raises(ValueError):
        forecast([1, 2], 0)


def test_forecast_is_damped_not_explosive():
    """A damped trend must not run away - a classic beginner bug."""
    result = forecast(SERIES, 50)
    assert result[-1] < SERIES[-1] * 10


def test_forecast_with_seasonality_varies():
    result = forecast(SEASONAL, 4, period=2)
    assert result[0] != result[1]


def test_backtest_reports_metrics():
    metrics = backtest(SERIES)
    assert metrics["n"] > 0
    assert metrics["rmse"] >= metrics["mae"] > 0


def test_backtest_on_perfect_linear_series_is_accurate():
    metrics = backtest([float(i) for i in range(20)], alpha=0.9)
    assert metrics["mae"] < 2


def test_backtest_short_series():
    assert backtest([1, 2])["n"] == 0


def test_decomposition_parts():
    result = decompose(SEASONAL, 2, window=3)
    assert len(result.trend) == len(SEASONAL)
    assert len(result.residual) == len(SEASONAL)
