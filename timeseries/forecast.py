"""Small forecasting toolkit: enough to be useful, honest about its error."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


def simple_moving_average(series: list[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window must be positive")
    out: list[float | None] = []
    for i in range(len(series)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(round(sum(series[i + 1 - window : i + 1]) / window, 4))
    return out


def exponential_moving_average(series: list[float], alpha: float) -> list[float]:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if not series:
        return []
    out = [series[0]]
    for value in series[1:]:
        out.append(round(alpha * value + (1 - alpha) * out[-1], 4))
    return out


def seasonal_indices(series: list[float], period: int) -> list[float]:
    """Multiplicative indices: how each position in the cycle compares to the local mean."""
    if period <= 1 or len(series) < period * 2:
        return [1.0] * period
    detrended = [v / m if (m := _centred_mean(series, i, period)) else 1.0 for i, v in enumerate(series)]
    buckets = [[] for _ in range(period)]
    for i, value in enumerate(detrended):
        buckets[i % period].append(value)
    indices = [statistics.fmean(b) if b else 1.0 for b in buckets]
    mean = statistics.fmean(indices) or 1.0
    return [round(i / mean, 4) for i in indices]


def _centred_mean(series: list[float], index: int, period: int) -> float:
    half = period // 2 or 1
    window = series[max(0, index - half) : index + half + 1]
    return statistics.fmean(window) if window else 0.0


def forecast(series: list[float], horizon: int, *, period: int = 0, window: int = 3, alpha: float = 0.4) -> list[float]:
    """Trend from an EMA, seasonality from the history. Deliberately simple and testable."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not series:
        raise ValueError("series must not be empty")
    smoothed = exponential_moving_average(series, alpha)
    base = smoothed[-1]
    trend = 0.0
    if len(smoothed) >= 2:
        trend = (smoothed[-1] - smoothed[-2]) * 0.5  # damped: half the last slope
    indices = seasonal_indices(series, period) if period > 1 else []
    out = []
    for step in range(1, horizon + 1):
        value = base + trend * step
        if indices:
            value *= indices[(len(series) + step - 1) % period]
        out.append(round(value, 4))
    return out


def backtest(series: list[float], *, horizon: int = 1, period: int = 0, alpha: float = 0.4) -> dict[str, float]:
    """Walk-forward validation: never look at future points when forecasting."""
    errors: list[float] = []
    for cut in range(max(3, len(series) // 2), len(series) - horizon + 1):
        predicted = forecast(series[:cut], horizon, period=period, alpha=alpha)
        actual = series[cut : cut + horizon]
        errors.extend(p - a for p, a in zip(predicted, actual))
    if not errors:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "n": 0}
    mae = statistics.fmean(abs(e) for e in errors)
    rmse = math.sqrt(statistics.fmean(e * e for e in errors))
    nonzero = [abs(e) for e in errors if e]
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(statistics.fmean(nonzero) * 100, 2) if nonzero else 0.0,
        "n": len(errors),
    }


@dataclass(slots=True, frozen=True)
class Decomposition:
    trend: list[float | None]
    seasonal: list[float]
    residual: list[float | None]


def decompose(series: list[float], period: int, window: int = 3) -> Decomposition:
    trend = simple_moving_average(series, window)
    seasonal = seasonal_indices(series, period) if period > 1 else [1.0] * max(1, len(series))
    residual = [
        round(series[i] - (trend[i] or series[i]) * (seasonal[i % len(seasonal)] if period > 1 else 1), 4)
        for i in range(len(series))
    ]
    return Decomposition(trend, seasonal, residual)
