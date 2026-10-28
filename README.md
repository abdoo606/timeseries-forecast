# 22 · timeseries

Moving averages, seasonal decomposition and walk-forward forecasts.

## Run

```bash
from timeseries.forecast import forecast, backtest
```

## Test

```bash
pytest projects/22-timeseries -q
```

## What it demonstrates

- SMA, EMA, multiplicative seasonal indices
- Damped trend so long horizons do not explode
- Walk-forward backtest: never sees the future it is predicting
- Reports MAE, RMSE and MAPE rather than a single flattering number
