"""Synthetic 5-minute NSE-like candles for testing the validation harness.

This is NOT market data and says nothing about real profitability. It exists so
the harness can be checked on worlds where the truth is known:
  * edge = 0.0  -> pure random walk: a sound gate must not pass.
  * edge > 0.0  -> continuation after high-volume opening bars is planted: a
                   sound gate should detect it given enough sessions.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

BARS_PER_DAY = 75  # 09:15 .. 15:25 bar starts


def make_synthetic_market(
    n_symbols: int = 30,
    n_sessions: int = 300,
    *,
    seed: int = 0,
    edge: float = 0.0,
    in_play_prob: float = 0.10,
    start: date = date(2023, 1, 2),
) -> dict[str, pd.DataFrame]:
    """Return {symbol: OHLCV frame} with an IST DatetimeIndex.

    ``edge`` is the per-bar drift, in units of the bar's own volatility, applied
    in the direction of the opening bar on "in play" days only (days with
    abnormally high opening volume). Everything else is driftless.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start=start, periods=n_sessions)
    offsets = pd.to_timedelta(np.arange(BARS_PER_DAY) * 5 + 9 * 60 + 15, unit="m")
    profile = np.ones(BARS_PER_DAY)
    profile[0], profile[1:6] = 4.0, 2.0          # heavy open
    profile[-6:] = 1.6                            # busy close
    frames: dict[str, pd.DataFrame] = {}
    for k in range(n_symbols):
        price0 = float(np.exp(rng.uniform(np.log(120), np.log(3000))))
        daily_sigma = float(rng.uniform(0.012, 0.025))
        base_volume = float(np.exp(rng.uniform(np.log(2e4), np.log(2e5))))
        bar_sigma_base = daily_sigma / np.sqrt(BARS_PER_DAY)
        price = price0
        parts: list[pd.DataFrame] = []
        for day in days:
            in_play = rng.random() < in_play_prob
            vol_mult = 1.6 if in_play else 1.0
            bar_sigma = bar_sigma_base * vol_mult
            rets = rng.normal(0.0, bar_sigma, BARS_PER_DAY)
            rets[0] = rng.normal(0.0, bar_sigma * 3.0)
            if in_play and edge != 0.0:
                direction = 1.0 if rets[0] > 0 else -1.0
                rets[1:] += direction * edge * bar_sigma
            closes = price * np.exp(np.cumsum(rets))
            opens = np.concatenate(([price], closes[:-1]))
            wick = np.abs(rng.normal(0.0, bar_sigma * 0.5, (2, BARS_PER_DAY)))
            highs = np.maximum(opens, closes) * (1 + wick[0])
            lows = np.minimum(opens, closes) * (1 - wick[1])
            volume = base_volume * profile * np.exp(rng.normal(0.0, 0.35, BARS_PER_DAY))
            if in_play:
                volume[:6] *= rng.uniform(2.5, 4.5)
            index = (pd.Timestamp(day).tz_localize("Asia/Kolkata") + offsets)
            parts.append(pd.DataFrame(
                {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume.round()},
                index=index,
            ))
            price = float(closes[-1])
        frames[f"SYM{k:02d}"] = pd.concat(parts)
    return frames
