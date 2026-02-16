from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def calibrate_gbm(spot_series: pd.Series, trading_days_per_year: int = 252) -> tuple[float, float]:
    log_returns = np.log(spot_series / spot_series.shift(1)).dropna()
    if len(log_returns) < 2:
        raise ValueError("Need at least 2 returns to calibrate GBM.")

    dt = 1.0 / 52.0
    mean_lr = float(log_returns.mean())
    var_lr = float(log_returns.var(ddof=1))

    sigma = math.sqrt(max(var_lr, 0.0) / dt)
    mu = (mean_lr / dt) + 0.5 * sigma * sigma
    return mu, sigma


def forward_price(spot: float, rd: float, rf: float, tenor_years: float) -> float:
    return float(spot * math.exp((rd - rf) * tenor_years))


def gk_option_price(
    spot: float,
    strike: float,
    rd: float,
    rf: float,
    vol: float,
    tenor_years: float,
    option_type: str = "call",
) -> float:
    if tenor_years <= 0.0:
        intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        return float(intrinsic)

    if vol <= 0.0:
        fwd = forward_price(spot, rd, rf, tenor_years)
        disc = math.exp(-rd * tenor_years)
        intrinsic_fwd = max(fwd - strike, 0.0) if option_type == "call" else max(strike - fwd, 0.0)
        return float(disc * intrinsic_fwd)

    sqrt_t = math.sqrt(tenor_years)
    d1 = (math.log(spot / strike) + (rd - rf + 0.5 * vol * vol) * tenor_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    spot_leg = spot * math.exp(-rf * tenor_years)
    strike_leg = strike * math.exp(-rd * tenor_years)

    if option_type == "call":
        return float(spot_leg * norm.cdf(d1) - strike_leg * norm.cdf(d2))
    if option_type == "put":
        return float(strike_leg * norm.cdf(-d2) - spot_leg * norm.cdf(-d1))
    raise ValueError("option_type must be 'call' or 'put'.")


def simulate_terminal_spot(
    spot_t0: float,
    mu: float,
    sigma: float,
    tenor_years: float,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    drift = (mu - 0.5 * sigma * sigma) * tenor_years
    diffusion = sigma * math.sqrt(tenor_years) * z
    return spot_t0 * np.exp(drift + diffusion)
