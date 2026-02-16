from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Config:
    asset: str = "EURUSD=X"
    start_date: date = date(2018, 1, 1)
    end_date: date = date(2024, 12, 31)

    num_hops: int = 52
    hop_interval_days: int = 14
    sim_paths: int = 1000
    random_seed: int = 42

    calibration_weeks: int = 156
    trading_days_per_year: int = 252

    fallback_domestic_rate: float = 0.04
    fallback_foreign_rate: float = 0.02

    iv_ticker: str = "^EVZ"
    iv_rolling_days: int = 30

    portfolio_label: str = "long_fx_strangle"
    portfolio_structure: str = "strangle"  # straddle | strangle
    position_side: str = "long"
    position_units: float = 1.0  # Applied to each leg
    trade_maturity_days: int = 21
    trade_moneyness: float = 1.0
    trade_strike_override: float | None = None
    call_moneyness: float = 1.05
    put_moneyness: float = 0.95
    call_strike_override: float | None = None
    put_strike_override: float | None = None
