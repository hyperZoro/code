from __future__ import annotations

import math


def translate_strike_spot_ratio(strike_today: float, spot_today: float, spot_hop: float) -> float:
    if spot_today <= 0.0 or spot_hop <= 0.0:
        raise ValueError("Spots must be positive for strike translation.")
    moneyness = strike_today / spot_today
    return float(moneyness * spot_hop)


def translate_strike_vol_adjusted(
    strike_today: float,
    spot_today: float,
    spot_hop: float,
    vol_today: float,
    vol_hop: float,
    tenor_years: float,
) -> float:
    if min(spot_today, spot_hop, vol_today, vol_hop, tenor_years) <= 0.0:
        raise ValueError("Inputs must be positive for vol-adjusted strike translation.")

    std_distance_today = math.log(strike_today / spot_today) / (vol_today * math.sqrt(tenor_years))
    translated = spot_hop * math.exp(std_distance_today * vol_hop * math.sqrt(tenor_years))
    return float(translated)
