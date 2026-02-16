from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .analytics import calibrate_gbm, gk_option_price, simulate_terminal_spot
from .config import Config
from .data_loader import MarketData
from .translator import translate_strike_spot_ratio, translate_strike_vol_adjusted


@dataclass
class BacktestResult:
    summary: pd.DataFrame


class BacktestEngine:
    def __init__(self, cfg: Config, market_data: MarketData) -> None:
        self.cfg = cfg
        self.md = market_data

    def _nearest_prior_index_date(self, target: pd.Timestamp) -> pd.Timestamp | None:
        idx = self.md.spot.index[self.md.spot.index <= target]
        return idx[-1] if len(idx) else None

    def _choose_hop_dates(self) -> list[pd.Timestamp]:
        t_now = self.md.spot.index.max()
        hops: list[pd.Timestamp] = []

        for i in range(1, self.cfg.num_hops + 1):
            target = t_now - pd.Timedelta(days=i * self.cfg.hop_interval_days)
            hop = self._nearest_prior_index_date(target)
            if hop is None:
                continue
            if not hops or hop < hops[-1]:
                hops.append(hop)
        return hops

    @staticmethod
    def _empirical_cdf(samples: np.ndarray, x: float) -> float:
        return float(np.mean(samples <= x))

    def _position_sign(self) -> float:
        side = self.cfg.position_side.lower().strip()
        if side == "long":
            return 1.0
        if side == "short":
            return -1.0
        raise ValueError("position_side must be 'long' or 'short'.")

    def _resolve_today_strikes(self, spot_now: float) -> tuple[float, float]:
        structure = self.cfg.portfolio_structure.lower().strip()

        if structure == "straddle":
            base = (
                float(self.cfg.trade_strike_override)
                if self.cfg.trade_strike_override is not None
                else spot_now * float(self.cfg.trade_moneyness)
            )
            strike_call = float(self.cfg.call_strike_override) if self.cfg.call_strike_override is not None else base
            strike_put = float(self.cfg.put_strike_override) if self.cfg.put_strike_override is not None else base
            return strike_call, strike_put

        if structure == "strangle":
            strike_call = (
                float(self.cfg.call_strike_override)
                if self.cfg.call_strike_override is not None
                else spot_now * float(self.cfg.call_moneyness)
            )
            strike_put = (
                float(self.cfg.put_strike_override)
                if self.cfg.put_strike_override is not None
                else spot_now * float(self.cfg.put_moneyness)
            )
            return strike_call, strike_put

        raise ValueError("portfolio_structure must be 'straddle' or 'strangle'.")

    def _build_current_trade(self) -> tuple[pd.Timestamp, float, float, float, float]:
        t_now = self.md.spot.index.max()
        spot_now = float(self.md.spot.loc[t_now, "Close"])
        vol_now = float(self.md.implied_vol.loc[t_now])
        strike_call_now, strike_put_now = self._resolve_today_strikes(spot_now)
        if min(strike_call_now, strike_put_now, vol_now, spot_now) <= 0.0:
            raise ValueError("Spot, vol, and call/put strikes must be strictly positive.")
        return t_now, spot_now, strike_call_now, strike_put_now, vol_now

    @staticmethod
    def _portfolio_value(spot: float, strike_call: float, strike_put: float, rd: float, rf: float, vol: float, tenor: float) -> float:
        call_val = gk_option_price(spot, strike_call, rd, rf, vol, tenor, "call")
        put_val = gk_option_price(spot, strike_put, rd, rf, vol, tenor, "put")
        return float(call_val + put_val)

    def run(self) -> BacktestResult:
        t_now, spot_now, strike_call_now, strike_put_now, vol_now = self._build_current_trade()
        position_scale = self._position_sign() * float(self.cfg.position_units)
        tenor_years = self.cfg.trade_maturity_days / self.cfg.trading_days_per_year
        std_distance_call_today = math.log(strike_call_now / spot_now) / (vol_now * math.sqrt(tenor_years))
        std_distance_put_today = math.log(strike_put_now / spot_now) / (vol_now * math.sqrt(tenor_years))
        hops = self._choose_hop_dates()

        rows: list[dict[str, float | str | pd.Timestamp]] = []

        for i, t_hop in enumerate(hops):
            t_end = self._nearest_prior_index_date(t_hop + pd.Timedelta(days=self.cfg.hop_interval_days))
            if t_end is None or t_end <= t_hop:
                continue

            wed_hist = self.md.wed_spot[self.md.wed_spot.index < t_hop].tail(self.cfg.calibration_weeks)
            if len(wed_hist) < self.cfg.calibration_weeks:
                continue

            mu, sigma = calibrate_gbm(wed_hist, self.cfg.trading_days_per_year)

            spot_hop = float(self.md.spot.loc[t_hop, "Close"])
            spot_end = float(self.md.spot.loc[t_end, "Close"])
            rd = float(self.md.domestic_rate.loc[t_hop])
            rf = float(self.md.foreign_rate.loc[t_hop])
            vol_hop = float(self.md.implied_vol.loc[t_hop])

            strike_call_a = translate_strike_spot_ratio(strike_call_now, spot_now, spot_hop)
            strike_put_a = translate_strike_spot_ratio(strike_put_now, spot_now, spot_hop)

            strike_call_b = translate_strike_vol_adjusted(
                strike_call_now, spot_now, spot_hop, vol_now, vol_hop, tenor_years
            )
            strike_put_b = translate_strike_vol_adjusted(
                strike_put_now, spot_now, spot_hop, vol_now, vol_hop, tenor_years
            )

            terminal_spots = simulate_terminal_spot(
                spot_t0=spot_hop,
                mu=mu,
                sigma=sigma,
                tenor_years=tenor_years,
                n_paths=self.cfg.sim_paths,
                seed=self.cfg.random_seed + i,
            )

            sim_vals_a = np.array(
                [
                    position_scale
                    * self._portfolio_value(s, strike_call_a, strike_put_a, rd, rf, vol_hop, tenor_years)
                    for s in terminal_spots
                ]
            )
            sim_vals_b = np.array(
                [
                    position_scale
                    * self._portfolio_value(s, strike_call_b, strike_put_b, rd, rf, vol_hop, tenor_years)
                    for s in terminal_spots
                ]
            )

            realized_a = position_scale * self._portfolio_value(
                spot_end, strike_call_a, strike_put_a, rd, rf, vol_hop, tenor_years
            )
            realized_b = position_scale * self._portfolio_value(
                spot_end, strike_call_b, strike_put_b, rd, rf, vol_hop, tenor_years
            )

            pit_a = self._empirical_cdf(sim_vals_a, realized_a)
            pit_b = self._empirical_cdf(sim_vals_b, realized_b)

            rows.append(
                {
                    "asof_date": t_now,
                    "hop_date": t_hop,
                    "horizon_date": t_end,
                    "tenor_years": tenor_years,
                    "spot_today": spot_now,
                    "vol_today": vol_now,
                    "portfolio_label": self.cfg.portfolio_label,
                    "portfolio_structure": self.cfg.portfolio_structure,
                    "position_side": self.cfg.position_side,
                    "position_units": self.cfg.position_units,
                    "strike_call_today": strike_call_now,
                    "strike_put_today": strike_put_now,
                    "call_moneyness_today": strike_call_now / spot_now,
                    "put_moneyness_today": strike_put_now / spot_now,
                    "std_distance_call_today": std_distance_call_today,
                    "std_distance_put_today": std_distance_put_today,
                    "mu": mu,
                    "sigma": sigma,
                    "spot_hop": spot_hop,
                    "vol_hop": vol_hop,
                    "spot_realized": spot_end,
                    "rd": rd,
                    "rf": rf,
                    "strike_call_a": strike_call_a,
                    "strike_put_a": strike_put_a,
                    "strike_call_b": strike_call_b,
                    "strike_put_b": strike_put_b,
                    "strike_call_diff_abs": strike_call_b - strike_call_a,
                    "strike_put_diff_abs": strike_put_b - strike_put_a,
                    "realized_a": realized_a,
                    "realized_b": realized_b,
                    "pit_a": pit_a,
                    "pit_b": pit_b,
                    "pit_diff": pit_b - pit_a,
                }
            )

        summary = pd.DataFrame(rows).sort_values("hop_date").reset_index(drop=True) if rows else pd.DataFrame()
        return BacktestResult(summary=summary)
