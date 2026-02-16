from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import io
from contextlib import redirect_stderr, redirect_stdout
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from .config import Config


@dataclass
class MarketData:
    spot: pd.DataFrame
    wed_spot: pd.Series
    domestic_rate: pd.Series
    foreign_rate: pd.Series
    implied_vol: pd.Series


def _business_day_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end)


def _fallback_spot_series(cfg: Config) -> pd.Series:
    idx = _business_day_index(pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date))
    rng = np.random.default_rng(cfg.random_seed)
    log_rets = rng.normal(loc=0.0, scale=0.006, size=len(idx))
    series = pd.Series(np.exp(np.cumsum(log_rets)), index=idx)
    return 1.05 * series / series.iloc[0]


def _fetch_fred_series(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    try:
        from pandas_datareader.data import DataReader

        out = DataReader(series_id, "fred", start, end)
        if out.empty:
            return None
        return out.iloc[:, 0]
    except Exception:
        return None


def _safe_yf_download(ticker: str, cfg: Config) -> pd.DataFrame:
    sink = io.StringIO()
    try:
        # yfinance may print provider/network errors to stdout/stderr.
        with redirect_stdout(sink), redirect_stderr(sink):
            return yf.download(
                ticker,
                start=cfg.start_date,
                end=cfg.end_date + timedelta(days=1),
                auto_adjust=False,
                progress=False,
                threads=False,
            )
    except Exception:
        return pd.DataFrame()


def _extract_close_column(data: pd.DataFrame, ticker: str) -> pd.Series | None:
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            return data[("Close", ticker)].dropna()
        if ("Adj Close", ticker) in data.columns:
            return data[("Adj Close", ticker)].dropna()
        return None
    if "Close" in data.columns:
        return data["Close"].dropna()
    if "Adj Close" in data.columns:
        return data["Adj Close"].dropna()
    return None


def _load_spot_from_stooq(cfg: Config) -> pd.Series | None:
    try:
        from pandas_datareader.data import DataReader

        raw = DataReader("EURUSD", "stooq", cfg.start_date, cfg.end_date)
        if raw.empty or "Close" not in raw.columns:
            return None
        return raw["Close"].sort_index().dropna()
    except Exception:
        return None


def _load_spot(cfg: Config) -> pd.DataFrame:
    data = _safe_yf_download(cfg.asset, cfg)
    close = _extract_close_column(data, cfg.asset)
    if close is not None and not close.empty:
        return pd.DataFrame({"Close": close})

    close = _load_spot_from_stooq(cfg)
    if close is not None and not close.empty:
        warnings.warn("Yahoo spot download failed; used Stooq fallback for spot.")
        return pd.DataFrame({"Close": close})

    warnings.warn("Spot download failed; using synthetic fallback spot series.")
    close = _fallback_spot_series(cfg)
    return pd.DataFrame({"Close": close})


def _load_rates(index: pd.DatetimeIndex, cfg: Config) -> tuple[pd.Series, pd.Series]:
    start, end = index.min(), index.max()

    usd_3m = _fetch_fred_series("DTB3", start, end)
    eur_3m = _fetch_fred_series("EUR3MTD156N", start, end)

    if usd_3m is None:
        warnings.warn("Could not fetch US 3M rates from FRED; using fallback scalar.")
        usd = pd.Series(cfg.fallback_domestic_rate, index=index)
    else:
        usd = (usd_3m / 100.0).reindex(index).ffill().bfill()

    if eur_3m is None:
        warnings.warn("Could not fetch EUR 3M rates from FRED; using fallback scalar.")
        eur = pd.Series(cfg.fallback_foreign_rate, index=index)
    else:
        eur = (eur_3m / 100.0).reindex(index).ffill().bfill()

    return usd, eur


def _load_implied_vol(spot_close: pd.Series, cfg: Config) -> pd.Series:
    iv_data = _safe_yf_download(cfg.iv_ticker, cfg)
    close = _extract_close_column(iv_data, cfg.iv_ticker)
    if close is not None and not close.empty:
        iv = (close / 100.0).reindex(spot_close.index).ffill().bfill()
        return iv

    warnings.warn("IV download failed; using rolling realized volatility proxy.")
    log_rets = np.log(spot_close / spot_close.shift(1)).dropna()
    rv = log_rets.rolling(cfg.iv_rolling_days).std() * np.sqrt(cfg.trading_days_per_year)
    rv = rv.reindex(spot_close.index).ffill().bfill()
    return rv


def load_market_data(cfg: Config) -> MarketData:
    spot = _load_spot(cfg)
    spot = spot[~spot.index.duplicated(keep="last")].sort_index()

    dom_rate, for_rate = _load_rates(spot.index, cfg)
    iv = _load_implied_vol(spot["Close"], cfg)

    wed_spot = spot.loc[spot.index.weekday == 2, "Close"].dropna()

    return MarketData(
        spot=spot,
        wed_spot=wed_spot,
        domestic_rate=dom_rate,
        foreign_rate=for_rate,
        implied_vol=iv,
    )
