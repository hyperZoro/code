from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .data_loader import load_market_data
from .engine import BacktestEngine
from .reporting import create_histograms, create_hop_bar_chart, evaluate


def run_backtest(output_dir: Path | str = "python/fx_trade_translation_backtest/output") -> None:
    cfg = Config()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    md = load_market_data(cfg)
    engine = BacktestEngine(cfg, md)
    result = engine.run()

    if result.summary.empty:
        raise RuntimeError("No backtest results were generated. Check date range and data availability.")

    summary_path = out / "backtest_summary.csv"
    translated_trades_path = out / "translated_trades_by_hop.csv"
    portfolio_path = out / "portfolio_definition.csv"
    stats_path = out / "ks_stats.csv"
    plot_path = out / "pit_histograms.png"
    hop_plot_path = out / "pit_hop_comparison.png"

    result.summary.to_csv(summary_path, index=False)
    translation_cols = [
        "asof_date",
        "hop_date",
        "horizon_date",
        "portfolio_label",
        "portfolio_structure",
        "position_side",
        "position_units",
        "spot_today",
        "vol_today",
        "strike_call_today",
        "strike_put_today",
        "call_moneyness_today",
        "put_moneyness_today",
        "std_distance_call_today",
        "std_distance_put_today",
        "spot_hop",
        "vol_hop",
        "strike_call_a",
        "strike_put_a",
        "strike_call_b",
        "strike_put_b",
        "strike_call_diff_abs",
        "strike_put_diff_abs",
        "pit_a",
        "pit_b",
        "pit_diff",
    ]
    result.summary[translation_cols].to_csv(translated_trades_path, index=False)
    portfolio_df = pd.DataFrame(
        [
            {
                "asof_date": result.summary.loc[0, "asof_date"],
                "asset": cfg.asset,
                "portfolio_label": cfg.portfolio_label,
                "portfolio_structure": cfg.portfolio_structure,
                "position_side": cfg.position_side,
                "position_units": cfg.position_units,
                "trade_maturity_days": cfg.trade_maturity_days,
                "trade_moneyness": cfg.trade_moneyness,
                "trade_strike_override": cfg.trade_strike_override,
                "call_moneyness": cfg.call_moneyness,
                "put_moneyness": cfg.put_moneyness,
                "call_strike_override": cfg.call_strike_override,
                "put_strike_override": cfg.put_strike_override,
                "spot_today": result.summary.loc[0, "spot_today"],
                "strike_call_today": result.summary.loc[0, "strike_call_today"],
                "strike_put_today": result.summary.loc[0, "strike_put_today"],
                "vol_today": result.summary.loc[0, "vol_today"],
                "std_distance_call_today": result.summary.loc[0, "std_distance_call_today"],
                "std_distance_put_today": result.summary.loc[0, "std_distance_put_today"],
                "pit_metric": "portfolio_value_cdf",
                "pit_note": "Portfolio is long call + long put (straddle/strangle), which is non-monotonic in spot.",
            }
        ]
    )
    portfolio_df.to_csv(portfolio_path, index=False)
    stats = evaluate(result.summary)
    stats.to_csv(stats_path, index=False)
    create_histograms(result.summary, plot_path)
    create_hop_bar_chart(result.summary, hop_plot_path)

    print(f"Saved backtest summary: {summary_path}")
    print(f"Saved translated-trade diagnostics: {translated_trades_path}")
    print(f"Saved portfolio definition: {portfolio_path}")
    print(f"Saved KS statistics: {stats_path}")
    print(f"Saved histogram plot: {plot_path}")
    print(f"Saved hop-date comparison plot: {hop_plot_path}")


if __name__ == "__main__":
    run_backtest()
