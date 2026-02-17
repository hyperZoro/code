from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kstest


def _position_sign(position_side: str) -> float:
    return 1.0 if str(position_side).lower() == "long" else -1.0


def _portfolio_payoff(spot_grid: np.ndarray, strike_call: float, strike_put: float, position_side: str, units: float) -> np.ndarray:
    sign = _position_sign(position_side)
    call_payoff = np.maximum(spot_grid - strike_call, 0.0)
    put_payoff = np.maximum(strike_put - spot_grid, 0.0)
    return sign * float(units) * (call_payoff + put_payoff)


def create_translation_dashboard(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = summary.sort_values("hop_date").reset_index(drop=True)
    if data.empty:
        return

    first = data.iloc[0]
    labels = pd.to_datetime(data["hop_date"]).dt.strftime("%Y-%m-%d").tolist()

    strike_candidates = [
        float(first["strike_call_today"]),
        float(first["strike_put_today"]),
        float(data["strike_call_a"].min()),
        float(data["strike_put_a"].min()),
        float(data["strike_call_b"].min()),
        float(data["strike_put_b"].min()),
        float(data["spot_hop"].min()),
        float(data["spot_hop"].max()),
    ]
    s_min = min(strike_candidates) * 0.7
    s_max = max(strike_candidates) * 1.3
    spot_grid = np.linspace(max(1e-6, s_min), s_max, 500)

    current_payoff = _portfolio_payoff(
        spot_grid,
        float(first["strike_call_today"]),
        float(first["strike_put_today"]),
        str(first["position_side"]),
        float(first["position_units"]),
    )

    payoff_a_all = []
    payoff_b_all = []
    for _, row in data.iterrows():
        payoff_a_all.append(
            _portfolio_payoff(
                spot_grid,
                float(row["strike_call_a"]),
                float(row["strike_put_a"]),
                str(row["position_side"]),
                float(row["position_units"]),
            )
        )
        payoff_b_all.append(
            _portfolio_payoff(
                spot_grid,
                float(row["strike_call_b"]),
                float(row["strike_put_b"]),
                str(row["position_side"]),
                float(row["position_units"]),
            )
        )

    y_min = min(float(np.min(current_payoff)), float(np.min(np.array(payoff_a_all))), float(np.min(np.array(payoff_b_all))))
    y_max = max(float(np.max(current_payoff)), float(np.max(np.array(payoff_a_all))), float(np.max(np.array(payoff_b_all))))
    y_pad = 0.1 * (y_max - y_min if y_max > y_min else 1.0)

    vol_series = data["vol_hop"].astype(float).values

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(
            "Spot vs Portfolio Payoff",
            "Implied Vol by Hop Date",
            "PIT Comparison by Hop Date",
        ),
        vertical_spacing=0.12,
    )

    # Static traces always visible.
    fig.add_trace(
        go.Scatter(x=spot_grid, y=current_payoff, mode="lines", name="Current Trade", line=dict(color="gray", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([spot_grid, [None]]),
            y=np.concatenate([np.zeros_like(spot_grid), [None]]),
            mode="lines",
            name="Zero Payoff",
            line=dict(color="lightgray", width=1, dash="dot"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=labels, y=vol_series, mode="lines+markers", name="Hop IV", line=dict(color="#1f77b4", width=2)),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[float(first["vol_today"])] * len(labels),
            mode="lines",
            name="Today IV",
            line=dict(color="gray", width=1.5, dash="dash"),
        ),
        row=1,
        col=2,
    )

    static_count = len(fig.data)
    frame_indices: list[list[int]] = []

    for i, row in data.iterrows():
        start_idx = len(fig.data)
        frame_trace_ids: list[int] = []

        fig.add_trace(
            go.Scatter(
                x=spot_grid,
                y=payoff_a_all[i],
                mode="lines",
                name="Method A",
                line=dict(color="#1f77b4", width=3),
                visible=(i == 0),
                showlegend=(i == 0),
            ),
            row=1,
            col=1,
        )
        frame_trace_ids.append(start_idx)

        fig.add_trace(
            go.Scatter(
                x=spot_grid,
                y=payoff_b_all[i],
                mode="lines",
                name="Method B",
                line=dict(color="#ff7f0e", width=3, dash="dash"),
                visible=(i == 0),
                showlegend=(i == 0),
            ),
            row=1,
            col=1,
        )
        frame_trace_ids.append(start_idx + 1)

        fig.add_trace(
            go.Scatter(
                x=[float(row["spot_hop"]), float(row["spot_hop"])],
                y=[y_min - y_pad, y_max + y_pad],
                mode="lines",
                name="Spot@Hop",
                line=dict(color="black", width=1, dash="dot"),
                visible=(i == 0),
                showlegend=(i == 0),
            ),
            row=1,
            col=1,
        )
        frame_trace_ids.append(start_idx + 2)

        fig.add_trace(
            go.Scatter(
                x=[labels[i]],
                y=[float(row["vol_hop"])],
                mode="markers",
                name="Selected Hop",
                marker=dict(color="red", size=10),
                visible=(i == 0),
                showlegend=(i == 0),
            ),
            row=1,
            col=2,
        )
        frame_trace_ids.append(start_idx + 3)

        colors_a = ["rgba(31,119,180,0.25)"] * len(data)
        colors_b = ["rgba(255,127,14,0.25)"] * len(data)
        colors_a[i] = "rgba(31,119,180,1.0)"
        colors_b[i] = "rgba(255,127,14,1.0)"

        fig.add_trace(
            go.Bar(
                x=labels,
                y=data["pit_a"].astype(float).values,
                name="PIT A",
                marker=dict(color=colors_a),
                visible=(i == 0),
                showlegend=(i == 0),
            ),
            row=2,
            col=1,
        )
        frame_trace_ids.append(start_idx + 4)

        fig.add_trace(
            go.Bar(
                x=labels,
                y=data["pit_b"].astype(float).values,
                name="PIT B",
                marker=dict(color=colors_b),
                visible=(i == 0),
                showlegend=(i == 0),
            ),
            row=2,
            col=1,
        )
        frame_trace_ids.append(start_idx + 5)

        frame_indices.append(frame_trace_ids)

    n_total = len(fig.data)
    steps = []
    for i, trace_ids in enumerate(frame_indices):
        visible = [False] * n_total
        for s in range(static_count):
            visible[s] = True
        for t in trace_ids:
            visible[t] = True

        row = data.iloc[i]
        title = (
            f"Translation Dashboard | Hop {labels[i]} | "
            f"IV={float(row['vol_hop']):.4f} | PIT A={float(row['pit_a']):.3f} | "
            f"PIT B={float(row['pit_b']):.3f} | diff={float(row['pit_diff']):+.3f}"
        )

        steps.append(
            {
                "method": "update",
                "args": [{"visible": visible}, {"title": {"text": title}}],
                "label": labels[i],
            }
        )

    fig.update_layout(
        title={"text": steps[0]["args"][1]["title"]["text"]},
        height=860,
        barmode="group",
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Hop Date: "},
                "pad": {"t": 15},
                "steps": steps,
            }
        ],
    )

    fig.update_xaxes(title_text="Terminal Spot", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Payoff", row=1, col=1, range=[y_min - y_pad, y_max + y_pad])
    fig.update_xaxes(title_text="Hop Date", row=1, col=2)
    fig.update_yaxes(title_text="Implied Vol", row=1, col=2)
    fig.update_xaxes(title_text="Hop Date", row=2, col=1)
    fig.update_yaxes(title_text="PIT", row=2, col=1, range=[0.0, 1.0])

    fig.add_hline(y=0.05, line_dash="dash", line_color="black", line_width=1, row=2, col=1)
    fig.add_hline(y=0.2, line_dash="dash", line_color="black", line_width=1, row=2, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="black", line_width=1, row=2, col=1)
    fig.add_hline(y=0.95, line_dash="dash", line_color="black", line_width=1, row=2, col=1)

    fig.write_html(str(output_path), include_plotlyjs=True)


def ks_uniform_score(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    stat, pval = kstest(clean.values, "uniform")
    return float(stat), float(pval)


def create_histograms(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0].hist(summary["pit_a"], bins=10, range=(0.0, 1.0), edgecolor="black")
    axes[0].set_title("Method A PIT")
    axes[0].set_xlabel("PIT")
    axes[0].set_ylabel("Count")

    axes[1].hist(summary["pit_b"], bins=10, range=(0.0, 1.0), edgecolor="black")
    axes[1].set_title("Method B PIT")
    axes[1].set_xlabel("PIT")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_hop_bar_chart(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = summary.sort_values("hop_date").copy()
    labels = data["hop_date"].dt.strftime("%Y-%m-%d")
    x = np.arange(len(data))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, data["pit_a"], width, label="Method A (Spot Ratio)")
    ax.bar(x + width / 2, data["pit_b"], width, label="Method B (Vol-Adjusted)")

    for y in [0.05, 0.2, 0.8, 0.95]:
        ax.axhline(y, color="black", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("PIT Value")
    ax.set_xlabel("Hop Date")
    ax.set_title("PIT by Hop Date: Method A vs Method B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate(summary: pd.DataFrame) -> pd.DataFrame:
    ks_a, p_a = ks_uniform_score(summary["pit_a"])
    ks_b, p_b = ks_uniform_score(summary["pit_b"])

    return pd.DataFrame(
        [
            {"method": "A_spot_ratio", "ks_stat": ks_a, "p_value": p_a},
            {"method": "B_vol_adjusted", "ks_stat": ks_b, "p_value": p_b},
        ]
    )
