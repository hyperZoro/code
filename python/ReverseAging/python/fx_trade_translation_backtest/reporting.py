from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest



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
