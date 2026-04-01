"""
Render a presentation-friendly figure from quantile benchmark CSV output.

Usage:
    python plot_quantile_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).parent
INPUT = ROOT / "quantile_benchmark_cv_results.csv"
OUTPUT = ROOT / "quantile_benchmark_plots.png"
TARGET_COVERAGE = 0.50


def main() -> None:
    df = pd.read_csv(INPUT)

    order = ["CatBoost", "LightGBM", "HistGB", "XGBoost"]
    order = [name for name in order if name in set(df["model"])]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)

    colors = []
    for model in df["model"]:
        if model == "CatBoost":
            colors.append("#2E7D32")
        elif model == "LightGBM":
            colors.append("#1565C0")
        else:
            colors.append("#B0BEC5")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle("5-Fold Quantile Regression Benchmark", fontsize=16, fontweight="bold")

    # Panel 1: MAE
    ax = axes[0]
    ax.bar(
        df["model"],
        df["q50_mae"],
        yerr=df["q50_mae_std"],
        color=colors,
        capsize=4,
        edgecolor="white",
    )
    ax.set_title("Q50 MAE\nLower is better")
    ax.set_ylabel("Mean Absolute Error ($)")
    ax.tick_params(axis="x", rotation=20)
    for i, val in enumerate(df["q50_mae"]):
        ax.text(i, val + 40, f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: Coverage
    ax = axes[1]
    ax.bar(
        df["model"],
        df["coverage_q25_q75"] * 100,
        yerr=df["coverage_q25_q75_std"] * 100,
        color=colors,
        capsize=4,
        edgecolor="white",
    )
    ax.axhline(TARGET_COVERAGE * 100, color="#D32F2F", linestyle="--", linewidth=1.5, label="Target 50%")
    ax.set_title("Q25-Q75 Coverage\nCloser to 50% is better")
    ax.set_ylabel("Coverage (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right", frameon=False)
    for i, val in enumerate(df["coverage_q25_q75"] * 100):
        ax.text(i, val + 0.6, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # Panel 3: Crossing
    ax = axes[2]
    ax.bar(
        df["model"],
        df["crossing_rate"] * 100,
        yerr=df["crossing_rate_std"] * 100,
        color=colors,
        capsize=4,
        edgecolor="white",
    )
    ax.set_title("Quantile Crossing Rate\nLower is better")
    ax.set_ylabel("Crossing rate (%)")
    ax.tick_params(axis="x", rotation=20)
    for i, val in enumerate(df["crossing_rate"] * 100):
        ax.text(i, val + 0.25, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.text(
        0.5,
        0.01,
        "CatBoost is not best on point MAE, but it is best on interval calibration and quantile consistency.",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
