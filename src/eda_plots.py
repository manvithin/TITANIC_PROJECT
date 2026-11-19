"""EDA charts for the Titanic project.

Every function builds one chart, saves it to ``outputs/figures/`` and returns
the matplotlib Figure so the notebook can display it as well.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:  # Support both "python src/train_model.py" and "python -m src.train_model".
    from .pipeline import FIGURES_DIR
except ImportError:
    from pipeline import FIGURES_DIR

SURVIVAL_PALETTE = {0: "#4C72B0", 1: "#DD8452"}
SURVIVAL_LABELS = ["Died", "Survived"]
ACCENT = "#2A6F97"
GRID_COLOR = "#D9E2EC"


def _style() -> None:
    """Apply one consistent, presentation-friendly style to every chart."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#B0BEC5",
            "grid.color": GRID_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, filename: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    return path


def _annotate_counts(ax: plt.Axes) -> None:
    """Write the height above every bar in a grouped count plot."""
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)


def plot_survival_by_sex(df: pd.DataFrame, outdir: Path = FIGURES_DIR) -> plt.Figure:
    """Grouped bar chart: how many men/women died or survived."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(
        data=df,
        x="Sex",
        hue="Survived",
        palette=SURVIVAL_PALETTE,
        ax=ax,
    )
    ax.set_title("Survival Count by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Number of passengers")
    ax.legend(title="Outcome", labels=SURVIVAL_LABELS, frameon=False)
    _annotate_counts(ax)
    _save(fig, "survival_by_sex.png", outdir)
    return fig


def plot_survival_by_pclass(
    df: pd.DataFrame, outdir: Path = FIGURES_DIR
) -> plt.Figure:
    """Grouped bar chart: survival counts for 1st, 2nd and 3rd class."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(
        data=df,
        x="Pclass",
        hue="Survived",
        palette=SURVIVAL_PALETTE,
        ax=ax,
    )
    ax.set_title("Survival Count by Passenger Class")
    ax.set_xlabel("Passenger class (1 = first, 3 = third)")
    ax.set_ylabel("Number of passengers")
    ax.legend(title="Outcome", labels=SURVIVAL_LABELS, frameon=False)
    _annotate_counts(ax)
    _save(fig, "survival_by_pclass.png", outdir)
    return fig


def plot_target_count(df: pd.DataFrame, outdir: Path = FIGURES_DIR) -> plt.Figure:
    """Bar chart of the target distribution (549 died, 342 survived)."""
    _style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.countplot(
        data=df,
        x="Survived",
        hue="Survived",
        palette=SURVIVAL_PALETTE,
        legend=False,
        ax=ax,
    )
    total = len(df)
    for container in ax.containers:
        labels = [
            f"{int(bar.get_height())}\n({bar.get_height() / total:.1%})"
            for bar in container
        ]
        ax.bar_label(container, labels=labels, padding=4, fontsize=10)
    ax.set_title("Target Distribution: Survived")
    ax.set_xlabel("Survived (0 = died, 1 = survived)")
    ax.set_ylabel("Number of passengers")
    ax.set_ylim(0, df["Survived"].value_counts().max() * 1.18)
    sns.despine(ax=ax)
    _save(fig, "target_count.png", outdir)
    return fig


def plot_age_distribution(
    df: pd.DataFrame, outdir: Path = FIGURES_DIR
) -> plt.Figure:
    """Histogram of passenger ages with the median marked."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(
        data=df,
        x="Age",
        bins=30,
        kde=True,
        color=ACCENT,
        edgecolor="white",
        ax=ax,
    )
    median_age = df["Age"].median()
    ax.axvline(
        median_age,
        color="#C0392B",
        linestyle="--",
        linewidth=1.5,
        label=f"Median age = {median_age:.1f}",
    )
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of passengers")
    ax.legend(frameon=False)
    _save(fig, "age_distribution.png", outdir)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, outdir: Path = FIGURES_DIR
) -> plt.Figure:
    """Correlation heatmap for the numeric and engineered features."""
    _style()
    numeric_columns = [
        "Survived",
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
    ]
    corr = df[numeric_columns].astype(float).corr()

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Numerical Features")
    _save(fig, "correlation_heatmap.png", outdir)
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    outdir: Path = FIGURES_DIR,
    top_n: int = 10,
) -> plt.Figure:
    """Horizontal bar chart of the Random Forest's top feature importances."""
    _style()
    series = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(series.index, series.values, color=ACCENT, height=0.65)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_ylabel("")
    ax.set_xlim(0, series.max() * 1.18)
    sns.despine(ax=ax)
    _save(fig, "feature_importance.png", outdir)
    return fig


def create_all_eda_plots(
    df: pd.DataFrame, outdir: Path = FIGURES_DIR
) -> list[Path]:
    """Create and save the five required EDA charts."""
    plot_survival_by_sex(df, outdir)
    plot_survival_by_pclass(df, outdir)
    plot_target_count(df, outdir)
    plot_age_distribution(df, outdir)
    plot_correlation_heatmap(df, outdir)

    filenames = [
        "survival_by_sex.png",
        "survival_by_pclass.png",
        "target_count.png",
        "age_distribution.png",
        "correlation_heatmap.png",
    ]
    return [outdir / name for name in filenames]
