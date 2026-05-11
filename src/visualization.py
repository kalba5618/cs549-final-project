"""Plotting utilities for final-report figures."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not <30", "Readmitted <30"],
    )
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()

    path = output_dir / f"confusion_matrix_{safe_filename(model_name)}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(roc_data: dict[str, tuple], output_dir: str | Path) -> Path | None:
    """Plot one ROC curve comparing all models with available scores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not roc_data:
        return None

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, (y_true, y_score) in roc_data.items():
        if y_score is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=model_name).plot(ax=ax)

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = output_dir / "roc_curve_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_comparison(results_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = ["accuracy", "precision", "recall", "f1"]
    plot_df = results_df.set_index("Model")[metric_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Model Metric Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="best")
    fig.tight_layout()

    path = output_dir / "model_metric_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_threshold_metrics(threshold_df: pd.DataFrame, model_name: str, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["precision", "recall", "f1"]:
        ax.plot(threshold_df["threshold"], threshold_df[metric], marker="o", label=metric)

    ax.set_title(f"Threshold Tuning: {model_name}")
    ax.set_xlabel("Positive-class threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()

    path = output_dir / f"threshold_metrics_{safe_filename(model_name)}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
