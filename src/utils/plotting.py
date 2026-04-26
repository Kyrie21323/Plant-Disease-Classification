"""
plotting.py
Plotting helpers for training diagnostics.

Functions:
    save_training_curves  — saves a loss-vs-epoch figure to outputs/figures/
    save_confusion_matrix — saves a labelled heatmap of the confusion matrix
"""

from __future__ import annotations

from pathlib import Path

import matplotlib  # type: ignore[import]
matplotlib.use("Agg")   # non-interactive backend — safe for WSL / headless envs
import matplotlib.pyplot as plt  # type: ignore[import]

# Default output directory (relative to repo root inside WSL).
FIGURES_DIR = Path(
    "/mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification"
    "/outputs/figures"
)


def save_training_curves(
    train_losses: list[float],
    val_losses:   list[float],
    filename:     str  = "baseline_training_curves.png",
    figures_dir:  Path = FIGURES_DIR,
    title:        str  = "BaselineCNN — Training Curves",
) -> Path:
    """
    Save a figure showing training and validation loss over epochs.

    Args:
        train_losses: Per-epoch training loss values.
        val_losses:   Per-epoch validation loss values.
        filename:     Output filename (saved inside figures_dir).
        figures_dir:  Directory to write the figure into.
        title:        Figure title shown above the plot.

    Returns:
        Path to the saved figure.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, train_losses, marker="o", linewidth=2, label="Train loss")
    ax.plot(epochs, val_losses,   marker="s", linewidth=2, label="Val loss",   linestyle="--")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Mark the epoch with the lowest validation loss.
    best_epoch = val_losses.index(min(val_losses)) + 1
    best_val   = min(val_losses)
    ax.axvline(x=best_epoch, color="grey", linestyle=":", linewidth=1.2)
    ax.annotate(
        f"best val\nepoch {best_epoch}\n{best_val:.4f}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 0.4, best_val + 0.05),
        fontsize=9,
        color="grey",
    )

    plt.tight_layout()
    out_path = figures_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def save_confusion_matrix(
    matrix:      list[list[int]],
    class_names: list[str],
    filename:    str  = "confusion_matrix.png",
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Save a labelled confusion-matrix heatmap.

    Each cell [true][pred] shows the raw count. The colour scale is
    row-normalised (i.e. recall per class) so that class-size imbalance
    does not wash out errors in smaller classes.

    Args:
        matrix:      num_classes × num_classes count matrix from
                     compute_confusion_matrix().
        class_names: Ordered list of class label strings (length == num_classes).
        filename:    Output filename saved inside figures_dir.
        figures_dir: Directory to write the figure into.

    Returns:
        Path to the saved figure.
    """
    import numpy as np

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cm = np.array(matrix, dtype=float)
    n  = cm.shape[0]

    # Row-normalise so each row sums to 1 (recall view).
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Tick labels.
    short_names = [name.replace("_", "\n") for name in class_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(short_names, fontsize=7)

    # Annotate each cell with the raw count.
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=13)

    plt.tight_layout()
    out_path = figures_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path
