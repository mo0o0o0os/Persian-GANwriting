"""Charts for the results tables. Labels are in English because matplotlib does not shape Persian text.

Every function returns the figure and, when ``save_dir`` is given, also writes a PNG there.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import SHEMO_CLASSES

KIND_COLORS = {"audio": "#4C72B0", "text": "#DD8452", "fusion": "#55A868", "reference": "#9E9E9E"}
REF_STYLES = [{"color": "#1f1f1f", "ls": "--"}, {"color": "#8B0000", "ls": ":"}, {"color": "#4B0082", "ls": "-."}]


def _save(fig, save_dir, name):
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


def score_bars(df: pd.DataFrame, metric: str = "UA", references: dict[str, float] | None = None, save_dir=None,
               name: str = "score_bars"):
    """Mean ± std over folds for every system, one panel per experiment, coloured by kind.

    ``references`` draws dashed vertical lines (e.g. the paper's numbers) when they fall near the data.
    """
    g = df.groupby(["experiment", "label", "kind"], sort=False)[metric]
    t = pd.DataFrame({"mean": g.mean() * 100, "std": g.std().fillna(0) * 100}).reset_index()
    exps = list(dict.fromkeys(t["experiment"]))
    lo = max(0.0, t["mean"].min() - 10)
    hi = min(100.0, (t["mean"] + t["std"]).max() + 6)
    refs = {k: v for k, v in (references or {}).items() if lo - 5 <= v <= hi + 5}
    if refs:
        lo, hi = min(lo, min(refs.values()) - 3), max(hi, max(refs.values()) + 3)
    heights = [max(2, (t["experiment"] == e).sum()) for e in exps]
    fig, axes = plt.subplots(len(exps), 1, figsize=(10, 0.42 * sum(heights) + 1.2 * len(exps)), squeeze=False,
                             gridspec_kw={"height_ratios": heights}, sharex=True)
    for ax, exp in zip(axes[:, 0], exps):
        part = t[t["experiment"] == exp].sort_values("mean")
        ax.barh(part["label"], part["mean"], xerr=part["std"], capsize=3, alpha=0.9,
                color=[KIND_COLORS.get(k, "grey") for k in part["kind"]])
        for y, (m, sd) in enumerate(zip(part["mean"], part["std"])):
            ax.text(min(m + sd + 0.3, hi - 0.5), y, f"{m:.2f}", va="center", fontsize=8)
        for (label, value), style in zip(refs.items(), REF_STYLES):
            ax.axvline(value, lw=1.2, alpha=0.8, **style)
        ax.set_title(exp, fontsize=10, loc="left")
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)
    axes[-1, 0].set_xlim(lo, hi)
    axes[-1, 0].set_xlabel(f"{metric} (%), mean ± std over folds")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in KIND_COLORS.values()]
    handles += [plt.Line2D([0], [0], lw=1.2, **style) for style in REF_STYLES[:len(refs)]]
    axes[-1, 0].legend(handles, list(KIND_COLORS) + list(refs), loc="lower right", fontsize=8)
    fig.tight_layout()
    return _save(fig, save_dir, name)


def per_fold_lines(df: pd.DataFrame, metric: str = "UA", save_dir=None, name: str = "per_fold"):
    """One panel per experiment; one line per system across folds."""
    exps = list(dict.fromkeys(df["experiment"]))
    fig, axes = plt.subplots(1, len(exps), figsize=(5.5 * len(exps), 4), squeeze=False, sharey=True)
    for ax, exp in zip(axes[0], exps):
        part = df[df["experiment"] == exp]
        for label, sys in part.groupby("label", sort=False):
            sys = sys.sort_values("fold")
            kind = sys["kind"].iloc[0]
            ax.plot(sys["fold"], sys[metric] * 100, marker="o", label=label, lw=2 if kind in ("audio", "text") else 1.3,
                    color=KIND_COLORS[kind] if kind in ("audio", "text") else None)
        ax.set_title(exp, fontsize=9)
        ax.set_xlabel("fold")
        ax.set_xticks(sorted(part["fold"].unique()))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel(f"{metric} (%)")
    fig.tight_layout()
    return _save(fig, save_dir, name)


def heatmap(table: pd.DataFrame, title: str = "", fmt: str = "{:.1f}", cmap: str = "YlGnBu", save_dir=None,
            name: str = "heatmap"):
    """Annotated heatmap of any numeric table (e.g. per-class recall: systems x classes)."""
    fig, ax = plt.subplots(figsize=(1.3 * table.shape[1] + 3, 0.45 * table.shape[0] + 1.5))
    im = ax.imshow(table.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns], rotation=30, ha="right")
    ylabels = [" | ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in table.index]
    ax.set_yticks(range(table.shape[0]), ylabels, fontsize=8)
    vmax = np.nanmax(table.values)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 * vmax else "black")
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, save_dir, name)


def confusion_grid(matrices: dict[str, np.ndarray], classes=SHEMO_CLASSES, save_dir=None, name: str = "confusions"):
    """Normalised confusion matrices side by side (title -> matrix)."""
    n = len(matrices)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows), squeeze=False)
    for ax, (title, m) in zip(axes.flat, matrices.items()):
        ax.imshow(m, cmap="YlGnBu", vmin=0, vmax=1)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(title, fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, save_dir, name)


def training_curves(curves: pd.DataFrame, save_dir=None, name: str = "training_curves"):
    """Validation loss and accuracy per epoch, one column per fine-tuned model, one line per fold."""
    groups = list(curves.groupby(["model", "folds"], sort=False))
    fig, axes = plt.subplots(2, len(groups), figsize=(4.2 * len(groups), 6), squeeze=False)
    for j, ((model, folds), part) in enumerate(groups):
        for fold, f in part.groupby("fold"):
            f = f.sort_values("epoch")
            axes[0][j].plot(f["epoch"], f["eval_loss"], marker="o", label=f"fold {fold}")
            axes[1][j].plot(f["epoch"], f["eval_accuracy"], marker="o", label=f"fold {fold}")
        axes[0][j].set_title(f"{model}\n{folds.split('-')[0]} folds", fontsize=9)
        axes[0][j].set_ylabel("validation loss")
        axes[1][j].set_ylabel("validation accuracy")
        axes[1][j].set_xlabel("epoch")
        for ax in (axes[0][j], axes[1][j]):
            ax.grid(alpha=0.3)
            ax.ticklabel_format(axis="y", useOffset=False)
        axes[0][j].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, save_dir, name)


def train_test_gap(table: pd.DataFrame, save_dir=None, name: str = "train_test_gap"):
    """Grouped bars of train vs. test UA (output of report.overfitting)."""
    t = table.sort_values("gap (train − test)")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(t) + 1.5))
    ax.barh(y - 0.2, t["train UA"], height=0.4, label="train UA", color="#C44E52")
    ax.barh(y + 0.2, t["test UA"], height=0.4, label="test UA", color="#4C72B0")
    ax.set_yticks(y, t.index, fontsize=8)
    ax.set_xlabel("UA (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, name)
