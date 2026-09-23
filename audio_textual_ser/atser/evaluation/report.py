"""Tables that summarise every experiment in the store.

All functions start from ``collect(store)``: one row per (experiment, system, fold) with the paper's
metrics. Scores are fractions in the data frame and percentages in the printed tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import REFERENCE_RESULTS, SHEMO_CLASSES, DataConfig, FinetuneConfig
from ..pipeline.store import ArtifactStore
from ..utils import read_json
from .metrics import fold_metrics, normalized_confusion, per_class_metrics
from .stats import holm, paired_test, verdict

SHOW = ["UA", "WA", "F1", "W-F1", "MCC"]


def configs_from_json(cfg: dict) -> tuple[DataConfig, dict[str, FinetuneConfig]]:
    """Rebuild the dataclasses from an experiment's stored json config."""
    data = DataConfig(**{**cfg["data"], "classes": tuple(cfg["data"]["classes"])})
    return data, {m: FinetuneConfig(**cfg[m]) for m in ("audio", "text")}


def _label(exp: dict, sys: dict) -> str:
    kind = sys.get("kind", "fusion")
    if kind == "audio":
        return f"audio · {exp.get('audio_model', sys.get('model', ''))}"
    if kind == "text":
        return f"text · {exp.get('text_model', sys.get('model', ''))}"
    if kind == "reference":
        return f"reference · {sys['system']}"
    return f"fusion · {sys['system']}"


def collect(store: ArtifactStore, experiments: list[str] | None = None, classes=SHEMO_CLASSES) -> pd.DataFrame:
    rows = []
    for exp, sys, fold, arr in store.iter_predictions(experiments):
        train_ua_file = store.prediction_path(exp["name"], sys["system"], fold).with_suffix(".train_ua.npy")
        rows.append({
            "experiment": exp["name"], "text_model": exp.get("text_model"), "audio_model": exp.get("audio_model"),
            "fold_scheme": exp.get("fold_scheme"), "feature_source": exp.get("feature_source"),
            "system": sys["system"], "kind": sys.get("kind", "fusion"), "label": _label(exp, sys), "fold": fold,
            **fold_metrics(arr["y_true"], arr["y_pred"], classes),
            "train UA": float(np.load(train_ua_file)) if train_ua_file.exists() else np.nan,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {"audio": 0, "text": 1, "reference": 2, "fusion": 3}
    df["_o"] = df["kind"].map(order)
    return df.sort_values(["experiment", "_o", "system", "fold"]).drop(columns="_o").reset_index(drop=True)


def summary(df: pd.DataFrame, metrics=SHOW) -> pd.DataFrame:
    """mean ± std over folds, in %, one row per (experiment, system)."""
    g = df.groupby(["experiment", "label"], sort=False)
    out = pd.DataFrame(index=g.size().index)
    for m in metrics:
        mean, std = g[m].mean() * 100, g[m].std() * 100
        out[m] = [f"{a:.2f} ± {b:.2f}" for a, b in zip(mean, std)]
    out["folds"] = g.size()
    return out


def summary_numeric(df: pd.DataFrame, metric: str = "UA") -> pd.DataFrame:
    g = df.groupby(["experiment", "label", "kind"], sort=False)[metric]
    return pd.DataFrame({"mean": g.mean() * 100, "std": g.std() * 100, "n": g.size()}).reset_index()


def leaderboard(df: pd.DataFrame, metric: str = "UA"):
    """All systems of all experiments sorted by the metric; a styled table (colour = score)."""
    table = df.groupby(["experiment", "label"], sort=False)[SHOW].mean().mul(100).round(2)
    table = table.sort_values(metric, ascending=False)
    try:
        return table.style.background_gradient(subset=[metric], cmap="RdYlGn").format("{:.2f}")
    except Exception:  # no matplotlib/jinja2
        return table


def per_fold(df: pd.DataFrame, metric: str = "UA") -> pd.DataFrame:
    t = df.pivot_table(index=["experiment", "label"], columns="fold", values=metric, sort=False) * 100
    t.columns = [f"fold {c}" for c in t.columns]
    t["mean"] = t.mean(axis=1)
    t["std"] = t.iloc[:, :-1].std(axis=1)
    return t.round(2)


def per_class(store: ArtifactStore, experiments: list[str] | None = None, metric: str = "Sensitivity",
              classes=SHEMO_CLASSES) -> pd.DataFrame:
    """Per-class score (recall by default), averaged over folds; rows = systems, columns = classes."""
    acc: dict[tuple, list[pd.Series]] = {}
    for exp, sys, fold, arr in store.iter_predictions(experiments):
        pc = per_class_metrics(arr["y_true"], arr["y_pred"], classes)[metric]
        acc.setdefault((exp["name"], _label(exp, sys)), []).append(pc)
    rows = {k: pd.concat(v, axis=1).mean(axis=1) * 100 for k, v in acc.items()}
    out = pd.DataFrame(rows).T.round(1)
    out.index.names = ["experiment", "label"]
    return out


def confusion(store: ArtifactStore, experiment: str, system: str, classes=SHEMO_CLASSES) -> np.ndarray:
    """Normalised confusion matrix pooled over all folds."""
    ys, ps = [], []
    for exp, sys, fold, arr in store.iter_predictions([experiment]):
        if sys["system"] == system:
            ys.append(arr["y_true"])
            ps.append(arr["y_pred"])
    return normalized_confusion(np.concatenate(ys), np.concatenate(ps), len(classes))


def compare_to(df: pd.DataFrame, baseline: str = "audio", metric: str = "UA") -> pd.DataFrame:
    """Every system vs. the baseline system of the same experiment (paired over folds, Holm-corrected)."""
    rows = []
    for exp, part in df.groupby("experiment", sort=False):
        base = part[part["system"] == baseline].set_index("fold")[metric]
        if base.empty:
            continue
        for label, sys in part[part["system"] != baseline].groupby("label", sort=False):
            s = sys.set_index("fold")[metric]
            common = base.index.intersection(s.index)
            test = paired_test(s[common].values, base[common].values)
            rows.append({"experiment": exp, "system": label, f"{metric} (%)": s.mean() * 100,
                         f"{baseline} (%)": base[common].mean() * 100, "difference": test["diff"] * 100,
                         "p (t-test)": test["p"], "p (Wilcoxon)": test["p_wilcoxon"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p (Holm)"] = holm(out["p (t-test)"].values)
    out["verdict"] = [verdict(d, p) for d, p in zip(out["difference"], out["p (t-test)"])]
    return out.round(4)


def regret(df: pd.DataFrame, metric: str = "UA") -> pd.DataFrame:
    """max(audio, text) − fusion per fold; positive = fusion is worse than the best single modality."""
    rows = []
    for exp, part in df.groupby("experiment", sort=False):
        uni = part[part["kind"].isin(["audio", "text"])].pivot_table(index="fold", columns="system", values=metric)
        if uni.shape[1] < 2:
            continue
        best = uni.max(axis=1)
        for label, sys in part[part["kind"] == "fusion"].groupby("label", sort=False):
            s = sys.set_index("fold")[metric].reindex(best.index)
            r = (best - s) * 100
            test = paired_test(best.values, s.values)
            rows.append({"experiment": exp, "system": label, "regret mean": r.mean(), "regret max": r.max(),
                         "folds with regret > 0": int((r > 0).sum()), "p": test["p"], "verdict":
                         "بدتر از بهترین تک‌وجهی" if test["p"] < 0.05 and r.mean() > 0 else
                         ("بهتر از بهترین تک‌وجهی" if test["p"] < 0.05 else "تفاوت معنی‌دار نیست")})
    return pd.DataFrame(rows).round(3)


def reproduction_check(df: pd.DataFrame, experiment: str, fusion_system: str = "paper_sum_svm",
                       tolerance: float = 2.5) -> pd.DataFrame:
    """Our UA (%) vs. the paper, the author's repo outputs and our earlier Kaggle run."""
    part = df[df["experiment"] == experiment]
    ours = {k: part[part["system"] == s].sort_values("fold")["UA"].values * 100
            for k, s in (("audio", "audio"), ("text", "text"), ("fusion", fusion_system))}
    rows = []
    for key, values in ours.items():
        if len(values) == 0:
            continue
        row = {"part": key, "this run": round(values.mean(), 2)}
        ok = True
        for ref_name, ref in REFERENCE_RESULTS.items():
            if key not in ref:
                continue
            ref_vals = np.atleast_1d(ref[key])
            row[ref_name] = round(float(np.mean(ref_vals)), 2)
            gap = values.mean() - np.mean(ref_vals)
            row[f"Δ {ref_name}"] = round(gap, 2)
            if len(ref_vals) == len(values) == 5:
                row[f"p vs {ref_name}"] = round(paired_test(values, ref_vals)["p"], 3)
            ok &= abs(gap) <= tolerance
        row["verdict"] = "بازتولید شد ✓" if ok else f"اختلاف بیش از {tolerance} واحد ✗"
        rows.append(row)
    return pd.DataFrame(rows).set_index("part")


def overfitting(store: ArtifactStore, df: pd.DataFrame) -> pd.DataFrame:
    """UA on the training rows vs. the test fold for every model and fusion system."""
    rows = []
    for exp in store.experiments().values():
        data_cfg, model_cfgs = configs_from_json(exp["config"])
        for modality, mcfg in model_cfgs.items():
            for fold in range(1, data_cfg.n_folds + 1):
                h = read_json(store.finetune_dir(mcfg, data_cfg, fold) / "history.json")
                if not h:
                    continue
                rows.append({"system": f"{modality} · {mcfg.name} [{data_cfg.fold_scheme}]", "fold": fold,
                             "train UA": h["summary"]["train"]["UA"] * 100, "valid UA": h["summary"]["valid"]["UA"] * 100,
                             "test UA": h["summary"]["test"]["UA"] * 100})
    fus = df[df["kind"] == "fusion"].dropna(subset=["train UA"])
    for (exp, label), part in fus.groupby(["experiment", "label"], sort=False):
        for _, r in part.iterrows():
            rows.append({"system": f"{label} [{exp}]", "fold": r["fold"], "train UA": r["train UA"] * 100,
                         "valid UA": np.nan, "test UA": r["UA"] * 100})
    out = pd.DataFrame(rows).drop_duplicates(subset=["system", "fold"])
    if out.empty:
        return out
    t = out.groupby("system", sort=False)[["train UA", "valid UA", "test UA"]].mean()
    t["gap (train − test)"] = t["train UA"] - t["test UA"]
    return t.round(2)


def training_curves(store: ArtifactStore) -> pd.DataFrame:
    """Per-epoch validation loss/accuracy of every fine-tuned fold in the store."""
    rows = []
    for hist in store.root.glob("finetune/*/*/*/fold*/history.json"):
        h = read_json(hist)
        name = f"{h['config']['modality']} · {h['config']['checkpoint'].split('/')[-1]}"
        for entry in h["log_history"]:
            if "eval_loss" in entry:
                rows.append({"model": name, "folds": hist.parent.parent.name, "fold": h["fold"],
                             "epoch": round(entry.get("epoch", 0)), "eval_loss": entry["eval_loss"],
                             "eval_accuracy": entry.get("eval_accuracy")})
    return pd.DataFrame(rows)


def key_systems(df: pd.DataFrame, top: int = 2) -> pd.DataFrame:
    """Per experiment: audio, text, the paper's fusion and the ``top`` best other fusion methods (for readable plots)."""
    keep = []
    for _, part in df.groupby("experiment", sort=False):
        means = part[part["kind"] == "fusion"].groupby("system")["UA"].mean().sort_values(ascending=False)
        chosen = {"audio", "text", "paper_sum_svm", *means.index[:top]}
        keep.append(part[part["system"].isin(chosen)])
    return pd.concat(keep) if keep else df
