"""Statistical comparison of systems over folds.

The paper compares systems with a paired t-test over the 5 folds (Comprehensive_Evaluation_Metrics.ipynb,
``run_paired_ttest``). We report the same test, plus the Wilcoxon signed-rank test as a check that
does not assume normality (with only 5 pairs its smallest possible two-sided p is 0.0625), and a Holm
correction when many systems are compared with one baseline.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def paired_test(a, b) -> dict:
    """Compare per-fold scores a (system) and b (baseline): mean difference, t-test and Wilcoxon."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    out = {"diff": diff.mean(), "t": np.nan, "p": np.nan, "p_wilcoxon": np.nan}
    if len(a) >= 2 and np.any(diff != 0):
        res = ttest_rel(a, b)
        out.update(t=float(res.statistic), p=float(res.pvalue))
        try:
            out["p_wilcoxon"] = float(wilcoxon(a, b).pvalue)
        except ValueError:
            pass
    elif len(a) >= 2:
        out.update(t=0.0, p=1.0, p_wilcoxon=1.0)
    return out


def holm(pvalues) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (NaNs are left as NaN)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def verdict(diff: float, p: float, alpha: float = 0.05) -> str:
    """Persian one-liner used in the tables."""
    if np.isnan(p):
        return "—"
    if p >= alpha:
        return "تفاوت معنی‌دار نیست"
    return "به‌طور معنی‌دار بهتر" if diff > 0 else "به‌طور معنی‌دار بدتر"
