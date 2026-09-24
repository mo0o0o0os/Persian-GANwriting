"""Pilot 2: cheap adaptation after drift. SEFR variants vs. standard online learners (prequential accuracy).

SEFR variants (all O(m) memory):
  static            trained on the warm-up window only
  cumulative        exact online SEFR (running class means)
  forget(l)         exponentially weighted class means, fixed decay l
  triggered         decay 1.0 normally; a Page-Hinkley alarm on the hard error switches to decay 0.99
                    for the next 300 samples (fast forgetting only when needed)
Baselines (river, online, same features): Gaussian Naive Bayes, logistic regression (SGD), Hoeffding tree.
"""

import sys

import numpy as np
from river import drift, linear_model, naive_bayes, optim, tree

from pilot import SEFR, T0, WARM, make_stream

POST = 1000  # accuracy measured from drift start + POST to the end


def prequential_sefr(Xs, Y, mode, decay=1.0):
    m = SEFR(Xs.shape[1], decay=decay if mode == "forget" else 1.0)
    m.fit(Xs[:WARM], Y[:WARM])
    ph = drift.PageHinkley(min_instances=30, delta=0.05, threshold=50.0)
    base_err = None
    fast_left, correct = 0, []
    warm_err = np.mean([(m.predict(x) != y) for x, y in zip(Xs[:WARM], Y[:WARM])])
    sd = np.sqrt(warm_err * (1 - warm_err)) + 1e-9
    for x, y in zip(Xs[WARM:], Y[WARM:]):
        pred = m.predict(x)
        correct.append(pred == y)
        if mode == "static":
            continue
        if mode == "triggered":
            ph.update(((pred != y) - warm_err) / sd)
            if ph.drift_detected:
                fast_left = 300
            m.decay = 0.99 if fast_left > 0 else 1.0
            fast_left = max(0, fast_left - 1)
        m.partial_fit(x, y)
    return np.array(correct, dtype=float)


def prequential_river(Xs, Y, model):
    feats = lambda x: {i: float(v) for i, v in enumerate(x)}
    for x, y in zip(Xs[:WARM], Y[:WARM]):
        model.learn_one(feats(x), int(y))
    correct = []
    for x, y in zip(Xs[WARM:], Y[WARM:]):
        correct.append(model.predict_one(feats(x)) == y)
        model.learn_one(feats(x), int(y))
    return np.array(correct, dtype=float)


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for name in ("SEA", "Sine", "Agrawal"):
        for kind in ("abrupt", "gradual"):
            rows = {}
            for seed in range(seeds):
                X, Y, start = make_stream(name, kind, seed)
                lo, hi = X[:WARM].min(0), X[:WARM].max(0)
                Xs = np.clip((X - lo) / (hi - lo + 1e-12), 0, 1)
                runs = {
                    "SEFR static": prequential_sefr(Xs, Y, "static"),
                    "SEFR cumulative": prequential_sefr(Xs, Y, "cumulative"),
                    "SEFR forget 0.999": prequential_sefr(Xs, Y, "forget", 0.999),
                    "SEFR forget 0.995": prequential_sefr(Xs, Y, "forget", 0.995),
                    "SEFR forget 0.99": prequential_sefr(Xs, Y, "forget", 0.99),
                    "SEFR triggered (PH)": prequential_sefr(Xs, Y, "triggered"),
                    "GaussianNB": prequential_river(Xs, Y, naive_bayes.GaussianNB()),
                    "LogReg SGD": prequential_river(Xs, Y, linear_model.LogisticRegression(optimizer=optim.SGD(0.1))),
                    "HoeffdingTree": prequential_river(Xs, Y, tree.HoeffdingTreeClassifier()),
                }
                b0 = start - WARM
                for k, c in runs.items():
                    rows.setdefault(k, []).append((c[:b0].mean(), c[b0 + POST:].mean(), c.mean()))
            print(f"== {name} {kind}: prequential accuracy (before drift | after drift+{POST} | whole stream)")
            for k, v in rows.items():
                v = np.array(v)
                print(f"   {k:22s} {v[:, 0].mean():.3f} | {v[:, 1].mean():.3f} | {v[:, 2].mean():.3f}")
