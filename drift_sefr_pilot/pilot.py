"""Pilot: does a soft (probabilistic) error signal detect drift earlier than the hard error rate
for a tiny linear classifier (SEFR)? And how much does cheap on-device adaptation recover?

Setting (supervised stream, labels arrive after prediction, as in DDM / PUDD):
  * SEFR trained on a 1000-sample warm-up window (features min-max scaled on that window, clipped to [0,1]).
  * Platt scaling p = sigmoid(a * (score - bias) + c) fitted on the same warm-up window.
  * signals per sample: hard error 1[y_hat != y]  vs  soft error 1 - p(y | x)   (both in [0, 1]).
  * detectors (river): ADWIN(delta=0.002) on each signal, Page-Hinkley on each signal (z-scored with warm-up
    mean/std, same parameters for both), and DDM on the hard error (classic baseline).
Streams: river synthetic generators with one drift at t0=5000 (abrupt: width 1; gradual: width 2000).
Metrics over 30 seeds: false alarms before the drift starts, missed detections, delay from drift start.
"""

import json
import sys
import time

import numpy as np
from river import drift
from river.datasets import synth

T0, N, WARM, HORIZON = 5000, 9000, 1000, 3000


class SEFR:
    """Exact online SEFR: per-class feature sums and counts (O(m) memory). decay<1 = forgetting."""

    def __init__(self, m, decay=1.0):
        self.s = np.zeros((2, m))
        self.n = np.zeros(2)
        self.decay = decay
        self.w = np.zeros(m)
        self.b = 0.0

    def partial_fit(self, x, y):
        if self.decay < 1:
            self.s *= self.decay
            self.n *= self.decay
        self.s[y] += x
        self.n[y] += 1
        self._refresh()

    def fit(self, X, Y):
        for c in (0, 1):
            self.s[c] = X[Y == c].sum(0)
            self.n[c] = (Y == c).sum()
        self._refresh()

    def _refresh(self):
        if self.n.min() == 0:
            return
        mu = self.s / self.n[:, None]
        self.w = (mu[1] - mu[0]) / (mu[1] + mu[0] + 1e-7)  # Eq. 5
        pos_avg, neg_avg = mu[1] @ self.w, mu[0] @ self.w  # Eq. 7-8 (the mean score is linear in the mean)
        self.b = (self.n[0] * pos_avg + self.n[1] * neg_avg) / self.n.sum()  # Eq. 9

    def margin(self, x):
        return x @ self.w - self.b

    def predict(self, x):
        return int(self.margin(x) > 0)


def platt(margins, y, iters=300, lr=0.5):
    """Fit p = sigmoid(a*m + c) by gradient descent on log-loss (2 parameters)."""
    z = (margins - margins.mean()) / (margins.std() + 1e-9)
    a, c = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * z + c)))
        a -= lr * np.mean((p - y) * z)
        c -= lr * np.mean(p - y)
    mu, sd = margins.mean(), margins.std() + 1e-9
    return lambda m: 1 / (1 + np.exp(-(a * (m - mu) / sd + c)))


def make_stream(name, kind, seed):
    width = 1 if kind == "abrupt" else 2000
    if name == "SEA":
        a, b = synth.SEA(variant=0, seed=seed), synth.SEA(variant=3, seed=seed + 1)
    elif name == "Sine":
        a, b = synth.Sine(classification_function=0, seed=seed), synth.Sine(classification_function=2, seed=seed + 1)
    elif name == "Agrawal":
        a, b = synth.Agrawal(classification_function=0, seed=seed), synth.Agrawal(classification_function=2, seed=seed + 1)
    # own composition (river's ConceptDriftStream overflows for width=1): linear mixing over the transition
    rng = np.random.default_rng(seed)
    start = T0 if width == 1 else T0 - width // 2
    ia, ib = iter(a), iter(b)
    X, Y = [], []
    for t in range(N):
        use_b = t >= start + width or (t >= start and rng.random() < (t - start) / width)
        x, y = next(ib if use_b else ia)
        X.append([float(v) for v in x.values()])
        Y.append(int(y))
    return np.array(X), np.array(Y), start


class HistChi2:
    """Chi-square test between the bucket histogram of the last W values and a warm-up reference.
    Memory: W bucket indices (uint8) + B counters. Buckets = warm-up quantiles (cf. PUDD's adaptive bucketing)."""

    def __init__(self, reference, bins, window=200, alpha=1e-3):
        from scipy.stats import chi2
        ref = np.asarray(reference, dtype=float)
        if bins == 2:
            self.edges = np.array([0.5])
        else:
            self.edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
        counts = np.bincount(np.searchsorted(self.edges, ref, side="right"), minlength=len(self.edges) + 1) + 1.0
        self.p_ref = counts / counts.sum()
        self.W, self.buf, self.counts = window, [], np.zeros(len(self.p_ref))
        self.crit = chi2.ppf(1 - alpha, df=len(self.p_ref) - 1)
        self.drift_detected = False

    def update(self, v):
        k = int(np.searchsorted(self.edges, v, side="right"))
        self.buf.append(k)
        self.counts[k] += 1
        if len(self.buf) > self.W:
            self.counts[self.buf.pop(0)] -= 1
        self.drift_detected = False
        if len(self.buf) == self.W:
            exp = self.p_ref * self.W
            if ((self.counts - exp) ** 2 / exp).sum() > self.crit:
                self.drift_detected = True
                self.buf, self.counts = [], np.zeros(len(self.p_ref))


def first_alarm(signal, make_detector, start):
    det = make_detector()
    alarms = []
    for t, v in enumerate(signal, start=WARM):
        det.update(v)
        if det.drift_detected:
            alarms.append(t)
    return alarms


def score(alarms, drift_start):
    fa = sum(1 for t in alarms if t < drift_start)
    after = [t for t in alarms if drift_start <= t < drift_start + HORIZON]
    return fa, (after[0] - drift_start) if after else None


def run(name, kind, seed):
    X, Y, drift_start = make_stream(name, kind, seed)
    lo, hi = X[:WARM].min(0), X[:WARM].max(0)
    Xs = np.clip((X - lo) / (hi - lo + 1e-12), 0, 1)
    model = SEFR(Xs.shape[1])
    model.fit(Xs[:WARM], Y[:WARM])
    m_warm = Xs[:WARM] @ model.w - model.b
    prob = platt(m_warm, Y[:WARM])

    margins = Xs[WARM:] @ model.w - model.b
    yhat = (margins > 0).astype(int)
    p1 = prob(margins)
    hard = (yhat != Y[WARM:]).astype(float)
    soft = np.where(Y[WARM:] == 1, 1 - p1, p1)

    p1w = prob(m_warm)
    soft_w = np.where(Y[:WARM] == 1, 1 - p1w, p1w)
    hard_w = ((m_warm > 0).astype(int) != Y[:WARM]).astype(float)
    z = lambda s, ref: (s - ref.mean()) / (ref.std() + 1e-9)

    out = {}
    detectors = {
        "ADWIN": (lambda: drift.ADWIN(delta=0.002), hard, soft),
        "PageHinkley": (lambda: drift.PageHinkley(min_instances=30, delta=0.05, threshold=50.0),
                        z(hard, hard_w), z(soft, soft_w)),
    }
    for det_name, (mk, h, s) in detectors.items():
        out[f"{det_name} | hard error"] = score(first_alarm(h, mk, WARM), drift_start)
        out[f"{det_name} | soft error"] = score(first_alarm(s, mk, WARM), drift_start)
    for W in (100, 200):
        out[f"Chi2 W={W} | hard error"] = score(first_alarm(hard, lambda: HistChi2(hard_w, 2, W), WARM), drift_start)
        out[f"Chi2 W={W} | soft error"] = score(first_alarm(soft, lambda: HistChi2(soft_w, 5, W), WARM), drift_start)
    out["DDM | hard error"] = score(first_alarm(hard.astype(int), lambda: drift.binary.DDM(), WARM), drift_start)

    # accuracy before / after the drift for the static model, and two O(m) adaptation strategies
    acc = {"static": (1 - hard[: T0 - WARM - 1000].mean(), 1 - hard[T0 - WARM + 1000:].mean())}
    for label, decay in (("online (cumulative)", 1.0), ("online + forgetting 0.995", 0.995)):
        mdl = SEFR(Xs.shape[1], decay=decay)
        mdl.fit(Xs[:WARM], Y[:WARM])
        correct = []
        for x, y in zip(Xs[WARM:], Y[WARM:]):
            correct.append(mdl.predict(x) == y)
            mdl.partial_fit(x, y)
        correct = np.array(correct, dtype=float)
        acc[label] = (correct[: T0 - WARM - 1000].mean(), correct[T0 - WARM + 1000:].mean())
    return out, acc


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    results = {}
    start = time.time()
    for name in ("SEA", "Sine", "Agrawal"):
        for kind in ("abrupt", "gradual"):
            key = f"{name} {kind}"
            runs = [run(name, kind, s) for s in range(seeds)]
            det_keys = runs[0][0].keys()
            summary = {}
            for k in det_keys:
                fas = [r[0][k][0] for r in runs]
                delays = [r[0][k][1] for r in runs if r[0][k][1] is not None]
                summary[k] = {"false_alarms_per_run": float(np.mean(fas)),
                              "missed_%": 100 * (1 - len(delays) / seeds),
                              "median_delay": float(np.median(delays)) if delays else None,
                              "mean_delay": float(np.mean(delays)) if delays else None}
            accs = {k: [float(np.mean([r[1][k][0] for r in runs])), float(np.mean([r[1][k][1] for r in runs]))]
                    for k in runs[0][1]}
            results[key] = {"detectors": summary, "accuracy_before_after": accs}
            print(f"== {key} ({time.time() - start:.0f}s)", flush=True)
            for k, v in summary.items():
                print(f"   {k:28s} FA/run={v['false_alarms_per_run']:.2f}  missed={v['missed_%']:.0f}%  "
                      f"median delay={v['median_delay']}  mean delay={v['mean_delay'] and round(v['mean_delay'])}")
            for k, (b, a) in accs.items():
                print(f"   acc {k:28s} before={b:.3f} after={a:.3f}")
    json.dump(results, open("pilot_results.json", "w"), indent=1)
