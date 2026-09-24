# Drift-aware SEFR — pilot experiments

Two small CPU-only experiments (a few minutes each) behind the topic report
"Drift-aware SEFR: early detection and low-cost adaptation for ultra-low-power classifiers".

```bash
pip install river numpy scipy
python pilot.py 30        # detection: hard vs soft (calibrated) error signals for SEFR, 6 drift scenarios x 30 seeds
python pilot_adapt.py 20  # adaptation: SEFR variants vs Gaussian NB, SGD logistic regression, Hoeffding tree
```

- `SEFR` here is an exact online version of the original classifier (per-class feature sums and counts,
  O(m) memory; `decay < 1` gives exponential forgetting). With `decay = 1` it equals batch SEFR.
- Streams: river's SEA, Sine and Agrawal generators with one abrupt (width 1) or gradual (width 2000)
  drift at sample 5000; the model is trained on the first 1000 samples.
- `pilot_results.json` holds the detection results of the run reported in the topic document.

Findings (synthetic data only, own implementation):

1. Forgetting after a Page-Hinkley alarm gives SEFR the best post-drift accuracy on SEA (0.975 vs 0.951 for a
   Hoeffding tree, 0.832 static) and matches the tree on Sine, with a few hundred bytes of state.
2. The soft (calibrated) error signal detects drift *later* than the hard error on SEA, the same on Sine, and only
   helps where the hard error is uninformative (Agrawal, where linear SEFR is at chance level).
3. SEFR's linearity is a real limitation (Agrawal: 0.50 vs 0.74–0.88 for a Hoeffding tree).
