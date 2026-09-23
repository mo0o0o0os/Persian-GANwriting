"""Decision-level (late) fusion of the fine-tuned unimodal classifiers' outputs.

These need no training on the fusion side, so they cannot overfit the training rows. They use the
``logits`` the fine-tuning stage saved for the test fold.
"""

from __future__ import annotations

import numpy as np

from .base import FusionInput, FusionMethod, register, softmax


@register("late_average")
class LateAverage(FusionMethod):
    description = "میانگین وزن‌دار احتمال‌های دو طبقه‌بند fine-tune شده (w_audio قابل تنظیم)"
    needs = ("logits",)

    def fit(self, train: FusionInput):
        self.check_inputs(train)
        return self

    def predict_proba(self, test: FusionInput):
        w = float(self.params.get("w_audio", 0.5))
        return w * softmax(test.logits["audio"]) + (1 - w) * softmax(test.logits["text"])


@register("late_product")
class LateProduct(LateAverage):
    description = "ضرب احتمال‌ها (جمع log-prob وزن‌دار)؛ هر مدلی که مطمئن‌تر است اثر بیشتری دارد"

    def predict_proba(self, test: FusionInput):
        w = float(self.params.get("w_audio", 0.5))
        logp = w * np.log(softmax(test.logits["audio"]) + 1e-12) + (1 - w) * np.log(softmax(test.logits["text"]) + 1e-12)
        return softmax(logp)


@register("late_confidence")
class LateConfidence(LateAverage):
    """Per-utterance weights from each classifier's own confidence (max probability) — the simplest
    form of reliability weighting (as in dynamic stream weighting for audio-visual ASR)."""

    description = "وزن هر وجه برای هر جمله = اطمینان خودش (بیشینهٔ احتمال)"

    def predict_proba(self, test: FusionInput):
        pa, pt = softmax(test.logits["audio"]), softmax(test.logits["text"])
        temp = float(self.params.get("temperature", 1.0))
        ca, ct = pa.max(1, keepdims=True) ** (1 / temp), pt.max(1, keepdims=True) ** (1 / temp)
        return (ca * pa + ct * pt) / (ca + ct)
