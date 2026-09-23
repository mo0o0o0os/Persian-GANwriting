"""Fusion with classical (scikit-learn) classifiers on the utterance embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base import FusionInput, FusionMethod, register


class _SklearnOnFeatures(FusionMethod):
    """Builds X from the embeddings, fits one sklearn estimator."""

    def build_x(self, data: FusionInput) -> np.ndarray:
        raise NotImplementedError

    def make_estimator(self):
        raise NotImplementedError

    def fit(self, train: FusionInput):
        self.check_inputs(train)
        self.model = self.make_estimator()
        self.model.fit(self.build_x(train), train.y.tolist())
        return self

    def predict(self, test: FusionInput) -> np.ndarray:
        return np.asarray(self.model.predict(self.build_x(test)))

    def predict_proba(self, test: FusionInput):
        return self.model.predict_proba(self.build_x(test)) if hasattr(self.model, "predict_proba") else None


@register("paper_sum_svm")
class PaperSumSVM(_SklearnOnFeatures):
    """The paper's early fusion (Early_Fusion_Summation.ipynb, cells 7 and 29): x = text + audio,
    ``svm.SVC(probability=True)`` with default settings, no scaling, labels from ``clf.predict``."""

    description = "مقاله: جمع بردار صوت و متن + SVM پیش‌فرض (بدون نرمال‌سازی)"

    def build_x(self, data):
        return data.features["text"] + data.features["audio"]

    def make_estimator(self):
        # random_state only affects predict_proba's internal calibration, never predict()
        return SVC(probability=True, random_state=self.seed)


@register("concat_svm")
class ConcatSVM(_SklearnOnFeatures):
    description = "الحاق بردارها (۱۵۳۶ بعد) + SVM پیش‌فرض"

    def build_x(self, data):
        return np.concatenate([data.features["audio"], data.features["text"]], axis=1)

    def make_estimator(self):
        return SVC(probability=True, random_state=self.seed, C=self.params.get("C", 1.0))


@register("scaled_concat_svm")
class ScaledConcatSVM(ConcatSVM):
    description = "الحاق + استانداردسازی + SVM (C قابل تنظیم، وزن کلاس اختیاری)"

    def make_estimator(self):
        return make_pipeline(StandardScaler(), SVC(probability=True, random_state=self.seed, C=self.params.get("C", 1.0),
                                                   class_weight=self.params.get("class_weight")))


@register("concat_logreg")
class ConcatLogReg(ConcatSVM):
    description = "الحاق + استانداردسازی + رگرسیون لجستیک با وزن کلاس متوازن"

    def make_estimator(self):
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.params.get("C", 1.0), class_weight="balanced",
                                                                  max_iter=5000, random_state=self.seed))


@register("audio_svm")
class AudioSVM(_SklearnOnFeatures):
    description = "فقط بردار صوت + SVM (مرجع تک‌وجهی روی همان ویژگی‌ها)"
    kind = "reference"

    def build_x(self, data):
        return data.features["audio"]

    def make_estimator(self):
        return SVC(probability=True, random_state=self.seed)


@register("text_svm")
class TextSVM(AudioSVM):
    description = "فقط بردار متن + SVM (مرجع تک‌وجهی روی همان ویژگی‌ها)"

    def build_x(self, data):
        return data.features["text"]


@register("stacking_oof")
class StackingOOF(FusionMethod):
    """Two-stage stacking done the standard way: the meta-classifier is trained on out-of-fold
    probabilities of the unimodal classifiers, never on predictions for rows they were fitted on.

    Caveat: with features from encoders fine-tuned on the same training rows (FeatureConfig.source =
    "finetuned") the embeddings themselves already carry that memorisation; the second stage is only
    fully leakage-free with source = "pretrained".
    """

    description = "stacking دومرحله‌ای با پیش‌بینی‌های out-of-fold (رگرسیون لجستیک برای هر وجه + متا)"

    def _unimodal(self):
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.params.get("C", 1.0), class_weight="balanced",
                                                                  max_iter=5000, random_state=self.seed))

    def fit(self, train: FusionInput):
        self.check_inputs(train)
        cv = StratifiedKFold(n_splits=self.params.get("inner_folds", 5), shuffle=True, random_state=self.seed)
        oof, self.unimodal = [], {}
        for m in ("audio", "text"):
            X = train.features[m]
            oof.append(cross_val_predict(self._unimodal(), X, train.y, cv=cv, method="predict_proba"))
            self.unimodal[m] = self._unimodal().fit(X, train.y)
        self.meta = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=self.seed)
        self.meta.fit(np.log(np.concatenate(oof, axis=1) + 1e-8), train.y)
        return self

    def predict_proba(self, test: FusionInput):
        probs = [self.unimodal[m].predict_proba(test.features[m]) for m in ("audio", "text")]
        return self.meta.predict_proba(np.log(np.concatenate(probs, axis=1) + 1e-8))
