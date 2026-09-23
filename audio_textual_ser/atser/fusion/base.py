"""Common interface of all fusion methods + a registry, so new methods plug in without touching the pipeline.

A fusion method receives, for the training rows and then for the test rows of one fold, a
``FusionInput`` with everything that might be useful:

    features["audio"], features["text"]   [N, H]      the paper's utterance embeddings
    layers["audio"],   layers["text"]     [N, L, H]   per-layer embeddings (if stored)
    logits["audio"],   logits["text"]     [N, C]      outputs of the fine-tuned unimodal classifiers

and must implement ``fit`` and ``predict`` (``predict_proba`` optional).

Adding a method::

    from atser.fusion import FusionMethod, register

    @register("my_method")
    class MyMethod(FusionMethod):
        description = "one line for the comparison table"
        def fit(self, train):  ...; return self
        def predict(self, test): ...

then add ``FusionSpec.of("my_method", some_param=1)`` to the experiment config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REGISTRY: dict[str, type["FusionMethod"]] = {}


def register(name: str):
    def wrap(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return wrap


def create(name: str, **params) -> "FusionMethod":
    if name not in REGISTRY:
        raise KeyError(f"روش ترکیب «{name}» تعریف نشده. روش‌های موجود: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


def available():
    import pandas as pd

    return pd.DataFrame([{"method": n, "needs": ", ".join(c.needs), "description": c.description}
                         for n, c in sorted(REGISTRY.items())])


@dataclass
class FusionInput:
    uids: np.ndarray
    y: np.ndarray
    n_classes: int
    features: dict[str, np.ndarray] = field(default_factory=dict)
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    logits: dict[str, np.ndarray] = field(default_factory=dict)

    def subset(self, rows: np.ndarray) -> "FusionInput":
        pick = lambda d: {k: v[rows] for k, v in d.items()}
        return FusionInput(self.uids[rows], self.y[rows], self.n_classes, pick(self.features), pick(self.layers),
                           pick(self.logits))


class FusionMethod:
    name = "base"
    description = ""
    needs: tuple[str, ...] = ("features",)  # which FusionInput fields must be present
    kind = "fusion"  # "reference" for single-modality baselines that live here for convenience

    def __init__(self, seed: int = 0, **params):
        self.seed = seed
        self.params = params

    def fit(self, train: FusionInput) -> "FusionMethod":
        raise NotImplementedError

    def predict(self, test: FusionInput) -> np.ndarray:
        proba = self.predict_proba(test)
        if proba is None:
            raise NotImplementedError
        return proba.argmax(1)

    def predict_proba(self, test: FusionInput) -> np.ndarray | None:
        return None

    def check_inputs(self, data: FusionInput) -> None:
        for need in self.needs:
            if not getattr(data, need):
                raise ValueError(f"روش {self.name} به «{need}» نیاز دارد ولی موجود نیست.")


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
