"""Small PyTorch fusion heads trained on the cached embeddings (seconds to minutes per fold).

``TorchFusion`` holds the whole training loop (standardisation, class-weighted loss, early stopping
on balanced accuracy of an inner validation split). A new architecture only needs ``inputs`` (which
arrays to feed) and ``build`` (the nn.Module); see ``MLPConcat`` for the smallest example.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from .base import FusionInput, FusionMethod, register


class TorchFusion(FusionMethod):
    defaults = dict(hidden=256, dropout=0.3, lr=1e-3, weight_decay=1e-4, epochs=200, batch_size=64, patience=20,
                    valid_fraction=0.1, class_weighted=True, modality_dropout=0.0)

    def hp(self, key):
        return self.params.get(key, self.defaults[key])

    # ---- to override ----------------------------------------------------------------------
    def inputs(self, data: FusionInput) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def build(self, shapes: dict[str, tuple[int, ...]], n_classes: int) -> nn.Module:
        raise NotImplementedError

    # ---- shared -----------------------------------------------------------------------------
    def _standardise(self, arrays: dict[str, np.ndarray], fit: bool) -> dict[str, torch.Tensor]:
        if fit:
            self.stats = {k: (v.astype(np.float32).mean(0), v.astype(np.float32).std(0) + 1e-6) for k, v in arrays.items()}
        return {k: torch.from_numpy((v.astype(np.float32) - self.stats[k][0]) / self.stats[k][1]) for k, v in arrays.items()}

    def fit(self, train: FusionInput):
        self.check_inputs(train)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = self._standardise(self.inputs(train), fit=True)
        y = torch.from_numpy(train.y.astype(np.int64))
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.hp("valid_fraction"), random_state=self.seed)
            fit_rows, val_rows = next(sss.split(np.zeros(len(y)), train.y))
        except ValueError:  # too few samples per class (quick tests)
            perm = np.random.RandomState(self.seed).permutation(len(y))
            cut = max(1, int(len(y) * self.hp("valid_fraction")))
            val_rows, fit_rows = perm[:cut], perm[cut:]

        self.module = self.build({k: tuple(v.shape[1:]) for k, v in X.items()}, train.n_classes).to(self.device)
        counts = np.bincount(train.y[fit_rows], minlength=train.n_classes).astype(np.float32)
        weights = (counts.sum() / (train.n_classes * np.maximum(counts, 1))) if self.hp("class_weighted") else np.ones_like(counts)
        loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(self.device))
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.hp("lr"), weight_decay=self.hp("weight_decay"))

        to_dev = lambda rows: {k: v[torch.as_tensor(rows, dtype=torch.long)].to(self.device) for k, v in X.items()}
        Xval, yval = to_dev(val_rows), train.y[val_rows]
        best, best_score, wait = None, -1.0, 0
        gen = torch.Generator().manual_seed(self.seed)
        self.history = []
        for epoch in range(self.hp("epochs")):
            self.module.train()
            order = torch.from_numpy(fit_rows)[torch.randperm(len(fit_rows), generator=gen)]
            for i in range(0, len(order), self.hp("batch_size")):
                rows = order[i:i + self.hp("batch_size")].numpy()
                batch = to_dev(rows)
                batch = self._modality_dropout(batch)
                loss = loss_fn(self.module(**batch), y[rows].to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            score = balanced_accuracy_score(yval, self._forward(Xval).argmax(1))
            self.history.append(score)
            if score > best_score:
                best, best_score, wait = copy.deepcopy(self.module.state_dict()), score, 0
            else:
                wait += 1
                if wait >= self.hp("patience"):
                    break
        self.module.load_state_dict(best)
        self.best_valid_ua = best_score
        return self

    def _modality_dropout(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        p = self.hp("modality_dropout")
        if p <= 0 or len(batch) < 2:
            return batch
        keys = sorted(batch)
        n = next(iter(batch.values())).shape[0]
        drop = torch.rand(n, device=self.device) < p
        which = torch.randint(0, len(keys), (n,), device=self.device)
        out = {}
        for j, k in enumerate(keys):
            mask = (drop & (which == j)).float().view(-1, *([1] * (batch[k].dim() - 1)))
            out[k] = batch[k] * (1 - mask)
        return out

    @torch.no_grad()
    def _forward(self, X: dict[str, torch.Tensor]) -> np.ndarray:
        self.module.eval()
        return torch.softmax(self.module(**X), dim=1).cpu().numpy()

    def predict_proba(self, test: FusionInput):
        X = {k: v.to(self.device) for k, v in self._standardise(self.inputs(test), fit=False).items()}
        return self._forward(X)


# ------------------------------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, dim_in, hidden, n_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, audio, text):
        return self.net(torch.cat([audio, text], dim=1))


@register("mlp_concat")
class MLPConcat(TorchFusion):
    description = "الحاق بردارها + شبکهٔ MLP دو لایه (توقف زودهنگام روی UA اعتبارسنجی)"

    def inputs(self, data):
        return {"audio": data.features["audio"], "text": data.features["text"]}

    def build(self, shapes, n_classes):
        return _MLP(shapes["audio"][0] + shapes["text"][0], self.hp("hidden"), n_classes, self.hp("dropout"))


class _GMU(nn.Module):
    """Gated Multimodal Unit (Arevalo et al., 2017): a learned, per-dimension gate between modalities."""

    def __init__(self, dim_a, dim_t, hidden, n_classes, dropout):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, hidden)
        self.proj_t = nn.Linear(dim_t, hidden)
        self.gate = nn.Linear(dim_a + dim_t, hidden)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, audio, text):
        h_a, h_t = torch.tanh(self.proj_a(audio)), torch.tanh(self.proj_t(text))
        z = torch.sigmoid(self.gate(torch.cat([audio, text], dim=1)))
        return self.out(z * h_a + (1 - z) * h_t)


@register("gmu")
class GMUFusion(MLPConcat):
    description = "Gated Multimodal Unit: دروازهٔ یادگرفتنی بین صوت و متن"

    def build(self, shapes, n_classes):
        return _GMU(shapes["audio"][0], shapes["text"][0], self.hp("hidden"), n_classes, self.hp("dropout"))


class _WeightedLayers(nn.Module):
    """Learnable softmax weights over the hidden layers of each encoder (SUPERB-style), then an MLP."""

    def __init__(self, shape_a, shape_t, hidden, n_classes, dropout):
        super().__init__()
        self.w_a = nn.Parameter(torch.zeros(shape_a[0]))
        self.w_t = nn.Parameter(torch.zeros(shape_t[0]))
        self.mlp = _MLP(shape_a[1] + shape_t[1], hidden, n_classes, dropout)

    def forward(self, audio, text):
        a = (torch.softmax(self.w_a, 0)[None, :, None] * audio).sum(1)
        t = (torch.softmax(self.w_t, 0)[None, :, None] * text).sum(1)
        return self.mlp(a, t)


@register("weighted_layers_mlp")
class WeightedLayersMLP(TorchFusion):
    description = "وزن یادگرفتنی برای لایه‌های هر encoder (به‌جای میانگین ساده) + MLP"
    needs = ("layers",)

    def inputs(self, data):
        return {"audio": data.layers["audio"], "text": data.layers["text"]}

    def build(self, shapes, n_classes):
        return _WeightedLayers(shapes["audio"], shapes["text"], self.hp("hidden"), n_classes, self.hp("dropout"))

    def layer_weights(self) -> dict[str, np.ndarray]:
        return {"audio": torch.softmax(self.module.w_a, 0).detach().cpu().numpy(),
                "text": torch.softmax(self.module.w_t, 0).detach().cpu().numpy()}
