"""Runs one experiment stage by stage, fold by fold, skipping whatever is already in the store.

Stages
    finetune("audio") / finetune("text")   train the unimodal classifiers (hours / minutes)
    extract("audio") / extract("text")     embeddings for the fusion stage (minutes)
    unimodal_systems()                     the fine-tuned classifiers' own test predictions
    fuse()                                 every fusion method in the config (seconds each)

A session that dies mid-way resumes at the next call: finished folds carry a DONE marker,
unfinished training resumes from the last epoch checkpoint.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from ..config import ExperimentConfig, FusionSpec, resolve_checkpoint
from ..data.shemo import ShEMO
from ..data.splits import FoldSplit, make_fold_splits
from ..features.extract import extract_audio, extract_text, load_features, save_features
from ..fusion import FusionInput, create, softmax
from ..models.common import load_fold_predictions
from ..models.finetune import finetuner_for
from ..utils import log, read_json
from .store import ArtifactStore

MODALITIES = ("audio", "text")


class Experiment:
    def __init__(self, cfg: ExperimentConfig, data: ShEMO, store: ArtifactStore):
        self.cfg = cfg
        self.data = data
        self.store = store
        self.classes = data.classes
        self.groups = data.groups(cfg.data)
        self._splits: dict[tuple, dict[int, FoldSplit]] = {}
        self._features: dict[tuple, dict[str, np.ndarray]] = {}
        store.register_experiment(cfg)

    # ------------------------------------------------------------------ helpers
    @property
    def folds(self) -> list[int]:
        return sorted(self.groups)

    def model_cfg(self, modality: str):
        return {"audio": self.cfg.audio, "text": self.cfg.text}[modality]

    def splits(self, modality: str) -> dict[int, FoldSplit]:
        m = self.model_cfg(modality)
        key = (m.seed, m.valid_fraction, m.stratify_valid)
        if key not in self._splits:
            self._splits[key] = make_fold_splits(self.groups, self.data.y, len(self.classes), m.seed,
                                                 m.valid_fraction, m.stratify_valid)
        return self._splits[key]

    def finetune_dir(self, modality: str, fold: int):
        return self.store.finetune_dir(self.model_cfg(modality), self.cfg.data, fold)

    # ------------------------------------------------------------------ stage 1: fine-tuning
    def finetune(self, modality: str, folds: list[int] | None = None, force: bool = False) -> pd.DataFrame:
        import torch

        mcfg = self.model_cfg(modality)
        tuner = finetuner_for(mcfg, self.classes)
        rows = []
        for k in folds or self.folds:
            d = self.finetune_dir(modality, k)
            if self.store.is_done(d) and not force:
                log(f"[{modality} {mcfg.name} fold {k}] قبلاً انجام شده؛ رد شد")
            else:
                if force and d.exists():
                    shutil.rmtree(d)
                if not torch.cuda.is_available():
                    log("هشدار: GPU در دسترس نیست؛ fine-tune روی CPU بسیار کند است.")
                tuner.run_fold(self.data, self.splits(modality)[k], d)
                self.store.mark_done(d, {"model": mcfg.checkpoint})
            summary = read_json(d / "history.json")["summary"]
            rows.append({"fold": k, **{f"{s} UA": round(summary[s]["UA"] * 100, 2) for s in ("train", "valid", "test")},
                         "test WA": round(summary["test"]["WA"] * 100, 2), "minutes": summary.get("minutes")})
        return pd.DataFrame(rows).set_index("fold")

    # ------------------------------------------------------------------ stage 2: embeddings
    def _extract(self, modality: str, model_path: str, uids: np.ndarray, desc: str):
        feat = self.cfg.features
        if modality == "audio":
            return extract_audio(model_path, self.data.audios(uids), feat.audio_batch_size, feat.pad_aware,
                                 feat.keep_layers, desc)
        return extract_text(model_path, self.data.texts(uids), feat.text_batch_size, feat.keep_layers,
                            self.cfg.text.max_length, desc)

    def extract(self, modality: str, folds: list[int] | None = None, force: bool = False) -> None:
        mcfg, feat = self.model_cfg(modality), self.cfg.features
        if feat.source == "pretrained":
            path = self.store.feature_path(modality, mcfg, feat, self.cfg.data, None)
            if path.exists() and not force:
                log(f"[{modality} {mcfg.name}] ویژگی‌های مدل پیش‌آموخته قبلاً استخراج شده")
                return
            uids = np.arange(len(self.data))
            avg, layers = self._extract(modality, resolve_checkpoint(mcfg.checkpoint), uids, f"{modality} (pretrained)")
            save_features(path, uids=uids, avg=avg, layers=layers)
            return
        for k in folds or self.folds:
            path = self.store.feature_path(modality, mcfg, feat, self.cfg.data, k)
            if path.exists() and not force:
                log(f"[{modality} {mcfg.name} fold {k}] ویژگی‌ها قبلاً استخراج شده")
                continue
            d = self.finetune_dir(modality, k)
            if not self.store.is_done(d):
                raise RuntimeError(f"مدل {modality} fold {k} هنوز آموزش ندیده؛ اول finetune(\"{modality}\") را اجرا کنید.")
            sp = self.splits(modality)[k]
            tr_avg, tr_layers = self._extract(modality, str(d / "model"), sp.feature_train, f"{modality} fold {k} train")
            te_avg, te_layers = self._extract(modality, str(d / "model"), sp.test, f"{modality} fold {k} test")
            save_features(path, train_uids=sp.feature_train, train_avg=tr_avg, train_layers=tr_layers,
                          test_uids=sp.test, test_avg=te_avg, test_layers=te_layers)

    def _load_features(self, modality: str, fold: int) -> dict[str, np.ndarray]:
        mcfg, feat = self.model_cfg(modality), self.cfg.features
        path = self.store.feature_path(modality, mcfg, feat, self.cfg.data, None if feat.source == "pretrained" else fold)
        key = (str(path),)
        if key not in self._features:
            if not path.exists():
                raise RuntimeError(f"ویژگی‌های {modality} برای fold {fold} نیست؛ اول extract(\"{modality}\") را اجرا کنید.")
            if len(self._features) >= 4:  # keep memory bounded (per-layer arrays are large)
                self._features.clear()
            self._features[key] = load_features(path)
        return self._features[key]

    def fusion_input(self, fold: int, part: str) -> FusionInput:
        """Everything a fusion method may use, for the 'train' rows (all other folds) or the 'test' fold."""
        split = self.splits("text")[fold]
        uids = split.feature_train if part == "train" else split.test
        features, layers, logits = {}, {}, {}
        for m in MODALITIES:
            f = self._load_features(m, fold)
            if self.cfg.features.source == "pretrained":
                rows = f["uids"].searchsorted(uids)
                features[m] = f["avg"][rows]
                if "layers" in f:
                    layers[m] = f["layers"][rows]
            else:
                if not np.array_equal(f[f"{part}_uids"], uids):
                    raise RuntimeError("ترتیب جمله‌ها در ویژگی‌ها با انتظار نمی‌خواند")
                features[m] = f[f"{part}_avg"]
                if f"{part}_layers" in f:
                    layers[m] = f[f"{part}_layers"]
            d = self.finetune_dir(m, fold)
            if self.store.is_done(d):
                preds = load_fold_predictions(d / "predictions.npz")
                lookup = {}
                for s in preds.values():
                    lookup.update(zip(s["uids"].tolist(), s["logits"]))
                logits[m] = np.stack([lookup[u] for u in uids.tolist()])
        return FusionInput(np.asarray(uids), self.data.labels(uids), len(self.classes), features, layers, logits)

    # ------------------------------------------------------------------ stage 3: systems
    def unimodal_systems(self) -> None:
        """Store the fine-tuned classifiers' test predictions as the systems 'audio' and 'text'."""
        for m in MODALITIES:
            mcfg = self.model_cfg(m)
            for k in self.folds:
                d = self.finetune_dir(m, k)
                if not self.store.is_done(d):
                    continue
                test = load_fold_predictions(d / "predictions.npz")["test"]
                self.store.save_predictions(self.cfg.name, m, k, test["uids"], test["y"], test["logits"].argmax(1),
                                            softmax(test["logits"]), kind=m, meta={"model": mcfg.name})

    def fuse(self, specs: list[FusionSpec] | None = None, folds: list[int] | None = None, force: bool = False) -> pd.DataFrame:
        rows = []
        for spec in specs or list(self.cfg.fusion):
            uas = []
            for k in folds or self.folds:
                if self.store.has_predictions(self.cfg.name, spec.system_name, k) and not force:
                    data = np.load(self.store.prediction_path(self.cfg.name, spec.system_name, k))
                    uas.append(balanced_accuracy_score(data["y_true"], data["y_pred"]))
                    continue
                train, test = self.fusion_input(k, "train"), self.fusion_input(k, "test")
                method = create(spec.method, **({"seed": 0} | spec.kwargs))
                method.fit(train)
                y_pred = method.predict(test)
                proba = method.predict_proba(test)
                train_ua = balanced_accuracy_score(train.y, method.predict(train))
                self.store.save_predictions(self.cfg.name, spec.system_name, k, test.uids, test.y, y_pred, proba,
                                            kind=method.kind, meta={"method": spec.method, "params": spec.kwargs,
                                                                 "description": method.description})
                np.save(self.store.prediction_path(self.cfg.name, spec.system_name, k).with_suffix(".train_ua.npy"), train_ua)
                uas.append(balanced_accuracy_score(test.y, y_pred))
                log(f"[{self.cfg.name}] {spec.system_name} fold {k}: UA تست={uas[-1] * 100:.2f}٪ (UA آموزش={train_ua * 100:.1f}٪)")
            rows.append({"system": spec.system_name, "UA mean": round(np.mean(uas) * 100, 2),
                         "UA std": round(np.std(uas, ddof=1) * 100, 2) if len(uas) > 1 else np.nan})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ everything
    def run(self, stages=("audio", "text", "features", "fusion"), folds: list[int] | None = None) -> None:
        if "audio" in stages:
            display_or_log(self.finetune("audio", folds), f"نتیجهٔ fine-tune صوت ({self.cfg.audio.name})")
        if "text" in stages:
            display_or_log(self.finetune("text", folds), f"نتیجهٔ fine-tune متن ({self.cfg.text.name})")
        if "features" in stages:
            self.extract("audio", folds)
            self.extract("text", folds)
        if "fusion" in stages:
            self.unimodal_systems()
            display_or_log(self.fuse(folds=folds), f"نتیجهٔ روش‌های ترکیب ({self.cfg.name})")

    def status(self) -> pd.DataFrame:
        rows = {}
        for k in self.folds:
            row = {}
            for m in MODALITIES:
                row[f"fine-tune {m}"] = "✓" if self.store.is_done(self.finetune_dir(m, k)) else "·"
                path = self.store.feature_path(m, self.model_cfg(m), self.cfg.features, self.cfg.data,
                                               None if self.cfg.features.source == "pretrained" else k)
                row[f"features {m}"] = "✓" if path.exists() else "·"
            for spec in self.cfg.fusion:
                row[spec.system_name] = "✓" if self.store.has_predictions(self.cfg.name, spec.system_name, k) else "·"
            rows[f"fold {k}"] = row
        return pd.DataFrame(rows).T


def display_or_log(df: pd.DataFrame, title: str) -> None:
    log(title)
    try:
        from IPython.display import display

        display(df)
    except ImportError:
        print(df.to_string())
