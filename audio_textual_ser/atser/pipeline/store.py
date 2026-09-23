"""On-disk layout of everything the pipeline produces, with "is this already done?" checks.

    atser_store/
      STORE.json                                    marker (lets a later session find this store)
      finetune/<modality>/<model>-<hash>/<folds>/fold<k>/
          model/  predictions.npz  history.json  DONE
      features/<modality>/<source>/<model>-<hash>/<feature-hash>/<folds>/fold<k>.npz
      features/<modality>/pretrained/<model>/<feature-hash>/<data-hash>/all.npz
      predictions/<experiment>/<system>/fold<k>.npz  (uids, y_true, y_pred, proba)
      experiments/<experiment>.json                 the full config of each experiment

``<folds>`` = fold scheme + hash of the data settings, so quick tests never mix with full runs.
Every hash comes from the settings that produced the artifact (see atser.config), so changing the
text model reuses the audio artifacts, and changing a hyper-parameter creates new ones.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import DataConfig, ExperimentConfig, FeatureConfig, FinetuneConfig
from ..utils import dir_size, human_size, log, read_json, slugify, write_json

MARKER = "STORE.json"


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root / MARKER).exists():
            write_json(self.root / MARKER, {"atser_store": 1, "created": datetime.now().isoformat(timespec="seconds")})

    # ------------------------------------------------------------------ paths
    @staticmethod
    def folds_tag(data: DataConfig) -> str:
        return f"{data.fold_scheme}-{data.key()}"

    def finetune_dir(self, cfg: FinetuneConfig, data: DataConfig, fold: int) -> Path:
        return self.root / "finetune" / cfg.modality / cfg.dirname / self.folds_tag(data) / f"fold{fold}"

    def feature_path(self, modality: str, model_cfg: FinetuneConfig, feat: FeatureConfig, data: DataConfig,
                     fold: int | None) -> Path:
        base = self.root / "features" / modality / feat.source
        if feat.source == "pretrained":
            return base / slugify(model_cfg.checkpoint) / feat.key() / data.key() / "all.npz"
        return base / model_cfg.dirname / feat.key() / self.folds_tag(data) / f"fold{fold}.npz"

    def prediction_path(self, experiment: str, system: str, fold: int) -> Path:
        return self.root / "predictions" / slugify(experiment) / slugify(system) / f"fold{fold}.npz"

    # ------------------------------------------------------------------ done markers
    @staticmethod
    def is_done(directory: Path) -> bool:
        return (Path(directory) / "DONE").exists()

    @staticmethod
    def mark_done(directory: Path, info: dict | None = None) -> None:
        write_json(Path(directory) / "DONE", {"finished": datetime.now().isoformat(timespec="seconds"), **(info or {})})

    # ------------------------------------------------------------------ experiments
    def register_experiment(self, cfg: ExperimentConfig) -> None:
        path = self.root / "experiments" / f"{slugify(cfg.name)}.json"
        old = read_json(path)
        new_keys = {"data": cfg.data.key(), "audio": cfg.audio.key(), "text": cfg.text.key(), "features": cfg.features.key()}
        if old and old.get("keys") != new_keys:
            raise ValueError(f"آزمایشی با نام «{cfg.name}» قبلاً با تنظیمات دیگری اجرا شده؛ برای تنظیمات جدید نام دیگری بگذارید.")
        write_json(path, {"name": cfg.name, "keys": new_keys, "config": cfg,
                          "audio_model": cfg.audio.name, "text_model": cfg.text.name,
                          "fold_scheme": cfg.data.fold_scheme, "feature_source": cfg.features.source})

    def experiments(self) -> dict[str, dict]:
        return {p.stem: read_json(p) for p in sorted((self.root / "experiments").glob("*.json"))}

    # ------------------------------------------------------------------ predictions
    def save_predictions(self, experiment: str, system: str, fold: int, uids, y_true, y_pred, proba=None,
                         kind: str = "fusion", meta: dict | None = None) -> None:
        path = self.prediction_path(experiment, system, fold)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(uids=np.asarray(uids), y_true=np.asarray(y_true), y_pred=np.asarray(y_pred))
        if proba is not None:
            arrays["proba"] = np.asarray(proba, dtype=np.float32)
        np.savez_compressed(path, **arrays)
        write_json(path.parent / "system.json", {"experiment": experiment, "system": system, "kind": kind, **(meta or {})})

    def has_predictions(self, experiment: str, system: str, fold: int) -> bool:
        return self.prediction_path(experiment, system, fold).exists()

    def iter_predictions(self, experiments: list[str] | None = None):
        """Yield (experiment_meta, system_meta, fold, arrays) for every stored prediction file."""
        exps = self.experiments()
        wanted = {slugify(e) for e in experiments} if experiments else None
        for exp_dir in sorted((self.root / "predictions").glob("*")):
            if wanted is not None and exp_dir.name not in wanted:
                continue
            exp_meta = exps.get(exp_dir.name, {"name": exp_dir.name})
            for sys_dir in sorted(exp_dir.glob("*")):
                sys_meta = read_json(sys_dir / "system.json", {"system": sys_dir.name, "kind": "fusion"})
                for f in sorted(sys_dir.glob("fold*.npz")):
                    data = np.load(f)
                    yield exp_meta, sys_meta, int(f.stem[4:]), {k: data[k] for k in data.files}

    # ------------------------------------------------------------------ housekeeping
    def usage(self):
        import pandas as pd

        rows = [{"folder": p.name, "size": human_size(dir_size(p))} for p in sorted(self.root.glob("*")) if p.is_dir()]
        return pd.DataFrame(rows)

    def restore_from(self, sources: list[Path], include_models: bool = True) -> int:
        """Copy artifacts from earlier runs (e.g. an attached Kaggle notebook output) that are not here yet."""
        copied = 0
        for src in sources:
            src = Path(src)
            for f in src.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(src)
                if not include_models and "model" in rel.parts:
                    continue
                dst = self.root / rel
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                copied += 1
        if copied:
            log(f"{copied} فایل از اجراهای قبلی به {self.root} کپی شد")
        return copied


def find_stores(search_roots: list[Path], exclude: Path | None = None) -> list[Path]:
    """Find ``atser_store`` folders (by their STORE.json marker) under the given roots."""
    found = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for marker in root.rglob(MARKER):
            store = marker.parent
            if exclude is not None and store.resolve() == Path(exclude).resolve():
                continue
            found.append(store)
    return found
