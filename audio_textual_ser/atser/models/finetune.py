"""Fine-tuning of the unimodal classifiers, one fold at a time.

AudioFinetuner reproduces ShEMO/Speech models/Speech_Emotion_Recognition_CV.ipynb (cells 33-45):
    AutoFeatureExtractor + ``map(preprocess_function, batched=True)`` with max 5 s, truncation and
    padding to the longest clip of every 1000-row chunk, AutoModelForAudioClassification with the
    CNN feature encoder frozen, best epoch by validation accuracy.

TextFinetuner reproduces ShEMO/Text models/Text_Emotion_Recogniton_CV.ipynb (cells 31-45):
    AutoTokenizer(truncation=True) + DataCollatorWithPadding, AutoModelForSequenceClassification,
    the last epoch is kept.

Each fold writes into its own directory:
    model/            the fine-tuned model + processor (used later for feature extraction)
    predictions.npz   logits of the train / valid / test splits, with uids and labels
    history.json      per-epoch log, settings, and accuracies of every split

To use a different architecture, subclass ``Finetuner`` and override ``load_processor``,
``build_dataset`` and ``load_model``; the training loop, resume and saving stay the same.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import FinetuneConfig, resolve_checkpoint
from ..data.shemo import ShEMO
from ..data.splits import FoldSplit
from ..utils import log, set_all_seeds, write_json
from .common import (EpochLogger, last_checkpoint, make_trainer, predict_logits, release_gpu, remove_checkpoints,
                     save_fold_predictions, split_report, training_arguments)


class Finetuner:
    modality = "base"

    def __init__(self, cfg: FinetuneConfig, classes: tuple[str, ...]):
        self.cfg = cfg
        self.classes = tuple(classes)
        self.label2id = {c: str(i) for i, c in enumerate(self.classes)}
        self.id2label = {str(i): c for i, c in enumerate(self.classes)}

    # ---- to override -------------------------------------------------------------------
    def load_processor(self):
        raise NotImplementedError

    def build_dataset(self, data: ShEMO, uids: np.ndarray, processor):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def data_collator(self, processor):
        return None

    # ---- shared loop -------------------------------------------------------------------
    def run_fold(self, data: ShEMO, split: FoldSplit, fold_dir: Path) -> dict:
        fold_dir = Path(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        tag = f"[{self.modality} {self.cfg.name} fold {split.fold}]"
        set_all_seeds(self.cfg.seed)

        processor = self.load_processor()
        datasets = {name: self.build_dataset(data, uids, processor)
                    for name, uids in (("train", split.train), ("valid", split.valid), ("test", split.test))}
        log(f"{tag} train={len(split.train)}، valid={len(split.valid)}، test={len(split.test)}")

        model = self.load_model()
        ckpt_dir = fold_dir / "checkpoints"
        args = training_arguments(self.cfg, ckpt_dir, len(datasets["train"]))
        trainer = make_trainer(model, args, datasets["train"], datasets["valid"], processor,
                               data_collator=self.data_collator(processor), callbacks=[EpochLogger(tag)])
        resume = last_checkpoint(ckpt_dir)
        if resume:
            log(f"{tag} ادامهٔ آموزش از {Path(resume).name}")
        trainer.train(resume_from_checkpoint=resume)

        logits = {name: predict_logits(trainer, ds) for name, ds in datasets.items()}
        uids = {"train": split.train, "valid": split.valid, "test": split.test}
        ys = {name: data.labels(u) for name, u in uids.items()}
        save_fold_predictions(fold_dir / "predictions.npz",
                              **{name: (uids[name], ys[name], logits[name]) for name in uids})
        trainer.save_model(str(fold_dir / "model"))
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(str(fold_dir / "model"))

        summary = {name: split_report(ys[name], logits[name]) for name in uids}
        summary["minutes"] = round((time.time() - start) / 60, 1)
        write_json(fold_dir / "history.json", {"config": self.cfg, "fold": split.fold, "summary": summary,
                                              "log_history": trainer.state.log_history})
        log(f"{tag} UA تست={summary['test']['UA'] * 100:.2f}٪، WA تست={summary['test']['WA'] * 100:.2f}٪ "
            f"(UA آموزش={summary['train']['UA'] * 100:.1f}٪) — {summary['minutes']} دقیقه")
        remove_checkpoints(ckpt_dir)
        del trainer, model, datasets
        release_gpu()
        return summary


class AudioFinetuner(Finetuner):
    modality = "audio"

    def load_processor(self):
        from transformers import AutoFeatureExtractor

        return AutoFeatureExtractor.from_pretrained(resolve_checkpoint(self.cfg.checkpoint))

    def encode(self, arrays: list[np.ndarray], feature_extractor, chunk: int = 1000) -> list[np.ndarray]:
        """Same as the author's ``dataset.map(preprocess_function, batched=True)`` (default 1000 rows per call)."""
        kwargs = dict(sampling_rate=feature_extractor.sampling_rate, padding=True)
        if self.cfg.max_duration_s:
            kwargs.update(max_length=int(feature_extractor.sampling_rate * self.cfg.max_duration_s), truncation=True)
        out: list[np.ndarray] = []
        for i in range(0, len(arrays), chunk):
            out.extend(feature_extractor(arrays[i:i + chunk], **kwargs)["input_values"])
        return out

    def build_dataset(self, data: ShEMO, uids: np.ndarray, processor):
        from datasets import Dataset

        values = self.encode(data.audios(uids), processor)
        return Dataset.from_dict({"input_values": values, "label": [int(v) for v in data.labels(uids)]})

    def load_model(self):
        from transformers import AutoModelForAudioClassification

        model = AutoModelForAudioClassification.from_pretrained(
            resolve_checkpoint(self.cfg.checkpoint), num_labels=len(self.classes),
            label2id=self.label2id, id2label=self.id2label)
        if self.cfg.freeze_feature_encoder:
            (model.freeze_feature_encoder if hasattr(model, "freeze_feature_encoder") else model.freeze_feature_extractor)()
        return model


class TextFinetuner(Finetuner):
    modality = "text"

    def load_processor(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(resolve_checkpoint(self.cfg.checkpoint))

    def build_dataset(self, data: ShEMO, uids: np.ndarray, processor):
        from datasets import Dataset

        ds = Dataset.from_dict({"text": data.texts(uids), "label": [int(v) for v in data.labels(uids)]})
        kwargs = {"truncation": True}
        if self.cfg.max_length:
            kwargs["max_length"] = self.cfg.max_length
        return ds.map(lambda batch: processor(batch["text"], **kwargs), batched=True)

    def data_collator(self, processor):
        from transformers import DataCollatorWithPadding

        return DataCollatorWithPadding(tokenizer=processor)

    def load_model(self):
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification.from_pretrained(
            resolve_checkpoint(self.cfg.checkpoint), num_labels=len(self.classes),
            label2id=self.label2id, id2label=self.id2label)


FINETUNERS = {"audio": AudioFinetuner, "text": TextFinetuner}


def finetuner_for(cfg: FinetuneConfig, classes: tuple[str, ...]) -> Finetuner:
    return FINETUNERS[cfg.modality](cfg, classes)
