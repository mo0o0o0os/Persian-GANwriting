"""Hugging Face Trainer helpers shared by the audio and text fine-tuners.

Handles the API differences between transformers 4.x and 5.x in one place
(``evaluation_strategy`` -> ``eval_strategy``, ``tokenizer`` -> ``processing_class``, ...), prints one
line per epoch, and saves a fold's predictions in a single ``.npz`` file.
"""

from __future__ import annotations

import gc
import inspect
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from transformers import Trainer, TrainerCallback, TrainingArguments

from ..config import FinetuneConfig
from ..utils import log

_TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)
_TRAINER_PARAMS = set(inspect.signature(Trainer.__init__).parameters)


def compute_metrics(pred) -> dict:
    """The author's metric function (accuracy + weighted F1); 'accuracy' selects the best audio epoch."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="weighted")}


def training_arguments(cfg: FinetuneConfig, output_dir: Path, n_train: int, **extra) -> TrainingArguments:
    """TrainingArguments with the author's values, written so it works on transformers 4.x and 5.x."""
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        save_strategy="epoch",  # the author saved per epoch (audio) or every 500 steps (text); only resume uses it
        save_total_limit=2,
        logging_steps=cfg.logging_steps or max(1, n_train // cfg.batch_size),
        load_best_model_at_end=cfg.select_best,
        report_to="none",
        push_to_hub=False,
        disable_tqdm=False,
    )
    if cfg.select_best:
        kwargs.update(metric_for_best_model="accuracy", greater_is_better=True)
    kwargs["eval_strategy" if "eval_strategy" in _TA_PARAMS else "evaluation_strategy"] = "epoch"
    if "warmup_ratio" in _TA_PARAMS:
        kwargs["warmup_ratio"] = cfg.warmup_ratio
    else:  # newer transformers: a float warmup_steps < 1 is interpreted as a ratio
        kwargs["warmup_steps"] = cfg.warmup_ratio
    if "average_tokens_across_devices" in _TA_PARAMS:
        kwargs["average_tokens_across_devices"] = False  # otherwise the logged loss is doubled with 2 GPUs
    kwargs.update(extra)
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in _TA_PARAMS})


def make_trainer(model, args, train_dataset, eval_dataset, processor, data_collator=None, callbacks=None) -> Trainer:
    kwargs = dict(model=model, args=args, compute_metrics=compute_metrics, train_dataset=train_dataset,
                  eval_dataset=eval_dataset, data_collator=data_collator, callbacks=callbacks or [])
    kwargs["processing_class" if "processing_class" in _TRAINER_PARAMS else "tokenizer"] = processor
    return Trainer(**kwargs)


class EpochLogger(TrainerCallback):
    """One readable line per epoch: train loss, validation loss/accuracy/F1."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        epoch = round(state.epoch or 0)
        train = f"train_loss={self.last_train_loss:.4f}, " if self.last_train_loss is not None else ""
        log(f"{self.prefix} epoch {epoch}/{int(args.num_train_epochs)}: {train}"
            f"eval_loss={metrics.get('eval_loss', float('nan')):.4f}, "
            f"eval_accuracy={metrics.get('eval_accuracy', float('nan')):.4f}, eval_f1={metrics.get('eval_f1', float('nan')):.4f}")


def last_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.is_dir():
        return None
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(output_dir))


def predict_logits(trainer: Trainer, dataset) -> np.ndarray:
    return np.asarray(trainer.predict(dataset).predictions, dtype=np.float32)


def split_report(y: np.ndarray, logits: np.ndarray) -> dict:
    pred = logits.argmax(-1)
    return {"UA": balanced_accuracy_score(y, pred), "WA": accuracy_score(y, pred), "n": int(len(y))}


def save_fold_predictions(path: Path, **splits: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """splits: name -> (uids, y, logits)."""
    arrays = {}
    for name, (uids, y, logits) in splits.items():
        arrays[f"{name}_uids"] = np.asarray(uids)
        arrays[f"{name}_y"] = np.asarray(y)
        arrays[f"{name}_logits"] = np.asarray(logits, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def load_fold_predictions(path: Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        split, field = key.rsplit("_", 1)
        out.setdefault(split, {})[field] = data[key]
    return out


def release_gpu() -> None:
    """Free cached GPU memory once the caller has dropped its references."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def remove_checkpoints(output_dir: Path) -> None:
    shutil.rmtree(output_dir, ignore_errors=True)
