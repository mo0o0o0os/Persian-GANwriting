"""Experiment settings.

Every stage of the pipeline reads its settings from one of these dataclasses. The directory an
artifact is cached in is derived from a hash of the settings that produced it, so:

* changing only the text model reuses the (expensive) audio models and features;
* changing any hyper-parameter produces a new cache entry instead of silently mixing results.

``paper_shemo()`` returns the exact settings of the author's ShEMO notebooks.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from .utils import short_model_name, slugify, stable_hash

SHEMO_CLASSES = ("anger", "happiness", "neutral", "sadness", "surprise")

# Hugging Face name -> local directory. Only needed offline or for tests with tiny models.
MODEL_OVERRIDES: dict[str, str] = {}


def resolve_checkpoint(name: str) -> str:
    return MODEL_OVERRIDES.get(name, name)


@dataclass(frozen=True)
class DataConfig:
    """Which utterances go into which fold.

    fold_scheme:
        "paper"   the author's five session CSVs (random 5-fold; the same speaker appears in
                  train and test, i.e. speaker-dependent).
        "speaker" five speaker-independent folds (StratifiedGroupKFold over speakers).
    """

    classes: tuple[str, ...] = SHEMO_CLASSES
    fold_scheme: str = "paper"
    n_folds: int = 5
    speaker_fold_seed: int = 0
    sample_rate: int = 16_000
    quick_rows_per_class: int = 0  # >0: keep only this many rows per class per fold (smoke tests)

    def key(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True)
class FinetuneConfig:
    """Fine-tuning recipe of one unimodal classifier (one per modality)."""

    modality: str  # "audio" | "text"
    checkpoint: str
    seed: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    batch_size: int
    grad_accum: int = 1
    valid_fraction: float = 0.1
    stratify_valid: bool = False
    select_best: bool = False  # keep the best epoch on the validation split (by accuracy)
    max_duration_s: float | None = None  # audio: truncate inputs to this many seconds
    freeze_feature_encoder: bool = False  # audio: freeze wav2vec2's CNN front-end
    max_length: int | None = None  # text: truncation length (None = model maximum)
    logging_steps: int | None = None  # None = one log line per epoch (author's text setting)

    def key(self) -> str:
        return stable_hash(self)

    @property
    def name(self) -> str:
        return short_model_name(self.checkpoint)

    @property
    def dirname(self) -> str:
        return f"{slugify(self.checkpoint)}-{self.key()}"

    def quick(self) -> "FinetuneConfig":
        """One epoch, for smoke tests."""
        return dataclasses.replace(self, epochs=1)


@dataclass(frozen=True)
class FeatureConfig:
    """How utterance embeddings are extracted for the fusion stage.

    source:
        "finetuned"  from each fold's fine-tuned model (the paper)
        "pretrained" from the original checkpoint, frozen (no fold-specific training, so no
                     leakage between the encoder and the fusion classifier)
    pad_aware:
        False reproduces the author's audio code, which averages over padded frames too.
        True averages only over real frames.
    """

    source: str = "finetuned"
    pooling: str = "layer_mean"  # mean over time for every hidden layer, then average over layers
    audio_batch_size: int = 25
    text_batch_size: int = 25
    pad_aware: bool = False
    keep_layers: bool = True  # also store the per-layer vectors (for learnable layer weights later)

    def key(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True)
class FusionSpec:
    """A fusion method from ``atser.fusion`` plus its parameters."""

    method: str
    params: tuple[tuple[str, Any], ...] = ()
    label: str | None = None

    @staticmethod
    def of(method: str, label: str | None = None, **params) -> "FusionSpec":
        return FusionSpec(method, tuple(sorted(params.items())), label)

    @property
    def kwargs(self) -> dict:
        return dict(self.params)

    @property
    def system_name(self) -> str:
        if self.label:
            return self.label
        if not self.params:
            return self.method
        return self.method + "(" + ",".join(f"{k}={v}" for k, v in self.params) + ")"


@dataclass(frozen=True)
class ExperimentConfig:
    """One complete experiment: data + audio model + text model + features + fusion methods."""

    name: str
    data: DataConfig
    audio: FinetuneConfig
    text: FinetuneConfig
    features: FeatureConfig = field(default_factory=FeatureConfig)
    fusion: tuple[FusionSpec, ...] = (FusionSpec.of("paper_sum_svm"),)

    # Every with_* method returns a new config whose name is derived from its settings
    # (see auto_name); use .named("...") afterwards for a custom name.
    def with_text_model(self, checkpoint: str, **overrides) -> "ExperimentConfig":
        return self._renamed(text=dataclasses.replace(self.text, checkpoint=checkpoint, **overrides))

    def with_audio_model(self, checkpoint: str, **overrides) -> "ExperimentConfig":
        return self._renamed(audio=dataclasses.replace(self.audio, checkpoint=checkpoint, **overrides))

    def with_fusion(self, *specs: FusionSpec) -> "ExperimentConfig":
        return dataclasses.replace(self, fusion=tuple(specs))

    def with_features(self, **changes) -> "ExperimentConfig":
        return self._renamed(features=dataclasses.replace(self.features, **changes))

    def with_fold_scheme(self, scheme: str) -> "ExperimentConfig":
        return self._renamed(data=dataclasses.replace(self.data, fold_scheme=scheme))

    def quick(self, rows_per_class: int = 20) -> "ExperimentConfig":
        """A small subset and one epoch: checks that everything runs, in minutes."""
        return self._renamed(audio=self.audio.quick(), text=self.text.quick(),
                             data=dataclasses.replace(self.data, quick_rows_per_class=rows_per_class))

    def named(self, name: str) -> "ExperimentConfig":
        return dataclasses.replace(self, name=name)

    def _renamed(self, **changes) -> "ExperimentConfig":
        new = dataclasses.replace(self, **changes)
        return dataclasses.replace(new, name=auto_name(new))


def auto_name(cfg: ExperimentConfig) -> str:
    """e.g. 'paper__bert-fa-base-uncased', 'speaker__bert-base-uncased__pretrained', '...__quick'.

    A 4-character hash is added when a model's hyper-parameters differ from the paper's, so two
    different recipes never share a name (the store refuses to mix them).
    """
    quick = bool(cfg.data.quick_rows_per_class)  # quick runs change epochs on purpose; the name says "quick"

    def model_part(m: FinetuneConfig, paper: FinetuneConfig) -> str:
        same = dataclasses.replace(m, checkpoint=paper.checkpoint, epochs=paper.epochs if quick else m.epochs) == paper
        return m.name + ("" if same else f"-{m.key()[:4]}")

    parts = [cfg.data.fold_scheme, model_part(cfg.text, PAPER_TEXT)]
    if cfg.audio.checkpoint != PAPER_AUDIO.checkpoint or model_part(cfg.audio, PAPER_AUDIO) != cfg.audio.name:
        parts.append("audio=" + model_part(cfg.audio, PAPER_AUDIO))
    if cfg.features != FeatureConfig():
        parts.append(cfg.features.source + ("" if cfg.features == FeatureConfig(source=cfg.features.source)
                                            else f"-{cfg.features.key()[:4]}"))
    if cfg.data.quick_rows_per_class:
        parts.append("quick")
    return "__".join(parts)


# --------------------------------------------------------------------------------------------
# The author's settings, copied from the notebooks (cell numbers refer to the original files).
# --------------------------------------------------------------------------------------------

# ShEMO/Speech models/Speech_Emotion_Recognition_CV.ipynb, cells 33, 40, 41, 45
PAPER_AUDIO = FinetuneConfig(
    modality="audio",
    checkpoint="facebook/wav2vec2-base",
    seed=1968,
    epochs=6,
    learning_rate=1e-4,
    weight_decay=0.00002,
    warmup_ratio=0.1,
    batch_size=16,
    grad_accum=2,
    valid_fraction=0.1,
    stratify_valid=False,
    select_best=True,
    max_duration_s=5.0,
    freeze_feature_encoder=True,
    logging_steps=10,
)

# ShEMO/Text models/Text_Emotion_Recogniton_CV.ipynb, cells 31, 40, 41, 45
PAPER_TEXT = FinetuneConfig(
    modality="text",
    checkpoint="bert-base-uncased",
    seed=42,
    epochs=4,
    learning_rate=5e-5,
    weight_decay=0.1,
    warmup_ratio=0.1,
    batch_size=32,
    grad_accum=1,
    valid_fraction=0.1,
    stratify_valid=True,
    select_best=False,
    max_length=None,
)


def paper_shemo(text_checkpoint: str = "bert-base-uncased", fold_scheme: str = "paper") -> ExperimentConfig:
    """The ShEMO pipeline exactly as in the paper's code (optionally with another text model)."""
    cfg = ExperimentConfig(
        name="",
        data=DataConfig(fold_scheme=fold_scheme),
        audio=PAPER_AUDIO,
        text=dataclasses.replace(PAPER_TEXT, checkpoint=text_checkpoint),
        features=FeatureConfig(),
        fusion=(FusionSpec.of("paper_sum_svm"),),
    )
    return cfg.named(auto_name(cfg))


# Numbers used by the reproduction check (UA, %, per fold 1..5).
REFERENCE_RESULTS = {
    "paper (Table, ShEMO)": {"audio": 74.83, "text": 40.35, "fusion": 76.37},
    "author repo outputs": {
        "audio": [75.97, 81.96, 68.10, 72.22, 75.88],
        "fusion": [76.11, 81.08, 71.52, 76.40, 76.74],
    },
    "our Kaggle run (repo notebooks, Sep 2026)": {
        "audio": [76.56, 81.07, 77.23, 73.87, 76.06],
        "text": [42.98, 41.19, 36.91, 39.47, 36.90],
        "fusion": [78.05, 80.61, 78.34, 72.76, 73.77],
    },
}
