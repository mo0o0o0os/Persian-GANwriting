"""ShEMO (modified version) — loading, 16 kHz resampling and fold membership.

What the author did (ShEMO/Preprocessing dataset/Preprocess_ShEMO_for_HuggingFace_with_splits.ipynb):
the modified ShEMO (https://github.com/aliyzd95/modified-shemo) was split into five random
"sessions" saved as ``ShEMO/Dataset/Sessions/session_{1..5}.csv`` in the paper's repo, and uploaded
to Hugging Face as ``Zahra99/ShEMO_Audio_Sessions`` / ``Zahra99/ShEMO_Text_Sessions`` (private).
We rebuild the same thing from those CSVs:

* text  = the ``transcription`` column, unchanged
* label = ClassLabel over ['anger', 'happiness', 'neutral', 'sadness', 'surprise']
* audio = the original wav, decoded like ``datasets.Audio(sampling_rate=16_000)`` does it
          (soundfile -> float64 -> librosa.resample), cached once as 16 kHz float64 wav files.

Each utterance gets a ``uid`` (its row number in the concatenated CSVs); everything downstream
refers to utterances by ``uid``.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import SHEMO_CLASSES, DataConfig
from ..utils import log

AUTHOR_REPO_URL = "https://github.com/ZahraDehghani99/Audio-Textual-Emotion-Recognition-using-Pre-trained-models.git"
AUTHOR_REPO_COMMIT = "09ab996377fd05064216a3d2d3c1a29a6a0f0a84"
MODIFIED_SHEMO_URL = "https://github.com/aliyzd95/modified-shemo.git"
MODIFIED_SHEMO_COMMIT = "e9fb83717ef03755e038b96aac111d352498773a"
SHEMO_KAGGLE_DATASET = "mansourehk/shemo-persian-speech-emotion-detection-database"

_SHEMO_NAME = re.compile(r"^[FM]\d{2}[A-Z]\d{2}$")


def git_clone(url: str, dest: str | os.PathLike, commit: str | None = None) -> Path:
    """Clone ``url`` into ``dest`` (once) and check out ``commit``."""
    dest = Path(dest)
    if not (dest / ".git").is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "-q", url, str(dest)])
    if commit:
        subprocess.check_call(["git", "-C", str(dest), "checkout", "-q", commit])
    head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "--short", "HEAD"], text=True).strip()
    log(f"{url.rsplit('/', 1)[-1]} → {dest} (commit {head})")
    return dest


def fetch_author_repo(dest: str | os.PathLike) -> Path:
    """The paper's repository at the commit we reproduced; only its session CSVs are used."""
    return git_clone(AUTHOR_REPO_URL, dest, AUTHOR_REPO_COMMIT)


def find_shemo_wavs(search_roots: list[str | os.PathLike]) -> dict[str, Path]:
    """Map 'F01A01' -> path for every ShEMO wav found under the given folders."""
    found: dict[str, Path] = {}
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*.wav"):
            if _SHEMO_NAME.match(p.stem) and p.stem not in found:
                found[p.stem] = p
    return found


def download_shemo_with_kagglehub() -> Path:
    """For Colab/local runs: download the Kaggle copy of ShEMO (needs Kaggle credentials)."""
    import kagglehub

    return Path(kagglehub.dataset_download(SHEMO_KAGGLE_DATASET))


def _decode_like_datasets(src: Path, target_sr: int) -> np.ndarray:
    """Same steps as ``datasets.Audio.decode_example`` (datasets 2.x/3.x): float64, mono, librosa resample."""
    import soundfile as sf

    array, sr = sf.read(str(src))  # float64
    array = array.T
    if array.ndim > 1:  # librosa.to_mono
        array = np.mean(array, axis=0)
    if sr != target_sr:
        if importlib.util.find_spec("librosa") is not None:
            import librosa

            array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
        else:  # fallback when librosa is missing: numerically very close, not bit-identical
            import torch
            import torchaudio

            array = torchaudio.functional.resample(torch.from_numpy(array), sr, target_sr).numpy()
    return np.asarray(array, dtype=np.float64)


class ShEMO:
    """The modified ShEMO corpus as used by the paper.

    ``table`` columns: uid, utterance_name, text, label, y, session, speaker, gender, wav16, seconds
    """

    def __init__(self, table: pd.DataFrame, classes: tuple[str, ...] = SHEMO_CLASSES):
        self.table = table.reset_index(drop=True)
        self.classes = tuple(classes)
        self._audio_cache: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- construction
    @classmethod
    def prepare(cls, sessions_dir: str | os.PathLike, wav_roots: list[str | os.PathLike], cache_dir: str | os.PathLike,
                data_cfg: DataConfig | None = None) -> "ShEMO":
        """Read the session CSVs, locate the wavs and cache 16 kHz copies."""
        data_cfg = data_cfg or DataConfig()
        sessions_dir = Path(sessions_dir)
        frames = []
        for s in range(1, data_cfg.n_folds + 1):
            df = pd.read_csv(sessions_dir / f"session_{s}.csv")
            if data_cfg.quick_rows_per_class:
                df = df.groupby("label").head(data_cfg.quick_rows_per_class)
            df = df.assign(session=s)
            frames.append(df)
        table = pd.concat(frames, ignore_index=True)
        table = table.rename(columns={"transcription": "text"})
        unknown = set(table["label"]) - set(data_cfg.classes)
        if unknown:
            raise ValueError(f"برچسب‌های ناشناخته در CSV: {unknown}")
        table["y"] = table["label"].map({c: i for i, c in enumerate(data_cfg.classes)}).astype(int)
        table["speaker"] = table["utterance_name"].str[:3]
        table["gender"] = table["utterance_name"].str[0].map({"F": "female", "M": "male"})
        table.insert(0, "uid", np.arange(len(table)))

        wavs = find_shemo_wavs(wav_roots)
        missing = sorted(set(table["utterance_name"]) - set(wavs))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} فایل صوتی ShEMO پیدا نشد (مثلاً {missing[:5]}). دیتاست ShEMO را به نوت‌بوک اضافه کنید "
                f"(Kaggle: Add Input → {SHEMO_KAGGLE_DATASET})."
            )
        cache_dir = Path(cache_dir) / "shemo_16k"
        cache_dir.mkdir(parents=True, exist_ok=True)
        table["wav16"] = [str(cache_dir / f"{n}.wav") for n in table["utterance_name"]]
        todo = [(wavs[n], Path(p)) for n, p in zip(table["utterance_name"], table["wav16"]) if not Path(p).exists()]
        if todo:
            import soundfile as sf
            from tqdm.auto import tqdm

            how = "librosa" if importlib.util.find_spec("librosa") else "torchaudio (librosa نصب نیست)"
            log(f"تبدیل {len(todo)} فایل به ۱۶ kHz با {how} ...")
            for src, dst in tqdm(todo, desc="resample 16 kHz"):
                sf.write(str(dst), _decode_like_datasets(src, data_cfg.sample_rate), data_cfg.sample_rate, subtype="DOUBLE")
        import soundfile as sf

        table["seconds"] = [sf.info(p).duration for p in table["wav16"]]
        log(f"ShEMO آماده است: {len(table)} جمله، {table['speaker'].nunique()} گوینده، {len(todo)} فایل جدید ۱۶ kHz")
        return cls(table, data_cfg.classes)

    # ---------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self.table)

    @property
    def y(self) -> np.ndarray:
        return self.table["y"].to_numpy()

    def texts(self, uids) -> list[str]:
        return self.table["text"].iloc[np.asarray(uids)].tolist()

    def labels(self, uids) -> np.ndarray:
        return self.table["y"].to_numpy()[np.asarray(uids)]

    def audio(self, uid: int) -> np.ndarray:
        """16 kHz waveform as float32 (the dtype wav2vec2's feature extractor converts to anyway)."""
        uid = int(uid)
        if uid not in self._audio_cache:
            import soundfile as sf

            array, _ = sf.read(self.table.at[uid, "wav16"], dtype="float64")
            self._audio_cache[uid] = array.astype(np.float32)
        return self._audio_cache[uid]

    def audios(self, uids) -> list[np.ndarray]:
        return [self.audio(u) for u in uids]

    def free_audio_cache(self) -> None:
        self._audio_cache.clear()

    # ---------------------------------------------------------------- folds
    def groups(self, data_cfg: DataConfig) -> dict[int, np.ndarray]:
        """fold number (1..n) -> uids of that fold, in a fixed order.

        "paper":   the author's session CSVs, in CSV order.
        "speaker": speaker-independent folds; no speaker appears in two folds.
        """
        if data_cfg.fold_scheme == "paper":
            return {s: self.table.index[self.table["session"] == s].to_numpy() for s in range(1, data_cfg.n_folds + 1)}
        if data_cfg.fold_scheme == "speaker":
            from sklearn.model_selection import StratifiedGroupKFold

            sgkf = StratifiedGroupKFold(n_splits=data_cfg.n_folds, shuffle=True, random_state=data_cfg.speaker_fold_seed)
            out = {}
            for k, (_, test) in enumerate(sgkf.split(self.table, self.table["y"], self.table["speaker"]), start=1):
                out[k] = np.sort(test)
            return out
        raise ValueError(f"fold_scheme ناشناخته: {data_cfg.fold_scheme}")

    # ---------------------------------------------------------------- reporting
    def summary(self, data_cfg: DataConfig | None = None) -> pd.DataFrame:
        """Utterances per class and fold, plus speakers per fold and overlap with the other folds."""
        data_cfg = data_cfg or DataConfig()
        groups = self.groups(data_cfg)
        rows = {}
        for k, uids in groups.items():
            part = self.table.iloc[uids]
            others = self.table.drop(index=uids)
            counts = part["label"].value_counts().reindex(self.classes, fill_value=0)
            row = counts.to_dict()
            row["total"] = len(part)
            row["speakers"] = part["speaker"].nunique()
            row["test speakers also in train (%)"] = round(100 * part["speaker"].isin(set(others["speaker"])).mean(), 1)
            row["hours"] = round(part["seconds"].sum() / 3600, 2)
            rows[f"fold {k}"] = row
        return pd.DataFrame(rows).T
