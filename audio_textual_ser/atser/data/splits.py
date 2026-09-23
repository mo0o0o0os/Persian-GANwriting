"""Train / valid / test membership of every fold, reproduced with the same `datasets` calls as the author.

Author's code (speech cell 41/45, text cell 41/45)::

    session_i = shemo["session_i"].train_test_split(test_size=0.1, seed=seed[, stratify_by_column="label"])
    train = concatenate_datasets([session_j["train"] for j != k]).shuffle(seed=seed)
    valid = concatenate_datasets([session_j["test"]  for j != k])
    test  = shemo["session_k"]

and for feature extraction (Extracting_*.ipynb, cell 20)::

    train = concatenate_datasets([shemo["session_j"] for j != k])     # not shuffled, includes valid rows
    test  = shemo["session_k"]

The permutations `datasets` draws depend only on the number of rows, the seed and (when stratified)
the labels, so running the same calls on a table of uids yields exactly the author's order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import log


@dataclass
class FoldSplit:
    fold: int
    train: np.ndarray  # uids used to fine-tune (shuffled, as the Trainer sees them)
    valid: np.ndarray  # uids used for per-epoch evaluation / best-epoch selection
    test: np.ndarray  # uids of the left-out fold
    feature_train: np.ndarray  # uids whose embeddings train the fusion classifier (all other folds)

    @property
    def feature_test(self) -> np.ndarray:
        return self.test


def _column(ds, name: str) -> np.ndarray:
    # works with datasets 2.x, 3.x and 4.x (4.x returns a lazy Column object for ds[name])
    return ds.to_pandas()[name].to_numpy()


def make_fold_splits(groups: dict[int, np.ndarray], y: np.ndarray, n_classes: int, seed: int,
                     valid_fraction: float = 0.1, stratify: bool = False) -> dict[int, FoldSplit]:
    """Reproduce the author's per-fold train/valid/test uids.

    groups: fold -> uids (in the order the author's per-session dataset has them)
    y:      label id of every uid
    """
    from datasets import ClassLabel, Dataset, Features, Value, concatenate_datasets

    features = Features({"uid": Value("int64"), "label": ClassLabel(num_classes=n_classes)})
    per_group = {}
    for g, uids in groups.items():
        ds = Dataset.from_dict({"uid": [int(u) for u in uids], "label": [int(v) for v in y[uids]]}, features=features)
        try:
            per_group[g] = ds.train_test_split(test_size=valid_fraction, seed=seed,
                                               stratify_by_column="label" if stratify else None)
        except ValueError as err:  # tiny quick-test subsets cannot always be stratified
            log(f"fold {g}: تقسیم طبقه‌بندی‌شده ممکن نشد ({err}); تقسیم ساده انجام شد")
            per_group[g] = ds.train_test_split(test_size=valid_fraction, seed=seed)

    order = sorted(groups)
    splits = {}
    for k in order:
        others = [g for g in order if g != k]
        train = concatenate_datasets([per_group[g]["train"] for g in others]).shuffle(seed=seed)
        valid = concatenate_datasets([per_group[g]["test"] for g in others])
        splits[k] = FoldSplit(
            fold=k,
            train=_column(train, "uid"),
            valid=_column(valid, "uid"),
            test=np.asarray(groups[k]),
            feature_train=np.concatenate([np.asarray(groups[g]) for g in others]),
        )
    return splits
