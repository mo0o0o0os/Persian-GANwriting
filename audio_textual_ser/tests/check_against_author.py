"""Run the author's original notebook code and atser side by side on the same data; compare the numbers.

    python tests/make_test_assets.py <author_repo> <assets>
    python tests/check_against_author.py <author_repo> <assets> [rows_per_class_per_session=20]

Uses tiny random models on CPU (deterministic), one epoch, a subset of ShEMO rows with fake audio.
The author's code is taken verbatim from the notebook cells; only Colab/Hub specifics are patched
(same list as in the Kaggle runner). Checked for folds 1 and 2:

* audio fine-tuning: test logits of the author's Trainer vs. atser's AudioFinetuner
* text fine-tuning:  test logits of the author's Trainer vs. atser's TextFinetuner
* feature extraction: the author's extract_weighted_layer_embedding vs. atser.features (same model)
* fusion: the author's SVM_results predictions vs. atser's paper_sum_svm
* metrics: the author's per_class_metrics (imblearn G-Mean) vs. atser.evaluation.metrics

Fold 2 re-seeds the author's side before training. The author ran folds one after another in one
kernel without re-seeding, so the classifier-head initialisation of folds 2-5 depended on the
previous fold's training; atser seeds every fold so a resumed run gives the same result.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

REPO, ASSETS = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
MODELS, FAKE = ASSETS / "models", ASSETS / "shemo_fake"
WORK = Path(tempfile.mkdtemp(prefix="atser_check_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.use_deterministic_algorithms(True)

import atser.config as C  # noqa: E402
from atser.data import ShEMO  # noqa: E402
from atser.evaluation.metrics import per_class_metrics as our_per_class  # noqa: E402
from atser.models import load_fold_predictions  # noqa: E402
from atser.pipeline import ArtifactStore, Experiment  # noqa: E402

C.MODEL_OVERRIDES.update({k: str(MODELS / k) for k in ("facebook/wav2vec2-base", "bert-base-uncased")})
CLASSES = list(C.SHEMO_CLASSES)
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def cell_sources(nb, idx):
    cells = json.load(open(REPO / nb, encoding="utf-8"))["cells"]
    return ["".join(cells[i]["source"]) for i in idx]


PATCHES = [("evaluation_strategy", "eval_strategy"), ("tokenizer=feature_extractor", "processing_class=feature_extractor"),
           ("tokenizer=tokenizer,", "processing_class=tokenizer,"), ("push_to_hub=True", "push_to_hub=False"),
           ("hub_private_repo=True,", ""), ("report_to='wandb'", "report_to='none'"), ("report_to = 'wandb'", "report_to='none'"),
           ("trainer.push_to_hub()", "pass"), ("model.freeze_feature_extractor()", "model.freeze_feature_encoder()"),
           ("num_train_epochs=6", "num_train_epochs=1"), ("num_train_epochs=4", "num_train_epochs=1"),
           ('"facebook/wav2vec2-base"', repr(str(MODELS / "facebook/wav2vec2-base"))),
           ('"bert-base-uncased"', repr(str(MODELS / "bert-base-uncased")))]


def run(sources, ns, extra=()):
    for src in sources:
        for a, b in list(PATCHES) + list(extra):
            src = src.replace(a, b)
        exec(compile(src, "<author>", "exec"), ns)


# ------------------------------------------------------------------ the author's datasets (as on the HF hub)
from datasets import Audio, ClassLabel, Dataset, DatasetDict, Features, Value, load_dataset  # noqa: E402

os.chdir(WORK)
sessions = {i: pd.read_csv(REPO / f"ShEMO/Dataset/Sessions/session_{i}.csv").groupby("label").head(N) for i in range(1, 6)}
(WORK / "text").mkdir()
for i, df in sessions.items():
    df.rename(columns={"transcription": "text"})[["text", "label"]].to_csv(WORK / f"text/session{i}.csv", index=False)
shemo_text = load_dataset("csv", data_files={f"session{i}": str(WORK / f"text/session{i}.csv") for i in sessions},
                          features=Features({"text": Value("string"), "label": ClassLabel(num_classes=5, names=CLASSES)}))
wav = {p.stem: p for p in FAKE.rglob("*.wav")}
shemo_audio = DatasetDict({
    f"session{i}": Dataset.from_dict({"audio": [str(wav[n]) for n in df["utterance_name"]], "label": df["label"].tolist()})
    .cast_column("audio", Audio(sampling_rate=16_000)).cast_column("label", ClassLabel(num_classes=5, names=CLASSES))
    for i, df in sessions.items()})

# ------------------------------------------------------------------ atser side
data = ShEMO.prepare(REPO / "ShEMO/Dataset/Sessions", [FAKE], WORK / "cache", C.DataConfig(quick_rows_per_class=N))
cfg = C.paper_shemo().quick(N)
store = ArtifactStore(WORK / "store")
exp = Experiment(cfg, data, store)
exp.finetune("audio", folds=[1, 2])
exp.finetune("text", folds=[1, 2])
exp.extract("audio", folds=[1])
exp.extract("text", folds=[1])
exp.unimodal_systems()
exp.fuse(folds=[1])

# ------------------------------------------------------------------ audio fine-tuning
SPEECH = "ShEMO/Speech models/Speech_Emotion_Recognition_CV.ipynb"
ns = {"shemo_audio": shemo_audio}
run(cell_sources(SPEECH, [8, 9, 17, 21, 31, 33, 40, 41, 42]), ns)
for fold, cell in ((1, 45), (2, 56)):
    if fold > 1:
        run(cell_sources(SPEECH, [40]), ns)
    run(cell_sources(SPEECH, [cell]), ns)
    ours = load_fold_predictions(exp.finetune_dir("audio", fold) / "predictions.npz")["test"]["logits"]
    theirs = ns["preds_output_test"].predictions
    check(f"audio fine-tune fold {fold}: identical test logits", np.allclose(ours, theirs, atol=1e-5),
          f"max |diff| = {np.abs(ours - theirs).max():.2e}")

# ------------------------------------------------------------------ text fine-tuning
TEXT = "ShEMO/Text models/Text_Emotion_Recogniton_CV.ipynb"
ns = {"shemo_text": shemo_text}
run(cell_sources(TEXT, [6, 7, 16, 20, 29, 31, 40, 41, 42]), ns)
for fold, cell in ((1, 45), (2, 61)):
    if fold > 1:
        run(cell_sources(TEXT, [40]), ns)
    run(cell_sources(TEXT, [cell]), ns)
    ours = load_fold_predictions(exp.finetune_dir("text", fold) / "predictions.npz")["test"]["logits"]
    theirs = ns["preds_output_test"].predictions
    check(f"text fine-tune fold {fold}: identical test logits", np.allclose(ours, theirs, atol=1e-5),
          f"max |diff| = {np.abs(ours - theirs).max():.2e}")

# ------------------------------------------------------------------ feature extraction (same fine-tuned model)
feats = {}
for modality, nb, sub, var in (("audio", "Extracting_weighted_sum_mean_pooling_all_hidden_states_Audio.ipynb", "Wav2vec2-base", "shemo_audio"),
                               ("text", "Extracting_weighted_sum_mean_pooling_all_hidden_states_Text.ipynb", "Bert-base", "shemo_text")):
    path = WORK / "feat" / sub
    path.mkdir(parents=True)
    ns = {var: shemo_audio if modality == "audio" else shemo_text, "PATH": str(path)}
    model_dir = str(exp.finetune_dir(modality, 1) / "model")
    src = cell_sources(f"ShEMO/Fusion/Early_fusion/{nb}", [5, 14, 15, 18, 20])
    src[-1] = src[-1].replace('model_checkpoint = f"Zahra99/', f'model_checkpoint = {model_dir!r} #')
    run(src, ns)
    ours = np.load(store.feature_path(modality, cfg.audio if modality == "audio" else cfg.text, cfg.features, cfg.data, 1))
    for part, var_name in (("train", "train_output"), ("test", "test_output")):
        theirs = ns[var_name].numpy()
        check(f"{modality} features fold 1 ({part}): identical", np.allclose(ours[f"{part}_avg"], theirs, atol=1e-5),
              f"max |diff| = {np.abs(ours[f'{part}_avg'] - theirs).max():.2e}")

# ------------------------------------------------------------------ SVM fusion and metrics
FUSION = "ShEMO/Fusion/Early_fusion/Early_Fusion_Summation.ipynb"
(WORK / "feat" / "best_multimodal").mkdir()
ns = {"PATH": str(WORK / "feat")}
run(cell_sources(FUSION, [3, 7, 9, 10, 13, 29]), ns, extra=[("plt.show()", "plt.close('all')")])
ns["SVM_results"](1)
theirs = np.load(WORK / "feat/best_multimodal/y_predicted_out1.npy")
ours = np.load(store.prediction_path(cfg.name, "paper_sum_svm", 1))
check("fusion (sum + SVM) fold 1: identical predictions", np.array_equal(ours["y_pred"], theirs),
      f"{(ours['y_pred'] != theirs).sum()} differences")
a = pd.DataFrame(ns["per_class_metrics"](ours["y_true"], theirs, class_names=CLASSES)).T.astype(float)
b = our_per_class(ours["y_true"], theirs, CLASSES)
check("per-class metrics (Precision, Sensitivity, Specificity, F1, G-Mean, MCC): identical",
      np.allclose(a[b.columns].values, b.values, atol=1e-9))

print("\n" + ("ALL CHECKS PASSED" if all(ok for _, ok in results) else "SOME CHECKS FAILED"), f"({WORK})")
sys.exit(0 if all(ok for _, ok in results) else 1)
