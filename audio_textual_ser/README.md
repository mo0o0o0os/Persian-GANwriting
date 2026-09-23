# atser — Audio-Textual Speech Emotion Recognition (ShEMO)

A Python-package rewrite of the ShEMO pipeline from *Audio-Textual Emotion Recognition Using Pre-trained
Models* (Dehghani et al., 2025,
[original code](https://github.com/ZahraDehghani99/Audio-Textual-Emotion-Recognition-using-Pre-trained-models)).
The logic lives in `atser/`. `main.ipynb` is the only notebook: it imports the package and runs it.

بازنویسی پروژه‌ای کد مقاله برای ShEMO. همهٔ منطق در پکیج `atser/` است و `main.ipynb` تنها نوت‌بوک پروژه است.

## Layout

```
audio_textual_ser/
├── main.ipynb                  the only notebook (Kaggle / Colab / local)
├── atser/
│   ├── config.py               settings dataclasses; paper_shemo() = the author's exact settings
│   ├── env.py                  platform detection, single GPU, secrets, paths
│   ├── data/shemo.py           session CSVs, wav discovery, 16 kHz resampling (like datasets.Audio)
│   ├── data/splits.py          train/valid/test per fold with the author's exact `datasets` calls
│   ├── models/finetune.py      audio (wav2vec2) and text (BERT) fine-tuning, per fold, resumable
│   ├── features/extract.py     "average of mean-pooled hidden states" + per-layer vectors
│   ├── fusion/                 registry of fusion methods (paper's sum+SVM and 12 others)
│   ├── evaluation/             the paper's metrics, paired tests, tables, plots
│   ├── pipeline/               artifact store (caching / resume) and the experiment runner
│   └── persist.py              Kaggle Dataset / Google Drive / Hugging Face export
└── tests/
    ├── make_test_assets.py     tiny random models + fake ShEMO audio (offline tests)
    ├── check_against_author.py runs the author's notebook cells and atser side by side
    └── run_notebook_smoke.py   executes main.ipynb end to end with tiny models
```

## Equivalence with the author's code

`tests/check_against_author.py` executes the author's notebook cells (patched only for Colab/Hub
specifics) and this package on the same data, on CPU with tiny models (deterministic). Every check is
identical, with a maximum absolute difference of 0.0:

| stage | result |
| --- | --- |
| audio fine-tuning, folds 1 and 2 | identical test logits |
| text fine-tuning, folds 1 and 2 | identical test logits |
| audio / text feature extraction | identical embeddings |
| sum + SVM fusion | identical predictions |
| per-class metrics | identical |

One deliberate difference: the author seeded once per notebook and trained the 5 folds in one kernel,
so the classifier-head initialisation of folds 2–5 depended on the previous fold. `atser` seeds every
fold, so an interrupted run resumes to the same result.

## Running on Kaggle

1. New notebook → File → Import Notebook → `main.ipynb` (or open it from GitHub).
2. Settings: GPU T4 x2 (one is used, as in the paper), Internet on.
3. Add Input: `mansourehk/shemo-persian-speech-emotion-detection-database`.
4. Edit section 0, Run All. To keep the outputs: Save Version → Save & Run All. Next time, attach that
   output (Add Input → Your Work) and completed stages are skipped.

## Extending

| to change | how |
| --- | --- |
| text model | add its Hugging Face name to `TEXT_MODELS` (audio is reused) |
| audio model | `paper_shemo().with_audio_model("microsoft/wavlm-base-plus")` |
| hyper-parameters | `cfg.with_text_model(name, epochs=2, select_best=True)`; the experiment name changes automatically |
| speaker-independent folds | `FOLD_SCHEME = "speaker"` |
| leakage-free frozen features | `FEATURE_SOURCE = "pretrained"` |
| new fusion method | subclass `atser.fusion.FusionMethod`, decorate with `@register("name")` |
| new neural fusion head | subclass `atser.fusion.TorchFusion`, implement `inputs` and `build` |
| different fine-tuning architecture | subclass `atser.models.Finetuner` (`load_processor`, `build_dataset`, `load_model`) |

## Tests

```bash
python tests/make_test_assets.py <author_repo_clone> <assets_dir>
python tests/check_against_author.py <author_repo_clone> <assets_dir>
python tests/run_notebook_smoke.py <assets_dir> <work_dir>
```
