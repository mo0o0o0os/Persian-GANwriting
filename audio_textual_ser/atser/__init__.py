"""atser — Audio-Textual Speech Emotion Recognition.

A Python-package rewrite of the ShEMO pipeline from
"Audio-Textual Emotion Recognition Using Pre-trained Models" (Dehghani et al., 2025),
https://github.com/ZahraDehghani99/Audio-Textual-Emotion-Recognition-using-Pre-trained-models

Layout
------
atser.config      experiment settings as dataclasses, plus the paper's exact preset
atser.env         platform detection (Kaggle / Colab / local), single-GPU setup, secrets
atser.data        ShEMO loading, 16 kHz resampling, fold definitions, train/valid/test splits
atser.models      fine-tuning of the audio (wav2vec2) and text (BERT) classifiers
atser.features    embedding extraction (the paper's "average of mean-pooled hidden states")
atser.fusion      pluggable fusion methods (the paper's sum+SVM and alternatives)
atser.evaluation  the paper's metrics, statistical tests, summary tables and plots
atser.pipeline    artifact store (caching / resume) and the experiment runner
atser.persist     saving models and results somewhere that survives the session

Importing ``atser`` does not import torch, so ``atser.env.use_single_gpu()`` can still
restrict the visible GPUs before CUDA is initialised.
"""

__version__ = "0.1.0"
