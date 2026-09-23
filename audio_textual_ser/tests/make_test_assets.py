"""Create offline test assets: tiny random models with the real names' formats, and fake ShEMO wavs.

    python tests/make_test_assets.py <author_repo_dir> <out_dir>

Writes <out_dir>/models/{facebook/wav2vec2-base, bert-base-uncased, HooshvareLab/bert-fa-base-uncased}
and <out_dir>/shemo_fake/{female,male}/<name>.wav (44.1 kHz noise, one file per CSV row).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from transformers import (BertConfig, BertModel, BertTokenizerFast, Wav2Vec2Config, Wav2Vec2FeatureExtractor,
                          Wav2Vec2Model)

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
models = out / "models"

w2v = models / "facebook" / "wav2vec2-base"
Wav2Vec2Model(Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=37,
                             conv_dim=(32, 32, 32), conv_stride=(5, 4, 4), conv_kernel=(8, 8, 8),
                             num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                             feat_extract_norm="group", do_stable_layer_norm=False)).save_pretrained(w2v)
Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                         return_attention_mask=False).save_pretrained(w2v)

for name, vocab in (("bert-base-uncased", list("abcdefghijklmnopqrstuvwxyz")),
                    ("HooshvareLab/bert-fa-base-uncased", list("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"))):
    d = models / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "?", "؟"] + vocab), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(d / "vocab.txt")).save_pretrained(d)
    BertModel(BertConfig(vocab_size=len(vocab) + 7, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=37)).save_pretrained(d)

names = pd.concat(pd.read_csv(repo / f"ShEMO/Dataset/Sessions/session_{i}.csv") for i in range(1, 6))["utterance_name"]
rng = np.random.default_rng(0)
for n in names:
    folder = out / "shemo_fake" / ("female" if n.startswith("F") else "male")
    folder.mkdir(parents=True, exist_ok=True)
    audio = (rng.standard_normal(int(44100 * rng.uniform(0.4, 1.2))) * 0.1).astype(np.float32)
    sf.write(folder / f"{n}.wav", audio, 44100, subtype="PCM_16")
print("models:", models, "| fake wavs:", len(names))
