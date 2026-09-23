"""Utterance embeddings for the fusion stage.

The paper's representation ("average of mean pooling of all hidden states"), from
ShEMO/Fusion/Early_fusion/Extracting_weighted_sum_mean_pooling_all_hidden_states_{Audio,Text}.ipynb:

audio (cells 14, 15, 20)
    feature_extractor(batch of 25 clips, padding=True)      # no truncation; pad to the longest clip
    hidden_states = AutoModel(fine-tuned)(input_values).hidden_states   # 13 layers for wav2vec2-base
    per layer: torch.mean(h, 1)                             # over ALL frames, padding included
    embedding = mean over layers                            # 768-d

text (cells 14, 15, 20)
    tokenizer(all texts of the split, padding=True, truncation=True)
    per layer: masked mean over real tokens (sum(h * mask) / clamp(sum(mask), 1e-9))
    embedding = mean over layers

Besides that average we also keep the per-layer vectors (``layers``, float16), so later methods can
learn layer weights without re-running the encoders.
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils import log


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _text_mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # exactly the author's function
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * mask, dim=1)
    count = torch.clamp(mask.sum(1), min=1e-9)
    return summed / count


@torch.no_grad()
def extract_audio(model_path: str, arrays: list[np.ndarray], batch_size: int = 25, pad_aware: bool = False,
                  keep_layers: bool = True, desc: str = "") -> tuple[np.ndarray, np.ndarray | None]:
    """Returns (embedding [N, H] float32, per-layer [N, L, H] float16 or None)."""
    from tqdm.auto import tqdm
    from transformers import AutoFeatureExtractor, AutoModel

    device = _device()
    fe = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True).to(device)
    model.eval()
    avgs, layers = [], []
    for i in tqdm(range(0, len(arrays), batch_size), desc=desc or "audio features", leave=False):
        batch = arrays[i:i + batch_size]
        enc = fe(batch, sampling_rate=fe.sampling_rate, padding=True)
        x = torch.from_numpy(np.stack(enc["input_values"])).to(device)
        hidden = torch.stack(model(input_values=x).hidden_states)  # [L, B, T, H]
        if pad_aware:
            lengths = torch.tensor([len(a) for a in batch], device=device)
            frames = model._get_feat_extract_output_lengths(lengths).clamp(max=hidden.shape[2])
            mask = (torch.arange(hidden.shape[2], device=device)[None, :] < frames[:, None]).float()  # [B, T]
            per_layer = torch.stack([(h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True) for h in hidden])
        else:
            per_layer = torch.stack([torch.mean(h, 1) for h in hidden])  # [L, B, H], the author's pooling
        avgs.append(per_layer.mean(dim=0).cpu().numpy().astype(np.float32))
        if keep_layers:
            layers.append(per_layer.permute(1, 0, 2).cpu().numpy().astype(np.float16))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(avgs), (np.concatenate(layers) if keep_layers else None)


@torch.no_grad()
def extract_text(model_path: str, texts: list[str], batch_size: int = 25, keep_layers: bool = True,
                 max_length: int | None = None, desc: str = "") -> tuple[np.ndarray, np.ndarray | None]:
    """Returns (embedding [N, H] float32, per-layer [N, L, H] float16 or None)."""
    from tqdm.auto import tqdm
    from transformers import AutoModel, AutoTokenizer

    device = _device()
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True).to(device)
    model.eval()
    kwargs = {"padding": True, "truncation": True}
    if max_length:
        kwargs["max_length"] = max_length
    enc = tok(texts, **kwargs)  # the author tokenises the whole split at once (batch_size=None)
    ids = torch.tensor(enc["input_ids"])
    att = torch.tensor(enc["attention_mask"])
    avgs, layers = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc or "text features", leave=False):
        # set_format(columns=["input_ids", "attention_mask", "label"]) in the author's code drops token_type_ids
        b_ids, b_att = ids[i:i + batch_size].to(device), att[i:i + batch_size].to(device)
        hidden = model(input_ids=b_ids, attention_mask=b_att).hidden_states
        per_layer = torch.stack([_text_mean_pooling(h, b_att) for h in torch.stack(hidden)])
        avgs.append(per_layer.mean(dim=0).cpu().numpy().astype(np.float32))
        if keep_layers:
            layers.append(per_layer.permute(1, 0, 2).cpu().numpy().astype(np.float16))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(avgs), (np.concatenate(layers) if keep_layers else None)


def save_features(path, **parts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: v for k, v in parts.items() if v is not None})
    log(f"ویژگی‌ها ذخیره شد: {path.name} " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in parts.items() if v is not None))


def load_features(path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}
