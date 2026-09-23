"""Keeping models and results after the session ends.

Options, from simplest to most flexible:

1. Kaggle "Save Version" -> "Save & Run All (Commit)": the whole /kaggle/working folder (our store)
   becomes the output of that notebook version. Next time: Add Input -> Your Work -> this notebook,
   and ``ArtifactStore.restore_from`` (the notebook does it) copies it back. Nothing to configure. (A "Quick Save" does NOT
   keep /kaggle/working.)
2. ``save_to_kaggle_dataset``: uploads a folder as a private Kaggle Dataset (a new version on every
   call). Needs the secrets KAGGLE_USERNAME and KAGGLE_KEY (kaggle.com -> Settings -> API -> Create
   New Token; then in the notebook: Add-ons -> Secrets).
3. ``save_to_drive``: Colab only, copies into Google Drive.
4. ``push_to_hub``: a model folder to a (private) Hugging Face repo; needs the secret HF_TOKEN.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .env import detect_platform, get_secret
from .pipeline.store import ArtifactStore
from .utils import dir_size, human_size, log

PARTS = ("experiments", "predictions", "features", "finetune")


def bundle(store: ArtifactStore, dest: str | Path, parts=PARTS, include_models: bool = True) -> Path:
    """Copy parts of the store into ``dest`` (a staging folder for uploading)."""
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    shutil.copy2(store.root / "STORE.json", dest / "STORE.json")
    ignore = None if include_models else shutil.ignore_patterns("model", "checkpoints")
    for part in parts:
        if (store.root / part).exists():
            shutil.copytree(store.root / part, dest / part, ignore=ignore)
    log(f"بستهٔ خروجی: {dest} ({human_size(dir_size(dest))})")
    return dest


def zip_folder(folder: str | Path, zip_path: str | Path) -> Path:
    """One zip file you can download from the notebook's Output panel."""
    zip_path = Path(zip_path)
    archive = shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(folder))
    log(f"فایل zip: {archive} ({human_size(Path(archive).stat().st_size)})")
    return Path(archive)


def _kaggle_env() -> dict:
    if shutil.which("kaggle") is None:
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    env = dict(os.environ)
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        value = get_secret(key)
        if value:
            env[key] = value
    if not env.get("KAGGLE_USERNAME") or not env.get("KAGGLE_KEY"):
        raise RuntimeError("برای ساخت Dataset روی Kaggle، secret های KAGGLE_USERNAME و KAGGLE_KEY لازم است "
                           "(kaggle.com → Settings → API → Create New Token؛ سپس Add-ons → Secrets).")
    return env


def save_to_kaggle_dataset(folder: str | Path, slug: str, title: str | None = None, public: bool = False,
                           notes: str = "atser update") -> str:
    """Upload ``folder`` as a Kaggle Dataset <username>/<slug>; creates it or adds a new version."""
    env = _kaggle_env()
    folder = Path(folder)
    dataset_id = f"{env['KAGGLE_USERNAME']}/{slug}"
    meta = {"title": title or slug.replace("-", " ").title(), "id": dataset_id, "licenses": [{"name": "CC0-1.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    exists = subprocess.run(["kaggle", "datasets", "status", dataset_id], env=env, capture_output=True, text=True).returncode == 0
    if exists:
        cmd = ["kaggle", "datasets", "version", "-p", str(folder), "-m", notes, "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(folder), "--dir-mode", "zip"] + (["--public"] if public else [])
    log(("نسخهٔ جدید" if exists else "ساخت") + f" Dataset {dataset_id} ...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(result.stdout[-2000:], result.stderr[-2000:])
    if result.returncode != 0:
        raise RuntimeError(f"آپلود به Kaggle ناموفق بود (کد {result.returncode})")
    log(f"ذخیره شد: https://www.kaggle.com/datasets/{dataset_id} — دفعهٔ بعد از Add Input → Datasets → Your Datasets اضافه‌اش کنید.")
    return dataset_id


def save_to_drive(folder: str | Path, drive_subdir: str = "atser") -> Path:
    if detect_platform() != "colab":
        raise RuntimeError("Google Drive فقط در Colab مستقیم در دسترس است.")
    from google.colab import drive

    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    dest = Path("/content/drive/MyDrive") / drive_subdir / Path(folder).name
    shutil.copytree(folder, dest, dirs_exist_ok=True)
    log(f"در Google Drive ذخیره شد: {dest}")
    return dest


def push_to_hub(model_dir: str | Path, repo_id: str, private: bool = True) -> str:
    from huggingface_hub import HfApi

    token = get_secret("HF_TOKEN")
    if not token:
        raise RuntimeError("secret با نام HF_TOKEN لازم است.")
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id)
    log(f"مدل در https://huggingface.co/{repo_id} آپلود شد")
    return repo_id
