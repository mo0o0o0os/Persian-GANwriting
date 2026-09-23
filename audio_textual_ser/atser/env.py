"""Where are we running (Kaggle / Colab / local), which GPU to use, and where files go.

The author trained on a single Colab GPU. Kaggle gives two T4s, and the HF Trainer would silently
switch to DataParallel (a different effective batch, a different result). ``use_single_gpu`` keeps
the setup identical to the paper; call it before anything imports torch.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .utils import human_size, log


def detect_platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return "kaggle"
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    return "local"


def use_single_gpu(index: int = 0) -> None:
    """Expose only one GPU to torch. Must run before torch initialises CUDA."""
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
            log("هشدار: torch قبلاً import شده و بیش از یک GPU می‌بیند؛ Kernel را restart کنید و این تابع را اول اجرا کنید.")
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)


@dataclass
class Paths:
    """All locations the pipeline writes to.

    work:   persistent outputs (models, features, predictions, results). On Kaggle this is
            /kaggle/working, which is kept when you use "Save Version -> Save & Run All".
    cache:  large temporary files that are cheap to rebuild (16 kHz wav copies).
    inputs: read-only places searched for datasets and for outputs of earlier runs.
    """

    work: Path
    cache: Path
    inputs: list[Path]

    @property
    def store(self) -> Path:
        return self.work / "atser_store"

    @property
    def results(self) -> Path:
        return self.work / "atser_results"

    @property
    def exports(self) -> Path:
        return self.work / "atser_exports"


def default_paths(platform: str | None = None) -> Paths:
    platform = platform or detect_platform()
    if platform == "kaggle":
        return Paths(Path("/kaggle/working"), Path("/tmp/atser_cache"), [Path("/kaggle/input")])
    if platform == "colab":
        return Paths(Path("/content/atser_work"), Path("/content/atser_cache"), [Path("/content"), Path("/root/.cache/kagglehub")])
    here = Path.cwd()
    return Paths(here / "atser_work", here / "atser_cache", [here, Path.home() / ".cache" / "kagglehub"])


def get_secret(name: str) -> str | None:
    """Read a secret from Kaggle Secrets, Colab userdata, or the environment (in that order)."""
    platform = detect_platform()
    if platform == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret(name)
        except Exception:
            pass
    if platform == "colab":
        try:
            from google.colab import userdata

            return userdata.get(name)
        except Exception:
            pass
    return os.environ.get(name)


def describe_hardware() -> dict:
    info = {"platform": detect_platform(), "python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}"
            info["gpu_memory"] = human_size(torch.cuda.get_device_properties(0).total_memory)
    except ImportError:
        info["torch"] = "not installed"
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import datasets

        info["datasets"] = datasets.__version__
    except ImportError:
        pass
    total, used, free = shutil.disk_usage(default_paths(info["platform"]).work.anchor or "/")
    info["disk_free"] = human_size(free)
    try:
        import psutil

        info["ram"] = human_size(psutil.virtual_memory().total)
    except ImportError:
        pass
    return info
