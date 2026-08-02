"""Immutable configuration and runtime qualification for the Balalaika recipe."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import re
import sys
from typing import Mapping


DEFAULT_DATASET_ROOT = Path("/workspace/balalaika_proprietary_v2")
DEFAULT_REPOSITORY_ROOT = Path("/workspace/CosyVoice")
DEFAULT_RUN_ROOT = Path("/workspace/cosyvoice3-balalaika-lora")
DEFAULT_BASE_MODEL_DIR = DEFAULT_REPOSITORY_ROOT / "pretrained_models/Fun-CosyVoice3-0.5B-2512"
DEFAULT_VISIBLE_DEVICES = (0, 1, 2, 3, 4, 5, 6, 7)
DEFAULT_SEED = 1986


@dataclass(frozen=True)
class RunPaths:
    """Filesystem and reproducibility defaults for one Balalaika run."""

    dataset_root: Path
    repository_root: Path
    run_root: Path
    base_model_dir: Path
    visible_devices: tuple[int, ...]
    seed: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RunPaths":
        values = os.environ if env is None else env
        base_model_dir = Path(values.get("BALALAIKA_BASE_MODEL_DIR", str(DEFAULT_BASE_MODEL_DIR)))
        if "_RL" in base_model_dir.name.upper():
            raise ValueError("BALALAIKA_BASE_MODEL_DIR must select the base/non-RL checkpoint")
        return cls(
            dataset_root=Path(values.get("BALALAIKA_DATASET_ROOT", str(DEFAULT_DATASET_ROOT))),
            repository_root=Path(values.get("BALALAIKA_REPOSITORY_ROOT", str(DEFAULT_REPOSITORY_ROOT))),
            run_root=Path(values.get("BALALAIKA_RUN_ROOT", str(DEFAULT_RUN_ROOT))),
            base_model_dir=base_model_dir,
            visible_devices=_parse_visible_devices(values.get("BALALAIKA_VISIBLE_DEVICES")),
            seed=_parse_seed(values.get("BALALAIKA_SEED")),
        )

    @property
    def stages_dir(self) -> Path:
        return self.run_root / "stages"

    def stage(self, name: str) -> Path:
        return self.stages_dir / f"{name}.json"


def _parse_visible_devices(value: str | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_VISIBLE_DEVICES
    try:
        devices = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("BALALAIKA_VISIBLE_DEVICES must be comma-separated integers") from exc
    if not devices or any(device < 0 for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("BALALAIKA_VISIBLE_DEVICES must contain unique non-negative integers")
    return devices


def _parse_seed(value: str | None) -> int:
    if value is None:
        return DEFAULT_SEED
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("BALALAIKA_SEED must be an integer") from exc


@dataclass(frozen=True)
class PhaseSpec:
    """Literal, immutable training boundary for one agreement phase."""

    number: int
    agreement_min: float | None
    agreement_max: float | None
    epochs: int
    learning_rate: float
    predicate: str

    @classmethod
    def for_phase(cls, number: int) -> "PhaseSpec":
        specs = {
            1: cls(1, None, 0.95, 2, 1e-4, "asr_agreement_mean < 0.95"),
            2: cls(2, 0.95, None, 3, 5e-5, "asr_agreement_mean >= 0.95"),
        }
        if number not in specs:
            raise ValueError(f"phase must be 1 or 2, got {number}")
        return specs[number]


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"missing required dependency: {name}") from exc


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"(\d+(?:\.\d+)*)", value)
    if match is None:
        raise RuntimeError(f"cannot parse version: {value}")
    return tuple(int(part) for part in match.group(1).split("."))


def collect_environment() -> dict[str, object]:
    """Return a qualified Blackwell runtime manifest or raise before GPU work."""

    torch = import_module("torch")
    onnxruntime = import_module("onnxruntime")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_count = torch.cuda.device_count()
    if device_count != 8:
        raise RuntimeError(f"Balalaika recipe requires exactly eight GPUs, found {device_count}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable")

    gpus: list[dict[str, object]] = []
    for index in range(device_count):
        name = torch.cuda.get_device_name(index)
        capability = tuple(torch.cuda.get_device_capability(index))
        if "RTX 5090" not in name:
            raise RuntimeError(f"GPU {index} is not an RTX 5090: {name}")
        if capability < (12, 0):
            raise RuntimeError(f"GPU {index} capability is below (12, 0): {capability}")
        gpus.append({"name": name, "capability": list(capability)})

    providers = list(onnxruntime.get_available_providers())
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
    onnxruntime_version = str(onnxruntime.__version__)
    if _version_tuple(onnxruntime_version) >= (1, 27):
        raise RuntimeError("onnxruntime-gpu must be below 1.27 for this CUDA-12 recipe")

    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpus": gpus,
        "accelerate": _distribution_version("accelerate"),
        "peft": _distribution_version("peft"),
        "transformers": _distribution_version("transformers"),
        "pyarrow": _distribution_version("pyarrow"),
        "wandb": _distribution_version("wandb"),
        "onnxruntime": onnxruntime_version,
        "onnx_asr": _distribution_version("onnx-asr"),
        "onnxruntime_providers": providers,
    }
