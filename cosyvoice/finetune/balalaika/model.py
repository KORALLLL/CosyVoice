"""CosyVoice3 base-model loading and broad adapter lifecycle helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
import json
import hashlib
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from typing import Literal, Mapping, Sequence, TypeAlias
import wave

import torch
from torch import nn

from cosyvoice.finetune.balalaika.artifacts import atomic_write_json, sha256_file
from cosyvoice.llm.llm import CosyVoice3LM


ADAPTER_NAME = "default"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_MANIFEST_NAME = "adapter_manifest.json"
APPROVED_BASE_MODEL_NAME = "Fun-CosyVoice3-0.5B-2512"
_INTERNAL_QWEN_LM_HEAD_PARTS = ("llm", "model", "lm_head")


@dataclass(frozen=True)
class LoraSettings:
    """The fixed broad-LoRA settings approved for the Balalaika recipe."""

    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    bias: Literal["none"] = "none"

    def __post_init__(self) -> None:
        if (self.r, self.alpha, self.dropout, self.bias) != (64, 128, 0.05, "none"):
            raise ValueError("Balalaika LoRA settings must be r=64, alpha=128, dropout=0.05, bias='none'")


@dataclass(frozen=True)
class TrainableAudit:
    """Stable inventory of adapter targets and trainable parameter names."""

    target_modules: tuple[str, ...]
    trainable_parameters: tuple[str, ...]
    unexpected_dense_parameters: tuple[str, ...]
    trainable_parameter_count: int
    total_parameter_count: int


@dataclass(frozen=True)
class AdapterManifest:
    """Checksum-bound metadata for one adapter-only checkpoint."""

    path: Path
    weights_path: Path
    weights_sha256: str
    base_checkpoint_sha256: str
    target_modules: tuple[str, ...]
    settings: LoraSettings
    code_revision: str


@dataclass(frozen=True)
class MergeReport:
    """Provenance and checksum of a standalone merged ``llm.pt``."""

    output: Path
    output_sha256: str
    base_checkpoint_sha256: str
    adapter_weights_sha256: str
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class ExportRequest:
    """Inputs whose successful phase-2 lineage may become a final model."""

    base_model_dir: Path
    phase2_checkpoint: Path
    validation_summary: Path
    output_dir: Path
    validation_request: object | None = None
    wandb_logger: object | None = None
    test_mode: bool = False
    expected_training_identity: Mapping[str, object] | None = None
    verification_voices: Sequence[Mapping[str, object]] | None = None
    recognizer: object | None = None
    pipeline_factory: object | None = None

    def __post_init__(self) -> None:
        for name in ("base_model_dir", "phase2_checkpoint", "validation_summary", "output_dir"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a pathlib.Path")
        if self.test_mode is not True and (self.validation_request is None or self.wandb_logger is None):
            raise ValueError("production export requires Task 10 evidence request and W&B logger")
        if self.test_mode is not True and not isinstance(self.expected_training_identity, Mapping):
            raise ValueError("production export requires the complete immutable training identity")
        if self.test_mode and (self.verification_voices is None or self.recognizer is None):
            raise ValueError("test export requires explicit verification voices and recognizer")
        if not self.test_mode and (self.pipeline_factory is not None or self.recognizer is not None):
            raise ValueError("production export constructs its own pipeline and recognizer")


@dataclass(frozen=True)
class FinalModelManifest:
    """Complete, checksum-bound standalone LLM export inventory."""

    path: Path
    llm_path: Path
    adapter_dir: Path
    llm_sha256: str
    base_checkpoint_sha256: str
    adapter_weights_sha256: str
    adapter_manifest_sha256: str
    target_modules: tuple[str, ...]
    phase2_checkpoint_sha256: str
    validation_summary_sha256: str
    mode: Literal["test", "production"]
    production_ready: bool = False
    base_assets_sha256: str = ""


@dataclass(frozen=True)
class VerifyRequest:
    """Dependencies for normal-path standalone CosyVoice3 smoke verification."""

    base_model_dir: Path
    final_manifest: FinalModelManifest
    recognizer: object
    voices: Sequence[Mapping[str, object]]
    output_dir: Path
    pipeline_factory: object | None = None
    test_mode: bool = False
    artifact_root: Path | None = None
    expected_local_rank: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.base_model_dir, Path) or not isinstance(self.output_dir, Path):
            raise TypeError("verification paths must be pathlib.Path values")
        if not isinstance(self.final_manifest, FinalModelManifest):
            raise TypeError("final_manifest must be FinalModelManifest")
        if not isinstance(self.voices, Sequence) or isinstance(self.voices, (str, bytes)):
            raise TypeError("voices must be a sequence of reserved voice mappings")
        if len(self.voices) != 20:
            raise ValueError("strict verification requires exactly twenty reserved voices")
        if self.test_mode is not True and self.pipeline_factory is not None:
            raise ValueError("production strict verification constructs normal CosyVoice3 directly")
        if self.artifact_root is not None and not isinstance(self.artifact_root, Path):
            raise TypeError("artifact_root must be a pathlib.Path")
        if isinstance(self.expected_local_rank, bool) or not isinstance(self.expected_local_rank, int) or self.expected_local_rank < 0:
            raise ValueError("expected_local_rank must be non-negative")


@dataclass(frozen=True)
class VerificationReport:
    """Durable evidence that a standalone LLM works without an adapter runtime."""

    path: Path
    strict_load: bool
    smoke_utterances: int
    audio_paths: tuple[Path, ...]
    audio_checksums: Mapping[str, str]
    asr_results: tuple[str, ...]
    library_identities: Mapping[str, object]


AdapterModel: TypeAlias = CosyVoice3LM


def _is_internal_qwen_lm_head_path(name: str) -> bool:
    """Return whether a module/tensor path is inside Qwen's internal LM head."""

    parts = tuple(name.split("."))
    width = len(_INTERNAL_QWEN_LM_HEAD_PARTS)
    return any(
        parts[index:index + width] == _INTERNAL_QWEN_LM_HEAD_PARTS
        for index in range(len(parts) - width + 1)
    )


def load_base_llm(model_dir: Path) -> CosyVoice3LM:
    """Load only the frozen base/non-RL CosyVoice3 language model."""

    model_dir = Path(model_dir)
    if any("_RL" in component.upper() for component in model_dir.parts):
        raise ValueError("model_dir must select the base/non-RL CosyVoice3 checkpoint")
    if model_dir.name != APPROVED_BASE_MODEL_NAME:
        raise ValueError(f"model_dir must be the approved {APPROVED_BASE_MODEL_NAME} model root")
    config_path = model_dir / "cosyvoice3.yaml"
    checkpoint_path = model_dir / "llm.pt"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing CosyVoice3 config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing CosyVoice3 LLM checkpoint: {checkpoint_path}")

    from hyperpyyaml import load_hyperpyyaml

    overrides = {
        "qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN"),
        "flow": None,
        "hift": None,
    }
    with config_path.open(encoding="utf-8") as stream:
        configs = load_hyperpyyaml(stream, overrides=overrides)
    model = configs.get("llm")
    if not isinstance(model, CosyVoice3LM):
        raise TypeError("cosyvoice3.yaml did not construct a CosyVoice3LM")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model._balalaika_base_model_dir = model_dir
    model._balalaika_base_checkpoint_sha256 = sha256_file(checkpoint_path)
    return model


def _discover_active_modules(model: CosyVoice3LM) -> tuple[str, ...]:
    from peft.tuners.tuners_utils import BaseTunerLayer

    if not isinstance(model, CosyVoice3LM):
        raise TypeError("LoRA injection requires a CosyVoice3LM, not the full CosyVoice pipeline")

    named_modules = tuple(model.named_modules())
    wrapper_names = tuple(name for name, module in named_modules if name and isinstance(module, BaseTunerLayer))
    targets = []
    for name, module in named_modules:
        if not name or any(name.startswith(f"{wrapper}.") for wrapper in wrapper_names):
            continue
        base_module = module.get_base_layer() if isinstance(module, BaseTunerLayer) else module
        if isinstance(base_module, (nn.Linear, nn.Embedding)) and not _is_internal_qwen_lm_head_path(name):
            targets.append(name)
    targets = tuple(targets)
    modules = dict(model.named_modules())
    embedding = model.llm.get_input_embeddings()
    embedding_names = tuple(name for name, module in modules.items() if module is embedding)
    required = {"speech_embedding", "llm_decoder"}
    missing = sorted(required.difference(targets))
    if not embedding_names or embedding_names[0] not in targets:
        missing.append("Qwen input embeddings")
    if missing:
        raise ValueError(f"missing required active LoRA targets: {', '.join(missing)}")
    if any(_is_internal_qwen_lm_head_path(name) for name in targets):
        raise RuntimeError("internal Qwen lm_head must not be an adapter target")
    return targets


def inject_lora(model: CosyVoice3LM, settings: LoraSettings) -> AdapterModel:
    """Freeze dense weights and inject PEFT LoRA into every active target in place."""

    from peft import LoraConfig, inject_adapter_in_model

    if hasattr(model, "_balalaika_target_modules"):
        raise ValueError("a Balalaika LoRA adapter is already injected")
    targets = _discover_active_modules(model)
    config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias=settings.bias,
        target_modules=list(targets),
        init_lora_weights=True,
    )
    adapted = inject_adapter_in_model(config, model, adapter_name=ADAPTER_NAME)
    adapted._balalaika_target_modules = targets
    adapted._balalaika_lora_settings = settings
    audit_trainable_parameters(adapted)
    return adapted


def audit_trainable_parameters(model: nn.Module) -> TrainableAudit:
    """Independently enforce adapter coverage and the adapter-only trainable boundary."""

    from peft.tuners.tuners_utils import BaseTunerLayer

    inventory = getattr(model, "_balalaika_target_modules", None)
    if not isinstance(inventory, tuple) or not inventory or not all(isinstance(name, str) and name for name in inventory):
        raise RuntimeError("LoRA target inventory must be a nonempty tuple of module names")
    if len(inventory) != len(set(inventory)):
        raise RuntimeError("LoRA target inventory contains duplicate module names")
    ambiguous = tuple(
        (left, right)
        for index, left in enumerate(inventory)
        for right in inventory[index + 1:]
        if left.startswith(f"{right}.") or right.startswith(f"{left}.")
    )
    if ambiguous:
        raise RuntimeError(f"LoRA target inventory contains ambiguous nested mappings: {ambiguous}")
    if any(_is_internal_qwen_lm_head_path(name) for name in inventory):
        raise RuntimeError("internal Qwen lm_head is forbidden in the LoRA target inventory")

    discovered = _discover_active_modules(model)
    if inventory != discovered:
        raise RuntimeError(f"stored LoRA target inventory does not match active targets: stored={inventory}, active={discovered}")

    wrappers = {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, BaseTunerLayer)
    }
    forbidden_heads = tuple(name for name in wrappers if _is_internal_qwen_lm_head_path(name))
    if forbidden_heads:
        raise RuntimeError(f"internal Qwen lm_head has a forbidden LoRA wrapper: {forbidden_heads}")
    missing_wrappers = tuple(name for name in inventory if name not in wrappers)
    extra_wrappers = tuple(name for name in wrappers if name not in inventory)
    if missing_wrappers or extra_wrappers:
        raise RuntimeError(
            f"LoRA wrapper inventory mismatch: missing={missing_wrappers}, extra={extra_wrappers}"
        )

    approved_parameter_names: set[str] = set()
    for target in inventory:
        wrapper = wrappers[target]
        adapter_parameters = _default_lora_parameters(wrapper)
        if not adapter_parameters:
            raise RuntimeError(f"required target {target} has no complete default LoRA parameter pair")
        if ADAPTER_NAME not in wrapper.active_adapters:
            raise RuntimeError(f"required target {target} does not have the default LoRA adapter active")
        frozen = tuple(name for name, parameter in adapter_parameters if not parameter.requires_grad)
        if frozen:
            raise RuntimeError(f"required target {target} has no complete trainable LoRA parameter pair: {frozen}")
        approved_parameter_names.update(f"{target}.{name}" for name, _ in adapter_parameters)

    trainable = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    for name in trainable:
        if _is_internal_qwen_lm_head_path(name):
            raise RuntimeError(f"internal Qwen lm_head has a forbidden trainable parameter: {name}")
        matches = tuple(target for target in inventory if name.startswith(f"{target}."))
        if "lora_" not in name:
            raise RuntimeError(f"dense trainable parameter is forbidden: {name}")
        if len(matches) != 1:
            raise RuntimeError(f"trainable LoRA parameter does not map to exactly one approved target: {name}")
        if name not in approved_parameter_names:
            raise RuntimeError(f"trainable LoRA parameter is not part of the approved default adapter: {name}")

    audit = TrainableAudit(
        target_modules=inventory,
        trainable_parameters=trainable,
        unexpected_dense_parameters=(),
        trainable_parameter_count=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        total_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    validate_trainable_audit_payload(asdict(audit))
    return audit


def validate_trainable_audit_payload(payload: Mapping[str, object]) -> None:
    """Validate the exact persisted production ``TrainableAudit`` schema."""

    required = {
        "target_modules",
        "trainable_parameters",
        "unexpected_dense_parameters",
        "trainable_parameter_count",
        "total_parameter_count",
    }
    if set(payload) != required:
        raise ValueError("trainable audit fields are invalid")
    targets = _audit_names(payload["target_modules"], "target_modules")
    trainable = _audit_names(payload["trainable_parameters"], "trainable_parameters")
    unexpected = payload["unexpected_dense_parameters"]
    trainable_count = payload["trainable_parameter_count"]
    total_count = payload["total_parameter_count"]
    if (
        not isinstance(unexpected, Sequence)
        or isinstance(unexpected, (str, bytes))
        or tuple(unexpected) != ()
        or isinstance(trainable_count, bool)
        or not isinstance(trainable_count, int)
        or isinstance(total_count, bool)
        or not isinstance(total_count, int)
        or trainable_count < len(trainable)
        or trainable_count >= total_count
    ):
        raise ValueError("trainable audit counts or dense inventory are invalid")
    required_targets = {"speech_embedding", "llm_decoder", "llm.model.model.embed_tokens"}
    if any(_is_internal_qwen_lm_head_path(name) for name in targets):
        raise ValueError("trainable audit target inventory contains the internal Qwen lm_head subtree")
    if not required_targets.issubset(targets):
        raise ValueError("trainable audit target coverage is invalid")
    for name in trainable:
        if _is_internal_qwen_lm_head_path(name):
            raise ValueError("trainable audit contains an internal Qwen lm_head tensor")
        matches = tuple(target for target in targets if name.startswith(f"{target}."))
        if len(matches) != 1:
            raise ValueError("trainable LoRA tensor does not map to exactly one approved target")
        suffix = name[len(matches[0]) + 1:]
        if suffix not in {
            "lora_A.default.weight",
            "lora_B.default.weight",
            "lora_embedding_A.default",
            "lora_embedding_B.default",
        }:
            raise ValueError("trainable audit contains an unapproved LoRA tensor")
    for target in targets:
        suffixes = {name[len(target) + 1:] for name in trainable if name.startswith(f"{target}.")}
        if suffixes not in (
            {"lora_A.default.weight", "lora_B.default.weight"},
            {"lora_embedding_A.default", "lora_embedding_B.default"},
        ):
            raise ValueError("approved target lacks one complete default LoRA tensor pair")


def _audit_names(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"trainable audit {field} must be a sequence")
    names = tuple(value)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"trainable audit {field} must contain unique nonempty names")
    return names


def _default_lora_parameters(wrapper: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    pairs = (
        ("lora_A", "lora_B"),
        ("lora_embedding_A", "lora_embedding_B"),
    )
    for left_name, right_name in pairs:
        left = getattr(wrapper, left_name, {})
        right = getattr(wrapper, right_name, {})
        left_has_adapter = ADAPTER_NAME in left
        right_has_adapter = ADAPTER_NAME in right
        if left_has_adapter or right_has_adapter:
            if not left_has_adapter or not right_has_adapter:
                return ()
            values = ((left_name, left[ADAPTER_NAME]), (right_name, right[ADAPTER_NAME]))
            parameters: list[tuple[str, nn.Parameter]] = []
            for collection_name, value in values:
                if isinstance(value, nn.Parameter):
                    parameters.append((f"{collection_name}.{ADAPTER_NAME}", value))
                else:
                    parameters.extend(
                        (f"{collection_name}.{ADAPTER_NAME}.{name}", parameter)
                        for name, parameter in value.named_parameters()
                    )
            return tuple(parameters)
    return ()


def save_adapter(model: nn.Module, path: Path) -> AdapterManifest:
    """Atomically save adapter tensors and their complete target inventory."""

    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    audit = audit_trainable_parameters(model)
    base_checksum = getattr(model, "_balalaika_base_checkpoint_sha256", None)
    if not isinstance(base_checksum, str) or len(base_checksum) != 64:
        raise ValueError("model is not bound to a base llm.pt checksum")
    settings = getattr(model, "_balalaika_lora_settings", None)
    if not isinstance(settings, LoraSettings):
        raise ValueError("model is missing Balalaika LoRA settings")

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    weights_path = path / ADAPTER_WEIGHTS_NAME
    state_dict = get_peft_model_state_dict(
        model,
        adapter_name=ADAPTER_NAME,
        save_embedding_layers=False,
    )
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in state_dict.items()}
    _atomic_safetensors(weights_path, tensors, save_file)
    weights_checksum = sha256_file(weights_path)
    revision = _code_revision()
    payload = {
        "format_version": 1,
        "weights": ADAPTER_WEIGHTS_NAME,
        "weights_sha256": weights_checksum,
        "base_checkpoint_sha256": base_checksum,
        "target_modules": list(audit.target_modules),
        "settings": asdict(settings),
        "code_revision": revision,
    }
    atomic_write_json(path / ADAPTER_MANIFEST_NAME, payload)
    return AdapterManifest(
        path=path / ADAPTER_MANIFEST_NAME,
        weights_path=weights_path,
        weights_sha256=weights_checksum,
        base_checkpoint_sha256=base_checksum,
        target_modules=audit.target_modules,
        settings=settings,
        code_revision=revision,
    )


def merge_adapter(base_dir: Path, adapter_dir: Path, output: Path) -> MergeReport:
    """Merge an adapter into a fresh base and atomically publish standalone ``llm.pt``."""

    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    base_dir = Path(base_dir)
    adapter_dir = Path(adapter_dir)
    output = Path(output)
    manifest, _ = _require_exact_adapter_directory(adapter_dir)
    weights_path = adapter_dir / manifest["weights"]

    model = load_base_llm(base_dir)
    base_checksum = getattr(model, "_balalaika_base_checkpoint_sha256", None)
    if base_checksum != manifest["base_checkpoint_sha256"]:
        raise ValueError("adapter base checkpoint checksum does not match the requested base")
    settings = LoraSettings(**manifest["settings"])
    adapted = inject_lora(model, settings)
    if list(adapted._balalaika_target_modules) != manifest["target_modules"]:
        raise ValueError("adapter target inventory does not match the fresh base model")
    load_result = set_peft_model_state_dict(
        adapted,
        load_file(str(weights_path), device="cpu"),
        adapter_name=ADAPTER_NAME,
    )
    missing_adapter_keys = [name for name in load_result.missing_keys if "lora_" in name]
    if missing_adapter_keys or load_result.unexpected_keys:
        raise ValueError(
            f"adapter state mismatch: missing={missing_adapter_keys}, unexpected={load_result.unexpected_keys}"
        )

    _merge_and_remove_adapter_layers(adapted, BaseTunerLayer)
    state_dict = {name: tensor.detach().cpu() for name, tensor in adapted.state_dict().items()}
    if any("lora_" in name or ".base_layer." in name for name in state_dict):
        raise RuntimeError("merged state dict still contains adapter wrapper keys")
    _atomic_torch_save(output, state_dict)
    return MergeReport(
        output=output,
        output_sha256=sha256_file(output),
        base_checkpoint_sha256=base_checksum,
        adapter_weights_sha256=manifest["weights_sha256"],
        target_modules=tuple(manifest["target_modules"]),
    )


def export_final_llm(request: ExportRequest) -> FinalModelManifest:
    """Publish one phase-2-provenance-bound, adapter-free final ``llm.pt``.

    The final directory is a single atomic publication.  It deliberately keeps
    a byte-for-byte copy of the adapter and its manifest for reproducibility,
    but the exported ``llm.pt`` itself has only the original base state names.
    """

    if not isinstance(request, ExportRequest):
        raise TypeError("request must be ExportRequest")
    lineage = _require_final_export_lineage(request)
    base_assets = build_base_asset_manifest(request.base_model_dir)
    base_assets_sha256 = _canonical_mapping_sha256(base_assets)
    code_identity = _code_identity()
    mode = "test" if request.test_mode else "production"
    if request.test_mode:
        verification_voices = _strict_smoke_voices(request.verification_voices)
    else:
        from cosyvoice.finetune.balalaika.evaluation import authenticated_prompt_inventory

        verification_voices, _ = authenticated_prompt_inventory(request.validation_request.prompts)
    prompt_inventory_sha256 = _prompt_inventory_sha256(verification_voices)
    task10_evidence = _final_validation_payload(lineage["validation_evidence"])
    expected_lineage = {
        "mode": mode,
        "production_ready": not request.test_mode,
        "base_checkpoint_sha256": sha256_file(request.base_model_dir / "llm.pt"),
        "adapter_weights_sha256": sha256_file(lineage["adapter_dir"] / ADAPTER_WEIGHTS_NAME),
        "adapter_manifest_sha256": lineage["adapter_manifest_sha256"],
        "phase2_checkpoint_manifest_sha256": lineage["checkpoint_sha256"],
        "phase2_model_state_sha256": lineage["model_state_sha256"],
        "validation_summary_sha256": lineage["validation_sha256"],
        "base_assets": base_assets,
        "base_assets_sha256": base_assets_sha256,
        "task10_evidence": task10_evidence,
        "prompt_inventory_sha256": prompt_inventory_sha256,
        "code_identity": code_identity,
    }
    output_dir = request.output_dir
    if output_dir.exists() or output_dir.is_symlink():
        return require_committed_final(
            output_dir,
            request.base_model_dir,
            expected_mode=mode,
            expected_lineage=expected_lineage,
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        copied_adapter = temporary / "adapter"
        shutil.copytree(lineage["adapter_dir"], copied_adapter)
        retained_adapter, retained_manifest_sha256 = _require_exact_adapter_directory(copied_adapter)
        if retained_manifest_sha256 != lineage["adapter_manifest_sha256"]:
            raise ValueError("retained adapter manifest checksum differs from the source")
        merged_path = temporary / "llm.pt"
        merge = merge_adapter(request.base_model_dir, copied_adapter, merged_path)
        _require_original_key_layout(request.base_model_dir / "llm.pt", merged_path)
        logit_evidence = _verify_adapter_active_logits(
            request.base_model_dir,
            copied_adapter,
            merged_path,
            temporary / "logit-verification.safetensors",
        )
        manifest_path = temporary / "final_model_manifest.json"
        staged_manifest = FinalModelManifest(
            path=manifest_path, llm_path=merged_path, adapter_dir=copied_adapter,
            llm_sha256=merge.output_sha256, base_checkpoint_sha256=merge.base_checkpoint_sha256,
            adapter_weights_sha256=merge.adapter_weights_sha256, adapter_manifest_sha256=retained_manifest_sha256,
            target_modules=merge.target_modules,
            phase2_checkpoint_sha256=lineage["checkpoint_sha256"], validation_summary_sha256=lineage["validation_sha256"],
            mode=mode, production_ready=not request.test_mode, base_assets_sha256=base_assets_sha256,
        )
        if request.test_mode:
            report = strict_verify_final_model(VerifyRequest(
                base_model_dir=request.base_model_dir, final_manifest=staged_manifest, recognizer=request.recognizer,
                voices=verification_voices, output_dir=temporary / "strict-verification",
                pipeline_factory=request.pipeline_factory, test_mode=True, artifact_root=temporary,
            ))
        else:
            from cosyvoice.finetune.balalaika.evaluation import GigaAmRecognizer

            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            report = strict_verify_final_model(VerifyRequest(
                base_model_dir=request.base_model_dir, final_manifest=staged_manifest, recognizer=GigaAmRecognizer(),
                voices=verification_voices, output_dir=temporary / "strict-verification",
                artifact_root=temporary, expected_local_rank=local_rank,
            ))
        strict_payload = _read_json_mapping(report.path, "strict verification report")
        if strict_payload.get("prompt_inventory_sha256") != prompt_inventory_sha256:
            raise ValueError("strict verification prompt inventory differs from Task 10")
        payload = {
            "format_version": 2,
            "llm": "llm.pt",
            "llm_sha256": merge.output_sha256,
            "adapter": "adapter",
            "adapter_weights_sha256": merge.adapter_weights_sha256,
            "adapter_manifest_sha256": retained_manifest_sha256,
            "adapter_code_revision": retained_adapter["code_revision"],
            "base_checkpoint_sha256": merge.base_checkpoint_sha256,
            "target_modules": list(merge.target_modules),
            "phase2_checkpoint": str(request.phase2_checkpoint),
            "phase2_checkpoint_manifest_sha256": lineage["checkpoint_sha256"],
            "phase2_model_state_sha256": lineage["model_state_sha256"],
            "validation_summary": str(request.validation_summary),
            "validation_summary_sha256": lineage["validation_sha256"],
            "mode": mode,
            "production_ready": not request.test_mode,
            "base_assets": base_assets,
            "base_assets_sha256": base_assets_sha256,
            "task10_evidence": task10_evidence,
            "code_identity": code_identity,
            "logit_verification": logit_evidence,
            "prompt_inventory_sha256": prompt_inventory_sha256,
            "selected_voices": strict_payload["selected_voices"],
            "strict_verification": {
                "report": "strict-verification/strict-verification.json",
                "report_sha256": sha256_file(report.path),
                "audio": [str(path.relative_to(temporary)) for path in report.audio_paths],
            },
        }
        atomic_write_json(manifest_path, payload)
        artifact_checksums = _regular_file_inventory(temporary, excluded={"final-success.json"})
        atomic_write_json(temporary / "final-success.json", {
            "format_version": 2,
            "mode": mode,
            "production_ready": not request.test_mode,
            "manifest": "final_model_manifest.json",
            "manifest_sha256": sha256_file(manifest_path),
            "logit_verification_sha256": _canonical_mapping_sha256(logit_evidence),
            "artifacts": artifact_checksums,
        })
        require_committed_final(
            temporary,
            request.base_model_dir,
            expected_mode=mode,
            expected_lineage=expected_lineage,
        )
        _publish_directory(temporary, output_dir)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    return require_committed_final(
        output_dir,
        request.base_model_dir,
        expected_mode=mode,
        expected_lineage=expected_lineage,
    )


def strict_verify_final_model(request: VerifyRequest) -> VerificationReport:
    """Strict-load and smoke-test a final LLM through normal CosyVoice3 APIs."""

    if not isinstance(request, VerifyRequest):
        raise TypeError("request must be VerifyRequest")
    expected_mode = "test" if request.test_mode else "production"
    if request.final_manifest.mode != expected_mode or request.final_manifest.production_ready != (not request.test_mode):
        raise ValueError("strict verification mode differs from the final manifest")
    _require_staged_final_inputs(request.final_manifest)
    if _canonical_mapping_sha256(build_base_asset_manifest(request.base_model_dir)) != request.final_manifest.base_assets_sha256:
        raise ValueError("strict verification base asset identity differs from exported base")
    if not request.test_mode:
        _require_production_recognizer(request.recognizer, request.expected_local_rank)
    inventory = _strict_smoke_voices(request.voices)
    voices = inventory[:4]
    if request.output_dir.exists() or request.output_dir.is_symlink():
        raise FileExistsError(f"strict verification destination already exists: {request.output_dir}")
    request.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{request.output_dir.name}.", dir=request.output_dir.parent))
    try:
        audio_dir = stage / "audio"
        with tempfile.TemporaryDirectory(prefix=".cosyvoice3-strict-", dir=stage) as raw_view:
            view = Path(raw_view)
            _make_verification_view(request.base_model_dir, request.final_manifest.llm_path, view)
            pipeline = _normal_cosyvoice3_pipeline(view, request.pipeline_factory)
            _require_frozen_inference_components(pipeline)
            _strict_load_pipeline_llm(pipeline, request.final_manifest.llm_path)
            audio_dir.mkdir()
            random.seed(1986)
            torch.manual_seed(1986)
            audio_paths = tuple(_synthesize_smoke_voice(pipeline, voice, prompt, audio_dir / f"smoke-{index + 1:02d}.wav") for index, (voice, prompt) in enumerate(zip(voices, _SMOKE_PROMPTS, strict=True)))
        _require_pcm_24khz_mono(audio_paths)
        transcribe = getattr(request.recognizer, "transcribe", None)
        if not callable(transcribe):
            raise TypeError("recognizer must expose transcribe(paths)")
        hypotheses = transcribe(audio_paths)
        if (
        not isinstance(hypotheses, Sequence)
        or isinstance(hypotheses, (str, bytes))
        or len(hypotheses) != 4
        or any(not isinstance(value, str) or not value.strip() for value in hypotheses)
        ):
            raise ValueError("GigaAM strict-verification transcription is invalid")
        checksums = {path.name: sha256_file(path) for path in audio_paths}
        libraries = _library_identities(request.recognizer, pipeline)
        prompt_inventory_sha256 = _canonical_mapping_sha256([
            {"voice_id": value["voice_id"], "prompt_text": value["prompt_text"], "prompt_sha256": value["prompt_sha256"]}
            for value in inventory
        ])
        artifact_root = request.artifact_root or request.output_dir
        try:
            relative_audio = [
                str((request.output_dir / "audio" / path.name).relative_to(artifact_root))
                for path in audio_paths
            ]
        except ValueError as exc:
            raise ValueError("strict verification output must remain under its artifact root") from exc
        payload = {
            "format_version": 2,
            "mode": expected_mode,
            "production_ready": not request.test_mode,
            "immutable_inputs": {
                "llm_sha256": request.final_manifest.llm_sha256,
                "base_assets_sha256": request.final_manifest.base_assets_sha256,
                "prompt_inventory_sha256": prompt_inventory_sha256,
            },
            "strict_load": True,
            "smoke_utterances": 4,
            "prompt_inventory_sha256": prompt_inventory_sha256,
            "selected_voices": [value["voice_id"] for value in voices],
            "prompt_source": "test_fixture" if request.test_mode else "task10_validation_request",
            "audio": [
                {"voice_id": voice["voice_id"], "prompt": prompt, "path": relative_path, "sha256": checksums[path.name], "asr": asr}
                for voice, prompt, path, relative_path, asr in zip(voices, _SMOKE_PROMPTS, audio_paths, relative_audio, hypotheses, strict=True)
            ],
            "recognizer": {
                "class": f"{type(request.recognizer).__module__}.{type(request.recognizer).__qualname__}",
                "provenance": dict(request.recognizer.provenance()) if callable(getattr(request.recognizer, "provenance", None)) else None,
            },
            "libraries": libraries,
        }
        report_path = stage / "strict-verification.json"
        atomic_write_json(report_path, payload)
        _publish_directory(stage, request.output_dir)
    except Exception:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise
    audio_paths = tuple(request.output_dir / "audio" / path.name for path in audio_paths)
    report_path = request.output_dir / "strict-verification.json"
    return VerificationReport(
        path=report_path,
        strict_load=True,
        smoke_utterances=4,
        audio_paths=audio_paths,
        audio_checksums=checksums,
        asr_results=tuple(hypotheses),
        library_identities=libraries,
    )


_SMOKE_PROMPTS = (
    "В Москве сегодня тихий августовский вечер.",
    "Пожалуйста, произнесите число сорок два внимательно.",
    "Русская речь должна звучать ясно и естественно.",
    "Это короткая проверка голосового клонирования.",
)


def build_base_asset_manifest(base_dir: Path) -> dict[str, object]:
    """Return a bounded fail-closed inventory for every regular base-model asset."""

    base_dir = Path(base_dir)
    required = {"cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt"}
    if base_dir.is_symlink() or not base_dir.is_dir():
        raise ValueError("base model directory must be a real directory")
    entries: list[dict[str, str]] = []
    for path in sorted(base_dir.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(base_dir.resolve()):
            raise ValueError("base model assets must not contain symlinks or path escapes")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("base model assets must be regular files")
        if len(entries) >= 10_000:
            raise ValueError("base model asset inventory exceeds safe bound")
        entries.append({"path": str(path.relative_to(base_dir)), "sha256": sha256_file(path)})
    names = {entry["path"] for entry in entries}
    if not required.issubset(names):
        raise ValueError("base model asset manifest lacks required CosyVoice3 files")
    return {"format_version": 1, "base_dir_name": base_dir.name, "files": entries}


def _canonical_mapping_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _prompt_inventory_sha256(voices: Sequence[Mapping[str, object]]) -> str:
    return _canonical_mapping_sha256([
        {
            "voice_id": voice["voice_id"],
            "prompt_text": voice["prompt_text"],
            "prompt_sha256": voice["prompt_sha256"],
        }
        for voice in voices
    ])


def _read_json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _regular_file_inventory(root: Path, *, excluded: set[str] | None = None) -> dict[str, str]:
    """Hash every bounded regular file below ``root`` without following links."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("final artifact root must be a real directory")
    excluded = excluded or set()
    result: dict[str, str] = {}
    directories: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("final artifact inventory contains a symlink or path escape")
        if path.is_dir():
            directories.add(str(path.relative_to(root)))
            continue
        if not path.is_file():
            raise ValueError("final artifact inventory contains a non-regular file")
        relative = str(path.relative_to(root))
        if relative in excluded:
            continue
        if len(result) >= 10_000:
            raise ValueError("final artifact inventory exceeds safe bound")
        result[relative] = sha256_file(path)
    if directories != {"adapter", "strict-verification", "strict-verification/audio"}:
        raise ValueError("final artifact directory inventory contains missing or extra directories")
    return result


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_final_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative path")
    relative = Path(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"{label} must be a safe relative path")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the final directory")
    return path


def _validate_base_asset_payload(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"format_version", "base_dir_name", "files"}:
        raise ValueError("final base asset manifest schema is invalid")
    files = value.get("files")
    if value.get("format_version") != 1 or not isinstance(value.get("base_dir_name"), str) or not value["base_dir_name"]:
        raise ValueError("final base asset manifest metadata is invalid")
    if not isinstance(files, list) or not files or len(files) > 10_000:
        raise ValueError("final base asset inventory size is invalid")
    names: list[str] = []
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError("final base asset entry schema is invalid")
        path = item.get("path")
        if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts or "\\" in path:
            raise ValueError("final base asset path is invalid")
        _require_digest(item.get("sha256"), "base asset checksum")
        names.append(path)
    if len(names) != len(set(names)) or not {"cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt"}.issubset(names):
        raise ValueError("final base asset inventory is incomplete or duplicated")


def _validate_code_identity(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"head", "dirty", "diff_sha256"}:
        raise ValueError("final code identity schema is invalid")
    if not isinstance(value.get("head"), str) or not value["head"] or type(value.get("dirty")) is not bool:
        raise ValueError("final code identity metadata is invalid")
    _require_digest(value.get("diff_sha256"), "code diff checksum")


def _validate_logit_evidence(
    root: Path,
    value: object,
    base_model_dir: Path,
    adapter_dir: Path,
    merged_path: Path,
) -> None:
    fields = {
        "format_version", "evidence", "evidence_sha256", "probe", "probe_sha256", "atol", "rtol",
        "max_abs_error", "max_relative_error", "max_tolerance_ratio", "finite", "pass",
    }
    if not isinstance(value, Mapping) or set(value) != fields or value.get("format_version") != 2:
        raise ValueError("logit verification schema is invalid")
    probe = {"token_ids": [[0, 0, 0]], "input_shape": [1, 3]}
    if value.get("probe") != probe or value.get("probe_sha256") != _canonical_mapping_sha256(probe):
        raise ValueError("logit verification probe identity changed")
    if value.get("evidence") != "logit-verification.safetensors":
        raise ValueError("logit verification tensor evidence path changed")
    _require_digest(value.get("evidence_sha256"), "logit verification tensor evidence checksum")
    if value.get("atol") != 0.02 or value.get("rtol") != 0.02 or value.get("finite") is not True or value.get("pass") is not True:
        raise ValueError("logit verification tolerance or result is invalid")
    for name in ("max_abs_error", "max_relative_error", "max_tolerance_ratio"):
        number = value.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not torch.isfinite(torch.tensor(float(number))) or number < 0:
            raise ValueError("logit verification error metrics are invalid")
    if value["max_tolerance_ratio"] > 1.0:
        raise ValueError("logit verification exceeds its elementwise tolerance")
    evidence_path = _safe_final_path(root, value["evidence"], "logit verification tensor evidence")
    if evidence_path.is_symlink() or not evidence_path.is_file() or sha256_file(evidence_path) != value["evidence_sha256"]:
        raise ValueError("logit verification tensor evidence checksum changed")
    from safetensors import SafetensorError
    from safetensors.torch import load_file

    try:
        tensors = load_file(str(evidence_path), device="cpu")
    except (OSError, RuntimeError, SafetensorError) as exc:
        raise ValueError("logit verification tensor evidence is invalid") from exc
    if set(tensors) != {"token_ids", "adapter_active_logits", "merged_logits"}:
        raise ValueError("logit verification tensor evidence schema is invalid")
    token_ids = tensors["token_ids"]
    expected = tensors["adapter_active_logits"]
    actual = tensors["merged_logits"]
    if (
        token_ids.dtype != torch.int64
        or token_ids.device.type != "cpu"
        or not token_ids.is_contiguous()
        or tuple(token_ids.shape) != (1, 3)
        or token_ids.tolist() != [[0, 0, 0]]
        or expected.dtype != torch.float32
        or actual.dtype != torch.float32
        or expected.device.type != "cpu"
        or actual.device.type != "cpu"
        or not expected.is_contiguous()
        or not actual.is_contiguous()
        or expected.ndim != 3
        or expected.shape != actual.shape
        or tuple(expected.shape[:2]) != (1, 3)
        or expected.shape[2] < 1
    ):
        raise ValueError("logit verification tensor evidence dtype or shape is invalid")
    recomputed = _logit_metrics(expected, actual, value["atol"], value["rtol"])
    for name in ("max_abs_error", "max_relative_error", "max_tolerance_ratio", "finite", "pass"):
        if value[name] != recomputed[name]:
            raise ValueError(f"logit verification {name} differs from tensor evidence")
    computed_ids, computed_expected, computed_actual = _compute_fixed_probe_tensors(
        base_model_dir,
        adapter_dir,
        merged_path,
    )
    if (
        not torch.equal(token_ids, computed_ids)
        or expected.dtype != computed_expected.dtype
        or expected.shape != computed_expected.shape
        or not torch.equal(expected, computed_expected)
        or actual.dtype != computed_actual.dtype
        or actual.shape != computed_actual.shape
        or not torch.equal(actual, computed_actual)
    ):
        raise ValueError("logit verification tensor evidence differs from authenticated model computation")


def _validate_task10_evidence(value: object, prompt_inventory_sha256: str) -> None:
    fields = {
        "evaluation_identity_sha256", "artifact_checksums", "checkpoint_sha256",
        "model_state_sha256", "prompt_inventory_sha256", "wandb",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("production final Task 10 evidence schema is invalid")
    for name in ("evaluation_identity_sha256", "checkpoint_sha256", "model_state_sha256"):
        _require_digest(value.get(name), f"Task 10 {name}")
    if value.get("prompt_inventory_sha256") != prompt_inventory_sha256:
        raise ValueError("production final prompt identity differs from Task 10")
    artifacts = value.get("artifact_checksums")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "results_jsonl", "summary_json", "panel_manifest", "validation_seal"
    }:
        raise ValueError("production final Task 10 artifact evidence is invalid")
    for digest in artifacts.values():
        _require_digest(digest, "Task 10 artifact checksum")
    wandb = value.get("wandb")
    if not isinstance(wandb, Mapping) or set(wandb) != {
        "run_id", "context", "marker", "run_manifest_sha256", "ledger_path",
        "ledger_sha256", "remote_markers",
    }:
        raise ValueError("production final W&B evidence schema is invalid")
    if not isinstance(wandb.get("run_id"), str) or not wandb["run_id"] or not isinstance(wandb.get("ledger_path"), str):
        raise ValueError("production final W&B identity is invalid")
    marker = _require_digest(wandb.get("marker"), "W&B commit marker")
    run_manifest_sha256 = _require_digest(wandb.get("run_manifest_sha256"), "W&B run manifest checksum")
    ledger_sha256 = _require_digest(wandb.get("ledger_sha256"), "W&B ledger checksum")
    ledger_path = Path(wandb["ledger_path"])
    run_manifest_path = ledger_path.parent.parent / "wandb-run.json"
    if (
        ledger_path.is_symlink() or not ledger_path.is_file() or sha256_file(ledger_path) != ledger_sha256
        or run_manifest_path.is_symlink() or not run_manifest_path.is_file()
        or sha256_file(run_manifest_path) != run_manifest_sha256
    ):
        raise ValueError("production final W&B local evidence changed")
    context = wandb.get("context")
    if not isinstance(context, Mapping) or set(context) != {
        "evaluation_identity_sha256", "assignment_sha256", "artifact_checksums", "metrics_sha256"
    } or not isinstance(wandb.get("remote_markers"), Mapping):
        raise ValueError("production final W&B context is invalid")
    for name in ("evaluation_identity_sha256", "assignment_sha256", "metrics_sha256"):
        _require_digest(context.get(name), f"W&B context {name}")
    context_artifacts = context.get("artifact_checksums")
    if not isinstance(context_artifacts, Mapping) or set(context_artifacts) != set(artifacts):
        raise ValueError("production final W&B artifact context is invalid")
    for digest in context_artifacts.values():
        _require_digest(digest, "W&B context artifact checksum")
    if value["evaluation_identity_sha256"] != context["evaluation_identity_sha256"]:
        raise ValueError("Task 10 evaluation identity differs from W&B context")
    if dict(artifacts) != dict(context_artifacts):
        raise ValueError("Task 10 artifact checksums differ from W&B context")
    expected_markers = {"validation/commit/40/scalars": marker, "validation/commit/40/media": marker}
    if dict(wandb["remote_markers"]) != expected_markers or _canonical_mapping_sha256(context) != marker:
        raise ValueError("production final W&B remote marker evidence changed")
    ledger = _read_json_mapping(ledger_path, "production W&B ledger")
    if set(ledger) != {
        "format_version", "validation_index", "run_id", "evaluation_identity_sha256",
        "assignment_sha256", "artifact_checksums", "metrics_sha256", "remote_marker_sha256",
        "media_logged", "scalars_logged", "committed",
    } or (
        ledger.get("format_version") != 2
        or ledger.get("validation_index") != 40
        or ledger.get("run_id") != wandb["run_id"]
        or ledger.get("evaluation_identity_sha256") != context["evaluation_identity_sha256"]
        or ledger.get("assignment_sha256") != context["assignment_sha256"]
        or ledger.get("artifact_checksums") != context_artifacts
        or ledger.get("metrics_sha256") != context["metrics_sha256"]
        or ledger.get("remote_marker_sha256") != marker
        or tuple(ledger.get(name) for name in ("media_logged", "scalars_logged", "committed")) != (True, True, True)
    ):
        raise ValueError("production W&B ledger differs from final evidence")
    run_manifest = _read_json_mapping(run_manifest_path, "production W&B run manifest")
    if run_manifest != {"format_version": 1, "run_id": wandb["run_id"]}:
        raise ValueError("production W&B run manifest differs from final evidence")


def require_committed_final(
    output_dir: Path,
    base_model_dir: Path,
    *,
    expected_mode: Literal["test", "production"] | None = None,
    expected_lineage: Mapping[str, object] | None = None,
) -> FinalModelManifest:
    """Fail closed unless ``output_dir`` is one complete immutable final export."""

    if not isinstance(base_model_dir, Path):
        raise TypeError("base_model_dir must be Path")
    output_dir = Path(output_dir)
    inventory = _regular_file_inventory(output_dir, excluded={"final-success.json"})
    expected_files = {
        "llm.pt",
        "adapter/adapter_manifest.json",
        "adapter/adapter_model.safetensors",
        "logit-verification.safetensors",
        "final_model_manifest.json",
        "strict-verification/strict-verification.json",
        *(f"strict-verification/audio/smoke-{index:02d}.wav" for index in range(1, 5)),
    }
    if set(inventory) != expected_files:
        raise ValueError("final artifact file inventory contains missing or unexpected files")
    seal_path = output_dir / "final-success.json"
    if seal_path.is_symlink() or not seal_path.is_file():
        raise ValueError("final success seal is missing or invalid")
    seal = _read_json_mapping(seal_path, "final success seal")
    if set(seal) != {
        "format_version", "mode", "production_ready", "manifest", "manifest_sha256",
        "logit_verification_sha256", "artifacts",
    } or seal.get("format_version") != 2:
        raise ValueError("final success seal schema is invalid")
    if seal.get("manifest") != "final_model_manifest.json" or seal.get("artifacts") != inventory:
        raise ValueError("final success seal artifact inventory changed")
    manifest_path = output_dir / "final_model_manifest.json"
    if inventory.get("final_model_manifest.json") != seal.get("manifest_sha256"):
        raise ValueError("final success seal manifest checksum changed")
    manifest = _read_json_mapping(manifest_path, "final model manifest")
    manifest_fields = {
        "format_version", "llm", "llm_sha256", "adapter", "adapter_weights_sha256",
        "adapter_manifest_sha256", "adapter_code_revision", "base_checkpoint_sha256",
        "target_modules", "phase2_checkpoint", "phase2_checkpoint_manifest_sha256",
        "phase2_model_state_sha256", "validation_summary", "validation_summary_sha256",
        "mode", "production_ready", "base_assets", "base_assets_sha256", "task10_evidence",
        "code_identity", "logit_verification", "prompt_inventory_sha256", "selected_voices",
        "strict_verification",
    }
    if set(manifest) != manifest_fields or manifest.get("format_version") != 2:
        raise ValueError("final model manifest schema is invalid")
    mode = manifest.get("mode")
    production_ready = manifest.get("production_ready")
    if mode not in ("test", "production") or production_ready is not (mode == "production"):
        raise ValueError("final model mode and production-ready state disagree")
    if seal.get("mode") != mode or seal.get("production_ready") is not production_ready:
        raise ValueError("final success seal mode differs from the manifest")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError("committed final mode differs from expected mode")
    for name in (
        "llm_sha256", "adapter_weights_sha256", "adapter_manifest_sha256",
        "base_checkpoint_sha256", "phase2_checkpoint_manifest_sha256",
        "phase2_model_state_sha256", "validation_summary_sha256", "base_assets_sha256",
        "prompt_inventory_sha256",
    ):
        _require_digest(manifest.get(name), f"final {name}")
    if manifest.get("llm") != "llm.pt" or manifest.get("adapter") != "adapter":
        raise ValueError("final model artifact paths changed")
    llm_path = output_dir / "llm.pt"
    if llm_path.is_symlink() or not llm_path.is_file() or sha256_file(llm_path) != manifest["llm_sha256"]:
        raise ValueError("final llm.pt checksum changed")
    adapter_dir = output_dir / "adapter"
    adapter, adapter_manifest_sha256 = _require_exact_adapter_directory(adapter_dir)
    if (
        adapter_manifest_sha256 != manifest["adapter_manifest_sha256"]
        or adapter["weights_sha256"] != manifest["adapter_weights_sha256"]
        or adapter["base_checkpoint_sha256"] != manifest["base_checkpoint_sha256"]
        or adapter["code_revision"] != manifest["adapter_code_revision"]
        or adapter["settings"] != asdict(LoraSettings())
        or adapter["target_modules"] != manifest.get("target_modules")
    ):
        raise ValueError("final retained adapter provenance changed")
    _validate_base_asset_payload(manifest["base_assets"])
    if _canonical_mapping_sha256(manifest["base_assets"]) != manifest["base_assets_sha256"]:
        raise ValueError("final base asset inventory checksum changed")
    current_base_assets = build_base_asset_manifest(base_model_dir)
    if (
        current_base_assets != manifest["base_assets"]
        or _canonical_mapping_sha256(current_base_assets) != manifest["base_assets_sha256"]
    ):
        raise ValueError("current base asset manifest differs from final embedded base assets")
    _validate_code_identity(manifest["code_identity"])
    _validate_logit_evidence(
        output_dir,
        manifest["logit_verification"],
        base_model_dir,
        adapter_dir,
        llm_path,
    )
    if seal.get("logit_verification_sha256") != _canonical_mapping_sha256(manifest["logit_verification"]):
        raise ValueError("final success seal logit evidence checksum changed")
    strict = manifest.get("strict_verification")
    if not isinstance(strict, Mapping) or set(strict) != {"report", "report_sha256", "audio"}:
        raise ValueError("final strict verification inventory schema is invalid")
    if strict.get("report") != "strict-verification/strict-verification.json":
        raise ValueError("final strict verification report path changed")
    report_path = _safe_final_path(output_dir, strict["report"], "strict report")
    if not report_path.is_file() or sha256_file(report_path) != strict.get("report_sha256"):
        raise ValueError("final strict verification report checksum changed")
    report = _read_json_mapping(report_path, "strict verification report")
    _validate_strict_report(output_dir, report, manifest)
    report_audio = [row["path"] for row in report["audio"]]
    if strict.get("audio") != report_audio:
        raise ValueError("final strict verification audio inventory changed")
    if mode == "production":
        _validate_task10_evidence(manifest.get("task10_evidence"), manifest["prompt_inventory_sha256"])
        task10 = manifest["task10_evidence"]
        if manifest["phase2_checkpoint_manifest_sha256"] != task10["checkpoint_sha256"]:
            raise ValueError("final phase-2 checkpoint identity differs from Task 10 evidence")
        if manifest["phase2_model_state_sha256"] != task10["model_state_sha256"]:
            raise ValueError("final phase-2 model state differs from Task 10 evidence")
        if manifest["validation_summary_sha256"] != task10["artifact_checksums"]["summary_json"]:
            raise ValueError("final validation summary differs from Task 10 evidence")
    elif manifest.get("task10_evidence") is not None:
        raise ValueError("test final cannot carry production Task 10 evidence")
    base_llm = next(
        (item["sha256"] for item in manifest["base_assets"]["files"] if item["path"] == "llm.pt"),
        None,
    )
    if manifest["base_checkpoint_sha256"] != base_llm:
        raise ValueError("final base checkpoint differs from the embedded base asset inventory")
    if expected_lineage is not None:
        if any(manifest.get(name) != value for name, value in expected_lineage.items()):
            raise ValueError("existing final export differs from the requested lineage")
    return FinalModelManifest(
        path=manifest_path,
        llm_path=llm_path,
        adapter_dir=adapter_dir,
        llm_sha256=manifest["llm_sha256"],
        base_checkpoint_sha256=manifest["base_checkpoint_sha256"],
        adapter_weights_sha256=manifest["adapter_weights_sha256"],
        adapter_manifest_sha256=manifest["adapter_manifest_sha256"],
        target_modules=tuple(manifest["target_modules"]),
        phase2_checkpoint_sha256=manifest["phase2_checkpoint_manifest_sha256"],
        validation_summary_sha256=manifest["validation_summary_sha256"],
        mode=mode,
        production_ready=production_ready,
        base_assets_sha256=manifest["base_assets_sha256"],
    )


def _validate_strict_report(root: Path, report: Mapping[str, object], manifest: Mapping[str, object]) -> None:
    fields = {
        "format_version", "mode", "production_ready", "immutable_inputs", "strict_load",
        "smoke_utterances", "prompt_inventory_sha256", "selected_voices", "prompt_source",
        "audio", "recognizer", "libraries",
    }
    if set(report) != fields or report.get("format_version") != 2:
        raise ValueError("strict verification report schema is invalid")
    mode = manifest["mode"]
    if report.get("mode") != mode or report.get("production_ready") is not manifest["production_ready"]:
        raise ValueError("strict verification report mode changed")
    immutable = report.get("immutable_inputs")
    expected_immutable = {
        "llm_sha256": manifest["llm_sha256"],
        "base_assets_sha256": manifest["base_assets_sha256"],
        "prompt_inventory_sha256": manifest["prompt_inventory_sha256"],
    }
    if immutable != expected_immutable or report.get("prompt_inventory_sha256") != manifest["prompt_inventory_sha256"]:
        raise ValueError("strict verification immutable input identity changed")
    selected = [f"voice_{index:02d}" for index in range(4)]
    if report.get("strict_load") is not True or report.get("smoke_utterances") != 4 or report.get("selected_voices") != selected:
        raise ValueError("strict verification completion evidence is invalid")
    if report.get("prompt_source") != ("task10_validation_request" if mode == "production" else "test_fixture"):
        raise ValueError("strict verification prompt source differs from mode")
    if manifest.get("selected_voices") != selected:
        raise ValueError("final selected voice inventory changed")
    audio = report.get("audio")
    if not isinstance(audio, list) or len(audio) != 4:
        raise ValueError("strict verification requires exactly four audio records")
    for index, row in enumerate(audio, 1):
        if not isinstance(row, Mapping) or set(row) != {"voice_id", "prompt", "path", "sha256", "asr"}:
            raise ValueError("strict verification audio record schema is invalid")
        expected_path = f"strict-verification/audio/smoke-{index:02d}.wav"
        if row.get("voice_id") != selected[index - 1] or row.get("prompt") != _SMOKE_PROMPTS[index - 1] or row.get("path") != expected_path:
            raise ValueError("strict verification audio identity changed")
        path = _safe_final_path(root, row["path"], "strict audio")
        if path.is_symlink() or not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise ValueError("strict verification audio checksum changed")
        if not isinstance(row.get("asr"), str) or not row["asr"].strip():
            raise ValueError("strict verification ASR result is empty")
        _require_pcm_24khz_mono((path,))
    recognizer = report.get("recognizer")
    if not isinstance(recognizer, Mapping) or set(recognizer) != {"class", "provenance"}:
        raise ValueError("strict verification recognizer schema is invalid")
    libraries = report.get("libraries")
    expected_libraries = {"torch", "pipeline_class", "recognizer_class", "peft", "transformers", "onnx-asr", "gigaam"}
    if not isinstance(libraries, Mapping) or set(libraries) != expected_libraries:
        raise ValueError("strict verification library identity schema is invalid")
    if libraries.get("recognizer_class") != recognizer.get("class"):
        raise ValueError("strict verification recognizer class identity changed")
    if mode == "production":
        if libraries.get("pipeline_class") != "cosyvoice.cli.cosyvoice.CosyVoice3":
            raise ValueError("production final was not verified by the exact CosyVoice3 pipeline class")
        if recognizer.get("class") != "cosyvoice.finetune.balalaika.evaluation.GigaAmRecognizer":
            raise ValueError("production final was not verified by Task 10 GigaAmRecognizer")
        provenance = recognizer.get("provenance")
        if not isinstance(provenance, Mapping) or set(provenance) != {"model", "provider", "device_id", "max_batch_size"}:
            raise ValueError("production strict recognizer provenance is invalid")
        if provenance.get("model") != "gigaam-v3-rnnt" or provenance.get("provider") != "CUDAExecutionProvider":
            raise ValueError("production strict recognizer is not GigaAM v3 RNN-T CUDA")
        device = provenance.get("device_id")
        batch = provenance.get("max_batch_size")
        if (
            isinstance(device, bool) or not isinstance(device, int) or device < 0
            or isinstance(batch, bool) or not isinstance(batch, int) or batch < 1
        ):
            raise ValueError("production strict recognizer CUDA rank or batch size is invalid")
        expected_gigaam = json.dumps(dict(provenance), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if libraries.get("gigaam") != expected_gigaam:
            raise ValueError("production strict recognizer library provenance changed")


def _final_validation_payload(evidence: object) -> object:
    if evidence is None:
        return None
    from cosyvoice.finetune.balalaika.evaluation import final_validation_evidence_payload

    return final_validation_evidence_payload(evidence)


def checkpoint_identity_sha256(manifest: Mapping[str, object]) -> str:
    """Stable phase-checkpoint identity, excluding its post-validation status bit."""

    if not isinstance(manifest, Mapping):
        raise TypeError("checkpoint manifest must be a mapping")
    value = dict(manifest)
    value.pop("validation_status", None)
    return _canonical_mapping_sha256(value)


def model_state_identity_sha256(state_files: Mapping[str, object]) -> str:
    """Stable identity of the authenticated immutable checkpoint state files."""

    if not isinstance(state_files, Mapping) or not state_files:
        raise ValueError("checkpoint state-file inventory is invalid")
    if any(not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64 for name, digest in state_files.items()):
        raise ValueError("checkpoint state-file inventory is invalid")
    return _canonical_mapping_sha256(dict(state_files))


def _require_final_export_lineage(request: ExportRequest) -> dict[str, object]:
    checkpoint = request.phase2_checkpoint / "checkpoint_manifest.json"
    try:
        checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        validation = json.loads(request.validation_summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("final export requires readable phase-2 and validation manifests") from exc
    if (
        not isinstance(checkpoint_payload, dict)
        or checkpoint_payload.get("format_version") != 1
        or checkpoint_payload.get("validation_status") != "succeeded"
    ):
        raise ValueError("phase-2 checkpoint is not validation-succeeded")
    identity = checkpoint_payload.get("identity")
    progress = checkpoint_payload.get("progress")
    if not isinstance(identity, Mapping) or not isinstance(progress, Mapping) or identity.get("phase") != 2 or progress.get("phase") != 2:
        raise ValueError("final export requires phase-2 checkpoint lineage")
    if progress.get("validation_index") != 40:
        raise ValueError("final export requires validation index 40")
    adapter_dir = request.phase2_checkpoint / "adapter"
    adapter, adapter_manifest_sha256 = _require_exact_adapter_directory(adapter_dir)
    _validate_phase2_training_identity(identity, adapter)
    if not request.test_mode:
        _validate_phase2_training_identity(request.expected_training_identity, adapter)
        if identity != request.expected_training_identity:
            raise ValueError("phase-2 checkpoint training identity differs from the workflow identity")
    settings = LoraSettings(**adapter["settings"])
    if identity.get("lora") != asdict(settings):
        raise ValueError("phase-2 LoRA rank/alpha settings differ from the final adapter")
    if identity.get("base_checkpoint_sha256") != adapter["base_checkpoint_sha256"]:
        raise ValueError("phase-2 base checksum differs from the final adapter")
    state_files = checkpoint_payload.get("state_files")
    if not isinstance(state_files, Mapping) or not state_files:
        raise ValueError("phase-2 checkpoint has no state checksum inventory")
    actual = {
        str(path.relative_to(request.phase2_checkpoint)): sha256_file(path)
        for path in sorted(request.phase2_checkpoint.rglob("*"))
        if path.is_file() and path != checkpoint
    }
    if dict(state_files) != actual:
        raise ValueError("phase-2 checkpoint state checksum inventory changed")
    checkpoint_sha256 = checkpoint_identity_sha256(checkpoint_payload)
    model_state_sha256 = model_state_identity_sha256(actual)
    if not isinstance(validation, Mapping) or validation.get("format_version") != 2 or validation.get("validation_index") != 40:
        raise ValueError("final export requires a successful validation-40 summary")
    evaluation_identity = validation.get("evaluation_identity")
    if not isinstance(evaluation_identity, Mapping):
        raise ValueError("validation-40 summary lacks evaluation provenance")
    if evaluation_identity.get("base_checkpoint_sha256") != adapter["base_checkpoint_sha256"]:
        raise ValueError("validation base checksum differs from the final adapter")
    if evaluation_identity.get("adapter_sha256") != adapter["weights_sha256"]:
        raise ValueError("validation adapter checksum differs from the final adapter")
    seal_path = request.validation_summary.with_name("validation-success.json")
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("validation-40 success seal is missing or invalid") from exc
    artifacts = seal.get("artifacts") if isinstance(seal, Mapping) else None
    if (
        not isinstance(seal, Mapping)
        or seal.get("format_version") != 1
        or not isinstance(seal.get("evaluation_identity_sha256"), str)
        or len(seal["evaluation_identity_sha256"]) != 64
        or set(seal["evaluation_identity_sha256"]) == {"0"}
        or not isinstance(artifacts, Mapping)
        or artifacts.get("summary_json") != sha256_file(request.validation_summary)
    ):
        raise ValueError("validation-40 success seal does not bind the summary")
    identity_sha256 = hashlib.sha256(
        json.dumps(dict(evaluation_identity), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if validation.get("evaluation_identity_sha256") != identity_sha256 or seal["evaluation_identity_sha256"] != identity_sha256:
        raise ValueError("validation-40 identity checksum is fabricated or changed")
    if not request.test_mode:
        from cosyvoice.finetune.balalaika.evaluation import verify_final_validation_evidence

        request_summary = getattr(request.validation_request, "summary_json", None)
        if not isinstance(request_summary, Path) or request_summary.resolve() != request.validation_summary.resolve():
            raise ValueError("production export validation_summary must be exactly Task 10 request.summary_json")

        evidence = verify_final_validation_evidence(
            request.validation_request,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_model_state_sha256=model_state_sha256,
            expected_base_checkpoint_sha256=str(adapter["base_checkpoint_sha256"]),
            expected_adapter_sha256=str(adapter["weights_sha256"]),
            wandb_logger=request.wandb_logger,
        )
    base_path = request.base_model_dir / "llm.pt"
    if not base_path.is_file() or sha256_file(base_path) != adapter["base_checkpoint_sha256"]:
        raise ValueError("requested base checkpoint differs from the final adapter")
    return {
        "adapter_dir": adapter_dir,
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": model_state_sha256,
        "validation_sha256": sha256_file(request.validation_summary),
        "validation_evidence": evidence if not request.test_mode else None,
    }


def _validate_phase2_training_identity(value: object, adapter: Mapping[str, object]) -> None:
    """Validate the exact immutable identity emitted by ``training._identity``."""

    from cosyvoice.finetune.balalaika.config import PhaseSpec

    fields = {
        "cache_manifest_sha256", "phase", "phase_spec", "eligible_samples",
        "base_checkpoint_sha256", "lora", "token_limit", "accumulation_steps",
        "max_grad_norm", "sampler_seed", "sampler_window_size", "dataloader_identity",
        "scheduler", "world_size",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("phase-2 training identity schema is invalid")
    _require_digest(value.get("cache_manifest_sha256"), "training cache manifest checksum")
    if value.get("phase") != 2 or value.get("phase_spec") != asdict(PhaseSpec.for_phase(2)):
        raise ValueError("phase-2 training schedule identity is invalid")
    if value.get("lora") != asdict(LoraSettings()) or value.get("lora") != adapter.get("settings"):
        raise ValueError("phase-2 training LoRA identity is invalid")
    if value.get("base_checkpoint_sha256") != adapter.get("base_checkpoint_sha256"):
        raise ValueError("phase-2 training base checksum differs from the adapter")
    if value.get("scheduler") != {"kind": "constant-v1"}:
        raise ValueError("phase-2 training scheduler identity is invalid")
    for name in ("eligible_samples", "token_limit", "accumulation_steps", "sampler_window_size"):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"phase-2 training {name} must be a positive integer")
    if value.get("world_size") != 8 or isinstance(value.get("world_size"), bool):
        raise ValueError("phase-2 training world_size must be exactly eight")
    seed = value.get("sampler_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("phase-2 training sampler_seed must be non-negative")
    norm = value.get("max_grad_norm")
    if isinstance(norm, bool) or not isinstance(norm, (int, float)) or not math.isfinite(float(norm)) or norm <= 0:
        raise ValueError("phase-2 training max_grad_norm must be finite and positive")
    loader = value.get("dataloader_identity")
    if not isinstance(loader, str) or not loader.strip():
        raise ValueError("phase-2 training dataloader identity must be nonempty")


def _require_original_key_layout(base_path: Path, merged_path: Path) -> None:
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    merged = torch.load(merged_path, map_location="cpu", weights_only=True)
    if not isinstance(base, Mapping) or not isinstance(merged, Mapping):
        raise ValueError("base and merged llm.pt files must contain state dictionaries")
    if set(base) != set(merged) or any("lora_" in name or ".base_layer." in name for name in merged):
        raise RuntimeError("merged checkpoint does not preserve the original state-dict key layout")


def _verify_adapter_active_logits(
    base_dir: Path,
    adapter_dir: Path,
    merged_path: Path,
    evidence_path: Path,
) -> dict[str, object]:
    """Prove the fresh adapter-active and standalone paths agree on fixed IDs."""

    from safetensors.torch import save_file

    token_ids, expected, actual = _compute_fixed_probe_tensors(base_dir, adapter_dir, merged_path)
    atol = 2e-2
    rtol = 2e-2
    metrics = _logit_metrics(expected, actual, atol, rtol)
    if not metrics["finite"]:
        raise RuntimeError("adapter-active and merged logits contain non-finite values")
    if not metrics["pass"]:
        raise RuntimeError("adapter-active and merged logits exceed BF16 tolerance")
    probe = {"token_ids": [[0, 0, 0]], "input_shape": [1, 3]}
    _atomic_safetensors(evidence_path, {
        "token_ids": token_ids,
        "adapter_active_logits": expected,
        "merged_logits": actual,
    }, save_file)
    return {
        "format_version": 2,
        "evidence": "logit-verification.safetensors",
        "evidence_sha256": sha256_file(evidence_path),
        "probe": probe,
        "probe_sha256": _canonical_mapping_sha256(probe),
        "atol": atol,
        "rtol": rtol,
        **metrics,
    }


def _compute_fixed_probe_tensors(
    base_dir: Path,
    adapter_dir: Path,
    merged_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh-load both model paths and return their exact fixed-probe evidence."""

    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter, _ = _require_exact_adapter_directory(adapter_dir)
    adapted = inject_lora(load_base_llm(base_dir), LoraSettings(**adapter["settings"])).eval()
    result = set_peft_model_state_dict(
        adapted,
        load_file(str(adapter_dir / adapter["weights"]), device="cpu"),
        adapter_name=ADAPTER_NAME,
    )
    missing = tuple(name for name in result.missing_keys if "lora_" in name)
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"adapter logits verification cannot load adapter: missing={missing}, unexpected={result.unexpected_keys}"
        )
    merged = load_base_llm(base_dir).eval()
    merged.load_state_dict(torch.load(merged_path, map_location="cpu", weights_only=True), strict=True)
    token_ids = torch.zeros((1, 3), dtype=torch.int64, device="cpu").contiguous()
    expected = _fixed_probe_logits(adapted).to(device="cpu", dtype=torch.float32).contiguous()
    actual = _fixed_probe_logits(merged).to(device="cpu", dtype=torch.float32).contiguous()
    return token_ids, expected, actual


def _logit_metrics(
    expected: torch.Tensor,
    actual: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    difference = (expected - actual).abs()
    denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).eps)
    tolerance = atol + rtol * actual.abs()
    max_abs_error = float(difference.max().item())
    max_relative_error = float((difference / denominator).max().item())
    max_tolerance_ratio = float((difference / tolerance).max().item())
    passed = finite and max_tolerance_ratio <= 1.0
    return {
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "max_tolerance_ratio": max_tolerance_ratio,
        "finite": finite,
        "pass": passed,
    }


def _fixed_probe_logits(model: CosyVoice3LM) -> torch.Tensor:
    embedding = model.llm.get_input_embeddings()
    device = embedding.weight.device
    token_ids = torch.zeros((1, 3), dtype=torch.long, device=device)
    embedded = embedding(token_ids)
    hidden, _ = model.llm(embedded, torch.tensor([3], dtype=torch.long, device=device))
    return model.llm_decoder(hidden).detach().float().cpu()


def _publish_directory(temporary: Path, destination: Path) -> None:
    directory_fd = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.replace(temporary, destination)
    parent_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _require_final_manifest(manifest: FinalModelManifest, base_model_dir: Path) -> None:
    committed = require_committed_final(
        manifest.path.parent,
        base_model_dir,
        expected_mode=manifest.mode,
    )
    if committed != manifest:
        raise ValueError("final model manifest object differs from committed evidence")


def _require_staged_final_inputs(manifest: FinalModelManifest) -> None:
    """Authenticate immutable inputs used before a final manifest exists."""

    if not manifest.llm_path.is_file() or manifest.llm_path.is_symlink() or sha256_file(manifest.llm_path) != manifest.llm_sha256:
        raise ValueError("staged final llm.pt checksum changed")
    adapter, adapter_manifest_sha256 = _require_exact_adapter_directory(manifest.adapter_dir)
    if (
        adapter_manifest_sha256 != manifest.adapter_manifest_sha256
        or adapter["weights_sha256"] != manifest.adapter_weights_sha256
        or adapter["base_checkpoint_sha256"] != manifest.base_checkpoint_sha256
        or tuple(adapter["target_modules"]) != manifest.target_modules
        or adapter["settings"] != asdict(LoraSettings())
    ):
        raise ValueError("staged retained adapter provenance differs from the final manifest")


def _require_production_recognizer(recognizer: object, expected_local_rank: int) -> dict[str, object]:
    from cosyvoice.finetune.balalaika.evaluation import GigaAmRecognizer, require_gigaam_cuda_runtime

    if type(recognizer) is not GigaAmRecognizer:
        raise TypeError("production strict verification requires the exact Task 10 GigaAmRecognizer class")
    provenance_method = getattr(recognizer, "provenance", None)
    if not callable(provenance_method):
        raise TypeError("production GigaAM recognizer has no provenance")
    provenance = provenance_method()
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "model", "provider", "device_id", "max_batch_size"
    }:
        raise ValueError("production GigaAM provenance schema is invalid")
    batch = provenance.get("max_batch_size")
    if (
        provenance.get("model") != "gigaam-v3-rnnt"
        or provenance.get("provider") != "CUDAExecutionProvider"
        or provenance.get("device_id") != expected_local_rank
        or getattr(recognizer, "local_rank", None) != expected_local_rank
        or isinstance(batch, bool)
        or not isinstance(batch, int)
        or batch < 1
        or getattr(recognizer, "max_batch_size", None) != batch
    ):
        raise ValueError("production GigaAM provenance does not match the expected CUDA local rank")
    require_gigaam_cuda_runtime(getattr(recognizer, "model", None), expected_local_rank)
    return dict(provenance)


def _strict_smoke_voices(voices: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    values: list[dict[str, object]] = []
    for voice in voices:
        if not isinstance(voice, Mapping):
            raise TypeError("reserved voices must be mappings")
        voice_id = voice.get("voice_id")
        prompt_text = voice.get("prompt_text")
        prompt_wav = voice.get("prompt_wav")
        prompt_sha256 = voice.get("prompt_sha256")
        path = Path(prompt_wav) if isinstance(prompt_wav, (str, Path)) else None
        if not isinstance(voice_id, str) or not voice_id or not isinstance(prompt_text, str) or not prompt_text.strip() or path is None or not path.is_file():
            raise ValueError("reserved voice is incomplete")
        values.append({"voice_id": voice_id, "prompt_text": prompt_text, "prompt_wav": path, "prompt_sha256": prompt_sha256})
    expected = {f"voice_{index:02d}" for index in range(20)}
    if {value["voice_id"] for value in values} != expected or len(values) != 20:
        raise ValueError("strict verification requires the approved voice_00 through voice_19 inventory")
    for value in values:
        checksum = value.get("prompt_sha256")
        if not isinstance(checksum, str) or checksum != sha256_file(value["prompt_wav"]):
            raise ValueError(f"reserved voice prompt checksum changed for {value['voice_id']}")
        if not _is_pcm_input(value["prompt_wav"]):
            raise ValueError(f"reserved voice prompt is not nonempty 24 kHz mono PCM for {value['voice_id']}")
    return tuple(sorted(values, key=lambda value: str(value["voice_id"])))


def _is_pcm_input(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as stream:
            return stream.getframerate() == 24_000 and stream.getnchannels() == 1 and stream.getnframes() > 0
    except wave.Error:
        return False


def _make_verification_view(base_dir: Path, llm_path: Path, view: Path) -> None:
    if not (base_dir / "cosyvoice3.yaml").is_file():
        raise FileNotFoundError("base CosyVoice3 configuration is missing")
    for source in base_dir.iterdir():
        destination = view / source.name
        if source.name == "llm.pt":
            shutil.copy2(llm_path, destination)
        else:
            os.symlink(source.resolve(), destination, target_is_directory=source.is_dir())


def _normal_cosyvoice3_pipeline(view: Path, factory: object | None) -> object:
    if factory is None:
        from cosyvoice.cli.cosyvoice import CosyVoice3

        factory = CosyVoice3
    if not callable(factory):
        raise TypeError("pipeline_factory must be callable")
    return factory(view)


def _require_frozen_inference_components(pipeline: object) -> None:
    if getattr(pipeline, "sample_rate", None) != 24_000:
        raise ValueError("normal CosyVoice3 pipeline must report 24 kHz audio")
    model = getattr(pipeline, "model", None)
    for name in ("flow", "hift"):
        component = getattr(model, name, None)
        parameters = getattr(component, "parameters", None)
        if component is None or not callable(parameters) or any(parameter.requires_grad for parameter in parameters()):
            raise ValueError(f"normal CosyVoice3 pipeline {name} is not frozen")


def _strict_load_pipeline_llm(pipeline: object, llm_path: Path) -> None:
    llm = getattr(getattr(pipeline, "model", None), "llm", None)
    loader = getattr(llm, "load_state_dict", None)
    if not callable(loader):
        raise TypeError("normal CosyVoice3 pipeline has no LLM strict loader")
    result = loader(torch.load(llm_path, map_location="cpu", weights_only=True), strict=True)
    missing = getattr(result, "missing_keys", ())
    unexpected = getattr(result, "unexpected_keys", ())
    if missing or unexpected:
        raise RuntimeError(f"normal CosyVoice3 strict load mismatch: missing={missing}, unexpected={unexpected}")


def _synthesize_smoke_voice(pipeline: object, voice: Mapping[str, object], prompt: str, destination: Path) -> Path:
    register = getattr(pipeline, "add_zero_shot_spk", None)
    synthesize = getattr(pipeline, "inference_zero_shot", None)
    if not callable(register) or not callable(synthesize):
        raise TypeError("normal CosyVoice3 pipeline lacks zero-shot voice-cloning APIs")
    speaker_id = f"strict-verification:{voice['voice_id']}"
    if register(voice["prompt_text"], str(voice["prompt_wav"]), speaker_id) is not True:
        raise RuntimeError(f"could not register reserved voice {voice['voice_id']}")
    chunks: list[torch.Tensor] = []
    for output in synthesize(prompt, voice["prompt_text"], str(voice["prompt_wav"]), zero_shot_spk_id=speaker_id, stream=False):
        speech = output.get("tts_speech") if isinstance(output, Mapping) else None
        if speech is None:
            raise RuntimeError("normal CosyVoice3 inference output lacks tts_speech")
        samples = torch.as_tensor(speech, dtype=torch.float32).detach().cpu().reshape(-1)
        if samples.numel() < 1 or not torch.isfinite(samples).all():
            raise RuntimeError("normal CosyVoice3 inference emitted invalid audio")
        chunks.append(samples)
    if not chunks:
        raise RuntimeError("normal CosyVoice3 inference emitted no audio")
    samples = torch.cat(chunks).clamp_(-1.0, 1.0)
    pcm = (samples * 32767.0).round().to(torch.int16).numpy().tobytes()
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24_000)
        stream.writeframes(pcm)
    return destination


def _require_pcm_24khz_mono(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            with wave.open(str(path), "rb") as stream:
                valid = stream.getframerate() == 24_000 and stream.getnchannels() == 1 and stream.getsampwidth() == 2 and stream.getnframes() > 0
        except wave.Error as exc:
            raise RuntimeError(f"strict verification audio is invalid: {path}") from exc
        if not valid:
            raise RuntimeError(f"strict verification audio is not nonempty 24 kHz mono PCM: {path}")


def _library_identities(recognizer: object, pipeline: object) -> dict[str, object]:
    identities: dict[str, object] = {
        "torch": torch.__version__,
        "pipeline_class": f"{type(pipeline).__module__}.{type(pipeline).__qualname__}",
        "recognizer_class": f"{type(recognizer).__module__}.{type(recognizer).__qualname__}",
    }
    for package in ("peft", "transformers", "onnx-asr"):
        try:
            identities[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            identities[package] = "unavailable"
    provenance = getattr(recognizer, "provenance", None)
    if callable(provenance):
        value = provenance()
        if isinstance(value, Mapping):
            identities["gigaam"] = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return identities


def _merge_and_remove_adapter_layers(model: nn.Module, tuner_layer_type: type[nn.Module]) -> None:
    target_names = [name for name, module in model.named_modules() if name and isinstance(module, tuner_layer_type)]
    for name in sorted(target_names, key=lambda value: value.count("."), reverse=True):
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = getattr(parent, child_name)
        layer.merge(safe_merge=True, adapter_names=[ADAPTER_NAME])
        setattr(parent, child_name, layer.get_base_layer())


def _read_adapter_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid adapter manifest: {path}") from exc
    required = {
        "format_version",
        "weights",
        "weights_sha256",
        "base_checkpoint_sha256",
        "target_modules",
        "settings",
        "code_revision",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"invalid adapter manifest: {path}")
    if value.get("format_version") != 1:
        raise ValueError(f"invalid adapter manifest format version: {path}")
    if value["weights"] != ADAPTER_WEIGHTS_NAME:
        raise ValueError("adapter manifest weights path must be a file name")
    for name in ("weights_sha256", "base_checkpoint_sha256"):
        digest = value.get(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"adapter {name} is invalid")
    if (
        not isinstance(value["target_modules"], list)
        or not value["target_modules"]
        or not all(isinstance(item, str) and item for item in value["target_modules"])
        or len(value["target_modules"]) != len(set(value["target_modules"]))
    ):
        raise ValueError("adapter target inventory is invalid")
    if any(_is_internal_qwen_lm_head_path(item) for item in value["target_modules"]):
        raise ValueError("adapter target inventory contains the internal Qwen lm_head subtree")
    if not isinstance(value["settings"], dict) or value["settings"] != asdict(LoraSettings()):
        raise ValueError("adapter settings are invalid")
    if not isinstance(value.get("code_revision"), str) or not value["code_revision"].strip():
        raise ValueError("adapter code metadata is invalid")
    return value


def _require_exact_adapter_directory(path: Path) -> tuple[dict[str, object], str]:
    """Authenticate the complete two-file adapter artifact without following links."""

    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("adapter directory must be a regular directory without symlinks")
    entries = tuple(path.iterdir())
    expected = {ADAPTER_MANIFEST_NAME, ADAPTER_WEIGHTS_NAME}
    if len(entries) != 2 or {entry.name for entry in entries} != expected:
        raise ValueError("adapter directory inventory must contain exactly manifest and weights")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("adapter directory inventory cannot contain symlinks or non-regular files")
    manifest_path = path / ADAPTER_MANIFEST_NAME
    manifest = _read_adapter_manifest(manifest_path)
    weights_path = path / ADAPTER_WEIGHTS_NAME
    if sha256_file(weights_path) != manifest["weights_sha256"]:
        raise ValueError("adapter weights checksum does not match its manifest")
    return manifest, sha256_file(manifest_path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor], save_file) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        save_file(tensors, str(temporary_path))
        _publish_temporary(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, value: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        torch.save(value, temporary_path)
        _publish_temporary(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_temporary(temporary: Path, destination: Path) -> None:
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _code_revision() -> str:
    repository = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else "unknown"


def _code_identity() -> dict[str, object]:
    """Return a JSON-safe identity for the exact checked-out code state."""

    repository = Path(__file__).resolve().parents[3]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=False, capture_output=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"], cwd=repository, check=False, capture_output=True
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "."], cwd=repository, check=False, capture_output=True
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if any(result.returncode != 0 for result in (head, status, diff, untracked)):
        raise RuntimeError("unable to determine final export code identity")
    revision = head.stdout.decode("utf-8", errors="strict").strip()
    if not revision:
        raise RuntimeError("unable to determine final export Git HEAD")
    dirty = bool(status.stdout)
    digest = hashlib.sha256()
    digest.update(status.stdout)
    digest.update(b"\0")
    digest.update(diff.stdout)
    paths = tuple(sorted(path for path in untracked.stdout.split(b"\0") if path))
    if len(paths) > 1_000:
        raise RuntimeError("code identity has too many untracked files")
    total_size = 0
    for raw_path in paths:
        relative = Path(os.fsdecode(raw_path))
        path = repository / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(repository):
            raise RuntimeError("code identity contains an unsafe untracked path")
        total_size += path.stat().st_size
        if total_size > 100 * 1024 * 1024:
            raise RuntimeError("code identity untracked content exceeds safe bound")
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return {
        "head": revision,
        "dirty": dirty,
        "diff_sha256": digest.hexdigest(),
    }
