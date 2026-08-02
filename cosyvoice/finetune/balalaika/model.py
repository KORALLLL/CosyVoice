"""CosyVoice3 base-model loading and broad adapter lifecycle helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
import json
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

    def __post_init__(self) -> None:
        for name in ("base_model_dir", "phase2_checkpoint", "validation_summary", "output_dir"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a pathlib.Path")


@dataclass(frozen=True)
class FinalModelManifest:
    """Complete, checksum-bound standalone LLM export inventory."""

    path: Path
    llm_path: Path
    adapter_dir: Path
    llm_sha256: str
    base_checkpoint_sha256: str
    adapter_weights_sha256: str
    target_modules: tuple[str, ...]
    phase2_checkpoint_sha256: str
    validation_summary_sha256: str


@dataclass(frozen=True)
class VerifyRequest:
    """Dependencies for normal-path standalone CosyVoice3 smoke verification."""

    base_model_dir: Path
    final_manifest: FinalModelManifest
    recognizer: object
    voices: Sequence[Mapping[str, object]]
    output_dir: Path
    pipeline_factory: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_model_dir, Path) or not isinstance(self.output_dir, Path):
            raise TypeError("verification paths must be pathlib.Path values")
        if not isinstance(self.final_manifest, FinalModelManifest):
            raise TypeError("final_manifest must be FinalModelManifest")
        if not isinstance(self.voices, Sequence) or isinstance(self.voices, (str, bytes)):
            raise TypeError("voices must be a sequence of reserved voice mappings")
        if len(self.voices) != 4:
            raise ValueError("strict verification requires exactly four reserved voices")


@dataclass(frozen=True)
class VerificationReport:
    """Durable evidence that a standalone LLM works without an adapter runtime."""

    path: Path
    strict_load: bool
    smoke_utterances: int
    audio_paths: tuple[Path, ...]
    audio_checksums: Mapping[str, str]
    asr_results: tuple[str, ...]
    library_identities: Mapping[str, str]


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
    manifest = _read_adapter_manifest(adapter_dir / ADAPTER_MANIFEST_NAME)
    weights_path = adapter_dir / manifest["weights"]
    if sha256_file(weights_path) != manifest["weights_sha256"]:
        raise ValueError("adapter weights checksum does not match its manifest")

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
    output_dir = request.output_dir
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"final model output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        copied_adapter = temporary / "adapter"
        shutil.copytree(lineage["adapter_dir"], copied_adapter)
        merged_path = temporary / "llm.pt"
        merge = merge_adapter(request.base_model_dir, copied_adapter, merged_path)
        _require_original_key_layout(request.base_model_dir / "llm.pt", merged_path)
        _verify_adapter_active_logits(request.base_model_dir, copied_adapter, merged_path)
        payload = {
            "format_version": 1,
            "llm": "llm.pt",
            "llm_sha256": merge.output_sha256,
            "adapter": "adapter",
            "adapter_weights_sha256": merge.adapter_weights_sha256,
            "base_checkpoint_sha256": merge.base_checkpoint_sha256,
            "target_modules": list(merge.target_modules),
            "phase2_checkpoint": str(request.phase2_checkpoint),
            "phase2_checkpoint_manifest_sha256": lineage["checkpoint_sha256"],
            "validation_summary": str(request.validation_summary),
            "validation_summary_sha256": lineage["validation_sha256"],
            "code_revision": _code_revision(),
        }
        manifest_path = temporary / "final_model_manifest.json"
        atomic_write_json(manifest_path, payload)
        _publish_directory(temporary, output_dir)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    return FinalModelManifest(
        path=output_dir / "final_model_manifest.json",
        llm_path=output_dir / "llm.pt",
        adapter_dir=output_dir / "adapter",
        llm_sha256=merge.output_sha256,
        base_checkpoint_sha256=merge.base_checkpoint_sha256,
        adapter_weights_sha256=merge.adapter_weights_sha256,
        target_modules=merge.target_modules,
        phase2_checkpoint_sha256=lineage["checkpoint_sha256"],
        validation_summary_sha256=lineage["validation_sha256"],
    )


def strict_verify_final_model(request: VerifyRequest) -> VerificationReport:
    """Strict-load and smoke-test a final LLM through normal CosyVoice3 APIs."""

    if not isinstance(request, VerifyRequest):
        raise TypeError("request must be VerifyRequest")
    _require_final_manifest(request.final_manifest)
    voices = _strict_smoke_voices(request.voices)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = request.output_dir / "audio"
    if audio_dir.exists() or audio_dir.is_symlink():
        raise FileExistsError(f"strict verification audio output already exists: {audio_dir}")
    with tempfile.TemporaryDirectory(prefix=".cosyvoice3-strict-", dir=request.output_dir) as raw_view:
        view = Path(raw_view)
        _make_verification_view(request.base_model_dir, request.final_manifest.llm_path, view)
        pipeline = _normal_cosyvoice3_pipeline(view, request.pipeline_factory)
        _require_frozen_inference_components(pipeline)
        _strict_load_pipeline_llm(pipeline, request.final_manifest.llm_path)
        audio_dir.mkdir()
        random.seed(1986)
        torch.manual_seed(1986)
        audio_paths = tuple(
            _synthesize_smoke_voice(pipeline, voice, prompt, audio_dir / f"smoke-{index + 1:02d}.wav")
            for index, (voice, prompt) in enumerate(zip(voices, _SMOKE_PROMPTS, strict=True))
        )
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
    libraries = _library_identities(request.recognizer)
    payload = {
        "format_version": 1,
        "final_model_manifest_sha256": sha256_file(request.final_manifest.path),
        "llm_sha256": request.final_manifest.llm_sha256,
        "strict_load": True,
        "smoke_utterances": 4,
        "audio": [
            {"voice_id": voice["voice_id"], "prompt": prompt, "path": str(path), "sha256": checksums[path.name], "asr": asr}
            for voice, prompt, path, asr in zip(voices, _SMOKE_PROMPTS, audio_paths, hypotheses, strict=True)
        ],
        "libraries": libraries,
    }
    report_path = request.output_dir / "strict-verification.json"
    atomic_write_json(report_path, payload)
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
    adapter = _read_adapter_manifest(adapter_dir / ADAPTER_MANIFEST_NAME)
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
        or not isinstance(artifacts, Mapping)
        or artifacts.get("summary_json") != sha256_file(request.validation_summary)
    ):
        raise ValueError("validation-40 success seal does not bind the summary")
    base_path = request.base_model_dir / "llm.pt"
    if not base_path.is_file() or sha256_file(base_path) != adapter["base_checkpoint_sha256"]:
        raise ValueError("requested base checkpoint differs from the final adapter")
    return {
        "adapter_dir": adapter_dir,
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_sha256": sha256_file(request.validation_summary),
    }


def _require_original_key_layout(base_path: Path, merged_path: Path) -> None:
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    merged = torch.load(merged_path, map_location="cpu", weights_only=True)
    if not isinstance(base, Mapping) or not isinstance(merged, Mapping):
        raise ValueError("base and merged llm.pt files must contain state dictionaries")
    if set(base) != set(merged) or any("lora_" in name or ".base_layer." in name for name in merged):
        raise RuntimeError("merged checkpoint does not preserve the original state-dict key layout")


def _verify_adapter_active_logits(base_dir: Path, adapter_dir: Path, merged_path: Path) -> None:
    """Prove the fresh adapter-active and standalone paths agree on fixed IDs."""

    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter = _read_adapter_manifest(adapter_dir / ADAPTER_MANIFEST_NAME)
    adapted = inject_lora(load_base_llm(base_dir), LoraSettings(**adapter["settings"])).eval()
    result = set_peft_model_state_dict(
        adapted,
        load_file(str(adapter_dir / adapter["weights"]), device="cpu"),
        adapter_name=ADAPTER_NAME,
    )
    missing = tuple(name for name in result.missing_keys if "lora_" in name)
    if missing or result.unexpected_keys:
        raise RuntimeError(f"adapter logits verification cannot load adapter: missing={missing}, unexpected={result.unexpected_keys}")
    merged = load_base_llm(base_dir).eval()
    merged.load_state_dict(torch.load(merged_path, map_location="cpu", weights_only=True), strict=True)
    expected = _fixed_probe_logits(adapted)
    actual = _fixed_probe_logits(merged)
    if not torch.allclose(expected, actual, rtol=2e-2, atol=2e-2):
        raise RuntimeError("adapter-active and merged logits exceed BF16 tolerance")


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


def _require_final_manifest(manifest: FinalModelManifest) -> None:
    try:
        payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("final model manifest is missing or invalid") from exc
    required = {
        "format_version", "llm", "llm_sha256", "adapter", "adapter_weights_sha256",
        "base_checkpoint_sha256", "target_modules", "phase2_checkpoint",
        "phase2_checkpoint_manifest_sha256", "validation_summary",
        "validation_summary_sha256", "code_revision",
    }
    if not isinstance(payload, Mapping) or set(payload) != required or payload.get("format_version") != 1:
        raise ValueError("final model manifest schema is invalid")
    if payload["llm"] != manifest.llm_path.name or payload["adapter"] != manifest.adapter_dir.name:
        raise ValueError("final model manifest paths changed")
    expected = {
        "llm_sha256": manifest.llm_sha256,
        "base_checkpoint_sha256": manifest.base_checkpoint_sha256,
        "adapter_weights_sha256": manifest.adapter_weights_sha256,
        "phase2_checkpoint_manifest_sha256": manifest.phase2_checkpoint_sha256,
        "validation_summary_sha256": manifest.validation_summary_sha256,
        "target_modules": list(manifest.target_modules),
    }
    if any(payload[name] != value for name, value in expected.items()):
        raise ValueError("final model manifest provenance changed")
    if not manifest.llm_path.is_file() or sha256_file(manifest.llm_path) != manifest.llm_sha256:
        raise ValueError("final llm.pt checksum changed")
    adapter = _read_adapter_manifest(manifest.adapter_dir / ADAPTER_MANIFEST_NAME)
    if (
        adapter["weights_sha256"] != manifest.adapter_weights_sha256
        or adapter["base_checkpoint_sha256"] != manifest.base_checkpoint_sha256
        or tuple(adapter["target_modules"]) != manifest.target_modules
    ):
        raise ValueError("retained adapter provenance differs from the final manifest")


def _strict_smoke_voices(voices: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    values: list[dict[str, object]] = []
    for voice in voices:
        if not isinstance(voice, Mapping):
            raise TypeError("reserved voices must be mappings")
        voice_id = voice.get("voice_id")
        prompt_text = voice.get("prompt_text")
        prompt_wav = voice.get("prompt_wav")
        path = Path(prompt_wav) if isinstance(prompt_wav, (str, Path)) else None
        if not isinstance(voice_id, str) or not voice_id or not isinstance(prompt_text, str) or not prompt_text.strip() or path is None or not path.is_file():
            raise ValueError("reserved voice is incomplete")
        values.append({"voice_id": voice_id, "prompt_text": prompt_text, "prompt_wav": path})
    if len({value["voice_id"] for value in values}) != 4:
        raise ValueError("strict verification voices must have four unique IDs")
    return tuple(values)


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


def _library_identities(recognizer: object) -> dict[str, str]:
    identities = {"torch": torch.__version__, "code_revision": _code_revision()}
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
        "weights",
        "weights_sha256",
        "base_checkpoint_sha256",
        "target_modules",
        "settings",
        "code_revision",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"invalid adapter manifest: {path}")
    if value.get("format_version") != 1:
        raise ValueError(f"invalid adapter manifest format version: {path}")
    if not isinstance(value["weights"], str) or Path(value["weights"]).name != value["weights"]:
        raise ValueError("adapter manifest weights path must be a file name")
    if not isinstance(value["target_modules"], list) or not all(isinstance(item, str) for item in value["target_modules"]):
        raise ValueError("adapter target inventory is invalid")
    if any(_is_internal_qwen_lm_head_path(item) for item in value["target_modules"]):
        raise ValueError("adapter target inventory contains the internal Qwen lm_head subtree")
    if not isinstance(value["settings"], dict):
        raise ValueError("adapter settings are invalid")
    return value


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
