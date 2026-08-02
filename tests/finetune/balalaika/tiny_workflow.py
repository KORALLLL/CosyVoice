"""Reusable CPU-only integration harness for the Balalaika recipe.

The expensive boundaries are deterministic fakes, while source routing, atomic
cache publication, mmap data reads, LoRA, checkpoint/resume, and merge use the
same implementation interfaces as production.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass, replace
from itertools import islice
import io
import json
from pathlib import Path
import tarfile
from typing import Any, Iterable, Mapping, Sequence
from unittest import mock

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cosyvoice.finetune.balalaika import cache as cache_api
from cosyvoice.finetune.balalaika import model as model_api
from cosyvoice.finetune.balalaika import sources as sources_api
from cosyvoice.finetune.balalaika import training as training_api
from cosyvoice.finetune.balalaika.artifacts import StageRequirementError, StageStore, atomic_write_json, sha256_file
from cosyvoice.finetune.balalaika.cache import AudioInput, CacheManifest, CacheShardRequest
from cosyvoice.finetune.balalaika.config import PhaseSpec
from cosyvoice.finetune.balalaika.data import CachedSpeechDataset
from cosyvoice.finetune.balalaika.memorization import SampleAccuracy, memorization_passed, select_memorization_rows
from cosyvoice.finetune.balalaika.model import LoraSettings, TrainableAudit
from cosyvoice.finetune.balalaika.sources import JoinedRow, assign_phases, reserve_prompt_ids
from cosyvoice.finetune.balalaika.training import TrainRequest, TrainingCallbacks, train_phase
from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder


@dataclass(frozen=True)
class TinyWorkflowResult:
    stage: str
    final_llm: Path
    strict_load_verified: bool
    validation_indices: tuple[int, ...]
    phase_epochs: tuple[int, int]
    boundaries_per_epoch: int
    pilot_stopped_before_approval: bool
    pilot_checksum_approved: bool
    memorization_passed: bool
    cache_resumed_after_interruption: bool
    phase_resumed_after_interruption: bool
    wandb_mode: str
    upload_calls: int


class _TinySpeechTokenizer:
    def extract(self, audio: list[AudioInput]) -> list[list[int]]:
        return [[1 + (index % 4) for index in range(sample.frames % 4 + 1)] for sample in audio]


class _TinyTextTokenizer:
    def encode(self, value: str, allowed_special: str = "all") -> list[int]:
        del allowed_special
        return [1 + (ord(character) % 29) for character in value]


class _FakeSynthesizer:
    def synthesize(self, text: str, voice_id: str) -> bytes:
        return f"{voice_id}\0{text}".encode("utf-8")


class _FakeAsr:
    def transcribe(self, audio: bytes) -> str:
        return audio.decode("utf-8").split("\0", 1)[1]


class _OfflineWandb:
    mode = "offline"

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def log(self, value: Mapping[str, object]) -> None:
        self.records.append(dict(value))


class _TinyEvaluator:
    def __init__(self, benchmark_rows: int) -> None:
        self.rows = tuple(f"число {index}" for index in range(benchmark_rows))
        self.synthesizer = _FakeSynthesizer()
        self.asr = _FakeAsr()
        self.wandb = _OfflineWandb()
        self.indices: list[int] = []

    def evaluate(self, validation_index: int) -> bool:
        exact = 0
        for row in self.rows:
            audio = self.synthesizer.synthesize(row, "voice-00")
            exact += int(self.asr.transcribe(audio) == row)
        self.indices.append(validation_index)
        self.wandb.log({"validation/index": validation_index, "utt-cer": 0.0, "exact": exact})
        return True


class _TrainableScalar(torch.nn.Module):
    """Tiny adapter-shaped model used only to exercise production trainer state."""

    def __init__(self) -> None:
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor(0.0))
        self._balalaika_base_checkpoint_sha256 = "b" * 64
        self._balalaika_lora_settings = LoraSettings()
        self._balalaika_target_modules = ("tiny",)

    def forward(self, batch: Mapping[str, Any], _device: torch.device) -> Mapping[str, torch.Tensor]:
        target = batch["target"].float()
        return {"loss": ((self.lora_weight - target) ** 2).mean()}


class _CpuAccelerator:
    """Eight-rank collective semantics represented on one CPU test process."""

    def __init__(self, **kwargs: object) -> None:
        self.gradient_accumulation_steps = int(kwargs.get("gradient_accumulation_steps", 1))
        self.num_processes = 8
        self.process_index = 0
        self.is_main_process = True
        self.device = torch.device("cpu")
        self.sync_gradients = True
        self._checkpointables: list[object] = []
        self._prepared: tuple[object, ...] = ()
        self._last_batch = False
        self._accumulated_batches = 0
        self._last_global_samples = 0

    def prepare(self, *values: object) -> tuple[object, ...]:
        self._prepared = values
        return values

    def register_for_checkpointing(self, value: object) -> None:
        self._checkpointables.append(value)

    @contextmanager
    def accumulate(self, _model: torch.nn.Module):
        self._accumulated_batches += 1
        self.sync_gradients = self._accumulated_batches % self.gradient_accumulation_steps == 0 or self._last_batch
        if self.sync_gradients:
            self._accumulated_batches = 0
        yield

    def prepare_rank_dataloader(self, dataloader: Iterable[Mapping[str, object]]):
        def prepared():
            iterator = iter(dataloader)
            try:
                current = next(iterator)
            except StopIteration:
                return
            while True:
                try:
                    following = next(iterator)
                except StopIteration:
                    self._last_batch = True
                    yield current
                    self._last_batch = False
                    return
                self._last_batch = False
                yield current
                current = following

        return prepared()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def clip_grad_norm_(self, parameters: Iterable[torch.nn.Parameter], max_norm: float):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def reduce(self, value: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        if reduction != "sum":
            raise AssertionError(reduction)
        self._last_global_samples = int(value.item())
        return value

    def gather_sample_counts(self, local_samples: int) -> tuple[int, ...]:
        return (local_samples, 0, 0, 0, 0, 0, 0, 0)

    def gather_object(self, value: Sequence[object]) -> list[object]:
        return [value[0], None, None, None, None, None, None, None]

    def wait_for_everyone(self) -> None:
        return None

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def save_state(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model, optimizer, scheduler = self._prepared
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "checkpointables": [value.state_dict() for value in self._checkpointables],
                "rng": torch.get_rng_state(),
            },
            output / "fake_accelerate_state.pt",
        )
        torch.save(torch.get_rng_state(), output / "random_states_0.pkl")

    def load_state(self, input_dir: str | Path) -> None:
        state = torch.load(Path(input_dir) / "fake_accelerate_state.pt", weights_only=False)
        model, optimizer, scheduler = self._prepared
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        for value, saved in zip(self._checkpointables, state["checkpointables"], strict=True):
            value.load_state_dict(saved)
        torch.set_rng_state(state["rng"])

    def skip_first_batches(self, dataloader: Iterable[Mapping[str, object]], count: int):
        return islice(dataloader, count, None)

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        return value.repeat(self.num_processes)


def run_tiny_workflow(
    root: Path,
    *,
    source_shards: int,
    phase1_rows: int,
    phase2_rows: int,
    reserved_prompts: int,
    benchmark_rows: int,
) -> TinyWorkflowResult:
    """Run a scaled workflow while retaining the production schedules."""

    if (source_shards, phase1_rows, phase2_rows, reserved_prompts, benchmark_rows) != (2, 8, 8, 2, 8):
        raise ValueError("the checked-in tiny fixture has one exact supported scale")
    root.mkdir(parents=True, exist_ok=True)
    stage_store = StageStore(root / "workflow_stages", {"fixture": "cpu-v1"}, {"seed": 1986})

    pilot_root = root / "pilot"
    pilot_root.mkdir()
    (pilot_root / "original.wav").write_bytes(b"RIFF-original")
    (pilot_root / "reconstructed.wav").write_bytes(b"RIFF-reconstructed")
    pilot_manifest = pilot_root / "manifest.json"
    atomic_write_json(
        pilot_manifest,
        {
            "original_sha256": sha256_file(pilot_root / "original.wav"),
            "reconstructed_sha256": sha256_file(pilot_root / "reconstructed.wav"),
        },
    )
    pilot = stage_store.publish("pilot", {"manifest": str(pilot_manifest), "manifest_sha256": sha256_file(pilot_manifest)})
    try:
        stage_store.require("pilot_approved")
    except StageRequirementError:
        pilot_stopped = True
    else:
        pilot_stopped = False
    approved_checksum = pilot.payload["manifest_sha256"]
    if approved_checksum != sha256_file(pilot_manifest):
        raise AssertionError("pilot checksum changed before approval")
    stage_store.publish("pilot_approved", {"pilot_manifest_sha256": approved_checksum})

    rows, selected = _tiny_join_rows(source_shards, phase1_rows, phase2_rows, reserved_prompts)
    plan_dir, tar_paths = _write_source_fixture(root, rows, selected, source_shards)
    cache_root = root / "cache"
    interrupted = False
    original_replace = cache_api._replace_staged

    def interrupt_once(source: Path, destination: Path) -> None:
        nonlocal interrupted
        if not interrupted and Path(destination).name == "shard_000000.parquet" and Path(destination).parent.name == "phase2":
            interrupted = True
            raise OSError("simulated cache interruption")
        original_replace(source, destination)

    with mock.patch.object(sources_api, "EXPECTED_SOURCE_TARS", source_shards), mock.patch.object(
        cache_api, "PROMPT_RESERVATION_COUNT", reserved_prompts
    ), mock.patch.object(cache_api, "_decode_audio", side_effect=_decode_fixture):
        request = CacheShardRequest(tar_paths[0], plan_dir, cache_root, 0, batch_size=3)
        with mock.patch.object(cache_api, "_replace_staged", side_effect=interrupt_once):
            try:
                cache_api.build_cache_shard(request, _TinySpeechTokenizer())
            except OSError as exc:
                if "simulated cache interruption" not in str(exc):
                    raise
            else:
                raise AssertionError("cache interruption fixture did not fire")
        cache_api.build_cache_shard(request, _TinySpeechTokenizer())
        for shard in range(1, source_shards):
            cache_api.build_cache_shard(
                CacheShardRequest(tar_paths[shard], plan_dir, cache_root, shard, batch_size=3),
                _TinySpeechTokenizer(),
            )
        cache = cache_api.verify_cache(cache_root)
    if list(cache_root.rglob("*.partial")) or list(cache_root.rglob("*.rollback")):
        raise AssertionError("cache resume left transaction debris")

    text_tokenizer = _TinyTextTokenizer()
    phase_datasets = {
        phase: CachedSpeechDataset(cache, phase, tokenizer=text_tokenizer)
        for phase in (1, 2)
    }
    selected_memory = select_memorization_rows(plan_dir, cache, seed=1986)
    accuracy_checks = tuple(
        tuple(SampleAccuracy(row.source_relative_path, row.speech_token_len, row.speech_token_len) for row in selected_memory)
        for _ in range(3)
    )
    memory_ok = memorization_passed(accuracy_checks)
    if not memory_ok:
        raise AssertionError("tiny four-audio memorization evidence did not pass")
    stage_store.publish("memorization_complete", {"rows": 4, "consecutive_exact_checks": 3})

    evaluator = _TinyEvaluator(benchmark_rows)
    evaluator.evaluate(0)
    train_model = _TrainableScalar()
    phase1_request = _training_request(root, train_model, cache, phase_datasets[1], PhaseSpec.for_phase(1), None)
    with mock.patch.object(training_api, "audit_trainable_parameters", side_effect=_tiny_audit), mock.patch.object(
        training_api, "save_adapter", side_effect=_tiny_save_adapter
    ):
        first = train_phase(
            phase1_request,
            TrainingCallbacks(validate=lambda event: evaluator.evaluate(event.validation_index) and event.validation_index < 3),
        )
        if first.completed or first.checkpoint is None:
            raise AssertionError("phase interruption fixture did not stop")
        resumed = train_phase(
            replace(phase1_request, resume_from=first.checkpoint),
            TrainingCallbacks(validate=lambda event: evaluator.evaluate(event.validation_index)),
        )
        if not resumed.completed or resumed.checkpoint is None:
            raise AssertionError("phase 1 did not complete after resume")
        phase1_adapter_sha = sha256_file(resumed.checkpoint / "adapter/adapter_model.safetensors")
        phase2 = train_phase(
            _training_request(
                root,
                train_model,
                cache,
                phase_datasets[2],
                PhaseSpec.for_phase(2),
                phase1_adapter_sha,
            ),
            TrainingCallbacks(validate=lambda event: evaluator.evaluate(event.validation_index)),
        )
    if not phase2.completed:
        raise AssertionError("phase 2 did not complete")

    final_llm = _merge_and_strict_load_tiny(root)
    stage_store.publish("complete", {"llm": str(final_llm), "llm_sha256": sha256_file(final_llm), "uploads": 0})
    completed = stage_store.require("complete")
    return TinyWorkflowResult(
        stage=completed.name,
        final_llm=final_llm,
        strict_load_verified=True,
        validation_indices=tuple(evaluator.indices),
        phase_epochs=(PhaseSpec.for_phase(1).epochs, PhaseSpec.for_phase(2).epochs),
        boundaries_per_epoch=8,
        pilot_stopped_before_approval=pilot_stopped,
        pilot_checksum_approved=stage_store.require("pilot_approved").payload["pilot_manifest_sha256"] == approved_checksum,
        memorization_passed=memory_ok,
        cache_resumed_after_interruption=interrupted,
        phase_resumed_after_interruption=first.progress.validation_index == 3 and resumed.progress.validation_index == 16,
        wandb_mode=evaluator.wandb.mode,
        upload_calls=0,
    )


def _tiny_join_rows(
    source_shards: int,
    phase1_rows: int,
    phase2_rows: int,
    reserved_prompts: int,
) -> tuple[list[JoinedRow], tuple[str, ...]]:
    rows: list[JoinedRow] = []
    for index in range(phase1_rows):
        shard = index % source_shards
        rows.append(JoinedRow(f"{shard:06d}/phase1_{index:02d}.mp3", f"низкий {index}", 0.5, text_token_count=2 + index))
    for index in range(phase2_rows + reserved_prompts):
        shard = index % source_shards
        rows.append(JoinedRow(f"{shard:06d}/phase2_{index:02d}.mp3", f"высокий {index}", 0.95, text_token_count=2 + index))
    selected = reserve_prompt_ids(rows, count=reserved_prompts, seed=1986)
    assigned = assign_phases(rows, set(selected))
    if sum(row.phase == 1 for row in assigned) != phase1_rows or sum(row.phase == 2 for row in assigned) != phase2_rows:
        raise AssertionError("tiny joined rows do not preserve the phase split")
    return assigned, selected


def _write_source_fixture(
    root: Path,
    rows: Sequence[JoinedRow],
    selected: Sequence[str],
    source_shards: int,
) -> tuple[Path, tuple[Path, ...]]:
    plan_dir = root / "split_plan"
    train_dir = root / "source"
    plan_dir.mkdir()
    train_dir.mkdir()
    tar_paths: list[Path] = []
    for shard in range(source_shards):
        shard_rows = [row for row in rows if row.source_relative_path.startswith(f"{shard:06d}/")]
        plan_path = plan_dir / f"shard_{shard:06d}.jsonl"
        with plan_path.open("w", encoding="utf-8") as handle:
            for row in shard_rows:
                value = {
                    "source_relative_path": row.source_relative_path,
                    "text": row.text,
                    "instruct": row.instruct,
                    "agreement": row.agreement,
                    "phase": row.phase,
                    "reserved": row.reserved,
                    "reservation_score": row.reservation_score,
                    "model_limit_exclusion": row.model_limit_exclusion,
                    "text_token_count": row.text_token_count,
                }
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        tar_path = train_dir / f"shard_{shard:06d}.tar"
        with tarfile.open(tar_path, "w") as archive:
            for row in shard_rows:
                stem = Path(row.source_relative_path).stem
                payloads = (
                    (f"{stem}.json", json.dumps({"source_relative_path": row.source_relative_path}).encode("utf-8")),
                    (f"{stem}.mp3", row.source_relative_path.encode("utf-8")),
                )
                for name, payload in payloads:
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
        tar_paths.append(tar_path)
    atomic_write_json(
        plan_dir / "manifest.json",
        {"reserved_prompts": [{"source_relative_path": source} for source in selected]},
    )
    return plan_dir, tuple(tar_paths)


def _decode_fixture(sample: object) -> AudioInput:
    source = str(getattr(sample, "source_relative_path"))
    index = int(Path(source).stem.rsplit("_", 1)[1])
    frames = index % 4 + 1
    return AudioInput(source, [0.0] * frames, 24_000, frames)


def _training_request(
    root: Path,
    model: _TrainableScalar,
    cache: CacheManifest,
    dataset: CachedSpeechDataset,
    phase: PhaseSpec,
    initial_adapter_sha256: str | None,
) -> TrainRequest:
    batches = [
        {"utts": [dataset[index].source_relative_path], "target": torch.tensor([float(index + 1)])}
        for index in range(len(dataset))
    ]
    return TrainRequest(
        model=model,
        phase=phase,
        eligible_samples=len(dataset),
        cache_checksum=sha256_file(cache.root / "manifest.json"),
        checkpoint_root=root / f"phase{phase.number}-checkpoints",
        token_limit=256,
        accumulation_steps=1,
        dataloader_factory=lambda _epoch, _accelerator: batches,
        accelerator_factory=_CpuAccelerator,
        initial_adapter_sha256=initial_adapter_sha256,
    )


def _tiny_audit(_model: torch.nn.Module) -> TrainableAudit:
    return TrainableAudit(("tiny",), ("lora_weight",), (), 1, 1)


def _tiny_save_adapter(model: _TrainableScalar, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    weights = path / "adapter_model.safetensors"
    weights.write_bytes(model.lora_weight.detach().cpu().numpy().tobytes())
    atomic_write_json(
        path / "adapter_manifest.json",
        {
            "base_checkpoint_sha256": model._balalaika_base_checkpoint_sha256,
            "weights": weights.name,
            "weights_sha256": sha256_file(weights),
            "settings": {"r": 64, "alpha": 128, "dropout": 0.05, "bias": "none"},
            "target_modules": ["tiny"],
        },
    )


def _tiny_cosyvoice3() -> CosyVoice3LM:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    encoder = Qwen2Encoder.__new__(Qwen2Encoder)
    torch.nn.Module.__init__(encoder)
    encoder.model = Qwen2ForCausalLM(config)
    return CosyVoice3LM(
        llm_input_size=8,
        llm_output_size=8,
        speech_token_size=6,
        llm=encoder,
        sampling=lambda scores, decoded, sampling: int(scores.argmax()),
        mix_ratio=[5, 15],
    )


def _merge_and_strict_load_tiny(root: Path) -> Path:
    base_dir = root / model_api.APPROVED_BASE_MODEL_NAME
    adapter_dir = root / "tiny-adapter"
    base_dir.mkdir()
    base = _tiny_cosyvoice3().eval()
    torch.save(base.state_dict(), base_dir / "llm.pt")
    base._balalaika_base_model_dir = base_dir
    base._balalaika_base_checkpoint_sha256 = sha256_file(base_dir / "llm.pt")
    pristine = copy.deepcopy(base)
    adapted = model_api.inject_lora(base, LoraSettings()).eval()
    first_parameter = next(parameter for name, parameter in adapted.named_parameters() if "lora_" in name)
    first_parameter.data.fill_(0.01)
    model_api.save_adapter(adapted, adapter_dir)
    final_llm = root / "final" / "llm.pt"
    with mock.patch.object(model_api, "load_base_llm", side_effect=lambda _path: copy.deepcopy(pristine)):
        model_api.merge_adapter(base_dir, adapter_dir, final_llm)
    reloaded = _tiny_cosyvoice3()
    state = torch.load(final_llm, map_location="cpu", weights_only=True)
    load = reloaded.load_state_dict(state, strict=True)
    if load.missing_keys or load.unexpected_keys:
        raise AssertionError("strict tiny merged load failed")
    return final_llm
