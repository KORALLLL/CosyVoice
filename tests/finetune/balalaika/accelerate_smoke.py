"""Two-process CPU smoke for uneven accumulation and pending resume."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from unittest import mock

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader

from cosyvoice.finetune.balalaika.artifacts import atomic_write_json
from cosyvoice.finetune.balalaika.config import PhaseSpec
from cosyvoice.finetune.balalaika.model import LoraSettings, TrainableAudit
from cosyvoice.finetune.balalaika import training


class DistributedToyAdapter(torch.nn.Module):
    def __init__(self, seen: list[str]) -> None:
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor(0.0))
        self.seen = seen
        self._balalaika_base_checkpoint_sha256 = "b" * 64
        self._balalaika_lora_settings = LoraSettings()

    def forward(self, batch, _device):
        self.seen.extend(batch["utts"])
        target = batch["target"].to(self.lora_weight.device).float()
        return {"loss": ((self.lora_weight - target) ** 2).mean()}


class EightRankIdentityAccelerator:
    """Expose production identity while delegating collectives to two CPU ranks."""

    num_processes = 8

    def __init__(self, accelerator: Accelerator) -> None:
        self._accelerator = accelerator

    def __getattr__(self, name):
        return getattr(self._accelerator, name)

    def gather(self, value):
        gathered = self._accelerator.gather(value).reshape(-1)
        if value.numel() == 1 and gathered.numel() == 2:
            gathered = torch.cat((gathered, torch.zeros(6, dtype=gathered.dtype, device=gathered.device)))
        return gathered


def _audit(_model) -> TrainableAudit:
    return TrainableAudit(("toy",), ("lora_weight",), (), 1, 1)


def _save_adapter(model, path) -> None:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "adapter_model.safetensors").write_bytes(
        model.lora_weight.detach().cpu().numpy().tobytes()
    )
    atomic_write_json(
        output / "adapter_manifest.json",
        {"base_checkpoint_sha256": model._balalaika_base_checkpoint_sha256},
    )


def _loader(rank: int) -> DataLoader:
    real = lambda name, target: {"utts": [name], "target": torch.tensor([target])}
    rows = [real("rank-0-a", 1.0), real("rank-0-b", 2.0)] if rank == 0 else [
        real("rank-1-a", 3.0),
        {"utts": [], "target": torch.empty(0)},
    ]
    return DataLoader(rows, batch_size=1, shuffle=False, collate_fn=lambda values: values[0])


def _worker() -> None:
    created: list[EightRankIdentityAccelerator] = []

    def accelerator_factory(**kwargs):
        value = EightRankIdentityAccelerator(Accelerator(cpu=True, **kwargs))
        created.append(value)
        return value

    seen: list[str] = []
    validated: list[tuple[int, int]] = []
    model = DistributedToyAdapter(seen)
    root = Path(os.environ["BALALAIKA_SMOKE_RUN_ROOT"])
    request = training.TrainRequest(
        model=model,
        phase=PhaseSpec.for_phase(1),
        eligible_samples=3,
        cache_checksum="c" * 64,
        checkpoint_root=root,
        token_limit=2000,
        accumulation_steps=3,
        scheduler_spec=training.SchedulerSpec(kind="constant-v1"),
        dataloader_factory=lambda _epoch, accelerator: _loader(accelerator.process_index),
        accelerator_factory=accelerator_factory,
    )

    def initial_validation(event):
        if event.validation_index == 3:
            raise RuntimeError("smoke validation interruption")
        validated.append((event.validation_index, event.boundary.ordinal))

    with mock.patch.object(training, "audit_trainable_parameters", _audit), mock.patch.object(
        training, "save_adapter", _save_adapter
    ):
        try:
            training.train_phase(request, training.TrainingCallbacks(validate=initial_validation))
        except RuntimeError as exc:
            if str(exc) != "smoke validation interruption":
                raise
        else:
            raise AssertionError("smoke validation interruption was not raised")

        pending = root / "phase-1-validation-03"

        def resumed_validation(event):
            validated.append((event.validation_index, event.boundary.ordinal))
            return event.validation_index < 8

        resumed = training.train_phase(
            training.TrainRequest(**{**request.__dict__, "resume_from": pending}),
            training.TrainingCallbacks(validate=resumed_validation),
        )

    expected_seen = ["rank-0-a", "rank-0-b"] if created[-1].process_index == 0 else ["rank-1-a"]
    if seen != expected_seen:
        raise AssertionError(f"rank {created[-1].process_index} replayed or skipped samples: {seen}")
    if validated != [(index, index) for index in range(1, 9)]:
        raise AssertionError(f"invalid validation sequence: {validated}")
    if resumed.progress.optimizer_steps != 1 or resumed.progress.global_samples != 3:
        raise AssertionError(f"invalid committed progress: {resumed.progress.state_dict()}")
    weights = created[-1]._accelerator.gather(model.lora_weight.detach().reshape(1)).cpu()
    if weights.numel() != 2 or not torch.equal(weights, weights[0].expand_as(weights)):
        raise AssertionError(f"model state diverged across ranks: {weights.tolist()}")

    artifact = Path(os.environ.get("BALALAIKA_ACCELERATE_SMOKE_ARTIFACT", "/tmp/balalaika-accelerate-smoke.json"))
    if created[-1].is_main_process:
        atomic_write_json(
            artifact,
            {
                "world_size": created[-1]._accelerator.num_processes,
                "boundaries": [
                    {"validation_index": index, "ordinal": ordinal}
                    for index, ordinal in validated
                ],
                "optimizer_steps": resumed.progress.optimizer_steps,
                "global_samples": resumed.progress.global_samples,
                "rank_weights": weights.tolist(),
            },
        )
    created[-1].wait_for_everyone()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if [item["ordinal"] for item in payload["boundaries"]] != list(range(1, 9)):
        raise AssertionError(f"invalid smoke artifact: {payload}")


def main() -> None:
    if os.environ.get("BALALAIKA_SMOKE_CHILD") == "1":
        _worker()
        return
    accelerator = Accelerator(cpu=True)
    if accelerator.num_processes == 1:
        _launch_cpu_workers()
        return
    if accelerator.num_processes != 2:
        raise AssertionError(f"smoke requires two processes, found {accelerator.num_processes}")
    _worker()


def _launch_cpu_workers() -> None:
    """Compensate for Accelerate 1.12's --cpu launcher ignoring num_processes."""

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    run_root = Path(tempfile.mkdtemp(prefix="balalaika-accelerate-smoke-"))
    base = os.environ.copy()
    base.update(
        {
            "ACCELERATE_USE_CPU": "true",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
            "BALALAIKA_SMOKE_CHILD": "1",
            "BALALAIKA_SMOKE_RUN_ROOT": str(run_root),
        }
    )
    workers = []
    try:
        for rank in range(2):
            env = dict(base, RANK=str(rank), LOCAL_RANK=str(rank))
            workers.append(
                subprocess.Popen(
                    [sys.executable, "-m", "tests.finetune.balalaika.accelerate_smoke"],
                    env=env,
                )
            )
        failures = [worker.wait() for worker in workers]
        if any(failures):
            raise AssertionError(f"two-process Accelerate smoke workers failed: {failures}")
    finally:
        shutil.rmtree(run_root)


if __name__ == "__main__":
    main()
