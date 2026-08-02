"""Two-process CPU smoke for collective real-sample boundary accounting."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from accelerate import Accelerator

from cosyvoice.finetune.balalaika.artifacts import atomic_write_json
from cosyvoice.finetune.balalaika.training import FractionBoundary, _global_sample_count


def main() -> None:
    accelerator = Accelerator(cpu=True)
    if accelerator.num_processes == 1 and os.environ.get("BALALAIKA_SMOKE_CHILD") != "1":
        _launch_cpu_workers()
        return
    if accelerator.num_processes != 2:
        raise AssertionError(f"smoke requires two processes, found {accelerator.num_processes}")
    boundaries = FractionBoundary.for_epoch(eligible_samples=8)
    completed = 0
    next_ordinal = 1
    records: list[dict[str, int]] = []
    for _ in range(4):
        completed += _global_sample_count(accelerator, 1)
        while next_ordinal <= 8 and completed >= boundaries[next_ordinal - 1].sample_target:
            boundary = boundaries[next_ordinal - 1]
            records.append(
                {
                    "ordinal": boundary.ordinal,
                    "sample_target": boundary.sample_target,
                    "global_samples": completed,
                }
            )
            next_ordinal += 1

    artifact = Path(os.environ.get("BALALAIKA_ACCELERATE_SMOKE_ARTIFACT", "/tmp/balalaika-accelerate-smoke.json"))
    if accelerator.is_main_process:
        atomic_write_json(
            artifact,
            {"world_size": accelerator.num_processes, "boundaries": records},
        )
    accelerator.wait_for_everyone()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload["world_size"] != 2 or [item["ordinal"] for item in payload["boundaries"]] != list(range(1, 9)):
        raise AssertionError(f"invalid smoke artifact: {payload}")


def _launch_cpu_workers() -> None:
    """Compensate for Accelerate 1.12's --cpu launcher ignoring num_processes."""

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = os.environ.copy()
    base.update(
        {
            "ACCELERATE_USE_CPU": "true",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
            "BALALAIKA_SMOKE_CHILD": "1",
        }
    )
    workers = []
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


if __name__ == "__main__":
    main()
