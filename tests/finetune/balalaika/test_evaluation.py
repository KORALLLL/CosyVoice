"""Fake-only tests for distributed hard-number synthesis and publication."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np


def _api():
    try:
        from cosyvoice.finetune.balalaika import evaluation
    except ModuleNotFoundError as exc:
        raise AssertionError("Balalaika evaluation module is missing") from exc
    return evaluation


def _rows(count: int = 2_000) -> list[dict[str, object]]:
    return [
        {
            "id": index,
            "stressed": f"Код {index} готов",
            "normalized_gold": "код семь готов",
            "hard_number": str(index),
            "category": "code",
            "number_span": {"reference_start": 1, "reference_end": 2},
        }
        for index in range(1, count + 1)
    ]


def _prompts(root: Path) -> list[dict[str, object]]:
    prompts = []
    for index in range(20):
        path = root / f"voice_{index:02d}.wav"
        _write_wav(path)
        prompts.append(
            {
                "voice_id": f"voice_{index:02d}",
                "audio_path": path,
                "text": f"prompt {index}",
                "wav_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            }
        )
    return prompts


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"\0\0" * 8)


def _remote_records(root: Path) -> list[dict[str, object]]:
    path = root / "remote.wav"
    _write_wav(path)
    checksum = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    return [
        {
            "benchmark_id": index,
            "hypothesis": "код семь готов",
            "audio_path": str(path),
            "audio_sha256": checksum,
            "generation_latency_seconds": 0.0,
            "asr_latency_seconds": 0.0,
        }
        for index in range(251, 2_001)
    ]


def _seal_remote_records(api, records, rows, prompts, provenance, validation_index=0):
    items = api.build_voice_assignment(rows, prompts)
    identity = api.build_evaluation_identity(validation_index, items, provenance)
    by_id = {item.benchmark_id: item for item in items}
    for record in records:
        record["evaluation_identity_sha256"] = identity.sha256
        record["item_sha256"] = api._item_sha256(by_id[record["benchmark_id"]])
        record["record_sha256"] = api._journal_record_sha256(record)
    return records


class _FakeRecognizer:
    local_rank = 0

    def transcribe(self, paths):
        return ["код семь готов" for _ in paths]

    def provenance(self):
        return {"model": "gigaam-v3-rnnt", "provider": "CUDAExecutionProvider", "device_id": self.local_rank, "max_batch_size": 8}


class _FakeSynthesizer:
    def __init__(self) -> None:
        self.generated: list[int] = []

    def synthesize(self, item, destination) -> None:
        self.generated.append(item.benchmark_id)
        _write_wav(destination)

    def provenance(self):
        return {"method": "inference_zero_shot", "sample_rate": 24_000, "stream": False}


def _provenance(api, recognizer=None, synthesizer=None, *, checkpoint="1" * 64):
    recognizer = recognizer or _FakeRecognizer()
    synthesizer = synthesizer or _FakeSynthesizer()
    return api.EvaluationProvenance(
        checkpoint_sha256=checkpoint,
        model_state_sha256="2" * 64,
        adapter_sha256="3" * 64,
        base_checkpoint_sha256="4" * 64,
        benchmark_snapshot_sha256="5" * 64,
        benchmark_revision="private-revision-1",
        asr_config=recognizer.provenance(),
        synthesis_config=synthesizer.provenance(),
        code_version="commit-a",
        config_version="balalaika-v1",
    )


class _FakeAccelerator:
    num_processes = 8
    process_index = 0
    local_process_index = 0
    is_main_process = True

    def __init__(self, remote_records=None) -> None:
        self.remote_records = list(remote_records or [])
        self.run = _FakeRun()
        self.scalar_logs = []

    @contextmanager
    def split_between_processes(self, items):
        yield list(items)[:250]

    def gather_object(self, records):
        return [*records, *self.remote_records]

    def wait_for_everyone(self) -> None:
        return None

    def broadcast_object_list(self, values, from_process=0) -> None:
        return None

    def get_tracker(self, name, unwrap=False):
        if (name, unwrap) != ("wandb", True):
            raise AssertionError((name, unwrap))
        return self.run

    def log(self, values, step=None, log_kwargs=None) -> None:
        if self.run.fail_scalars:
            raise RuntimeError("scalar network failure")
        self.scalar_logs.append((dict(values), step))
        self.run.pending.update(values)
        if (log_kwargs or {}).get("wandb", {}).get("commit"):
            self.run.history.append({**self.run.pending, "_step": step})
            self.run.pending.clear()


class _FakeRun:
    id = "run-123"
    resumed = False

    def __init__(self) -> None:
        self.dir = "wandb/run-123/files"
        self.media_logs = []
        self.pending = {}
        self.history = []
        self.fail_media = False
        self.fail_scalars = False
        self.scan_calls = 0

    def log(self, values, *, step=None, commit=None) -> None:
        if self.fail_media:
            raise RuntimeError("media network failure")
        self.media_logs.append((dict(values), {"step": step, "commit": commit}))
        self.pending.update(values)
        if commit:
            self.history.append({**self.pending, "_step": step})
            self.pending.clear()

    def scan_history(self, keys=None):
        self.scan_calls += 1
        for row in self.history:
            yield {key: row.get(key) for key in keys or row}


def _logger(api, accelerator, root):
    return api.WandbValidationLogger(
        accelerator,
        root / "wandb-run.json",
        table_factory=lambda rows: ("table", tuple(row["benchmark_id"] for row in rows)),
        audio_factory=lambda path: ("audio", Path(path).name),
        remote_history_reader=lambda run, keys: run.scan_history(keys=keys),
    )


def _wandb_report(api, root, prompts):
    output = root / "results.jsonl"
    summary = root / "summary.json"
    panel_dir = root / "panel"
    panel_dir.mkdir(exist_ok=True)
    panel_manifest = panel_dir / "manifest.json"
    seal = root / "validation-success.json"
    output.write_text("{}\n", encoding="utf-8")
    summary.write_text("{}\n", encoding="utf-8")
    panel_manifest.write_text("{}\n", encoding="utf-8")
    seal.write_text("{}\n", encoding="utf-8")
    checksums = {
        "results_jsonl": __import__("hashlib").sha256(output.read_bytes()).hexdigest(),
        "summary_json": __import__("hashlib").sha256(summary.read_bytes()).hexdigest(),
        "panel_manifest": __import__("hashlib").sha256(panel_manifest.read_bytes()).hexdigest(),
        "validation_seal": __import__("hashlib").sha256(seal.read_bytes()).hexdigest(),
    }
    return api.EvaluationReport(
        validation_index=0,
        output_jsonl=output,
        summary_json=summary,
        panel_dir=panel_dir,
        assignment_checksum="a" * 64,
        identity_checksum="b" * 64,
        artifact_checksums=checksums,
        metrics={
            "utt-wer": 0.1,
            "utt-cer": 0.2,
            "num-wer": 0.3,
            "num-cer": 0.4,
            "macro-utt-wer": 0.5,
            "by_category": {"code": {"num-wer": 0.6}},
        },
        row_count=2_000,
        worst_errors=({"benchmark_id": 1, "hypothesis": "семь"},),
        panel_audio={f"voice_{index:02d}": prompts[index]["audio_path"] for index in range(20)},
        generation_latency_seconds=1.25,
        asr_latency_seconds=0.25,
    )


class _FailingLogger:
    def log(self, report, validation_index) -> None:
        raise _api().WandbSyncError("offline")


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows_2000 = _rows()
        self.prompts_20 = _prompts(self.root / "prompts")
        self.environment = mock.patch.dict(os.environ, {"WANDB_API_KEY": "unit-test-key"})
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary.cleanup()

    def test_each_benchmark_row_is_generated_once(self) -> None:
        api = _api()

        items = api.build_voice_assignment(reversed(self.rows_2000), reversed(self.prompts_20))

        self.assertEqual(len(items), 2_000)
        self.assertEqual(len({item.benchmark_id for item in items}), 2_000)
        self.assertEqual(
            Counter(item.voice_id for item in items),
            {f"voice_{index:02d}": 100 for index in range(20)},
        )
        self.assertEqual(items[0].benchmark_id, 1)
        self.assertEqual(items[0].voice_id, "voice_00")
        self.assertEqual(items[20].voice_id, "voice_00")

    def test_assignment_refuses_anything_except_exact_fixed_inputs(self) -> None:
        api = _api()

        with self.assertRaisesRegex(api.EvaluationIntegrityError, "exactly 2000"):
            api.build_voice_assignment(self.rows_2000[:-1], self.prompts_20)
        with self.assertRaisesRegex(api.EvaluationIntegrityError, "exactly 20"):
            api.build_voice_assignment(self.rows_2000, self.prompts_20[:-1])
        duplicate = [*self.rows_2000[:-1], dict(self.rows_2000[0])]
        with self.assertRaisesRegex(api.EvaluationIntegrityError, "IDs"):
            api.build_voice_assignment(duplicate, self.prompts_20)

    def test_evaluation_identity_binds_every_semantic_and_model_input(self) -> None:
        api = _api()
        items = api.build_voice_assignment(self.rows_2000, self.prompts_20)
        provenance = api.EvaluationProvenance(
            checkpoint_sha256="1" * 64,
            model_state_sha256="2" * 64,
            adapter_sha256="3" * 64,
            base_checkpoint_sha256="4" * 64,
            benchmark_snapshot_sha256="5" * 64,
            benchmark_revision="private-revision-1",
            asr_config={"model": "gigaam-v3-rnnt", "provider": "CUDAExecutionProvider", "device_id": 0, "max_batch_size": 8},
            synthesis_config={"method": "inference_zero_shot", "sample_rate": 24_000, "stream": False},
            code_version="commit-a",
            config_version="balalaika-v1",
        )

        identity = api.build_evaluation_identity(0, items, provenance)

        self.assertEqual(
            set(identity.payload),
            {
                "format_version",
                "validation_index",
                "checkpoint_sha256",
                "model_state_sha256",
                "adapter_sha256",
                "base_checkpoint_sha256",
                "benchmark_snapshot_sha256",
                "benchmark_revision",
                "semantic_assignment_sha256",
                "asr_batch_size",
                "asr_config",
                "synthesis_config",
                "code_version",
                "config_version",
            },
        )
        changed = [
            replace(provenance, checkpoint_sha256="a" * 64),
            replace(provenance, model_state_sha256="a" * 64),
            replace(provenance, adapter_sha256="a" * 64),
            replace(provenance, base_checkpoint_sha256="a" * 64),
            replace(provenance, benchmark_snapshot_sha256="a" * 64),
            replace(provenance, benchmark_revision="private-revision-2"),
            replace(provenance, asr_config={**provenance.asr_config, "max_batch_size": 4}),
            replace(provenance, synthesis_config={**provenance.synthesis_config, "stream": True}),
            replace(provenance, code_version="commit-b"),
            replace(provenance, config_version="balalaika-v2"),
        ]
        self.assertEqual(len({api.build_evaluation_identity(0, items, value).sha256 for value in changed}), len(changed))
        self.assertNotIn(identity.sha256, {api.build_evaluation_identity(0, items, value).sha256 for value in changed})
        self.assertNotEqual(identity.sha256, api.build_evaluation_identity(1, items, provenance).sha256)
        semantic_change = list(items)
        semantic_change[0] = replace(items[0], hard_number="изменено")
        self.assertNotEqual(identity.sha256, api.build_evaluation_identity(0, semantic_change, provenance).sha256)
        with self.assertRaises(TypeError):
            identity.payload["asr_config"]["max_batch_size"] = 4

    def test_evaluation_identity_rejects_changed_asr_batch_size(self) -> None:
        api = _api()
        items = api.build_voice_assignment(self.rows_2000, self.prompts_20)
        provenance = _provenance(api)

        default = api.build_evaluation_identity(0, items, provenance)

        self.assertEqual(default.payload.get("asr_batch_size"), 8)
        self.assertNotEqual(
            default.sha256,
            api.build_evaluation_identity(0, items, provenance, asr_batch_size=4).sha256,
        )

    def test_evaluation_identity_rejects_changed_number_span_category(self) -> None:
        api = _api()
        items = api.build_voice_assignment(self.rows_2000, self.prompts_20)
        changed = list(items)
        changed[0] = replace(
            items[0],
            number_span=replace(items[0].number_span, category="changed-span-category"),
        )

        self.assertNotEqual(
            api.build_evaluation_identity(0, items, _provenance(api)).sha256,
            api.build_evaluation_identity(0, changed, _provenance(api)).sha256,
        )

    def test_wandb_remote_history_uses_api_run_not_active_run(self) -> None:
        api = _api()
        rows = [{"marker": "sealed"}]
        api_run = type("ApiRun", (), {"scan_history": lambda self, keys=None: iter(rows)})()
        client = type("Api", (), {"run": lambda self, path: api_run})()
        active = type("Run", (), {"entity": "team", "project": "project", "id": "run-123"})()

        fake_wandb = type("Wandb", (), {"Api": staticmethod(lambda: client)})()
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            self.assertEqual(list(api._wandb_api_history(active, ["marker"])), rows)

    def test_gigaam_loads_v3_rnnt_on_local_cuda_and_preserves_batch_order(self) -> None:
        api = _api()
        captured: list[tuple[object, object]] = []

        class Session:
            def get_providers(self):
                return ["CUDAExecutionProvider"]

            def get_provider_options(self):
                return {"CUDAExecutionProvider": {"device_id": "3"}}

        class FakeModel:
            def __init__(self) -> None:
                self.asr = type(
                    "Asr",
                    (),
                    {
                        "runtime_config": {
                            "providers": ["CUDAExecutionProvider"],
                            "provider_options": [{"device_id": "3"}],
                        }
                    },
                )()
                self.asr._encoder = Session()
                self.asr._decoder = Session()
                self.asr._joiner = Session()

            def recognize(self, paths):
                captured.append(("recognize", tuple(Path(path).name for path in paths)))
                if len(paths) > 2:
                    raise RuntimeError("CUDA out of memory")
                return [Path(path).stem for path in paths]

        def load_model(name, **kwargs):
            captured.append((name, kwargs))
            return FakeModel()

        recognizer = api.GigaAmRecognizer(local_rank=3, max_batch_size=4, model_loader=load_model)
        result = recognizer.transcribe([Path(f"{index}.wav") for index in range(4)])

        self.assertEqual(result, ["0", "1", "2", "3"])
        self.assertEqual(
            captured[0],
            (
                "gigaam-v3-rnnt",
                {
                    "providers": [("CUDAExecutionProvider", {"device_id": 3})],
                },
            ),
        )
        self.assertEqual(
            captured[1:],
            [
                ("recognize", ("0.wav", "1.wav", "2.wav", "3.wav")),
                ("recognize", ("0.wav", "1.wav")),
                ("recognize", ("2.wav", "3.wav")),
            ],
        )

    def test_gigaam_refuses_cpu_fallback(self) -> None:
        api = _api()
        cpu_session = type(
            "Session",
            (),
            {
                "get_providers": lambda self: ["CPUExecutionProvider"],
                "get_provider_options": lambda self: {"CPUExecutionProvider": {}},
            },
        )()
        cpu_model = type(
            "Model",
            (),
            {
                "asr": type(
                    "Asr",
                    (),
                    {
                        "runtime_config": {
                            "providers": [("CUDAExecutionProvider", {"device_id": 0})],
                            "provider_options": None,
                        },
                        "_encoder": cpu_session,
                        "_decoder": cpu_session,
                        "_joiner": cpu_session,
                    },
                )()
            },
        )()

        with self.assertRaisesRegex(api.GigaAmError, "CUDAExecutionProvider"):
            api.GigaAmRecognizer(model=cpu_model, local_rank=0)

    def test_results_publish_before_wandb_failure_and_resume_without_regeneration(self) -> None:
        api = _api()
        synthesizer = _FakeSynthesizer()
        provenance = _provenance(api, synthesizer=synthesizer)
        remote = _seal_remote_records(api, _remote_records(self.root), self.rows_2000, self.prompts_20, provenance)
        accelerator = _FakeAccelerator(remote)
        accelerator.run.fail_media = True
        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=accelerator,
            synthesizer=synthesizer,
            recognizer=_FakeRecognizer(),
            provenance=provenance,
            validation_index=0,
            output_jsonl=self.root / "validation-00/results.jsonl",
            summary_json=self.root / "validation-00/summary.json",
            panel_dir=self.root / "validation-00/listening-panel",
            temporary_audio_dir=self.root / "validation-00/audio",
            assignment_manifest=self.root / "voice-assignment.json",
            memorization_path=self.root / "memorization",
            wandb_logger=_logger(api, accelerator, self.root),
        )

        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            with self.assertRaises(api.WandbSyncError):
                api.evaluate_checkpoint(request)

        self.assertTrue(request.output_jsonl.exists())
        self.assertTrue(request.summary_json.exists())
        self.assertEqual(len(request.output_jsonl.read_text(encoding="utf-8").splitlines()), 2_000)
        first = json.loads(request.output_jsonl.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(
            set(first),
            {
                "benchmark_id",
                "evaluation_identity_sha256",
                "voice_id",
                "prompt_wav",
                "prompt_text",
                "prompt_sha256",
                "stressed",
                "normalized_gold",
                "hard_number",
                "hypothesis",
                "category",
                "number_span",
                "edit_counts",
                "audio_path",
                "audio_sha256",
                "generation_latency_seconds",
                "asr_latency_seconds",
                "row_sha256",
            },
        )
        self.assertEqual(first["number_span"]["hypothesis_text"], "семь")
        self.assertEqual(first["edit_counts"]["number"]["word"]["distance"], 0)
        self.assertEqual(len(list(request.panel_dir.glob("*.wav"))), 20)
        self.assertEqual(synthesizer.generated, list(range(1, 251)))
        self.assertFalse(request.temporary_audio_dir.exists())

        accelerator.run.fail_media = False
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            report = api.evaluate_checkpoint(request)
        self.assertEqual(synthesizer.generated, list(range(1, 251)))
        self.assertEqual(report.metrics["utt-wer"], 0.0)
        self.assertEqual(report.metrics["num-wer"], 0.0)

    def test_distributed_evaluation_refuses_wrong_local_shard_count(self) -> None:
        api = _api()

        class ShortAccelerator(_FakeAccelerator):
            @contextmanager
            def split_between_processes(self, items):
                yield list(items)[:249]

        accelerator = ShortAccelerator()
        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=accelerator,
            synthesizer=_FakeSynthesizer(),
            recognizer=_FakeRecognizer(),
            provenance=_provenance(api),
            validation_index=0,
            output_jsonl=self.root / "results.jsonl",
            summary_json=self.root / "summary.json",
            panel_dir=self.root / "panel",
            temporary_audio_dir=self.root / "audio",
            assignment_manifest=self.root / "mapping.json",
            memorization_path=self.root / "memorization",
            wandb_logger=_logger(api, accelerator, self.root),
        )
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            with self.assertRaisesRegex(api.EvaluationIntegrityError, "250"):
                api.evaluate_checkpoint(request)

    def test_unjournaled_valid_wav_is_replaced_not_reused(self) -> None:
        api = _api()
        synthesizer = _FakeSynthesizer()
        stale = self.root / "validation/audio/rank-00/benchmark-0001.wav"
        _write_wav(stale)
        stale.write_bytes(stale.read_bytes() + b"stale")
        provenance = _provenance(api, synthesizer=synthesizer)
        remote = _seal_remote_records(api, _remote_records(self.root), self.rows_2000, self.prompts_20, provenance)
        accelerator = _FakeAccelerator(remote)
        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=accelerator,
            synthesizer=synthesizer,
            recognizer=_FakeRecognizer(),
            provenance=provenance,
            validation_index=0,
            output_jsonl=self.root / "validation/results.jsonl",
            summary_json=self.root / "validation/summary.json",
            panel_dir=self.root / "validation/listening-panel",
            temporary_audio_dir=self.root / "validation/audio",
            assignment_manifest=self.root / "voice-assignment.json",
            memorization_path=self.root / "memorization",
            wandb_logger=_logger(api, accelerator, self.root),
        )
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            api.evaluate_checkpoint(request)

        self.assertEqual(synthesizer.generated, list(range(1, 251)))

    def test_evaluation_requires_wandb_before_distributed_work(self) -> None:
        api = _api()

        class ExplodingAccelerator(_FakeAccelerator):
            @contextmanager
            def split_between_processes(self, items):
                raise AssertionError("generation must not start")
                yield

        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=ExplodingAccelerator(),
            synthesizer=_FakeSynthesizer(),
            recognizer=_FakeRecognizer(),
            provenance=_provenance(api),
            validation_index=0,
            output_jsonl=self.root / "results.jsonl",
            summary_json=self.root / "summary.json",
            panel_dir=self.root / "panel",
            temporary_audio_dir=self.root / "audio",
            assignment_manifest=self.root / "mapping.json",
            memorization_path=self.root / "memorization",
            wandb_logger=None,
        )
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            with self.assertRaisesRegex(api.WandbSyncError, "mandatory"):
                api.evaluate_checkpoint(request)

    def test_recognizer_local_rank_must_match_accelerator_local_process(self) -> None:
        api = _api()
        recognizer = _FakeRecognizer()
        recognizer.local_rank = 1
        accelerator = _FakeAccelerator()
        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=accelerator,
            synthesizer=_FakeSynthesizer(),
            recognizer=recognizer,
            provenance=_provenance(api, recognizer=recognizer),
            validation_index=0,
            output_jsonl=self.root / "results.jsonl",
            summary_json=self.root / "summary.json",
            panel_dir=self.root / "panel",
            temporary_audio_dir=self.root / "audio",
            assignment_manifest=self.root / "mapping.json",
            memorization_path=self.root / "memorization",
            wandb_logger=_logger(api, accelerator, self.root),
        )
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            with self.assertRaisesRegex(api.EvaluationIntegrityError, "local rank"):
                api.evaluate_checkpoint(request)

    def test_partial_local_publication_resumes_without_regeneration(self) -> None:
        api = _api()
        synthesizer = _FakeSynthesizer()
        provenance = _provenance(api, synthesizer=synthesizer)
        remote = _seal_remote_records(api, _remote_records(self.root), self.rows_2000, self.prompts_20, provenance)
        accelerator = _FakeAccelerator(remote)
        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=accelerator,
            synthesizer=synthesizer,
            recognizer=_FakeRecognizer(),
            provenance=provenance,
            validation_index=0,
            output_jsonl=self.root / "validation/results.jsonl",
            summary_json=self.root / "validation/summary.json",
            panel_dir=self.root / "validation/listening-panel",
            temporary_audio_dir=self.root / "validation/audio",
            assignment_manifest=self.root / "voice-assignment.json",
            memorization_path=self.root / "memorization",
            wandb_logger=_logger(api, accelerator, self.root),
        )
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            with mock.patch.object(api, "_publish_listening_panel", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    api.evaluate_checkpoint(request)
            self.assertTrue(request.output_jsonl.exists())
            report = api.evaluate_checkpoint(request)

        self.assertEqual(synthesizer.generated, list(range(1, 251)))
        self.assertEqual(report.row_count, 2_000)
        self.assertTrue(request.summary_json.exists())

    def test_changed_identity_discards_sealed_journal_and_its_wav(self) -> None:
        api = _api()
        item = api.build_voice_assignment(self.rows_2000, self.prompts_20)[0]
        provenance = _provenance(api)
        identity = api.build_evaluation_identity(0, api.build_voice_assignment(self.rows_2000, self.prompts_20), provenance)
        changed = api.build_evaluation_identity(
            0,
            api.build_voice_assignment(self.rows_2000, self.prompts_20),
            replace(provenance, checkpoint_sha256="a" * 64),
        )
        journal = self.root / "audio/rank-00/records.jsonl"
        wav = journal.parent / "benchmark-0001.wav"
        _write_wav(wav)
        record = {
            "benchmark_id": 1,
            "evaluation_identity_sha256": identity.sha256,
            "item_sha256": api._item_sha256(item),
            "hypothesis": "код семь готов",
            "audio_path": str(wav),
            "audio_sha256": __import__("hashlib").sha256(wav.read_bytes()).hexdigest(),
            "generation_latency_seconds": 0.0,
            "asr_latency_seconds": 0.0,
        }
        record["record_sha256"] = api._journal_record_sha256(record)
        api._atomic_write_jsonl(journal, [record])

        loaded = api._load_rank_journal(journal, [item], changed)

        self.assertEqual(loaded, {})
        self.assertFalse(wav.exists())

    def test_duplicate_sealed_journal_id_invalidates_first_record_and_wav(self) -> None:
        api = _api()
        item = api.build_voice_assignment(self.rows_2000, self.prompts_20)[0]
        identity = api.build_evaluation_identity(
            0,
            api.build_voice_assignment(self.rows_2000, self.prompts_20),
            _provenance(api),
        )
        journal = self.root / "audio/rank-00/records.jsonl"
        wav = journal.parent / "benchmark-0001.wav"
        _write_wav(wav)
        record = {
            "benchmark_id": 1,
            "evaluation_identity_sha256": identity.sha256,
            "item_sha256": api._item_sha256(item),
            "hypothesis": "код семь готов",
            "audio_path": str(wav),
            "audio_sha256": __import__("hashlib").sha256(wav.read_bytes()).hexdigest(),
            "generation_latency_seconds": 0.0,
            "asr_latency_seconds": 0.0,
        }
        record["record_sha256"] = api._journal_record_sha256(record)
        api._atomic_write_jsonl(journal, [record, dict(record)])

        loaded = api._load_rank_journal(journal, [item], identity)

        self.assertEqual(loaded, {})
        self.assertFalse(wav.exists())

    def test_collective_wandb_preflight_allows_nonmain_without_tracker(self) -> None:
        api = _api()

        class BarrierReached(RuntimeError):
            pass

        bus: dict[str, object] = {}

        class RankAccelerator(_FakeAccelerator):
            def __init__(self, *, main: bool) -> None:
                super().__init__()
                self.is_main_process = main
                self.process_index = 0 if main else 1
                self.barriers = 0
                self.tracker_queries = 0

            def broadcast_object_list(self, values, from_process=0) -> None:
                if self.is_main_process:
                    bus["payload"] = values[0]
                else:
                    values[0] = bus["payload"]

            def get_tracker(self, name, unwrap=False):
                self.tracker_queries += 1
                if not self.is_main_process:
                    raise AssertionError("non-main rank must not query W&B")
                return super().get_tracker(name, unwrap)

            def wait_for_everyone(self) -> None:
                self.barriers += 1
                raise BarrierReached("first generation barrier reached")

        def request(accelerator, logger):
            return api.EvaluationRequest(
                rows=self.rows_2000,
                prompts=self.prompts_20,
                accelerator=accelerator,
                synthesizer=_FakeSynthesizer(),
                recognizer=_FakeRecognizer(),
                provenance=_provenance(api),
                validation_index=0,
                output_jsonl=self.root / "results.jsonl",
                summary_json=self.root / "summary.json",
                panel_dir=self.root / "panel",
                temporary_audio_dir=self.root / "audio",
                assignment_manifest=self.root / "mapping.json",
                memorization_path=self.root / "memorization",
                wandb_logger=logger,
            )

        main = RankAccelerator(main=True)
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            with self.assertRaisesRegex(BarrierReached, "generation barrier"):
                api.evaluate_checkpoint(request(main, _logger(api, main, self.root)))
        self.assertEqual(
            bus.get("payload"),
            {"format_version": 1, "phase": "wandb-preflight", "ok": True, "error": None},
        )

        nonmain = RankAccelerator(main=False)
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            with self.assertRaisesRegex(BarrierReached, "generation barrier"):
                api.evaluate_checkpoint(request(nonmain, None))
        self.assertEqual(nonmain.tracker_queries, 0)
        self.assertEqual(nonmain.barriers, 1)

    def test_collective_wandb_preflight_broadcasts_same_main_error(self) -> None:
        api = _api()
        bus: dict[str, object] = {}

        class RankAccelerator(_FakeAccelerator):
            def __init__(self, *, main: bool) -> None:
                super().__init__()
                self.is_main_process = main
                self.process_index = 0 if main else 1
                self.barriers = 0

            def broadcast_object_list(self, values, from_process=0) -> None:
                if self.is_main_process:
                    bus["payload"] = values[0]
                else:
                    values[0] = bus["payload"]

            def wait_for_everyone(self) -> None:
                self.barriers += 1

            def get_tracker(self, name, unwrap=False):
                if not self.is_main_process:
                    raise AssertionError("non-main rank must not query W&B")
                return super().get_tracker(name, unwrap)

        def request(accelerator):
            return api.EvaluationRequest(
                rows=self.rows_2000,
                prompts=self.prompts_20,
                accelerator=accelerator,
                synthesizer=_FakeSynthesizer(),
                recognizer=_FakeRecognizer(),
                provenance=_provenance(api),
                validation_index=0,
                output_jsonl=self.root / "results.jsonl",
                summary_json=self.root / "summary.json",
                panel_dir=self.root / "panel",
                temporary_audio_dir=self.root / "audio",
                assignment_manifest=self.root / "mapping.json",
                memorization_path=self.root / "memorization",
                wandb_logger=None,
            )

        errors = []
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            for main in (True, False):
                accelerator = RankAccelerator(main=main)
                with self.assertRaises(api.WandbSyncError) as caught:
                    api.evaluate_checkpoint(request(accelerator))
                errors.append(str(caught.exception))
                self.assertEqual(accelerator.barriers, 0)

        self.assertEqual(errors[0], errors[1])
        self.assertEqual(
            bus.get("payload"),
            {
                "format_version": 1,
                "phase": "wandb-preflight",
                "ok": False,
                "error": errors[0],
            },
        )

    def test_main_rank_rejects_missing_or_checksum_changed_remote_audio(self) -> None:
        api = _api()
        synthesizer = _FakeSynthesizer()
        provenance = _provenance(api, synthesizer=synthesizer)
        remote = _seal_remote_records(api, _remote_records(self.root), self.rows_2000, self.prompts_20, provenance)
        Path(remote[0]["audio_path"]).unlink()
        accelerator = _FakeAccelerator(remote)
        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=accelerator,
            synthesizer=synthesizer,
            recognizer=_FakeRecognizer(),
            provenance=provenance,
            validation_index=0,
            output_jsonl=self.root / "validation/results.jsonl",
            summary_json=self.root / "validation/summary.json",
            panel_dir=self.root / "validation/listening-panel",
            temporary_audio_dir=self.root / "validation/audio",
            assignment_manifest=self.root / "voice-assignment.json",
            memorization_path=self.root / "memorization",
            wandb_logger=_logger(api, accelerator, self.root),
        )
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            with self.assertRaisesRegex(api.EvaluationIntegrityError, "record/audio"):
                api.evaluate_checkpoint(request)

    def test_committed_resume_recomputes_rows_metrics_and_panel_content(self) -> None:
        api = _api()
        synthesizer = _FakeSynthesizer()
        provenance = _provenance(api, synthesizer=synthesizer)
        remote = _seal_remote_records(api, _remote_records(self.root), self.rows_2000, self.prompts_20, provenance)
        accelerator = _FakeAccelerator(remote)
        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=accelerator,
            synthesizer=synthesizer,
            recognizer=_FakeRecognizer(),
            provenance=provenance,
            validation_index=0,
            output_jsonl=self.root / "validation/results.jsonl",
            summary_json=self.root / "validation/summary.json",
            panel_dir=self.root / "validation/listening-panel",
            temporary_audio_dir=self.root / "validation/audio",
            assignment_manifest=self.root / "voice-assignment.json",
            memorization_path=self.root / "memorization",
            wandb_logger=_logger(api, accelerator, self.root),
        )
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            api.evaluate_checkpoint(request)
        tracked = [request.output_jsonl, request.summary_json, request.panel_dir / "voice_00.wav", request.panel_dir / "manifest.json", request.summary_json.with_name("validation-success.json")]
        original = {path: path.read_bytes() for path in tracked}

        def restore():
            for path, content in original.items():
                path.write_bytes(content)

        def reseal():
            seal_path = request.summary_json.with_name("validation-success.json")
            seal = json.loads(seal_path.read_text())
            seal["artifacts"]["results_jsonl"] = __import__("hashlib").sha256(request.output_jsonl.read_bytes()).hexdigest()
            seal["artifacts"]["summary_json"] = __import__("hashlib").sha256(request.summary_json.read_bytes()).hexdigest()
            api.atomic_write_json(seal_path, seal)

        with self.subTest("summary"):
            summary = json.loads(request.summary_json.read_text())
            summary["metrics"]["utt-wer"] = 99.0
            api.atomic_write_json(request.summary_json, summary)
            reseal()
            with mock.patch.object(api, "require_memorization_gate", return_value=object()):
                with self.assertRaisesRegex(api.EvaluationIntegrityError, "recomputed"):
                    api.evaluate_checkpoint(request)
            restore()

        with self.subTest("jsonl"):
            rows = [json.loads(line) for line in request.output_jsonl.read_text().splitlines()]
            rows[0]["hypothesis"] = "подмена"
            api._atomic_write_jsonl(request.output_jsonl, rows)
            summary = json.loads(request.summary_json.read_text())
            summary["results_sha256"] = __import__("hashlib").sha256(request.output_jsonl.read_bytes()).hexdigest()
            api.atomic_write_json(request.summary_json, summary)
            reseal()
            with mock.patch.object(api, "require_memorization_gate", return_value=object()):
                with self.assertRaisesRegex(api.EvaluationIntegrityError, "row semantics"):
                    api.evaluate_checkpoint(request)
            restore()

        with self.subTest("panel"):
            panel = request.panel_dir / "voice_00.wav"
            panel.write_bytes(panel.read_bytes() + b"tampered-but-still-pcm")
            with mock.patch.object(api, "require_memorization_gate", return_value=object()):
                with self.assertRaisesRegex(api.EvaluationIntegrityError, "panel WAV"):
                    api.evaluate_checkpoint(request)
            restore()

    def test_existing_assignment_manifest_rejects_changed_mapping(self) -> None:
        api = _api()
        accelerator = _FakeAccelerator()
        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=accelerator,
            synthesizer=_FakeSynthesizer(),
            recognizer=_FakeRecognizer(),
            provenance=_provenance(api),
            validation_index=0,
            output_jsonl=self.root / "results.jsonl",
            summary_json=self.root / "summary.json",
            panel_dir=self.root / "panel",
            temporary_audio_dir=self.root / "audio",
            assignment_manifest=self.root / "mapping.json",
            memorization_path=self.root / "memorization",
            wandb_logger=_logger(api, accelerator, self.root),
        )
        request.assignment_manifest.write_text(
            json.dumps({"format_version": 1, "assignment_sha256": "0" * 64, "rows": 2_000, "voices": 20}),
            encoding="utf-8",
        )
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            with self.assertRaisesRegex(api.EvaluationIntegrityError, "assignment"):
                api.evaluate_checkpoint(request)

    def test_wandb_logger_uses_existing_run_and_commits_each_index_once(self) -> None:
        api = _api()
        report = _wandb_report(api, self.root, self.prompts_20)
        accelerator = _FakeAccelerator()
        logger = _logger(api, accelerator, self.root)

        logger.log(report, 0)
        self.assertGreaterEqual(accelerator.run.scan_calls, 3)
        logger.log(report, 0)

        self.assertEqual(len(accelerator.run.media_logs), 1)
        self.assertEqual(accelerator.run.media_logs[0][1], {"step": 0, "commit": False})
        self.assertEqual(len(accelerator.scalar_logs), 1)
        self.assertEqual(accelerator.scalar_logs[0][1], 0)
        self.assertEqual(accelerator.scalar_logs[0][0]["utt-wer"], 0.1)
        self.assertEqual(accelerator.scalar_logs[0][0]["category/code/num-wer"], 0.6)
        manifest = json.loads((self.root / "wandb-run.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], "run-123")
        commit = json.loads((self.root / "wandb-validation-commits/validation-00.json").read_text(encoding="utf-8"))
        self.assertTrue(commit["committed"])
        self.assertNotIn("secret", json.dumps(manifest) + json.dumps(commit))

    def test_wandb_remote_marker_survives_success_then_local_ledger_crash(self) -> None:
        api = _api()
        report = _wandb_report(api, self.root, self.prompts_20)
        accelerator = _FakeAccelerator()
        writes = 0

        def crash_once(path, value):
            nonlocal writes
            writes += 1
            if writes == 1:
                raise OSError("ledger disk failure")
            api.atomic_write_json(path, value)

        logger = api.WandbValidationLogger(
            accelerator,
            self.root / "wandb-run.json",
            table_factory=lambda rows: ("table",),
            audio_factory=lambda path: ("audio", Path(path).name),
            ledger_writer=crash_once,
            remote_history_reader=lambda run, keys: run.scan_history(keys=keys),
        )
        with self.assertRaisesRegex(api.WandbSyncError, "preserved"):
            logger.log(report, 0)
        self.assertEqual(len(accelerator.run.history), 1)
        scalar_calls = len(accelerator.scalar_logs)
        media_calls = len(accelerator.run.media_logs)

        _logger(api, accelerator, self.root).log(report, 0)

        self.assertEqual(len(accelerator.scalar_logs), scalar_calls)
        self.assertEqual(len(accelerator.run.media_logs), media_calls)
        self.assertTrue(json.loads((self.root / "wandb-validation-commits/validation-00.json").read_text())["committed"])

    def test_wandb_scalar_failure_after_media_attempt_resumes_one_remote_step(self) -> None:
        api = _api()
        report = _wandb_report(api, self.root, self.prompts_20)
        accelerator = _FakeAccelerator()
        accelerator.run.fail_scalars = True
        logger = _logger(api, accelerator, self.root)

        with self.assertRaises(api.WandbSyncError):
            logger.log(report, 0)
        accelerator.run.fail_scalars = False
        logger.log(report, 0)

        self.assertEqual(len(accelerator.run.history), 1)
        history = accelerator.run.history[0]
        self.assertEqual(history["_step"], 0)
        self.assertEqual(history["validation/commit/00/scalars"], history["validation/commit/00/media"])

    def test_wandb_later_launch_requires_resume_must_for_persisted_run(self) -> None:
        api = _api()
        manifest = self.root / "wandb-run.json"
        manifest.write_text(json.dumps({"format_version": 1, "run_id": "run-123"}), encoding="utf-8")
        run = type("Run", (), {"id": "run-123", "dir": str(self.root / "wandb"), "resumed": False})()
        accelerator = type("Accelerator", (), {"get_tracker": lambda self, name, unwrap=False: run})()
        logger = api.WandbValidationLogger(accelerator, manifest)
        report = type("Report", (), {"validation_index": 1, "output_jsonl": self.root / "missing"})()

        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "secret"}, clear=True):
            with self.assertRaisesRegex(api.WandbSyncError, "resume=.must"):
                logger.log(report, 1)

        self.assertEqual(
            api.WandbValidationLogger.resume_init_kwargs(manifest),
            {"wandb": {"id": "run-123", "resume": "must"}},
        )

    def test_wandb_requires_environment_credential_before_logging(self) -> None:
        api = _api()
        accelerator = type("Accelerator", (), {})()
        logger = api.WandbValidationLogger(accelerator, self.root / "wandb-run.json")

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(api.WandbSyncError, "WANDB_API_KEY"):
                logger.log(type("Report", (), {})(), 0)

    def test_cosyvoice_synthesizer_uses_current_llm_and_frozen_flow_hift(self) -> None:
        api = _api()

        class Frozen:
            def parameters(self):
                return [type("Parameter", (), {"requires_grad": False})()]

        full_model = type("FullModel", (), {"llm": object(), "flow": Frozen(), "hift": Frozen()})()

        class Pipeline:
            sample_rate = 24_000
            model = full_model

            def __init__(self) -> None:
                self.calls = []

            def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
                self.calls.append(("add", prompt_text, prompt_wav, zero_shot_spk_id))
                return True

            def inference_zero_shot(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                yield {"tts_speech": np.asarray([[0.0, 0.25, -0.25]], dtype=np.float32)}

        pipeline = Pipeline()
        current_llm = object()
        accelerator = type("Accelerator", (), {"unwrap_model": lambda self, model: model})()
        synthesizer = api.CosyVoiceSynthesizer(pipeline, current_llm, accelerator)
        item = api.build_voice_assignment(self.rows_2000, self.prompts_20)[0]
        output = self.root / "synthesized.wav"

        synthesizer.synthesize(item, output)

        self.assertIs(full_model.llm, current_llm)
        self.assertTrue(api._is_pcm_24khz_mono(output))
        self.assertEqual(
            pipeline.calls,
            [
                ("add", item.prompt_text, str(item.prompt_wav), "validation:voice_00"),
                (
                    (item.stressed, item.prompt_text, str(item.prompt_wav)),
                    {"zero_shot_spk_id": "validation:voice_00", "stream": False},
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
