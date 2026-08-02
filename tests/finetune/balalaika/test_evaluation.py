"""Fake-only tests for distributed hard-number synthesis and publication."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
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


class _FakeRecognizer:
    def transcribe(self, paths):
        return ["код семь готов" for _ in paths]


class _FakeSynthesizer:
    def __init__(self) -> None:
        self.generated: list[int] = []

    def synthesize(self, item, destination) -> None:
        self.generated.append(item.benchmark_id)
        _write_wav(destination)


class _FakeAccelerator:
    num_processes = 8
    process_index = 0
    local_process_index = 0
    is_main_process = True

    def __init__(self, remote_records=None) -> None:
        self.remote_records = list(remote_records or [])

    @contextmanager
    def split_between_processes(self, items):
        yield list(items)[:250]

    def gather_object(self, records):
        return [*records, *self.remote_records]

    def wait_for_everyone(self) -> None:
        return None

    def broadcast_object_list(self, values, from_process=0) -> None:
        return None


class _FailingLogger:
    def log(self, report, validation_index) -> None:
        raise _api().WandbSyncError("offline")


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows_2000 = _rows()
        self.prompts_20 = _prompts(self.root / "prompts")

    def tearDown(self) -> None:
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
        remote = [
            {
                "benchmark_id": index,
                "hypothesis": "код семь готов",
                "audio_path": str(self.root / "missing-remote-audio.wav"),
                "audio_sha256": "a" * 64,
                "generation_latency_seconds": 0.0,
                "asr_latency_seconds": 0.0,
            }
            for index in range(251, 2_001)
        ]
        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=_FakeAccelerator(remote),
            synthesizer=synthesizer,
            recognizer=_FakeRecognizer(),
            validation_index=0,
            output_jsonl=self.root / "validation-00/results.jsonl",
            summary_json=self.root / "validation-00/summary.json",
            panel_dir=self.root / "validation-00/listening-panel",
            temporary_audio_dir=self.root / "validation-00/audio",
            assignment_manifest=self.root / "voice-assignment.json",
            memorization_path=self.root / "memorization",
            wandb_logger=_FailingLogger(),
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
            },
        )
        self.assertEqual(first["number_span"]["hypothesis_text"], "семь")
        self.assertEqual(first["edit_counts"]["number"]["word"]["distance"], 0)
        self.assertEqual(len(list(request.panel_dir.glob("*.wav"))), 20)
        self.assertEqual(synthesizer.generated, list(range(1, 251)))
        self.assertFalse(request.temporary_audio_dir.exists())

        successful = []
        resumed = api.EvaluationRequest(**{**request.__dict__, "wandb_logger": type("Logger", (), {"log": lambda _, report, index: successful.append((report.row_count, index))})()})
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            report = api.evaluate_checkpoint(resumed)
        self.assertEqual(synthesizer.generated, list(range(1, 251)))
        self.assertEqual(successful, [(2_000, 0)])
        self.assertEqual(report.metrics["utt-wer"], 0.0)
        self.assertEqual(report.metrics["num-wer"], 0.0)

    def test_distributed_evaluation_refuses_wrong_local_shard_count(self) -> None:
        api = _api()

        class ShortAccelerator(_FakeAccelerator):
            @contextmanager
            def split_between_processes(self, items):
                yield list(items)[:249]

        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=ShortAccelerator(),
            synthesizer=_FakeSynthesizer(),
            recognizer=_FakeRecognizer(),
            validation_index=0,
            output_jsonl=self.root / "results.jsonl",
            summary_json=self.root / "summary.json",
            panel_dir=self.root / "panel",
            temporary_audio_dir=self.root / "audio",
            assignment_manifest=self.root / "mapping.json",
            memorization_path=self.root / "memorization",
        )
        with mock.patch.object(api, "require_memorization_gate", return_value=object()):
            with self.assertRaisesRegex(api.EvaluationIntegrityError, "250"):
                api.evaluate_checkpoint(request)

    def test_partial_local_publication_resumes_without_regeneration(self) -> None:
        api = _api()
        synthesizer = _FakeSynthesizer()
        remote = [
            {
                "benchmark_id": index,
                "hypothesis": "код семь готов",
                "audio_path": str(self.root / "remote.wav"),
                "audio_sha256": "a" * 64,
                "generation_latency_seconds": 0.0,
                "asr_latency_seconds": 0.0,
            }
            for index in range(251, 2_001)
        ]
        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=_FakeAccelerator(remote),
            synthesizer=synthesizer,
            recognizer=_FakeRecognizer(),
            validation_index=0,
            output_jsonl=self.root / "validation/results.jsonl",
            summary_json=self.root / "validation/summary.json",
            panel_dir=self.root / "validation/listening-panel",
            temporary_audio_dir=self.root / "validation/audio",
            assignment_manifest=self.root / "voice-assignment.json",
            memorization_path=self.root / "memorization",
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

    def test_existing_assignment_manifest_rejects_changed_mapping(self) -> None:
        api = _api()
        request = api.EvaluationRequest(
            rows=self.rows_2000,
            prompts=self.prompts_20,
            accelerator=_FakeAccelerator(),
            synthesizer=_FakeSynthesizer(),
            recognizer=_FakeRecognizer(),
            validation_index=1,
            output_jsonl=self.root / "results.jsonl",
            summary_json=self.root / "summary.json",
            panel_dir=self.root / "panel",
            temporary_audio_dir=self.root / "audio",
            assignment_manifest=self.root / "mapping.json",
            memorization_path=self.root / "memorization",
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
        output = self.root / "results.jsonl"
        output.write_text("{}\n", encoding="utf-8")
        panel = {f"voice_{index:02d}": self.prompts_20[index]["audio_path"] for index in range(20)}
        report = api.EvaluationReport(
            validation_index=0,
            output_jsonl=output,
            summary_json=self.root / "summary.json",
            panel_dir=self.root / "panel",
            assignment_checksum="a" * 64,
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
            panel_audio=panel,
            generation_latency_seconds=1.25,
            asr_latency_seconds=0.25,
        )

        class Run:
            id = "run-123"
            dir = str(self.root / "wandb/run-123/files")
            resumed = False

            def __init__(self) -> None:
                self.media_logs = []

            def log(self, values, **kwargs) -> None:
                self.media_logs.append((values, kwargs))

        run = Run()

        class Accelerator:
            def __init__(self) -> None:
                self.scalar_logs = []

            def get_tracker(self, name, unwrap=False):
                if (name, unwrap) != ("wandb", True):
                    raise AssertionError((name, unwrap))
                return run

            def log(self, values, step=None) -> None:
                self.scalar_logs.append((values, step))

        accelerator = Accelerator()
        logger = api.WandbValidationLogger(
            accelerator,
            self.root / "wandb-run.json",
            table_factory=lambda rows: ("table", tuple(row["benchmark_id"] for row in rows)),
            audio_factory=lambda path: ("audio", Path(path).name),
        )
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "secret"}, clear=True):
            logger.log(report, 0)
            logger.log(report, 0)

        self.assertEqual(len(run.media_logs), 1)
        self.assertEqual(run.media_logs[0][1], {"step": 0, "commit": False})
        self.assertEqual(len(accelerator.scalar_logs), 1)
        self.assertEqual(accelerator.scalar_logs[0][1], 0)
        self.assertEqual(accelerator.scalar_logs[0][0]["utt-wer"], 0.1)
        self.assertEqual(accelerator.scalar_logs[0][0]["category/code/num-wer"], 0.6)
        manifest = json.loads((self.root / "wandb-run.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], "run-123")
        commit = json.loads((self.root / "wandb-validation-commits/validation-00.json").read_text(encoding="utf-8"))
        self.assertTrue(commit["committed"])
        self.assertNotIn("secret", json.dumps(manifest) + json.dumps(commit))

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
