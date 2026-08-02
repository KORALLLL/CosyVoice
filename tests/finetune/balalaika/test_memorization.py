"""Tests for the mandatory four-sample token memorization gate."""

from __future__ import annotations

import unittest
import json
import hashlib
from pathlib import Path
import tempfile
from contextlib import nullcontext

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cosyvoice.finetune.balalaika.cache import CacheManifest
from cosyvoice.finetune.balalaika.memorization import (
    SampleAccuracy,
    MemorizationCheck,
    MemorizationReport,
    MemorizationRequest,
    _publish_success,
    memorization_passed,
    require_memorization_gate,
    run_memorization_gate,
    select_memorization_rows,
)


class MemorizationPassTests(unittest.TestCase):
    def test_aggregate_100_percent_cannot_hide_sample_failure(self):
        checks = [SampleAccuracy("a", 10, 10), SampleAccuracy("b", 9, 10)]

        self.assertFalse(memorization_passed([checks, checks, checks]))

    def test_requires_three_consecutive_exact_checks(self):
        exact = [SampleAccuracy(key, 10, 10) for key in "abcd"]
        almost = [SampleAccuracy("a", 9, 10), *exact[1:]]

        self.assertFalse(memorization_passed([exact, almost, exact]))
        self.assertTrue(memorization_passed([exact, exact, exact]))


class MemorizationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan = self.root / "split_plan"
        self.plan.mkdir()
        self.cache_root = self.root / "cache"
        self.cache_root.mkdir()
        self.rows = [
            ("000000/p1-short-a.mp3", 1, 0.50, 3),
            ("000000/p1-short-b.mp3", 1, 0.60, 3),
            ("000000/p1-long-a.mp3", 1, 0.70, 9),
            ("000000/p1-long-b.mp3", 1, 0.80, 9),
            ("000000/p2-short-a.mp3", 2, 0.95, 4),
            ("000000/p2-short-b.mp3", 2, 0.96, 4),
            ("000000/p2-long-a.mp3", 2, 0.99, 10),
            ("000000/p2-long-b.mp3", 2, 1.00, 10),
        ]
        with (self.plan / "shard_000000.jsonl").open("w", encoding="utf-8") as handle:
            for source, phase, agreement, length in self.rows:
                handle.write(json.dumps({
                    "source_relative_path": source,
                    "text": f"text {source}",
                    "instruct": "You are a helpful assistant.<|endofprompt|>",
                    "agreement": agreement,
                    "phase": phase,
                    "reserved": False,
                    "reservation_score": None,
                    "model_limit_exclusion": None,
                    "text_token_count": 2,
                }) + "\n")
        for phase in (1, 2):
            phase_rows = [row for row in self.rows if row[1] == phase]
            directory = self.cache_root / f"phase{phase}"
            directory.mkdir()
            pq.write_table(
                pa.table({
                    "source_relative_path": [row[0] for row in phase_rows],
                    "text": [f"text {row[0]}" for row in phase_rows],
                    "instruct": ["You are a helpful assistant.<|endofprompt|>"] * len(phase_rows),
                    "speech_token": [list(range(row[3])) for row in phase_rows],
                    "speech_token_len": [row[3] for row in phase_rows],
                }),
                directory / "shard_000000.parquet",
            )
        manifest = self.cache_root / "shard_manifests" / "shard_000000.json"
        manifest.parent.mkdir()
        manifest.write_text("{}", encoding="utf-8")
        self.cache = CacheManifest(
            self.cache_root, {0: manifest}, {1: 4, 2: 4}, 0,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_selects_seed_bound_short_and_long_rows_from_both_phases(self):
        selected = select_memorization_rows(self.plan, self.cache, seed=1986)

        self.assertEqual(len(selected), 4)
        self.assertEqual({row.source_relative_path.split("-")[0].split("/")[-1] for row in selected}, {"p1", "p2"})
        self.assertEqual(sorted(row.speech_token_len for row in selected if "p1-" in row.source_relative_path), [3, 9])
        self.assertEqual(sorted(row.speech_token_len for row in selected if "p2-" in row.source_relative_path), [4, 10])
        self.assertEqual(selected, select_memorization_rows(self.plan, self.cache, seed=1986))

    def test_rejects_cached_row_that_no_longer_matches_split_plan(self):
        path = self.cache_root / "phase1" / "shard_000000.parquet"
        table = pq.read_table(path).to_pydict()
        table["text"][0] = "changed"
        pq.write_table(pa.table(table), path)

        with self.assertRaisesRegex(ValueError, "split plan"):
            select_memorization_rows(self.plan, self.cache)

    def test_gate_requires_three_full_per_sample_checks_and_publishes_only_success(self):
        selected = select_memorization_rows(self.plan, self.cache)
        output = self.root / "run"
        model = _MemorizingModel({row.source_relative_path: row.speech_token_len for row in selected})
        request = MemorizationRequest(
            split_plan=self.plan,
            cache=self.cache,
            output_root=output,
            base_model_dir=self.root / "base",
            max_steps=3,
            check_every=1,
            batch_size=2,
            learning_rate=2e-4,
            max_grad_norm=0.7,
            model_loader=lambda _: model,
            adapter_injector=lambda value, _: value,
            audit_fn=lambda _: {"trainable_parameters": ["lora.fixture"]},
            dataloader_factory=_fixture_loader,
            accelerator_factory=lambda **_: _FakeAccelerator(),
            wav_writer=lambda _, rows, root: [
                (root / f"{index}.wav").write_bytes(b"RIFF") for index, _ in enumerate(rows)
            ],
        )

        report = run_memorization_gate(request)

        self.assertEqual(report.steps, 3)
        self.assertEqual([check.check_index for check in report.checks], [1, 2, 3])
        self.assertTrue(all(sample.exact for check in report.checks for sample in check.samples))
        manifest = json.loads((output / "memorization" / "memorization_manifest.json").read_text())
        self.assertEqual(manifest["steps"], report.steps)
        seal = json.loads((report.path / "memorization_success.json").read_text())
        self.assertEqual(seal["manifest_sha256"], hashlib.sha256((report.path / "memorization_manifest.json").read_bytes()).hexdigest())
        self.assertEqual(manifest["checks"][-1]["check_index"], 3)
        config = manifest["provenance"]["run_config"]
        self.assertEqual(
            config,
            {
                "max_steps": 3,
                "learning_rate": 2e-4,
                "max_grad_norm": 0.7,
                "optimizer": {
                    "class": "torch.optim.adamw.AdamW",
                    "betas": [0.9, 0.999],
                    "eps": 1e-08,
                    "weight_decay": 0.01,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": None,
                },
                "scheduler": {"kind": "constant-v1"},
                "mixed_precision": "bf16",
                "gradient_accumulation_steps": 1,
                "world_size": 8,
                "local_batch_size": 2,
                "effective_batch_size": 16,
                "check_every": 1,
                "required_consecutive_checks": 3,
                "seed": 1986,
                "dataloader_identity": "cosyvoice.balalaika.memorization-loader:v1",
                "tokenizer_identity": "cosyvoice.qwen-tokenizer:v3",
            },
        )
        self.assertEqual(
            manifest["provenance"]["adapter"],
            {
                "settings": {"r": 64, "alpha": 128, "dropout": 0.05, "bias": "none"},
                "trainable_inventory": {"trainable_parameters": ["lora.fixture"]},
            },
        )
        verified = require_memorization_gate(report.path, expected_provenance=manifest["provenance"])
        self.assertEqual(verified.path, report.path)
        self.assertEqual(set(manifest["evidence"]), {"cross_entropy.json", "teacher_forced_tokens.json", *manifest["generated_wavs"]})
        token_evidence = json.loads((output / "memorization" / "teacher_forced_tokens.json").read_text())
        self.assertEqual(set(token_evidence), {row.source_relative_path for row in selected})
        self.assertTrue(all("predictions" in item and "targets" in item for item in token_evidence.values()))
        self.assertFalse((output / ".memorization.incomplete").exists())

    def test_evidence_verifier_rejects_tampered_artifact_and_provenance(self):
        selected = select_memorization_rows(self.plan, self.cache)
        request = MemorizationRequest(
            split_plan=self.plan,
            cache=self.cache,
            output_root=self.root / "tamper-run",
            base_model_dir=self.root / "base",
            max_steps=3,
            check_every=1,
            model_loader=lambda _: _MemorizingModel({row.source_relative_path: row.speech_token_len for row in selected}),
            adapter_injector=lambda value, _: value,
            audit_fn=lambda _: {"trainable_parameters": ["lora.fixture"]},
            dataloader_factory=_fixture_loader,
            accelerator_factory=lambda **_: _FakeAccelerator(),
        )
        report = run_memorization_gate(request)
        manifest_path = report.path / "memorization_manifest.json"
        manifest = json.loads(manifest_path.read_text())

        with self.assertRaisesRegex(ValueError, "provenance"):
            require_memorization_gate(report.path, expected_provenance={"changed": True})
        (report.path / "cross_entropy.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            require_memorization_gate(report.path, expected_provenance=manifest["provenance"])

    def test_publish_success_returns_completed_report(self):
        selected = select_memorization_rows(self.plan, self.cache)
        root = self.root / "direct-publish"
        root.mkdir()
        request = MemorizationRequest(
            split_plan=self.plan,
            cache=self.cache,
            output_root=self.root,
            base_model_dir=self.root / "base",
            max_steps=3,
            check_every=1,
            model_loader=lambda _: _MemorizingModel({}),
            dataloader_factory=_fixture_loader,
            accelerator_factory=lambda **_: _FakeAccelerator(),
        )
        exact = tuple(SampleAccuracy(row.source_relative_path, row.speech_token_len, row.speech_token_len) for row in selected)

        report = _publish_success(
            root, request, selected,
            [MemorizationCheck(index, index, exact) for index in (1, 2, 3)],
            {}, [0.0, 0.0, 0.0], {"trainable_parameters": ["lora.fixture"]},
            "b" * 64, _MemorizingModel({}), 3,
        )

        self.assertIsInstance(report, MemorizationReport)
        self.assertEqual(report.steps, 3)

    def test_verifier_rejects_seal_and_internal_trajectory_or_provenance_tampering(self):
        selected = select_memorization_rows(self.plan, self.cache)
        request = MemorizationRequest(
            split_plan=self.plan, cache=self.cache, output_root=self.root / "self-validate",
            base_model_dir=self.root / "base", max_steps=3, check_every=1,
            model_loader=lambda _: _MemorizingModel({row.source_relative_path: row.speech_token_len for row in selected}),
            adapter_injector=lambda value, _: value, audit_fn=lambda _: {"trainable_parameters": ["lora.fixture"]},
            dataloader_factory=_fixture_loader, accelerator_factory=lambda **_: _FakeAccelerator(),
        )
        report = run_memorization_gate(request)
        manifest_path = report.path / "memorization_manifest.json"
        original = json.loads(manifest_path.read_text())
        (report.path / "memorization_success.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "seal"):
            require_memorization_gate(report.path)

        mutations = {
            "check-index": lambda value: value["checks"][1].update(check_index=9),
            "optimizer-step": lambda value: value["checks"][1].update(optimizer_step=99),
            "steps": lambda value: value.update(steps=2),
            "row-length": lambda value: value["provenance"]["rows"][0].update(speech_token_len=0),
            "top-row": lambda value: value["rows"][0].update(speech_token_len=999),
            "row-hash": lambda value: value["provenance"]["rows"][0].update(speech_token_sha256="BAD"),
            "cache": lambda value: value["provenance"].update(cache_checksum="BAD"),
            "base": lambda value: value["provenance"].update(base_checkpoint_sha256="BAD"),
            "tokenizer": lambda value: value["provenance"]["run_config"].update(tokenizer_identity=""),
            "adapter": lambda value: value["provenance"].update(adapter={"settings": {}, "trainable_inventory": []}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = json.loads(json.dumps(original))
                mutate(changed)
                manifest_path.write_text(json.dumps(changed), encoding="utf-8")
                _write_test_seal(report.path)
                with self.assertRaises(ValueError):
                    require_memorization_gate(report.path)

    def test_gate_does_not_publish_a_failed_attempt(self):
        output = self.root / "failed-run"
        request = MemorizationRequest(
            split_plan=self.plan,
            cache=self.cache,
            output_root=output,
            base_model_dir=self.root / "base",
            max_steps=2,
            check_every=1,
            model_loader=lambda _: _MemorizingModel({}, exact_after=99),
            adapter_injector=lambda value, _: value,
            audit_fn=lambda _: {"trainable_parameters": ["lora.fixture"]},
            dataloader_factory=_fixture_loader,
            accelerator_factory=lambda **_: _FakeAccelerator(),
            wav_writer=lambda *_: None,
        )

        with self.assertRaisesRegex(RuntimeError, "did not reach"):
            run_memorization_gate(request)

        self.assertFalse((output / "memorization").exists())

    def test_nonfinite_training_loss_stops_before_backward_or_optimizer_progress(self):
        selected = select_memorization_rows(self.plan, self.cache)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                model = _MemorizingModel(
                    {row.source_relative_path: row.speech_token_len for row in selected},
                    loss_multiplier=value,
                )
                accelerator = _FakeAccelerator()
                request = MemorizationRequest(
                    split_plan=self.plan,
                    cache=self.cache,
                    output_root=self.root / f"nonfinite-{value}",
                    base_model_dir=self.root / "base",
                    max_steps=3,
                    check_every=1,
                    model_loader=lambda _, model=model: model,
                    adapter_injector=lambda value, _: value,
                    audit_fn=lambda _: {"trainable_parameters": ["lora.fixture"]},
                    dataloader_factory=_fixture_loader,
                    accelerator_factory=lambda **_: accelerator,
                )

                with self.assertRaisesRegex(RuntimeError, "loss must be finite"):
                    run_memorization_gate(request)

                self.assertEqual(accelerator.backward_calls, 0)
                self.assertEqual(accelerator.clip_calls, 0)
                self.assertEqual(model.weight.item(), 1.0)
                self.assertFalse((request.output_root / "memorization").exists())

    def test_wave_artifacts_do_not_decide_the_token_accuracy_gate(self):
        selected = select_memorization_rows(self.plan, self.cache)
        request = MemorizationRequest(
            split_plan=self.plan,
            cache=self.cache,
            output_root=self.root / "no-wavs",
            base_model_dir=self.root / "base",
            max_steps=3,
            check_every=1,
            model_loader=lambda _: _MemorizingModel({row.source_relative_path: row.speech_token_len for row in selected}),
            adapter_injector=lambda value, _: value,
            audit_fn=lambda _: {"trainable_parameters": ["lora.fixture"]},
            dataloader_factory=_fixture_loader,
            accelerator_factory=lambda **_: _FakeAccelerator(),
        )

        report = run_memorization_gate(request)

        self.assertEqual(report.steps, 3)
        manifest = json.loads((report.path / "memorization_manifest.json").read_text())
        self.assertEqual(manifest["generated_wavs"], [])


class _FakeAccelerator:
    device = torch.device("cpu")
    num_processes = 8
    is_main_process = True

    def __init__(self):
        self.backward_calls = 0
        self.clip_calls = 0

    def prepare(self, *values):
        return values

    def accumulate(self, _):
        return nullcontext()

    def backward(self, loss):
        self.backward_calls += 1
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        self.clip_calls += 1
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        return None


class _MemorizingModel(torch.nn.Module):
    def __init__(self, token_lengths, exact_after=0, loss_multiplier=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.token_lengths = token_lengths
        self.exact_after = exact_after
        self.loss_multiplier = loss_multiplier
        self.calls = 0
        self._balalaika_base_checkpoint_sha256 = "a" * 64
        self._balalaika_lora_settings = object()

    def forward(self, batch, device):
        self.calls += int(self.training)
        lengths = [self.token_lengths.get(source, 1) for source in batch["utts"]]
        totals = torch.tensor(lengths, dtype=torch.int64, device=device)
        correct = totals if self.calls >= self.exact_after else torch.clamp(totals - 1, min=0)
        return {
            "loss": self.weight * self.loss_multiplier,
            "correct_tokens_per_sample": correct,
            "target_tokens_per_sample": totals,
            "teacher_forced_predictions": torch.zeros((len(lengths), 1), dtype=torch.int64),
            "teacher_forced_targets": torch.zeros((len(lengths), 1), dtype=torch.int64),
        }


def _fixture_loader(rows, **_):
    return [{"utts": [row.source_relative_path]} for row in rows]


def _write_test_seal(root: Path) -> None:
    manifest = root / "memorization_manifest.json"
    (root / "memorization_success.json").write_text(json.dumps({
        "format_version": 1,
        "manifest": "memorization_manifest.json",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
