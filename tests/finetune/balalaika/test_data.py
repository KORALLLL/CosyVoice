"""Fixture-only tests for cached CosyVoice3 LLM training batches."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cosyvoice.finetune.balalaika.cache import CacheManifest
from cosyvoice.finetune.balalaika.config import PhaseSpec


class FakeTokenizer:
    """Small tokenizer fake that exposes the CosyVoice3 end-of-prompt token."""

    end_of_prompt_id = 151646

    def __init__(self) -> None:
        self.allowed_special: list[str] = []

    def encode(self, value: str, *, allowed_special: str) -> list[int]:
        self.allowed_special.append(allowed_special)
        if "<|endofprompt|>" in value:
            return [11, self.end_of_prompt_id]
        return [index + 1 for index, _ in enumerate(value.split())] or [1]


class EmptyInstructionTokenizer(FakeTokenizer):
    def encode(self, value: str, *, allowed_special: str) -> list[int]:
        if value.startswith("You are"):
            return []
        return super().encode(value, allowed_special=allowed_special)


class FakeAccelerator:
    def __init__(self, rank: int, *, even_batches: bool = False) -> None:
        self.num_processes = 8
        self.process_index = rank
        self.even_batches = even_batches


def _api():
    try:
        return import_module("cosyvoice.finetune.balalaika.data")
    except ModuleNotFoundError as exc:
        raise AssertionError("Balalaika cached-data module is missing") from exc


class CachedDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tokenizer = FakeTokenizer()
        self.cache = self._write_cache(17)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dataset_materializes_requested_verified_parquet_row(self) -> None:
        api = _api()
        dataset = api.CachedSpeechDataset(self.cache, PhaseSpec.for_phase(1), self.tokenizer)

        row = dataset[7]

        self.assertEqual(len(dataset), 17)
        self.assertEqual(row.source_relative_path, "000000/clip-07.mp3")
        self.assertEqual(row.text_token_len, 2)
        self.assertEqual(row.speech_token, (7, 8, 9))

    def test_dataset_reads_one_row_from_a_multi_row_group_without_fragment_table(self) -> None:
        api = _api()
        cache = self._write_cache(6, row_group_size=6)
        dataset = api.CachedSpeechDataset(cache, PhaseSpec.for_phase(1), self.tokenizer)

        class WholeGroupRead:
            def to_table(self, **_):
                raise AssertionError("__getitem__ must not materialize a whole row group")

        dataset._groups[0] = replace(dataset._groups[0], fragment=WholeGroupRead())
        row = dataset[4]

        self.assertEqual(row.source_relative_path, "000000/clip-04.mp3")
        self.assertEqual(row.speech_token, (4, 5, 6))

    def test_dataset_rejects_cached_speech_length_mismatch_on_access(self) -> None:
        api = _api()
        cache = self._write_cache(1, token_length=4)
        dataset = api.CachedSpeechDataset(cache, PhaseSpec.for_phase(1), self.tokenizer)

        with self.assertRaisesRegex(ValueError, "speech_token_len"):
            dataset[0]

    def test_dataset_rejects_empty_instruction_tokenization_on_access(self) -> None:
        api = _api()
        dataset = api.CachedSpeechDataset(self.cache, PhaseSpec.for_phase(1), EmptyInstructionTokenizer())

        with self.assertRaisesRegex(ValueError, "instruction tokenization"):
            dataset[0]

    def test_collator_emits_only_llm_fields(self) -> None:
        api = _api()
        dataset = api.CachedSpeechDataset(self.cache, PhaseSpec.for_phase(1), self.tokenizer)
        collator = api.CosyVoice3Collator(self.tokenizer)

        batch = collator([dataset[0], dataset[1]])

        self.assertEqual(
            set(batch),
            {
                "utts",
                "text",
                "text_token",
                "text_token_len",
                "instruct_token",
                "instruct_token_len",
                "speech_token",
                "speech_token_len",
            },
        )
        self.assertEqual(batch["utts"], ["000000/clip-00.mp3", "000000/clip-01.mp3"])
        self.assertEqual(batch["speech_token"].dtype, torch.int64)
        self.assertEqual(batch["speech_token_len"].tolist(), [3, 3])
        self.assertTrue(all(value == "all" for value in self.tokenizer.allowed_special))

    def test_collator_rejects_instruction_without_end_of_prompt_token(self) -> None:
        api = _api()
        dataset = api.CachedSpeechDataset(self.cache, PhaseSpec.for_phase(1), self.tokenizer)
        bad = dataset[0].__class__(
            source_relative_path="bad",
            text="text",
            instruct="instruction without marker",
            speech_token=(1,),
            speech_token_len=1,
            text_token_len=1,
        )

        with self.assertRaisesRegex(ValueError, "endofprompt"):
            api.CosyVoice3Collator(self.tokenizer)([bad])

    def test_batches_are_rank_disjoint_and_epoch_complete(self) -> None:
        api = _api()
        dataset = api.CachedSpeechDataset(self.cache, PhaseSpec.for_phase(1), self.tokenizer)
        rank_batches = [
            list(
                api.TokenBatchSampler(
                    dataset,
                    rank=rank,
                    seed=1986,
                    epoch=3,
                    max_tokens_per_gpu=12,
                    window_size=5,
                )
            )
            for rank in range(8)
        ]
        flattened = [index for batches in rank_batches for batch in batches for index in batch]

        self.assertEqual(sorted(flattened), list(range(len(dataset))))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(rank_batches, [
            list(api.TokenBatchSampler(dataset, rank=rank, seed=1986, epoch=3, max_tokens_per_gpu=12, window_size=5))
            for rank in range(8)
        ])
        self.assertNotEqual(
            rank_batches,
            [list(api.TokenBatchSampler(dataset, rank=rank, seed=1986, epoch=4, max_tokens_per_gpu=12, window_size=5)) for rank in range(8)],
        )

    def test_packing_charges_independently_padded_text_and_speech(self) -> None:
        api = _api()
        rows = [
            api.CachedRow("text-long", "text", "You are a helpful assistant.<|endofprompt|>", (1,), 1, 100),
            api.CachedRow("speech-long", "text", "You are a helpful assistant.<|endofprompt|>", tuple(range(100)), 100, 1),
        ]
        sampler = api.TokenBatchSampler(rows, rank=0, max_tokens_per_gpu=202, window_size=2)

        self.assertEqual(sampler._pack([0, 1]), [[0], [1]])

    def test_even_accelerate_steps_use_explicit_empty_sync_batches(self) -> None:
        api = _api()
        dataset = api.CachedSpeechDataset(self.cache, PhaseSpec.for_phase(1), self.tokenizer)
        samplers = [
            api.TokenBatchSampler(dataset, rank=rank, seed=1986, epoch=0, max_tokens_per_gpu=12, window_size=5, synchronize_steps=True)
            for rank in range(8)
        ]

        batches = [list(sampler) for sampler in samplers]

        self.assertEqual({len(value) for value in batches}, {max(map(len, batches))})
        self.assertTrue(any([] in value for value in batches))

    def test_phase_dataloader_requires_exactly_eight_accelerate_ranks(self) -> None:
        api = _api()
        accelerator = FakeAccelerator(0)
        accelerator.num_processes = 7

        with self.assertRaisesRegex(ValueError, "exactly eight"):
            api.build_phase_dataloader(self.cache, PhaseSpec.for_phase(1), accelerator, 0, tokenizer=self.tokenizer)

    def _write_cache(
        self,
        count: int,
        *,
        token_length: int | None = None,
        row_group_size: int = 1,
    ) -> CacheManifest:
        phase_dir = self.root / "phase1"
        phase_dir.mkdir(exist_ok=True)
        rows = [
            {
                "source_relative_path": f"000000/clip-{index:02d}.mp3",
                "text": f"sample {index}",
                "instruct": "You are a helpful assistant.<|endofprompt|>",
                "agreement": 0.99,
                "speech_token": [index, index + 1, index + 2],
                "speech_token_len": token_length if index == 0 and token_length is not None else 3,
            }
            for index in range(count)
        ]
        table = pa.table(
            {
                "source_relative_path": pa.array([row["source_relative_path"] for row in rows], type=pa.string()),
                "text": pa.array([row["text"] for row in rows], type=pa.string()),
                "instruct": pa.array([row["instruct"] for row in rows], type=pa.string()),
                "agreement": pa.array([row["agreement"] for row in rows], type=pa.float64()),
                "speech_token": pa.array([row["speech_token"] for row in rows], type=pa.list_(pa.int32())),
                "speech_token_len": pa.array([row["speech_token_len"] for row in rows], type=pa.int32()),
            }
        )
        pq.write_table(table, phase_dir / "shard_000000.parquet", row_group_size=row_group_size)
        return CacheManifest(self.root, {0: self.root / "shard_manifests/shard_000000.json"}, {1: count, 2: 0}, 20)


if __name__ == "__main__":
    unittest.main()
