"""Audio-free, verified cache access and deterministic CosyVoice3 batches."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from .cache import CacheManifest, TOKEN_MAX, TOKEN_MIN
from .config import DEFAULT_SEED, PhaseSpec, RunPaths
from .sources import TEXT_TOKEN_MAX_LENGTH


WORLD_SIZE = 8
DEFAULT_MAX_TOKENS_PER_GPU = 12_000
DEFAULT_LENGTH_WINDOW = 512
_PHASE_COLUMNS = ("source_relative_path", "text", "instruct", "speech_token", "speech_token_len")
_END_OF_PROMPT = "<|endofprompt|>"


class _Tokenizer(Protocol):
    def encode(self, value: str, *, allowed_special: str) -> Sequence[int]: ...


@dataclass(frozen=True)
class CachedRow:
    """One materialized metadata/token cache row, with no audio-bearing fields."""

    source_relative_path: str
    text: str
    instruct: str
    speech_token: tuple[int, ...]
    speech_token_len: int
    text_token_len: int


@dataclass(frozen=True)
class _RowGroup:
    """One Arrow row group and its exclusive global dataset range."""

    fragment: Any
    path: Path
    row_group: int
    stop: int


class CachedSpeechDataset(Dataset[CachedRow]):
    """Access verified phase rows by row group without loading a phase into memory."""

    def __init__(
        self,
        cache: CacheManifest,
        phase: PhaseSpec | int,
        tokenizer: _Tokenizer | None = None,
    ) -> None:
        self.cache = cache
        self.phase = phase.number if isinstance(phase, PhaseSpec) else phase
        if self.phase not in (1, 2):
            raise ValueError(f"phase must be 1 or 2, got {self.phase}")
        self._tokenizer = tokenizer or _load_cosyvoice3_tokenizer()
        paths = _phase_paths(cache, self.phase)
        filesystem = pa.fs.LocalFileSystem(use_mmap=True)
        self._arrow = ds.dataset([str(path) for path in paths], format="parquet", filesystem=filesystem)
        self._groups: list[_RowGroup] = []
        total = 0
        for fragment in self._arrow.get_fragments():
            for group in fragment.split_by_row_group():
                count = sum(row_group.num_rows for row_group in group.row_groups)
                if count < 1:
                    continue
                total += count
                self._groups.append(_RowGroup(group, Path(group.path), group.row_groups[0].id, total))
        expected = cache.phase_rows.get(self.phase)
        if expected is None or total != expected:
            raise ValueError(f"verified phase {self.phase} row count does not match cache manifest")
        self._row_count = total
        self._stops = [group.stop for group in self._groups]

    def __len__(self) -> int:
        return self._row_count

    def __getitem__(self, index: int) -> CachedRow:
        if index < 0:
            index += self._row_count
        if index < 0 or index >= self._row_count:
            raise IndexError(index)
        group_index = bisect_right(self._stops, index)
        group = self._groups[group_index]
        start = 0 if group_index == 0 else self._groups[group_index - 1].stop
        return _cached_row(_read_one_row(group, index - start), self._tokenizer)


def _read_one_row(group: _RowGroup, offset: int) -> dict[str, Any]:
    """Read one row from a memory-mapped Parquet row group without a table scan."""

    reader = pq.ParquetFile(group.path, memory_map=True)
    for position, batch in enumerate(
        reader.iter_batches(batch_size=1, row_groups=[group.row_group], columns=list(_PHASE_COLUMNS))
    ):
        if position == offset:
            values = batch.to_pylist()
            if len(values) == 1:
                return values[0]
            break
    raise ValueError(f"cached row-group offset {offset} could not be materialized")


class TokenBatchSampler(Sampler[list[int]]):
    """Deterministically pack complete batches, then assign batches to eight ranks."""

    def __init__(
        self,
        dataset: Dataset[CachedRow],
        *,
        rank: int,
        seed: int = DEFAULT_SEED,
        epoch: int = 0,
        max_tokens_per_gpu: int = DEFAULT_MAX_TOKENS_PER_GPU,
        window_size: int = DEFAULT_LENGTH_WINDOW,
        synchronize_steps: bool = False,
    ) -> None:
        if rank < 0 or rank >= WORLD_SIZE:
            raise ValueError(f"rank must be in [0, {WORLD_SIZE - 1}]")
        if max_tokens_per_gpu < 1 or window_size < 1:
            raise ValueError("max_tokens_per_gpu and window_size must be positive")
        self.dataset = dataset
        self.rank = rank
        self.seed = seed
        self.epoch = epoch
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.window_size = window_size
        self.synchronize_steps = synchronize_steps
        self._batches = self._rank_batches()

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)

    def _rank_batches(self) -> list[list[int]]:
        indices = list(range(len(self.dataset)))
        random.Random(self.seed + self.epoch).shuffle(indices)
        complete: list[list[int]] = []
        for offset in range(0, len(indices), self.window_size):
            window = indices[offset:offset + self.window_size]
            window.sort(key=self._combined_length)
            complete.extend(self._pack(window))
        ranks = [complete[rank::WORLD_SIZE] for rank in range(WORLD_SIZE)]
        if self.synchronize_steps:
            steps = max((len(value) for value in ranks), default=0)
            for value in ranks:
                value.extend([[] for _ in range(steps - len(value))])
        return ranks[self.rank]

    def _combined_length(self, index: int) -> int:
        row = self.dataset[index]
        return row.text_token_len + row.speech_token_len

    def _pack(self, indices: Sequence[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        max_text = 0
        max_speech = 0
        for index in indices:
            row = self.dataset[index]
            candidate_text = max(max_text, row.text_token_len)
            candidate_speech = max(max_speech, row.speech_token_len)
            if current and (len(current) + 1) * (candidate_text + candidate_speech) > self.max_tokens_per_gpu:
                batches.append(current)
                current = []
                max_text = 0
                max_speech = 0
            current.append(index)
            max_text = max(max_text, row.text_token_len)
            max_speech = max(max_speech, row.speech_token_len)
        if current:
            batches.append(current)
        return batches


class CosyVoice3Collator:
    """Produce exactly the cached-token fields consumed by ``CosyVoice3LM``."""

    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer
        self._end_of_prompt_tokens = tuple(int(token) for token in tokenizer.encode(_END_OF_PROMPT, allowed_special="all"))
        if not self._end_of_prompt_tokens:
            raise ValueError("CosyVoice3 tokenizer did not encode <|endofprompt|>")

    def __call__(self, rows: Sequence[CachedRow]) -> dict[str, Any]:
        if not rows:
            return _empty_batch()
        text_tokens = [self._encode(row.text, "text") for row in rows]
        instruct_tokens = [self._encode_instruction(row.instruct) for row in rows]
        speech_tokens = [torch.tensor(row.speech_token, dtype=torch.long) for row in rows]
        return {
            "utts": [row.source_relative_path for row in rows],
            "text": [row.text for row in rows],
            "text_token": pad_sequence(text_tokens, batch_first=True, padding_value=0),
            "text_token_len": torch.tensor([len(tokens) for tokens in text_tokens], dtype=torch.int64),
            "instruct_token": pad_sequence(instruct_tokens, batch_first=True, padding_value=0),
            "instruct_token_len": torch.tensor([len(tokens) for tokens in instruct_tokens], dtype=torch.int64),
            "speech_token": pad_sequence(speech_tokens, batch_first=True, padding_value=0),
            "speech_token_len": torch.tensor([row.speech_token_len for row in rows], dtype=torch.int64),
        }

    def _encode(self, value: str, field: str) -> torch.Tensor:
        tokens = tuple(int(token) for token in self.tokenizer.encode(value, allowed_special="all"))
        if not tokens:
            raise ValueError(f"{field} tokenization is empty")
        return torch.tensor(tokens, dtype=torch.long)

    def _encode_instruction(self, value: str) -> torch.Tensor:
        if _END_OF_PROMPT not in value:
            raise ValueError("instruction must contain <|endofprompt|>")
        tokens = self._encode(value, "instruction")
        if not _contains(tokens.tolist(), self._end_of_prompt_tokens):
            raise ValueError("instruction tokenization lacks <|endofprompt|>")
        return tokens


def build_phase_dataloader(
    cache: CacheManifest,
    phase: PhaseSpec,
    accelerator: Any,
    epoch: int,
    *,
    tokenizer: _Tokenizer | None = None,
    seed: int = DEFAULT_SEED,
    max_tokens_per_gpu: int = DEFAULT_MAX_TOKENS_PER_GPU,
    window_size: int = DEFAULT_LENGTH_WINDOW,
) -> DataLoader:
    """Build the local rank's deterministic cached-data loader for one phase."""

    if getattr(accelerator, "num_processes", None) != WORLD_SIZE:
        raise ValueError("Balalaika training requires exactly eight Accelerate ranks")
    rank = getattr(accelerator, "process_index", None)
    if not isinstance(rank, int):
        raise ValueError("Accelerate process_index must be an integer")
    resolved_tokenizer = tokenizer or _load_cosyvoice3_tokenizer()
    dataset = CachedSpeechDataset(cache, phase, resolved_tokenizer)
    sampler = TokenBatchSampler(
        dataset,
        rank=rank,
        seed=seed,
        epoch=epoch,
        max_tokens_per_gpu=max_tokens_per_gpu,
        window_size=window_size,
        synchronize_steps=bool(getattr(accelerator, "even_batches", False)),
    )
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=CosyVoice3Collator(resolved_tokenizer), num_workers=0)


def build_memorization_dataloader(
    rows: Sequence[CachedRow],
    *,
    tokenizer: _Tokenizer | None = None,
    batch_size: int = 1,
) -> DataLoader:
    """Build a small deterministic loader for the approved metadata-only prompt rows."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    resolved_tokenizer = tokenizer or _load_cosyvoice3_tokenizer()
    return DataLoader(list(rows), batch_size=batch_size, shuffle=False, collate_fn=CosyVoice3Collator(resolved_tokenizer), num_workers=0)


def _phase_paths(cache: CacheManifest, phase: int) -> tuple[Path, ...]:
    paths = tuple(cache.root / f"phase{phase}" / f"shard_{shard:06d}.parquet" for shard in sorted(cache.shards))
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError(f"verified cache is missing phase {phase} Parquet files")
    return paths


def _cached_row(value: dict[str, Any], tokenizer: _Tokenizer) -> CachedRow:
    source = value.get("source_relative_path")
    text = value.get("text")
    instruct = value.get("instruct")
    tokens = value.get("speech_token")
    length = value.get("speech_token_len")
    if not all(isinstance(item, str) and item for item in (source, text, instruct)):
        raise ValueError("cached row text/instruction/source fields must be nonempty strings")
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("cached row speech_token_len must be a positive integer")
    if not isinstance(tokens, list) or len(tokens) != length:
        raise ValueError("cached row speech_token_len does not match speech_token")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < TOKEN_MIN or token > TOKEN_MAX for token in tokens):
        raise ValueError("cached row has speech token outside the CosyVoice3 vocabulary")
    text_token_len = len(tuple(tokenizer.encode(text, allowed_special="all")))
    if text_token_len < 1 or text_token_len > TEXT_TOKEN_MAX_LENGTH:
        raise ValueError(f"cached row text token length must be in [1, {TEXT_TOKEN_MAX_LENGTH}]")
    if not tuple(tokenizer.encode(instruct, allowed_special="all")):
        raise ValueError("cached row instruction tokenization is empty")
    return CachedRow(source, text, instruct, tuple(tokens), length, text_token_len)


def _contains(tokens: Sequence[int], expected: Sequence[int]) -> bool:
    width = len(expected)
    return any(tuple(tokens[index:index + width]) == tuple(expected) for index in range(len(tokens) - width + 1))


def _empty_batch() -> dict[str, Any]:
    empty_tokens = torch.empty((0, 0), dtype=torch.long)
    empty_lengths = torch.empty((0,), dtype=torch.int64)
    return {
        "utts": [],
        "text": [],
        "text_token": empty_tokens,
        "text_token_len": empty_lengths,
        "instruct_token": empty_tokens.clone(),
        "instruct_token_len": empty_lengths.clone(),
        "speech_token": empty_tokens.clone(),
        "speech_token_len": empty_lengths.clone(),
    }


def _load_cosyvoice3_tokenizer() -> _Tokenizer:
    """Load the same production Qwen tokenizer used by split-plan preflight."""

    from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

    paths = RunPaths.from_env()
    return get_qwen_tokenizer(
        token_path=str(paths.base_model_dir / "CosyVoice-BlankEN"),
        skip_special_tokens=True,
        version="cosyvoice3",
    )
