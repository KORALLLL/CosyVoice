"""Deterministic, fail-closed metrics for Russian hard-number evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import unicodedata
from typing import Iterable, Mapping, Sequence


class NumberSpanError(ValueError):
    """Raised when a number phrase cannot be anchored without guessing."""


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def distance(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.distance / self.reference_units


@dataclass(frozen=True)
class EditCounts:
    """Parallel word and character edit counts for one text region."""

    word: ErrorCounts
    character: ErrorCounts


@dataclass(frozen=True)
class NumberSpan:
    """A half-open token interval in normalized reference text."""

    reference_start: int
    reference_end: int
    category: str = "unknown"


@dataclass(frozen=True)
class RowScores:
    utterance: EditCounts
    number: EditCounts
    category: str


@dataclass(frozen=True)
class MetricSummary:
    """Micro and macro WER/CER, including equivalent category breakdowns."""

    micro: Mapping[str, float]
    macro: Mapping[str, float]
    by_category: Mapping[str, Mapping[str, float]]

    def __getitem__(self, key: str) -> float:
        return self.micro[key]

    def as_dict(self) -> dict[str, object]:
        values: dict[str, object] = dict(self.micro)
        values.update({f"macro-{name}": value for name, value in self.macro.items()})
        values["by_category"] = {name: dict(metrics) for name, metrics in self.by_category.items()}
        return values


def normalize_asr_text(text: str) -> str:
    """Apply the fixed case, ``ё``, punctuation, and whitespace ASR policy."""

    if not isinstance(text, str):
        raise TypeError("ASR text must be a string")
    normalized = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    normalized = "".join(" " if unicodedata.category(char).startswith("P") or char == "_" else char for char in normalized)
    return " ".join(normalized.split())


def extract_number_span(row: Mapping[str, object]) -> NumberSpan:
    """Locate one spoken number phrase using its raw-text context anchors."""

    raw_text = _required_string(row, "stressed")
    hard_number = _required_string(row, "hard_number")
    gold = _required_string(row, "normalized_gold")
    raw_tokens = normalize_asr_text(raw_text).split()
    number_tokens = normalize_asr_text(hard_number).split()
    gold_tokens = normalize_asr_text(gold).split()
    if not number_tokens:
        raise NumberSpanError("hard number is empty after normalization")

    raw_occurrences = _subsequence_starts(raw_tokens, number_tokens)
    if not raw_occurrences:
        raise NumberSpanError("hard number is absent from raw text")

    candidates: set[tuple[int, int]] = set()
    for raw_start in raw_occurrences:
        raw_end = raw_start + len(number_tokens)
        prefix, suffix = raw_tokens[:raw_start], raw_tokens[raw_end:]
        prefix_starts = (0,) if not prefix else _subsequence_starts(gold_tokens, prefix)
        suffix_starts = (len(gold_tokens),) if not suffix else None
        for prefix_start in prefix_starts:
            span_start = prefix_start + len(prefix)
            candidate_suffix_starts = suffix_starts if suffix_starts is not None else _subsequence_starts(gold_tokens, suffix, minimum=span_start)
            for suffix_start in candidate_suffix_starts:
                if suffix_start > span_start:
                    candidates.add((span_start, suffix_start))

    if len(candidates) != 1:
        kind = "ambiguous" if candidates else "unanchorable"
        raise NumberSpanError(f"{kind} number span")
    start, end = candidates.pop()
    category = row.get("category", "unknown")
    return NumberSpan(start, end, category if isinstance(category, str) and category else "unknown")


def score_row(reference: str, hypothesis: str, number_span: NumberSpan) -> RowScores:
    """Score full utterance and the exactly projected number subspan."""

    reference_tokens = normalize_asr_text(reference).split()
    hypothesis_tokens = normalize_asr_text(hypothesis).split()
    if not (0 <= number_span.reference_start < number_span.reference_end <= len(reference_tokens)):
        raise NumberSpanError("number span is outside normalized reference text")

    word_alignment = _levenshtein_alignment(reference_tokens, hypothesis_tokens)
    number_hypothesis = _project_hypothesis_tokens(word_alignment, hypothesis_tokens, number_span)
    reference_number = reference_tokens[number_span.reference_start : number_span.reference_end]
    return RowScores(
        utterance=_edit_counts(reference_tokens, hypothesis_tokens),
        number=_edit_counts(reference_number, number_hypothesis),
        category=number_span.category,
    )


def aggregate_scores(rows: Iterable[RowScores]) -> MetricSummary:
    """Aggregate exact row counts into micro/macro utterance and number rates."""

    scored_rows = tuple(rows)
    if not scored_rows:
        raise ValueError("cannot aggregate zero scored rows")
    return MetricSummary(
        micro=_rates(scored_rows),
        macro=_macro_rates(scored_rows),
        by_category={
            category: _category_rates(group)
            for category, group in sorted(_group_by_category(scored_rows).items())
        },
    )


def _required_string(row: Mapping[str, object], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise NumberSpanError(f"missing required string: {name}")
    return value


def _subsequence_starts(tokens: Sequence[str], needle: Sequence[str], minimum: int = 0) -> list[int]:
    if not needle:
        return list(range(minimum, len(tokens) + 1))
    return [index for index in range(minimum, len(tokens) - len(needle) + 1) if list(tokens[index : index + len(needle)]) == list(needle)]


def _levenshtein_alignment(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[tuple[int | None, int | None, str], ...]:
    rows, columns = len(reference), len(hypothesis)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    for index in range(1, rows + 1):
        costs[index][0] = index
    for index in range(1, columns + 1):
        costs[0][index] = index
    for ref_index in range(1, rows + 1):
        for hyp_index in range(1, columns + 1):
            diagonal = costs[ref_index - 1][hyp_index - 1] + (reference[ref_index - 1] != hypothesis[hyp_index - 1])
            costs[ref_index][hyp_index] = min(diagonal, costs[ref_index - 1][hyp_index] + 1, costs[ref_index][hyp_index - 1] + 1)

    operations: list[tuple[int | None, int | None, str]] = []
    ref_index, hyp_index = rows, columns
    while ref_index or hyp_index:
        if (
            ref_index
            and hyp_index
            and reference[ref_index - 1] == hypothesis[hyp_index - 1]
            and costs[ref_index][hyp_index] == costs[ref_index - 1][hyp_index - 1]
        ):
            operations.append((ref_index - 1, hyp_index - 1, "equal"))
            ref_index -= 1
            hyp_index -= 1
        elif hyp_index and costs[ref_index][hyp_index] == costs[ref_index][hyp_index - 1] + 1:
            operations.append((None, hyp_index - 1, "insertion"))
            hyp_index -= 1
        elif ref_index and costs[ref_index][hyp_index] == costs[ref_index - 1][hyp_index] + 1:
            operations.append((ref_index - 1, None, "deletion"))
            ref_index -= 1
        else:
            operations.append((ref_index - 1, hyp_index - 1, "substitution"))
            ref_index -= 1
            hyp_index -= 1
    return tuple(reversed(operations))


def _project_hypothesis_tokens(
    alignment: Sequence[tuple[int | None, int | None, str]], hypothesis_tokens: Sequence[str], span: NumberSpan
) -> list[str]:
    projected = [hypothesis_index for reference_index, hypothesis_index, _ in alignment if reference_index is not None and span.reference_start <= reference_index < span.reference_end and hypothesis_index is not None]
    if not projected:
        return []
    first, last = min(projected), max(projected)
    return list(hypothesis_tokens[first : last + 1])


def _edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    word = _counts_from_alignment(_levenshtein_alignment(reference, hypothesis), len(reference))
    reference_characters = tuple("".join(reference))
    hypothesis_characters = tuple("".join(hypothesis))
    character = _counts_from_alignment(_levenshtein_alignment(reference_characters, hypothesis_characters), len(reference_characters))
    return EditCounts(word=word, character=character)


def _counts_from_alignment(alignment: Sequence[tuple[int | None, int | None, str]], reference_units: int) -> ErrorCounts:
    return ErrorCounts(
        substitutions=sum(operation == "substitution" for _, _, operation in alignment),
        deletions=sum(operation == "deletion" for _, _, operation in alignment),
        insertions=sum(operation == "insertion" for _, _, operation in alignment),
        reference_units=reference_units,
    )


def _rates(rows: Sequence[RowScores]) -> dict[str, float]:
    counts = _sum_counts(rows)
    return {name: count.rate for name, count in counts.items()}


def _macro_rates(rows: Sequence[RowScores]) -> dict[str, float]:
    return {
        "utt-wer": sum(row.utterance.word.rate for row in rows) / len(rows),
        "utt-cer": sum(row.utterance.character.rate for row in rows) / len(rows),
        "num-wer": sum(row.number.word.rate for row in rows) / len(rows),
        "num-cer": sum(row.number.character.rate for row in rows) / len(rows),
    }


def _sum_counts(rows: Sequence[RowScores]) -> dict[str, ErrorCounts]:
    selected = {
        "utt-wer": [row.utterance.word for row in rows],
        "utt-cer": [row.utterance.character for row in rows],
        "num-wer": [row.number.word for row in rows],
        "num-cer": [row.number.character for row in rows],
    }
    return {
        name: ErrorCounts(
            substitutions=sum(count.substitutions for count in values),
            deletions=sum(count.deletions for count in values),
            insertions=sum(count.insertions for count in values),
            reference_units=sum(count.reference_units for count in values),
        )
        for name, values in selected.items()
    }


def _group_by_category(rows: Sequence[RowScores]) -> dict[str, list[RowScores]]:
    grouped: dict[str, list[RowScores]] = defaultdict(list)
    for row in rows:
        grouped[row.category].append(row)
    return grouped


def _category_rates(rows: Sequence[RowScores]) -> dict[str, float]:
    values = _rates(rows)
    values.update({f"macro-{name}": value for name, value in _macro_rates(rows).items()})
    return values
