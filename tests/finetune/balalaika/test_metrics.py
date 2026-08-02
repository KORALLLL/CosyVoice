"""Contract tests for deterministic hard-number evaluation metrics."""

import unittest

from cosyvoice.finetune.balalaika.metrics import (
    NumberSpanError,
    aggregate_scores,
    extract_number_span,
    normalize_asr_text,
    score_row,
)


def validation_row(*, text: str, hard_number: str, normalized_gold: str, category: str = "cardinal") -> dict[str, object]:
    return {"stressed": text, "hard_number": hard_number, "normalized_gold": normalized_gold, "category": category}


class MetricsTests(unittest.TestCase):
    def test_normalization_matches_asr_policy(self):
        self.assertEqual(normalize_asr_text("Ёлка,  ПЯТЬ!"), "елка пять")

    def test_number_only_error_ignores_outside_word(self):
        row = validation_row(
            text="Оплатите 25 рублей завтра.",
            hard_number="25",
            normalized_gold="Оплатите двадцать пять рублей завтра.",
        )
        span = extract_number_span(row)
        scores = score_row(row["normalized_gold"], "внесите двадцать пять рублей завтра", span)
        self.assertGreater(scores.utterance.word.distance, 0)
        self.assertEqual(scores.number.word.distance, 0)

    def test_anchors_spans_for_supported_number_forms(self):
        cases = [
            ("Позвоните +7 (999) 123-45-67 сейчас", "+7 (999) 123-45-67", "Позвоните плюс семь девятьсот девяносто девять сто двадцать три сорок пять шестьдесят семь сейчас", 1, 13),
            ("Цена 12,5 килограмма", "12,5", "Цена двенадцать целых пять десятых килограмма", 1, 5),
            ("Дата 01.02.2025 подтверждена", "01.02.2025", "Дата первое февраля две тысячи двадцать пятого года подтверждена", 1, 8),
            ("Диапазон 3-5 метров", "3-5", "Диапазон от трех до пяти метров", 1, 5),
            ("Код 1111 принят", "1111", "Код одна тысяча сто одиннадцать принят", 1, 5),
            ("Занял 21-е место", "21-е", "Занял двадцать первое место", 1, 3),
        ]
        for text, hard_number, gold, start, end in cases:
            with self.subTest(hard_number=hard_number):
                span = extract_number_span(validation_row(text=text, hard_number=hard_number, normalized_gold=gold))
                self.assertEqual((span.reference_start, span.reference_end), (start, end))

    def test_ambiguous_prefix_raises_instead_of_guessing_a_number_span(self):
        row = validation_row(
            text="Код 25 готов",
            hard_number="25",
            normalized_gold="Код двадцать пять готов код двадцать пять готов",
        )
        with self.assertRaisesRegex(NumberSpanError, "ambiguous"):
            extract_number_span(row)

    def test_start_and_end_number_anchors_are_constrained_to_utterance_boundaries(self):
        starts = validation_row(text="25 рублей", hard_number="25", normalized_gold="двадцать пять рублей")
        ends = validation_row(text="Оплатите 25", hard_number="25", normalized_gold="Оплатите двадцать пять")
        self.assertEqual((extract_number_span(starts).reference_start, extract_number_span(starts).reference_end), (0, 2))
        self.assertEqual((extract_number_span(ends).reference_start, extract_number_span(ends).reference_end), (1, 3))

    def test_number_span_requires_stressed_instead_of_falling_back_to_text(self):
        row = {"text": "Оплатите 25 рублей", "hard_number": "25", "normalized_gold": "Оплатите двадцать пять рублей"}
        with self.assertRaisesRegex(NumberSpanError, "missing required string: stressed"):
            extract_number_span(row)

    def test_alignment_counts_substitution_deletion_and_insertion(self):
        row = validation_row(
            text="Сумма 25 рублей", hard_number="25", normalized_gold="Сумма двадцать пять рублей"
        )
        scores = score_row(row["normalized_gold"], "сумма тридцать рублей сегодня", extract_number_span(row))
        self.assertEqual(scores.utterance.word.distance, 3)
        self.assertEqual(scores.number.word.substitutions, 1)
        self.assertEqual(scores.number.word.deletions, 1)
        self.assertEqual(scores.number.word.insertions, 0)

    def test_aggregate_exposes_micro_macro_and_category_metrics(self):
        first = validation_row(text="Сумма 25 рублей", hard_number="25", normalized_gold="Сумма двадцать пять рублей", category="money")
        second = validation_row(text="Код 7 готов", hard_number="7", normalized_gold="Код семь готов", category="code")
        rows = [
            score_row(first["normalized_gold"], "сумма двадцать пять рублей", extract_number_span(first)),
            score_row(second["normalized_gold"], "код восемь готов", extract_number_span(second)),
        ]
        summary = aggregate_scores(rows)
        self.assertEqual(set(summary.micro), {"utt-wer", "utt-cer", "num-wer", "num-cer"})
        self.assertEqual(summary["num-wer"], 1 / 3)
        self.assertEqual(summary.macro["num-wer"], 0.5)
        self.assertEqual(summary.by_category["money"]["num-wer"], 0.0)
        self.assertEqual(summary.by_category["code"]["num-wer"], 1.0)


if __name__ == "__main__":
    unittest.main()
