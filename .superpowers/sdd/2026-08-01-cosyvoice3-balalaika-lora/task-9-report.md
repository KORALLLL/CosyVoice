# Task 9 report: Four-sample 100%-accuracy memorization gate

## Scope

- Added the isolated `memorization.py` gate and its success-only evidence bundle.
- Extended `CosyVoice3LM.forward` with teacher-forced prediction/target tensors so the gate retains the actual discrete-token evidence; existing `loss`, `acc`, and per-sample count behavior remains intact.
- Added focused tests for the exact-per-sample streak, deterministic two-phase short/long selection, split/cache reconciliation, successful publication, failure cleanup, and token evidence.

## TDD evidence

1. RED: `rtk python -m unittest tests.finetune.balalaika.test_memorization -v` failed with `ModuleNotFoundError: cosyvoice.finetune.balalaika.memorization` before the initial per-sample gate implementation.
2. GREEN: the same command passed the two specified aggregate/streak tests after adding `SampleAccuracy` and `memorization_passed`.
3. RED: the selection tests failed with `ImportError: cannot import name 'select_memorization_rows'` before the deterministic cache-reconciled selector existed.
4. GREEN: the selection suite passed after implementing hash-tiebroken shortest/longest selection for each phase.
5. RED: the runner tests failed with `ImportError: cannot import name 'MemorizationRequest'` before the isolated production-path runner existed.
6. GREEN: the runner tests passed after adding fresh base/LoRA creation, Accelerate/AdamW/constant-scheduler operation, no-dropout every-10-step evaluation, strict three-check success, and temporary-only failure artifacts.
7. RED: token-evidence test failed with missing `teacher_forced_tokens.json`; model integration then failed with missing `teacher_forced_predictions`.
8. GREEN: `rtk python -m unittest tests.finetune.balalaika.test_memorization tests.finetune.balalaika.test_model -v` passed 27 tests after retaining teacher-forced predictions/targets and writing the evidence bundle.
9. RED/GREEN: the no-WAV test initially failed because a missing WAV writer rejected an otherwise exact token proof; the writer is now diagnostic-only, so waveform generation never determines the gate result.

## Self-review

- Gate decisions inspect all four `correct == total` and `total > 0` pairs; no aggregate accuracy or waveform criterion participates.
- Selection reconciles plan/cache row identities, text, instruction, phase predicates, token lengths, and phase totals; success records cache artifact and speech-token checksums.
- Output is staged under `.memorization.incomplete`, removed on every exception, and atomically renamed only after three exact checks. Existing evidence is rejected rather than reused.
- The temporary adapter is audited and never saved or exposed as a phase input. Generated WAVs are retained only as diagnostics and are never a pass condition.
- `rtk git diff --check` passed.

## Deliberately not run

No real model, GPU, corpus, training run, install, upload, or credential access was performed.
