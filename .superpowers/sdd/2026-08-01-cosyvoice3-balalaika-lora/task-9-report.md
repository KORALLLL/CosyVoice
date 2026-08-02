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

## Fix Round 1: audit hardening

### Root cause

- The runner accepted the model loss as any tensor and immediately called `Accelerator.backward`; NaN and infinities therefore reached optimizer/scheduler progress and only surfaced when strict JSON evidence publication rejected the non-finite float.
- The v1 manifest recorded only selected provenance fields and listed diagnostic names without checksums; there was no downstream require function to validate artifacts and compare expected provenance.

### TDD evidence

1. RED: `rtk python -m unittest tests.finetune.balalaika.test_memorization.MemorizationSelectionTests.test_nonfinite_training_loss_stops_before_backward_or_optimizer_progress -v` reproduced NaN, `+Inf`, and `-Inf` reaching `_publish_success`; JSON rejected each value after optimization, not before it.
2. GREEN: `_loss` now requires exactly one finite value before backward, clipping, AdamW/scheduler stepping, evaluation, or publication. The test passes and proves zero fake-Accelerate backward/clip calls, unchanged model weight, and no success directory for all three values.
3. RED: the new identity/evidence tests initially failed with missing `require_memorization_gate` and absent manifest `provenance`/`evidence` fields.
4. GREEN: the v2 success manifest binds validated `max_steps`, LR, grad norm, fixed AdamW hyperparameters, constant scheduler spec, bf16, one accumulation step, eight ranks, local/effective batch sizes, evaluation cadence, three-check rule, seed, loader/tokenizer identities, base checksum, selected rows, fixed LoRA settings, and audited trainables. It records relative paths and SHA-256 values for every diagnostic artifact.
5. RED: a final provenance test failed until tokenizer identity and adapter audit were moved into the compared provenance object.
6. GREEN: `require_memorization_gate` now rejects missing/invalid manifests, changed expected provenance, missing/extra/tampered evidence, incomplete row provenance, and non-exact final checks. Focused memorization/model tests passed 30 tests.
7. RED/GREEN: the complete-config assertion exposed omitted AdamW execution defaults; `foreach=None`, `capturable=False`, `differentiable=False`, and `fused=None` are now explicit, immutable request fields and serialized with the remaining optimizer trajectory settings.

### Verification

- `rtk python -m unittest tests.finetune.balalaika.test_memorization tests.finetune.balalaika.test_model -v` — 30 passed.
- `rtk python -m unittest discover -s tests/finetune/balalaika -p 'test_*.py' -v` — 125 passed.
- `rtk python -m compileall -q cosyvoice/finetune/balalaika/memorization.py cosyvoice/llm/llm.py` and `rtk git diff --check` passed.
