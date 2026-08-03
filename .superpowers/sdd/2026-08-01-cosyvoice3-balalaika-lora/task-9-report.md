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

## Fix Round 2: sealed manifest and self-validating trajectory

### Root cause

- `_publish_success` wrote the manifest but its `MemorizationReport` return had been stranded after `require_memorization_gate`, making the declared return unreachable.
- The evidence map intentionally omitted its own manifest, but no separate immutable binding protected the manifest contents. The verifier relied too heavily on optional external provenance and did not independently validate all trajectory and row invariants.

### TDD evidence

1. RED: direct `_publish_success` coverage returned `None`; success output lacked `steps` and the expected seal. The existing verifier accepted a resealed top-level-row mismatch because it only read provenance rows.
2. GREEN: publication now records completed `steps`, atomically writes `memorization_manifest.json`, then atomically writes `memorization_success.json` containing the manifest path and SHA-256. `_publish_success` returns the completed `MemorizationReport` on its reachable path.
3. RED: seal/internal-tamper coverage initially failed with an evidence-checksum error rather than a required seal failure; the old verifier did not self-reject check-index, optimizer-step, completed-step, row length/hash, cache/base checksum, tokenizer, adapter, or duplicate top-level-row mutations when the manifest was rehashed.
4. GREEN: `require_memorization_gate` validates the seal before parsing or trusting the manifest, then enforces v3 manifest, evidence, exact row fields/checksums, base/cache checksums, fixed adapter settings/audit shape, full run-config types/ranges, monotonic check indices, check cadence, completed-step alignment, and a terminal streak of exactly three all-sample-exact checks. Each listed tamper case now fails even without `expected_provenance`.

### Verification

- `rtk python -m unittest tests.finetune.balalaika.test_memorization tests.finetune.balalaika.test_model -v` — 32 passed.
- Full Balalaika/compile/diff verification was run after the final implementation changes before commit.

## Fix Round 3: exact adapter audit and root-only evidence exclusions

### Root cause

- The manifest verifier accepted a minimal `trainable_parameters` list instead of the complete production `TrainableAudit` schema, allowing unverifiable target coverage and unapproved tensor names.
- `_evidence_checksums` excluded every file whose basename was `memorization_manifest.json` or `memorization_success.json`, including nested diagnostic files that must remain sealed evidence.

### TDD evidence

1. RED: a production-shaped audit fixture plus forbidden `llm.model.lm_head`, dense-tensor, and malformed-audit mutations showed that the old verifier accepted audit structures beyond the production contract. A nested generated diagnostic named `memorization_manifest.json` and a sibling named `memorization_success.json` were absent from the evidence map.
2. GREEN: `model.validate_trainable_audit_payload` is the shared production validator. It enforces the exact five-field schema, required active targets, complete approved LoRA pairs, no dense trainables, unique names, forbidden-Qwen-head exclusion, and coherent parameter totals. `audit_trainable_parameters` invokes it before returning; the memorization verifier uses the same helper and requires exact fixed `LoraSettings`.
3. GREEN: evidence exclusion now compares only `root / _MANIFEST_NAME` and `root / _SEAL_NAME`. Nested diagnostics of any basename are checksummed. Regression coverage verifies nested manifest/seal-named diagnostics are recorded and that tampering, removing, or adding any nested diagnostic causes `require_memorization_gate` to reject the evidence.

### Verification

- `rtk python -m unittest tests.finetune.balalaika.test_memorization tests.finetune.balalaika.test_model -q` — 32 passed.
- `rtk python -m unittest discover -s tests/finetune/balalaika -p 'test_*.py' -q` — 127 passed.
- `rtk python -m compileall -q cosyvoice/finetune/balalaika/memorization.py cosyvoice/finetune/balalaika/model.py cosyvoice/llm/llm.py` and `rtk git diff --check` passed.

## Fix Round 4: internal Qwen lm_head subtree exclusion

### Root cause

- Active discovery, the live trainable audit, and the persisted `TrainableAudit` validator used `endswith("llm.model.lm_head")`. That excluded only the internal head module itself; a linear descendant such as `llm.model.lm_head.child` remained an active LoRA target.
- The same exact-only predicate let a prefixed descendant target and its complete LoRA A/B tensor pair pass persisted-audit and memorization self-validation. Adapter manifest parsing had no direct forbidden-subtree check, so merge rejected a tampered descendant inventory only later as a generic fresh-base mismatch.

### TDD evidence

1. RED: four focused regressions failed. Active discovery included `llm.model.lm_head.child`; strict audit validation accepted `base_model.model.llm.model.lm_head.child` with complete trainable A/B names; merge did not identify the forbidden subtree; and a coherently duplicated/resealed memorization audit accepted the descendant pair.
2. GREEN: `_is_internal_qwen_lm_head_path` now matches the canonical `llm.model.lm_head` segment sequence at any wrapper-prefix depth and therefore covers both the head and every descendant. It does not match the outer CosyVoice `llm_decoder`.
3. GREEN: active target discovery, stored target inventory checks, live wrapper and trainable-name checks, strict persisted audit validation, and adapter manifest parsing all use the canonical helper. Memorization self-validation inherits the same rejection through `validate_trainable_audit_payload`.
4. GREEN: the focused RED set passed all four regressions after the minimal production change.

### Verification

- `rtk python -m unittest tests.finetune.balalaika.test_memorization tests.finetune.balalaika.test_model -v` — 35 passed.
- `rtk python -m unittest discover -s tests/finetune/balalaika -p 'test_*.py' -v` — 130 passed.
- `rtk python -m compileall -q cosyvoice/finetune/balalaika/memorization.py cosyvoice/finetune/balalaika/model.py cosyvoice/llm/llm.py tests/finetune/balalaika/test_memorization.py tests/finetune/balalaika/test_model.py` and `rtk git diff --check` passed.
- Code/tests commit: `4bfb3bb` (`fix: reject internal lm head descendants`).
