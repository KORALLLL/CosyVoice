# Task 10 report: Distributed synthesis, GigaAM validation, and W&B

## Scope

- Added a checksum-bound 2,000-row/20-voice round-robin assignment shared by validation indices 0 through 40.
- Added eight-rank synthesis, durable per-rank journals, bounded GigaAM v3 RNN-T ASR with CUDA-session qualification and OOM bisection, global reconciliation, four micro metrics plus diagnostics, atomic JSONL/summary publication, and a deterministic 20-WAV listening panel.
- Added an idempotent logger for Task 8's existing W&B tracker, including environment-only credential preflight, persisted run identity, later-launch `resume="must"`, media/scalar phase ledgers, and syncable-directory failure errors.
- Added only fake CPU unit coverage; no private benchmark fetch, model/GPU execution, W&B network, credential use, install, upload, or 2,000-item real synthesis occurred.

## TDD evidence

1. RED: `rtk python -m unittest tests.finetune.balalaika.test_evaluation -v` failed with `0 != 2000` and missing exact-input rejection while `build_voice_assignment` was an empty stub. GREEN: the two assignment tests passed after exact ID/voice/checksum validation and sorted round-robin assignment.
2. RED: focused GigaAM tests failed because `GigaAmRecognizer`/`GigaAmError` were absent. GREEN: both passed after exact `onnx_asr.load_model("gigaam-v3-rnnt")`, local-rank CUDA provider binding, bounded ordered batches, and identical-item OOM bisection were implemented.
3. RED: the CPU-fallback mutation passed incorrectly when only requested provider settings were inspected. GREEN: active encoder, decoder, and joiner ONNX sessions must each report `CUDAExecutionProvider` on the local rank.
4. RED: assignment durability/publication tests failed because `EvaluationRequest` and `evaluate_checkpoint` were absent. GREEN: three tests passed after exact 250-row local sharding, journals/gather reconciliation, assignment reuse, atomic local results, and publication-before-W&B behavior.
5. RED: retry after a panel-publication failure stopped on `validation publication is partial`. GREEN: uncommitted local output now resumes from rank journals without regenerating valid WAVs; `summary.json` is the final publication commit.
6. RED: three W&B tests failed because `WandbValidationLogger` was absent. GREEN: existing-tracker use, baseline run-ID persistence, later-launch `resume="must"`, environment-only key preflight, step-indexed media/scalars, and an idempotent per-index commit ledger passed.
7. RED: the CosyVoice bridge test failed because `CosyVoiceSynthesizer` was absent. GREEN: it installs the current unwrapped LoRA LLM, verifies frozen flow/HiFT, writes 24 kHz mono PCM, and restores training mode.
8. RED: self-review coverage exposed that nonempty zero-shot IDs were used before registration. GREEN: each of the 20 fixed voices is now registered once with `add_zero_shot_spk` before inference.
9. RED: the audit-row assertion exposed missing `hard_number` and `prompt_text`. GREEN: each JSONL row now includes those inputs alongside prompt identity/checksum, reference/hypothesis spans, exact edit counts, audio checksum, and latency diagnostics.

## Self-review

- Exactly 2,000 generations are assigned per point, not a cross product; integer IDs and `voice_id` ordering yield 100 rows per fixed voice.
- Validation indices are closed to 0..40. Eight ranks and exactly 250 local items are mandatory, and gathered IDs must cover 1..2000 exactly once.
- The Task 9 memorization seal is required before assignment or generation. Valid generated WAVs/rank journals survive transient work failures; published JSONL/summary/panel survive W&B failure; successful publication removes temporary audio unless `KEEP_EVAL_AUDIO=1`.
- W&B never initializes another run and never reads, stores, or logs an API key value. A validation callback raises until media, scalar logging, and the local commit ledger all succeed, so Task 8 leaves the checkpoint pending.
- Mutation review covers wrong assignment cardinality, wrong CUDA activation, wrong local shard count, altered mapping, partial publication, duplicate W&B logging, missing credentials, and missing prompt registration.

## Verification

- `rtk python -m unittest tests.finetune.balalaika.test_evaluation tests.finetune.balalaika.test_metrics -v` — 20 passed before the final zero-shot registration fix; rerun after the fix recorded below.
- `rtk python -m unittest discover -s tests/finetune/balalaika -p 'test_*.py' -q` — 142 passed before the final zero-shot registration fix; rerun after the fix recorded below.
- `rtk python -m compileall -q cosyvoice/finetune/balalaika/evaluation.py tests/finetune/balalaika/test_evaluation.py` and `rtk git diff --check` passed before the final fix and were rerun after it.
- Final fresh verification after all production/test edits: exact Task 10/metrics suite 20/20 passed; complete Balalaika suite 142/142 passed; compileall and diff-check exited 0.
