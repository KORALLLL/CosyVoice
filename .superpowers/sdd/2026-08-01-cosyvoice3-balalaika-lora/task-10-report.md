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

## Fix Round 1: immutable validation evidence and remote W&B commits

### Root cause and changes

- The original resume boundary trusted syntactically valid WAVs and a shallow local summary instead of binding every reusable artifact to one immutable evaluation identity. The identity now recursively freezes and hashes checkpoint/model/adapter/base checksums, benchmark snapshot/revision, validation index, all row/voice/prompt semantics, ASR and synthesis configuration, and code/config versions.
- Rank journals now carry identity, item, audio, and record seals. Only a fully sealed journal row may authorize WAV reuse; orphan or mismatched WAVs are removed and regenerated. Main-rank reconciliation additionally verifies every gathered WAV is present, checksum-identical, mono 24 kHz PCM.
- Published JSONL rows, the summary, the deterministic listening-panel manifest/content, and a final success seal are checksum-linked. A committed resume revalidates all 2,000 rows, recalculates edit counts and aggregate metrics, checks the exact panel mapping and WAV bytes, and rejects any changed artifact.
- W&B is a mandatory preflight dependency. Local and remote commit evidence binds the evaluation identity, assignment, artifact checksums, and metrics. Deterministic media/scalar markers are queried before logging and verified afterward; media uses the underlying active run while scalars use `Accelerator.log`. Retry after scalar or local-ledger failure is idempotent. Durable history queries use `wandb.Api().run(...).scan_history`, because the active SDK run has no `scan_history` method.
- GigaAM construction is bound to `accelerator.local_process_index`, and runtime recognizer/synthesizer provenance must exactly match the requested identity.

### TDD and adversarial evidence

1. RED: an unjournaled valid WAV skipped synthesis; missing W&B reached distributed work; a recognizer/local-rank mismatch reached gather. GREEN: all three focused tests passed after sealed journal-only reuse and pre-generation W&B/rank validation.
2. RED: W&B crash tests lacked an injectable ledger writer and a scalar retry produced no durable remote marker group. GREEN: remote-success/local-ledger-crash and media-pending/scalar-retry both commit exactly one remote step and avoid duplicate calls.
3. RED: marker-query instrumentation observed only two reads around a commit. GREEN: remote state is queried before each component and after the atomic scalar commit.
4. RED: the identity's nested ASR configuration remained mutable, and the production logger attempted the unsupported active-run `scan_history` API. GREEN: the identity is recursively immutable and remote reads use the public API run; focused SDK-boundary and crash tests pass.
5. Adversarial validation passes for identity changes, orphan WAVs, missing/checksum-changed gathered audio, resealed-but-corrupt JSONL/summary data, and modified listening-panel content.

### Final verification

- `rtk python -m py_compile cosyvoice/finetune/balalaika/evaluation.py tests/finetune/balalaika/test_evaluation.py` — passed.
- `rtk git diff --check` — passed.
- `rtk python -m unittest tests.finetune.balalaika.test_evaluation tests.finetune.balalaika.test_metrics -v` — 30/30 passed.
- `rtk python -m unittest discover -s tests/finetune/balalaika -p 'test_*.py' -q` — 152/152 passed (exit 0).
- Verification remained fake/local only: no private benchmark, GPU/model run, W&B network call, credential use, or upload occurred.

## Fix Round 2: complete identity, fail-closed journals, and collective W&B gates

### Root cause and changes

- The evaluation identity did not receive `EvaluationRequest.asr_batch_size`, and the semantic assignment recorded only the benchmark category, not `NumberSpan.category`. The canonical payload now contains the positive ASR batch size and the semantic span category, so either change produces a new identity and invalidates prior evidence.
- Duplicate journal detection removed the duplicate row's WAV but retained the first sealed row in memory, leaving a reusable record that pointed at deleted audio. Journal loading now tracks every seen expected ID, removes the first record as soon as any duplicate appears, keeps that ID invalid, and performs a final seal/WAV/checksum validation over every returned record.
- W&B preflight previously ran independently on every rank before collective communication. Only the main process now validates/owns the real `WandbValidationLogger` and tracker. It broadcasts a structured success/error payload through the Accelerator-compatible object-broadcast path before the first barrier; every rank either continues or raises the same error. Post-log media/scalar commit status uses the same collective structure, so main still must finish both remote markers and the local ledger before validation succeeds.

### Strict TDD evidence

1. RED: two identity tests reported the missing `asr_batch_size` payload and an unchanged checksum after mutating `NumberSpan.category`. GREEN: all identity-focused tests passed after canonicalizing both values and threading the request batch size into evaluation.
2. RED: a journal containing two independently valid sealed rows for benchmark 1 returned the first row after deleting its WAV. GREEN: the duplicate regression returns no reusable record and confirms the WAV is removed.
3. RED: the simulated main rank reached the first barrier without broadcasting preflight success, while a main preflight failure produced no collective error payload. GREEN: the faithful two-rank broadcast tests show a non-main rank with no logger/tracker reaches the generation barrier on success, and both ranks receive the same `WandbSyncError` on main failure.
4. Mutation review covers omitting either new identity field, retaining the first duplicate, returning a record with missing/changed WAV bytes, querying W&B from non-main, skipping the preflight broadcast, and accepting a malformed phase/status payload.

### Verification

- `rtk python -m py_compile cosyvoice/finetune/balalaika/evaluation.py tests/finetune/balalaika/test_evaluation.py` — passed.
- `rtk git diff --check` — passed.
- `rtk python -m unittest tests.finetune.balalaika.test_evaluation tests.finetune.balalaika.test_metrics -v` — 35/35 passed.
- `rtk python -m unittest discover -s tests/finetune/balalaika -p 'test_*.py' -q` — 157/157 passed (exit 0).
- Testing remains fake/local, including the faithful distributed object-broadcast simulation; no private benchmark, GPU/model execution, W&B network call, credential use, or upload occurred.
