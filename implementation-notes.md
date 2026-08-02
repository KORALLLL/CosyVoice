# Implementation Notes

## 2026-08-02 - Execution setup

- Decision: Implement the approved two-phase CosyVoice3 LoRA plan on the isolated `feature/cosyvoice3-balalaika-lora` worktree.
- Changed: Proceed despite one unrelated baseline failure, with the user's explicit approval; the baseline result is 23/24 passing.
- Validation: The failing test is `OpenAICompatibilityTests.test_streaming_response_has_pcm_headers_and_sample_rate`; it expects an absent ignored file, `asset/qwen_ref_4.wav`, while production returns `None` for a missing prompt file.
- Follow-up: Do not alter that unrelated baseline behavior as part of this training recipe.

## 2026-08-02 - Task 1 runtime preflight

- Decision: Runtime qualification requires exactly eight visible RTX 5090 GPUs, not merely at least eight.
- Decision: Reject any base-model path containing an `_RL` path component, including nested checkpoint paths.
- Validation: Task 1 focused unit tests pass 14/14; Python compilation and whitespace checks pass.
- Follow-up: The specified Flake8 check could not run because neither the executable nor module is installed; no package was installed during implementation.

## 2026-08-02 - Task 2 corpus split planning

- Decision: Apply the CosyVoice3 recipe's authoritative 1–200 Qwen text-token limit before prompt reservation; load the production tokenizer lazily and keep a private injection boundary for fixture tests.
- Decision: Accept only the canonical `asr_agreement_mean` field for phase assignment; legacy `agreement` input is rejected.
- Validation: Task 1+2 focused unit tests pass 27/27; scoped re-review approved the model-limit and canonical-schema fixes.
- Follow-up: Actual 4,075,032-row source reconciliation remains a real-data qualification step; Task 2 tests intentionally use fixtures.

## 2026-08-02 - Task 3 hard-number validation contract

- Decision: The global credential policy overrides the draft helper signature; `fetch_validation_rows` reads `HF_TOKEN` internally and accepts no credential argument.
- Decision: Treat empty raw-text anchors as strict utterance boundaries so valid leading and trailing number phrases remain scoreable.
- Decision: Validate the temporary private-dataset Parquet and all 2,000 number spans before publishing either final artifact.
- Validation: All Balalaika unit tests pass 39/39; scoped re-review approved atomic cleanup, typed schema provenance, boundary spans, and environment-only authentication.
- Follow-up: The real authenticated 2,000-row fetch remains deferred to qualification and must use a rotated environment token.

## 2026-08-02 - Task 4 compact cache publication

- Decision: Enforce exactly 20 unique reserved prompt identities globally while allowing each source shard to contain only its own subset.
- Decision: Treat the shard manifest publication/fsync as the transaction commit point; pre-commit failures restore prior finals, while post-commit backup cleanup is best-effort and cannot roll back valid outputs.
- Decision: Reject directory and other non-file tar members as unexpected input rather than silently skipping them.
- Validation: Cache tests pass 11/11 and all Balalaika tests pass 50/50; scoped re-review approved prompt reconciliation and transactional recovery.
- Follow-up: Real source decoding and cache generation remain gated behind tokenizer pilot approval and hardware qualification.

## 2026-08-02 - Task 5 tokenizer and pilot gate

- Decision: Treat missing, erroring, empty, or CPU-only ONNX provider reports as fatal; a tokenizer session is usable only when CUDA execution is positively verified.
- Decision: Pilot approval is revalidated against every recorded listening artifact checksum, not only the pilot stage JSON.
- Decision: Cache completion is reconciled against the canonical source-archive shard set; worker results and pre-existing manifests cannot introduce or omit shards.
- Validation: Tokenizer tests pass 12/12 and all Balalaika tests pass 62/62; scoped re-review approved provider, pilot-artifact, lease-result, and exact-shard-set controls.
- Follow-up: No real pilot or approval exists yet; real CUDA qualification and manual listening remain mandatory before corpus tokenization.

## 2026-08-02 - Task 6 LoRA model boundary

- Decision: Per-sample memorization statistics count only target IDs in `[0, speech_token_size)`; EOS, fill, ignored, and padded targets are excluded.
- Decision: Require the exact approved model-root name `Fun-CosyVoice3-0.5B-2512`, in addition to rejecting `_RL` paths.
- Decision: Treat the LoRA audit as an enforcing gate that independently maps every trainable adapter tensor to one unique approved active target and rejects missing, frozen, forbidden, orphan, duplicate, ambiguous, or dense trainables.
- Validation: Model tests pass 21/21, the required combined suite passes 23/23, and all Balalaika tests pass 83/83; scoped re-review approved statistics, audit, identity, and corrupted-adapter rejection.
- Follow-up: Production HyperPyYAML loading and the actual 0.5B target inventory remain part of real hardware qualification.

## 2026-08-02 - Task 7 cached batching

- Decision: Random Parquet access uses bounded one-row batches within retained row-group offsets rather than materializing whole row groups.
- Decision: Dynamic batch cost follows actual independent padding: `batch_size * (max_text_len + max_speech_len)`.
- Validation: Data tests pass 10/10 and all Balalaika tests pass 93/93; scoped re-review approved bounded row access and complementary-length budget handling.

## 2026-08-02 - Task 8 resumable Accelerate training

- Decision: Represent every one-eighth point as an exact rational fraction and use the smallest completed-sample threshold at or above it; small datasets may share a sample threshold, but retain eight distinct monotone boundary indices.
- Decision: Atomically publish a checksum-complete checkpoint as validation-pending before invoking validation, then atomically mark its manifest succeeded; restart retries pending validation before consuming another batch and skips succeeded validation.
- Assumption: Production callers supply the Task 7 epoch dataloader through a factory; the trainer owns Accelerate preparation, adapter-only optimizer construction, progress, and checkpoint state.
- Decision: Keep all eight fractional event identities even when a small epoch maps several exact fractions to the same completed-sample threshold; release every due event once at the next accumulation boundary.
- Validation: Training tests pass 13/13, all Balalaika tests pass 106/106, the two-process Accelerate CPU smoke writes eight ordered boundary records, and compilation/whitespace checks pass.
- Follow-up: Real eight-GPU BF16 execution, production model loading, and DDP behavior of explicit empty synchronization batches remain hardware qualification work.

### Fix round 1

- Decision: Preserve the pre-sharded Task 7 sampler by wrapping each rank loader through Accelerate with `num_processes=1`; this supplies end-of-dataloader accumulation synchronization without a second distributed shard.
- Decision: When any rank is empty, gather one CPU single-sample template and run the prepared model on a zero-contribution dummy at empty ranks, keeping DDP forward/backward collectives aligned. A globally empty step performs no training or sample-progress operation.
- Decision: Advance sample and sampler cursors only when an optimizer accumulation window commits; the sampler cursor also covers earlier globally empty physical batches so resume skips the exact consumed prefix.
- Decision: Resume identity now includes the complete approved phase, eligible count, cache manifest checksum, token budget, clipping, accumulation, world size, sampler seed/window, dataloader identity, base/LoRA provenance, and scheduler class plus initial configuration; Accelerate state checksums bind the evolving scheduler state.
- Changed: Stale checkpoint staging at the exact derived `.incomplete` path is removed before a retry; neighboring completed checkpoints are never touched.
- Validation: Focused training tests pass 18/18, all Balalaika tests pass 111/111, and the two-process DDP smoke passes uneven empty-rank work, a 2-of-3 final accumulation, validation crash/resume, exact boundaries, no replay, and equal rank weights.
- Follow-up: The smoke is intentionally two-process CPU validation; real eight-GPU BF16 capacity and performance qualification remains outside fixture scope.

### Fix round 2

- Decision: Lambda-based scheduler identity hashes stable executable semantics—bytecode, constants, names, argument shape, defaults, closure values, and referenced globals—rather than trusting `state_dict()`, source paths, line numbers, or a caller label.
- Decision: Reject scheduler callables containing recursive or unsupported configuration instead of publishing an identity that may collapse distinct future learning-rate behavior.
- Validation: The regression proves two LambdaLR instances have equal initial `state_dict()` values but distinct semantic identities, and changed future multipliers are refused on resume. Focused tests pass 20/20, all Balalaika tests pass 113/113, and the two-process smoke plus compilation/whitespace checks pass.

### Architectural resolution after review breaker

- Changed: With user approval, removed arbitrary scheduler callables and their fingerprint machinery after repeated reflection bypasses demonstrated that arbitrary Python behavior cannot be completely checksum-bound.
- Decision: `TrainRequest` accepts only the frozen, declarative `SchedulerSpec(kind="constant-v1")`; the trainer constructs the constant schedule internally from the exact phase learning rate and stores the spec verbatim in resume identity.
- Validation: Task 8 tests pass 23/23, all Balalaika tests pass 116/116, and the real two-process CPU `train_phase` smoke passes with equal rank weights and boundary ordinals 1–8. Regression tests reject both the retired `scheduler_factory` keyword and callable `scheduler_spec` values.
- Follow-up: Real eight-GPU BF16 qualification remains mandatory before production training.

## 2026-08-02 - Task 9 memorization gate

- Decision: Gate success requires exactly four deterministic phase-balanced rows to achieve per-sample real speech-token accuracy of 100% for three consecutive checks; waveform diagnostics never affect the decision.
- Decision: Seal the success manifest with an external SHA-256 record and checksum every evidence file; the verifier independently validates step cadence, all row token hashes, complete run configuration, base/cache/tokenizer provenance, and the strict production LoRA audit.
- Decision: Reject the entire internal Qwen `llm.model.lm_head` subtree across discovery, live/persisted audits, adapter manifests, merge, and memorization evidence while retaining the outer `llm_decoder` target.
- Validation: Task 9/model tests pass 35/35 and all Balalaika tests pass 130/130; scoped re-review approved the sealed evidence and forbidden-subtree controls.
- Follow-up: The real four-sample gate has not run; it remains mandatory after pilot approval and before any phase training.

## 2026-08-02 - Task 10 hard-number evaluation

- Decision: Bind every validation artifact and W&B commit to one immutable identity spanning checkpoint/model/adapter/base, benchmark semantics and number spans, all prompt text/audio checksums, ASR/provider/batch configuration, synthesis settings, code/config versions, and validation index.
- Decision: Reuse a generated row only from a sealed exact-identity journal with a matching 24 kHz mono WAV checksum; orphan, duplicate, stale, or changed-identity audio regenerates.
- Decision: W&B is a mandatory main-rank gate. Scalar and media components use remote commit markers written with their payloads and queried before retry; structured preflight/log outcomes are broadcast so all ranks proceed or fail together.
- Validation: Evaluation tests pass 27/27, exact evaluation/metrics tests pass 35/35, and all Balalaika tests pass 157/157; scoped re-review approved identity, journal, artifact-resume, remote WAV, rank, and collective W&B controls.
- Follow-up: Real eight-GPU synthesis, private benchmark access, GigaAM CUDA inference, and online W&B synchronization remain hardware qualification work.

## 2026-08-02 - Task 11 final model transaction

- Decision: A production-ready export is one atomic directory transaction that includes the adapter merge, original-key check, strict normal-path load, fixed-logit equivalence, four authenticated voice-cloning generations, GigaAM ASR, exact manifest, and success seal before the final rename.
- Decision: Finalization revalidates the exact Task 10 validation-40 publication and live W&B scalar/media markers, derives checkpoint and model-state identities from the actual phase-2 checkpoint, and requires the trainer's complete 14-field phase-2 identity.
- Decision: Retain exactly the adapter manifest and safetensors weights, reject symlinks/extras, bind the complete base-asset and 20-prompt inventories, and require the exact normal CosyVoice3 pipeline plus a freshly CUDA-qualified GigaAM v3 RNN-T recognizer for production evidence.
- Decision: Store fixed-probe IDs and both logit tensors in checksum-bound safetensors. Committed validation requires the authenticated base directory, recomputes adapter-active and merged logits from the actual models, and derives every tolerance metric from those tensors.
- Decision: Validate the complete staged final directory before publication and again afterward. Matching retries are idempotent; corrupt, incomplete, differently sourced, or consistently resealed-but-semantically-invalid artifacts are refused.
- Validation: Merge tests pass 36/36, evaluation plus merge tests pass 70/70, model tests pass 24/24, and all Balalaika tests pass 200/200. Independent scoped review found no remaining issues.
- Follow-up: Real CosyVoice3/GigaAM CUDA inference remains part of hardware qualification; no production export or training has run.

## 2026-08-02 - Task 12 two-phase launch workflow

- Decision: Expose exactly two operator shell launchers; both route phase execution through the pinned eight-process BF16 Accelerate config, while `--status` uses the same read-only Python state machine without launching workers.
- Decision: Preserve the pilot-before-preprocessing contract with a deterministic four-row memorization mini-cache. Rank zero retains a bounded hash sample from each split, extracts duration-short/long rows, tokenizes only those four, and publishes a checksum-bound 2+2 cache before the full 519-shard cache is allowed.
- Decision: Synchronize every main-only operation result/error and gather every collective result/error through one Accelerator context, preventing rank-local exceptions from letting peers enter a later collective.
- Decision: Extend the immutable Task 8 identity with the exact global validation base (`0` for phase 1, `16` for phase 2) and phase-bound adapter lineage (`None` for phase 1, sealed phase-1 adapter SHA-256 for phase 2). Task 11 now requires the resulting exact 16-field phase-2 identity.
- Decision: Store every Task 10 validation report identity and artifact inventory in phase evidence. Resume authentication uses a non-generating public Task 10 verifier to recompute the sealed report and require the live local ledger plus both remote W&B markers for validation 0 and the exact phase ranges 1–16 and 17–40.
- Decision: Reuse one `WandbValidationLogger` per Accelerate launch. The separate phase-2 launch resumes the persisted W&B run with `resume="must"`; it never creates a second logical validation run.
- Security: Hugging Face and W&B credentials remain environment-only. CLI/status configuration is recursively redacted for keys containing `TOKEN`, `KEY`, or `SECRET`, and the workflow has no Hub publication surface.
- Validation: Focused Task 8/workflow/public-Task-10 tests pass 40/40; targeted Task 11 identity/export tests pass 17/17; all Balalaika tests pass 217/217; both shell launchers pass `bash -n`; Python compilation and whitespace checks pass.
- Follow-up: No real network, CUDA, cache build, training, evaluation, export, or upload ran in Task 12. Real qualification still stops for manual pilot listening approval.

### Fix round 1

- Changed: Resume reconstructs and reauthenticates every earlier committed Task 10 validation before accepting the next boundary. A pending cursor excludes the interrupted boundary; a succeeded cursor includes it, and only actual phase validation indices are accepted.
- Decision: Load resume evidence lazily after `train_phase` has authenticated the exact training identity, all checkpoint-state checksums, and restored Accelerate state. This avoids trusting evaluation evidence first and avoids hashing multi-gigabyte state twice.
- Changed: Memorization, capacity qualification, and phase training each receive an isolated lifecycle scope around the workflow-owned Accelerator. Prepared models, optimizers, schedulers, dataloaders, and custom checkpoint objects are cleared on entry and on both successful and exceptional exit.
- Changed: CUDA tokenizer qualification restores the rank's original device in a `finally` block, and final export consumes the already committed validation-40 request instead of retaining or rebuilding the training-time evaluation model.
- Validation: Focused workflow/tokenizer tests pass 30/30 and training/memorization tests pass 35/35, including resume-cursor, lifecycle-exception, and CUDA-device regressions.

### Fix round 2

- Changed: `TrainingCallbacks` now has a resume-only post-load hook. The trainer invokes it on every rank after immutable identity and state-file authentication plus `Accelerator.load_state`, but before pending-boundary revalidation, boundary release, dataloader iteration, optimizer work, or checkpoint publication.
- Decision: Resume evidence verification is a rank-zero operation wrapped in the workflow's synchronized main-rank call. Rank zero alone accesses the live W&B tracker; the complete prior-validation evidence list or structured failure is broadcast and unwrapped identically on every rank.
- Decision: The hook itself performs that synchronized broadcast while phase training is already inside the workflow's collective operation. Its barrier completes before a shared exception is raised, after which every rank enters the outer collective error gather without leaving peers behind.
- Validation: New regressions cover main/non-main evidence and error broadcast, post-state-load/pre-forward/pre-save ordering, pending-current revalidation order, fresh-run exclusion, and final succeeded resume behavior.

## 2026-08-02 - Task 13 end-to-end fixture and operator documentation

- Decision: The CPU integration harness calls the actual `run_phase1` and `run_phase2` orchestration. It scales only corpus and prompt counts while retaining the literal production phase specifications (2/3 epochs), all eight boundaries per epoch, and global validation indices 0 through 40.
- Decision: Exercise real phase assignment/reservation, pilot stop/approval, atomic cache transactions, cache verification, mmap cache reads, `run_memorization_gate`, restart-safe trainer checkpoints, all 41 `evaluate_checkpoint` calls, adapter loading, and `export_final_llm(test_mode=True)`. Only expensive tokenization, synthesis, ASR, W&B transport, and distributed collectives are bounded CPU seams.
- Decision: The memorization seam injects the production broad-LoRA target set into a tiny CosyVoice3 model, uses the real trainable-parameter audit, performs real optimizer updates on a LoRA parameter, and requires three exact post-update checks plus independently verified sealed evidence.
- Decision: Inject a cache rename failure and stop phase 1 after validation 3, then require clean cache retry and authenticated checkpoint resume without duplicated validation indices. Phase 2 constructs a fresh model, loads the exact phase-1 adapter, and feeds its actual terminal checkpoint to final export.
- Security: A narrow internal `allow_test_export` workflow option permits the committed non-production test export to complete in this injected harness. It defaults false, has no CLI flag, is rejected by `ProductionBackend`, and cannot relax normal production export requirements.
- Security: The operator guide contains placeholders only, requires rotation of the credential exposed during design, keeps HF/W&B secrets environment-only, and explicitly defers every upload.
- Validation: The review-strengthened integration test passes 1/1 in 78.539 seconds; focused workflow tests pass 20/20; and all Balalaika tests pass 227/227 in 250.756 seconds. Both shell launchers pass `bash -n`; Python compilation, exact two-script inventory, diff/whitespace, and long-line checks pass. Top-level discovery remains at the approved baseline 23/24 because `asset/qwen_ref_4.wav` is absent. `flake8` is not installed in the current environment, so that optional check could not run.
- Follow-up: No network, real tokenization, CUDA/GPU training, real validation generation, online W&B logging, production export, or upload was run. Task 14 must stop after generating the real pilot and wait for explicit user listening approval.
