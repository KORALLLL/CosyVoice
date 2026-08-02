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
