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
