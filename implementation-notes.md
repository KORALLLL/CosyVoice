# Implementation Notes

## 2026-08-02 - Execution setup

- Decision: Implement the approved two-phase CosyVoice3 LoRA plan on the isolated `feature/cosyvoice3-balalaika-lora` worktree.
- Changed: Proceed despite one unrelated baseline failure, with the user's explicit approval; the baseline result is 23/24 passing.
- Validation: The failing test is `OpenAICompatibilityTests.test_streaming_response_has_pcm_headers_and_sample_rate`; it expects an absent ignored file, `asset/qwen_ref_4.wav`, while production returns `None` for a missing prompt file.
- Follow-up: Do not alter that unrelated baseline behavior as part of this training recipe.
