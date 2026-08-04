# Separated Cache and Phase-1 Operations Design

Date: 2026-08-04

## Purpose

Split the current monolithic Phase-1 workflow into independently operable
preparation, memorization, and training actions. This makes the full-corpus
speech-token extraction observable and restartable without beginning model
optimization.

## Operator contract

The recipe exposes three scripts:

1. `run_phase1.sh --memorize`
   - Requires the approved pilot and the existing four-row memorization cache.
   - Runs only the mandatory eight-GPU BF16 four-audio memorization proof.
   - Publishes only the sealed memorization evidence on success.

2. `prepare_cache.sh`
   - Requires the approved pilot.
   - Runs only the full-corpus CosyVoice3 speech-token cache build.
   - Publishes only the authenticated `cache/` artifacts and never initializes
     W&B, runs validation, optimizer steps, or Phase 1.

3. `run_phase1.sh --train`
   - Requires sealed memorization evidence and a verified complete cache.
   - Runs the capacity smoke, untouched-base 2,000-generation validation, and
     Phase-1 training for the existing two-epoch specification.

`run_phase2.sh` remains unchanged: it requires Phase-1 completion and performs
only Phase 2 and its existing finalization path.

## Capacity smoke

Capacity smoke is not training. It loads the real LoRA model and cache on all
eight GPUs, runs bounded forward/backward probes at fixed token-budget
candidates, and selects the largest candidate that leaves two GiB of VRAM
headroom on every GPU. It remains part of `run_phase1.sh --train`, immediately
before baseline validation, because it selects the batch token limit used by
training.

## Safety and compatibility

- The existing default no-flag `run_phase1.sh` invocation must fail with a
  clear mode-selection error; it must not silently run all stages.
- Each mode is independently restart-safe through the existing authenticated
  stage store. Existing valid stage evidence is reused; missing prerequisites
  fail closed.
- Production modes remain exactly eight Accelerate BF16 ranks. The explicit
  three-GPU smoke diagnostic remains unchanged and separate from these modes.
- The 20 reserved prompt clips, full cache schema, W&B run identity, 2,000
  generation baseline, Phase-1 hyperparameters, and no-upload policy do not
  change.
- `prepare_cache.sh` is the only additional user-facing script. It shares the
  existing internal Python workflow CLI rather than duplicating data logic.

## Internal structure

Add three narrow workflow commands:

- `phase1-memorize`: validates prior preflight/tokenizer/pilot stages, records
  the pilot approval when supplied, prepares the four-row cache if needed, and
  runs the memorization gate.
- `prepare-cache`: validates prior stages and approved pilot, then builds the
  full cache only.
- `phase1-train`: requires the prior sealed memorization and full-cache stages,
  runs capacity qualification, baseline validation, and Phase-1 training.

The existing all-in-one `phase1` command is removed from the public parser to
avoid a bypass. Common stage authentication remains shared and no stage’s
evidence format is weakened.

## Validation

- Test each mode’s exact stage sequence and prerequisite rejection.
- Test all three launchers: mode forwarding, eight-GPU validation, no
  accidental W&B/training in cache mode, and no default Phase-1 action.
- Keep the end-to-end fixture’s semantic coverage by driving the three Phase-1
  operations in sequence before Phase 2.
- Run focused workflow/cache tests, the full Balalaika suite, shell syntax, and
  Python compilation before committing.
