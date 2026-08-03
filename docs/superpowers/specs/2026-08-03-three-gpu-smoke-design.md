# Three-GPU CosyVoice3 LoRA Smoke Design

Date: 2026-08-03

## Purpose

Run a bounded, real three-GPU Accelerate smoke test while other local GPUs are
busy. The test must exercise the frozen CosyVoice3 base LLM, the production
broad-LoRA injection and audit, the already-authenticated four-row
memorization cache, distributed data loading, forward/backward computation,
and optimizer stepping.

## Boundaries

- Production keeps its exact eight-GPU contract. The normal launchers, phase
  manifests, memorization gate, cache, evaluation, W&B run, checkpoints, final
  export, and completion semantics remain eight-rank-only.
- The smoke invocation is opt-in through an explicit test-only launcher mode
  and rejects any world size other than three.
- It writes only under a dedicated `three_gpu_smoke/` directory below the run
  root. It must never read, create, authenticate, or modify
  `workflow_stages/`, `memorization/`, `cache/`, `evaluation/`, adapter
  directories, W&B manifests, or final exports.
- It consumes the existing `memorization_cache/` after integrity validation and
  never rebuilds it or scans the full corpus.
- It runs a fixed, small number of optimizer updates (default two) with BF16
  and the production LoRA/AdamW settings. It records only non-secret,
  checksum-bound diagnostic evidence: visible devices, world size, cache
  identity, LoRA audit summary, per-step finite losses, and success state.
- It does not claim memorization success, production capacity qualification,
  validation quality, or readiness to begin either training phase.

## Operator interface

The two production scripts stay the only user-facing scripts. Each gains one
explicit smoke route, selected only when all of these conditions hold:

1. `BALALAIKA_THREE_GPU_SMOKE=1`;
2. `CUDA_VISIBLE_DEVICES` contains exactly three unique physical IDs; and
3. the command is phase 1 (phase 2 rejects this mode).

The launcher exports the matching logical IDs `0,1,2`, launches exactly three
Accelerate processes, and invokes an internal `three-gpu-smoke` CLI command.
Without this opt-in flag, current eight-device validation and eight-process
launching are unchanged.

For this immediate diagnostic, the command will use `CUDA_VISIBLE_DEVICES=0,3,7`.

## Implementation shape

1. Add an immutable smoke request with strict validation: three ranks, BF16,
   two positive steps, and a distinct output root.
2. Add a smoke runner that validates the real four-row cache, loads the base
   LLM, injects and audits the production LoRA configuration, shards the real
   rows through Accelerate, then performs two synchronized finite-loss AdamW
   updates.
3. Gather rank-local evidence; rank zero atomically writes a manifest only
   after every rank reports success. Any error cleans temporary output and
   leaves no success manifest.
4. Wire the internal command and test-only launcher branch. The production
   state-machine code paths remain unmodified except for rejecting test mode
   where necessary.

## Validation

- Unit tests first prove that the ordinary launch path still rejects three
  GPUs, the explicit smoke path accepts exactly three, and phase 2 refuses it.
- Unit tests prove a smoke request cannot target the production run artifacts
  and that a successful mock distributed run writes only the smoke manifest.
- Run focused launcher/workflow/smoke tests, shell syntax checks, Python
  compilation, and whitespace checks before committing.
- After push, run the real command on GPUs 0, 3, and 7. Inspect its manifest
  for three ranks and finite loss records, then report it as a smoke diagnostic
  only.

## Failure handling

An out-of-memory, non-finite loss, rank mismatch, cache-integrity failure, or
collective error fails the smoke command without retrying automatically. It
cannot affect the production state machine; diagnosis and any configuration
change require a new explicit user instruction.
