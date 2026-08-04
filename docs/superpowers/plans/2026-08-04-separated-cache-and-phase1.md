# Separated Cache and Phase-1 Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split Phase-1 preparation into explicit memorization, cache-build, and training operations while preserving every existing eight-GPU integrity gate.

**Architecture:** `workflow.py` gains narrow internal commands that reuse the existing authenticated stage store rather than duplicating backend logic. `run_phase1.sh` becomes an explicit mode router for memorization and training; new `prepare_cache.sh` owns corpus-wide token extraction. Phase 2 remains a separate unchanged operator path.

**Tech Stack:** Python 3.11, Accelerate BF16, PyArrow, W&B, Bash, unittest.

## Global Constraints

- Production modes use exactly eight unique CUDA GPUs through the existing Accelerate BF16 configuration.
- `run_phase1.sh --memorize` performs only pilot approval (when supplied), four-row cache preparation, and the 100%-accuracy memorization gate.
- `prepare_cache.sh` performs only approved-pilot prerequisite checks and the full speech-token cache build; it must not initialize W&B, evaluate, run capacity smoke, or optimize model weights.
- `run_phase1.sh --train` requires authenticated memorization and cache stages, then performs capacity smoke, baseline validation 0, and the existing two-epoch Phase-1 training.
- `run_phase2.sh` retains its existing authenticated Phase-2/export behavior.
- The no-flag `run_phase1.sh` command fails clearly. `--status` remains read-only.
- Preserve agreement split, LoRA configuration, 2,000-generation validation, W&B run identity, no-upload policy, and all production stage evidence formats.
- Add exactly one user-facing script: `prepare_cache.sh`. Prefix shell commands with `rtk`.

---

## File Structure

- Modify: `cosyvoice/finetune/balalaika/workflow.py` — isolated Phase-1 internal commands and parser dispatch.
- Modify: `tests/finetune/balalaika/test_workflow.py` — state-order, prerequisite, parser, and launcher behavior tests.
- Modify: `tests/finetune/balalaika/tiny_workflow.py` — end-to-end fixture drives the three operations in order.
- Modify: `tests/finetune/balalaika/test_integration.py` — assert the revised operation trace.
- Modify: `examples/balalaika/cosyvoice3_lora/run_phase1.sh` — explicit `--memorize` / `--train` routing and preserved smoke/status routes.
- Create: `examples/balalaika/cosyvoice3_lora/prepare_cache.sh` — eight-GPU cache-only launcher.
- Modify: `examples/balalaika/cosyvoice3_lora/README.md` — operational sequence and capacity-smoke explanation.

### Task 1: Split the internal Phase-1 state machine

**Files:**

- Modify: `cosyvoice/finetune/balalaika/workflow.py`
- Modify: `tests/finetune/balalaika/test_workflow.py`

**Interfaces:**

- Produces: `run_phase1_memorize(args) -> int`, `run_prepare_cache(args) -> int`, and `run_phase1_train(args) -> int`.
- Consumes: existing `_ensure_stage`, `_require_stage`, `ProductionBackend`, `WorkflowOptions`, `PhaseSpec.for_phase(1)`, and all existing stage-authentication logic.

- [ ] **Step 1: Write failing mode-sequence and prerequisite tests**

```python
def test_phase1_memorize_stops_after_sealed_memorization(self):
    backend = FakeBackend()
    self.assertEqual(run_phase1_memorize(_args(root, backend, "a" * 64)), ExitCode.SUCCESS)
    operations = [item for item in backend.calls if isinstance(item, str)]
    self.assertEqual(operations, ["preflight", "qualify_tokenizer", "ensure_pilot", "prepare_memorization", "memorize"])
    self.assertNotIn("build_cache", operations)
    self.assertNotIn("capacity_smoke", operations)

def test_prepare_cache_requires_sealed_memorization_and_builds_only_cache(self):
    with self.assertRaises(StageRequirementError):
        run_prepare_cache(_args(root, FakeBackend()))
    backend = FakeBackend()
    _publish_required_memorization_stages(root, backend)
    self.assertEqual(run_prepare_cache(_args(root, backend)), ExitCode.SUCCESS)
    self.assertEqual([item for item in backend.calls if isinstance(item, str)], ["build_cache"])

def test_phase1_train_requires_cache_and_runs_baseline_before_training(self):
    backend = FakeBackend()
    _publish_required_memorization_and_cache_stages(root, backend)
    self.assertEqual(run_phase1_train(_args(root, backend)), ExitCode.SUCCESS)
    operations = [item for item in backend.calls if isinstance(item, str)]
    self.assertEqual(operations[:3], ["capacity_smoke", "ensure_logging", "evaluate_0"])
    self.assertTrue(any(isinstance(item, tuple) and item[0] == "train" for item in backend.calls))
```

- [ ] **Step 2: Run the new tests and confirm they fail because the split operations do not exist**

Run:

```bash
rtk python -m unittest \
  tests.finetune.balalaika.test_workflow.WorkflowTests.test_phase1_memorize_stops_after_sealed_memorization \
  tests.finetune.balalaika.test_workflow.WorkflowTests.test_prepare_cache_requires_sealed_memorization_and_builds_only_cache \
  tests.finetune.balalaika.test_workflow.WorkflowTests.test_phase1_train_requires_cache_and_runs_baseline_before_training -v
```

Expected: import/name failure for the new operation functions.

- [ ] **Step 3: Implement `run_phase1_memorize`**

Implement the current Phase-1 prefix through `memorization_complete` only:

1. ensure `preflight_complete`, `tokenizer_qualified`, and `pilot_ready`;
2. validate the exact pilot checksum;
3. require `--approve-pilot-sha256` only when `pilot_approved` is not already sealed, then publish the existing approval stage;
4. ensure `memorization_cache_ready` and collective `memorization_complete`;
5. return success without cache build, W&B, validation, capacity smoke, or training.

Keep the existing pilot-review-required return when no approval exists. Reuse the same stage names and evidence payloads so a later cache or training command can authenticate them.

- [ ] **Step 4: Implement `run_prepare_cache` and `run_phase1_train`**

`run_prepare_cache` must use `_require_stage` for `preflight_complete`, `tokenizer_qualified`, `pilot_ready`, `pilot_approved`, and `memorization_complete`, then `_ensure_stage(..., "cache_complete", collective=False, operation=backend.build_cache)`. It must make no logging, evaluation, capacity, or training calls.

`run_phase1_train` must require those same prerequisites plus `cache_complete`, then preserve the current suffix exactly: collective `capacity_smoke_complete`, W&B initialization, collective `validation_00` with `VALIDATION_GENERATIONS`, collective Phase-1 training, `_validate_phase_result(..., 1, 16)`, and `phase1_complete` publication.

Retain `run_phase1` as a private compatibility composition for existing non-CLI callers only, implemented as `memorize → prepare-cache → train`; remove it from the public parser and do not route a normal shell invocation to it.

- [ ] **Step 5: Add parser tests and public command dispatch**

Add parser subcommands `phase1-memorize`, `prepare-cache`, and `phase1-train`. Only `phase1-memorize` accepts `--approve-pilot-sha256`; both cache/train reject it as an unknown option. Update `main()` dispatch to call the three functions. Keep `phase2`, `status`, and `three-gpu-smoke` behavior unchanged.

```python
with self.assertRaises(SystemExit):
    main(["phase1"])
with self.assertRaises(SystemExit):
    main(["prepare-cache", "--approve-pilot-sha256", "a" * 64])
```

- [ ] **Step 6: Run focused workflow tests and commit**

Run:

```bash
rtk python -m unittest tests.finetune.balalaika.test_workflow -v
rtk python -m py_compile cosyvoice/finetune/balalaika/workflow.py
rtk git diff --check
```

Then commit and push:

```bash
rtk git add cosyvoice/finetune/balalaika/workflow.py tests/finetune/balalaika/test_workflow.py
rtk git commit -m "feat: split phase1 workflow operations"
rtk git push
```

### Task 2: Expose explicit shell operations and update end-to-end coverage

**Files:**

- Modify: `examples/balalaika/cosyvoice3_lora/run_phase1.sh`
- Create: `examples/balalaika/cosyvoice3_lora/prepare_cache.sh`
- Modify: `examples/balalaika/cosyvoice3_lora/README.md`
- Modify: `tests/finetune/balalaika/test_workflow.py`
- Modify: `tests/finetune/balalaika/tiny_workflow.py`
- Modify: `tests/finetune/balalaika/test_integration.py`

**Interfaces:**

- Consumes: Task 1 internal commands `phase1-memorize`, `prepare-cache`, and `phase1-train`.
- Produces: explicit operator scripts and an integration trace `("run_phase1 --memorize", "prepare_cache", "run_phase1 --train", "run_phase2")`.

- [ ] **Step 1: Write failing launcher tests**

```python
def test_phase1_launcher_requires_explicit_operation(self):
    result = subprocess.run(["bash", str(phase1)], text=True, capture_output=True)
    self.assertEqual(result.returncode, 2)
    self.assertIn("--memorize or --train", result.stderr)

def test_phase1_launcher_forwards_memorize_and_train_to_distinct_commands(self):
    self.assertEqual(captured_launch(phase1, "--memorize"), "phase1-memorize")
    self.assertEqual(captured_launch(phase1, "--train"), "phase1-train")

def test_prepare_cache_launcher_uses_eight_rank_cache_command(self):
    arguments = captured_launch(prepare_cache)
    self.assertIn("--num_processes 8", arguments)
    self.assertIn("prepare-cache", arguments)
```

- [ ] **Step 2: Run launcher tests and confirm expected failures**

Run:

```bash
rtk python -m unittest tests.finetune.balalaika.test_workflow.WorkflowTests.test_phase1_launcher_requires_explicit_operation tests.finetune.balalaika.test_workflow.WorkflowTests.test_phase1_launcher_forwards_memorize_and_train_to_distinct_commands tests.finetune.balalaika.test_workflow.WorkflowTests.test_prepare_cache_launcher_uses_eight_rank_cache_command -v
```

Expected: failures because `run_phase1.sh` still launches `phase1` unconditionally and `prepare_cache.sh` does not exist.

- [ ] **Step 3: Implement the launchers without weakening GPU validation**

Keep the current three-GPU smoke block and `--status` behavior. For ordinary eight-GPU Phase 1:

```bash
case "${1:-}" in
  --memorize) command="phase1-memorize"; shift ;;
  --train) command="phase1-train"; shift ;;
  *) echo "choose --memorize or --train" >&2; exit 2 ;;
esac
exec accelerate launch --config_file "$config" --num_processes 8 --mixed_precision bf16 \
  -m cosyvoice.finetune.balalaika.workflow "$command" "$@"
```

Create `prepare_cache.sh` by reusing the existing strict eight-device and Matcha `PYTHONPATH` setup, then launch only `prepare-cache`. It accepts no operation flag. It must not contain `torchrun`, W&B credentials, or upload logic.

- [ ] **Step 4: Update recipe documentation**

Replace the monolithic Phase-1 instructions with these ordered invocations:

```bash
bash examples/balalaika/cosyvoice3_lora/run_phase1.sh --memorize \
  --approve-pilot-sha256 <pilot-manifest-sha256>
bash examples/balalaika/cosyvoice3_lora/prepare_cache.sh
bash examples/balalaika/cosyvoice3_lora/run_phase1.sh --train
```

Document that cache preparation performs full speech-token extraction only, and explain that capacity smoke selects the safe token budget with two GiB headroom before baseline validation and training.

- [ ] **Step 5: Revise the tiny end-to-end fixture**

Make `run_tiny_workflow()` invoke the three new internal operations in order after pilot approval, then `run_phase2`. Update result metadata and `test_integration.py` to require the new trace while retaining all existing memorization, cache-resume, 41-validation, adapter-lineage, and no-upload assertions.

- [ ] **Step 6: Run verification and commit**

Run:

```bash
rtk python -m unittest tests.finetune.balalaika.test_workflow tests.finetune.balalaika.test_integration -v
rtk bash -n examples/balalaika/cosyvoice3_lora/run_phase1.sh
rtk bash -n examples/balalaika/cosyvoice3_lora/prepare_cache.sh
rtk bash -n examples/balalaika/cosyvoice3_lora/run_phase2.sh
rtk python -m py_compile cosyvoice/finetune/balalaika/workflow.py tests/finetune/balalaika/tiny_workflow.py
rtk git diff --check
```

Then commit and push:

```bash
rtk git add examples/balalaika/cosyvoice3_lora/run_phase1.sh examples/balalaika/cosyvoice3_lora/prepare_cache.sh examples/balalaika/cosyvoice3_lora/README.md tests/finetune/balalaika/test_workflow.py tests/finetune/balalaika/tiny_workflow.py tests/finetune/balalaika/test_integration.py
rtk git commit -m "feat: separate full cache preparation"
rtk git push
```

### Task 3: Full regression and operational handoff

**Files:**

- Modify: `implementation-notes.md`

**Interfaces:**

- Consumes: complete Task 1/2 implementation.
- Produces: an operator-facing verified split with no real training/cache run.

- [ ] **Step 1: Run the complete Balalaika suite**

```bash
rtk python -m unittest discover -s tests/finetune/balalaika
```

Expected: all tests pass. Do not run full tokenization, memorization, W&B, validation generation, or training for this code-only change.

- [ ] **Step 2: Perform read-only launcher command capture**

Use the existing fake-Python launcher-test harness to verify that `--memorize`, `--train`, and `prepare_cache.sh` preserve the repository/Matcha `PYTHONPATH`, exact eight-process Accelerate options, and their distinct internal commands.

- [ ] **Step 3: Record implementation decision and commit**

Append the nonsecret operational split, capacity-smoke meaning, and verification outcome to `implementation-notes.md`. Then:

```bash
rtk git diff --check
rtk git add implementation-notes.md
rtk git commit -m "docs: record separated phase1 operations"
rtk git push
```

## Plan self-review

- Spec coverage: Task 1 separates and authenticates each internal operation; Task 2 exposes exactly the requested memorization flag and cache script while retaining baseline validation inside training; Task 3 verifies the full recipe and documents the operator handoff.
- Placeholder scan: no deferred implementation markers or generic testing instructions remain.
- Type consistency: Task 2 launcher commands exactly match Task 1 parser commands; Task 3 validates the same scripts and workflow operations.
