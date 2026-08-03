# Three-GPU CosyVoice3 LoRA Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, isolated three-GPU Accelerate smoke test that performs real broad-LoRA optimizer updates without relaxing any eight-GPU production gate.

**Architecture:** A new `smoke.py` owns only the non-production diagnostic and atomically publishes evidence under `three_gpu_smoke/`. `workflow.py` exposes an internal command that never enters the production state machine. `run_phase1.sh` branches only under an explicit environment flag and `run_phase2.sh` rejects that flag; their ordinary eight-GPU launch paths remain literal and unchanged.

**Tech Stack:** Python 3.11, PyTorch BF16, Accelerate 1.12, PEFT LoRA, PyArrow, unittest, Bash.

## Global Constraints

- Production requires exactly eight unique GPUs, BF16, and the existing two phase scripts; no production check may be parameterized to three ranks.
- The smoke requires exactly three ranks, BF16, the real frozen non-RL base model, `LoraSettings()`, the production LoRA audit, and the verified four-row `memorization_cache/`.
- Smoke output is exclusively `<run_root>/three_gpu_smoke/`; it must neither publish nor modify `workflow_stages/`, `memorization/`, `cache/`, evaluation/W&B/checkpoint/adapter directories, or exports.
- Smoke runs exactly two optimizer steps by default with the production AdamW settings and must gather a finite loss from every rank for every step.
- Smoke never scans or tokenizes the corpus, runs validation, starts W&B, asserts memorization, or promotes a production phase.
- Preserve the existing `run_phase1.sh` and `run_phase2.sh` names as the only user-facing scripts. The smoke command uses `BALALAIKA_THREE_GPU_SMOKE=1`; phase 2 must refuse it.
- Never print or persist credentials. Prefix shell commands with `rtk`.

---

## File Structure

- Create: `cosyvoice/finetune/balalaika/smoke.py` — validated request, three-rank runner, temporary-output cleanup, and success-manifest publication.
- Modify: `cosyvoice/finetune/balalaika/workflow.py` — internal `three-gpu-smoke` CLI only; it must bypass `ProductionBackend` and all workflow stages.
- Modify: `examples/balalaika/cosyvoice3_lora/run_phase1.sh` — explicit smoke branch with three-device validation and `accelerate launch --num_processes 3`.
- Modify: `examples/balalaika/cosyvoice3_lora/run_phase2.sh` — fail closed if smoke mode is requested; retain normal eight-GPU logic.
- Modify: `tests/finetune/balalaika/test_workflow.py` — CLI/launcher behavior tests.
- Create: `tests/finetune/balalaika/test_smoke.py` — request, isolation, manifest, rank aggregation, and failure-cleanup tests using fixture models and accelerators.
- Modify: `examples/balalaika/cosyvoice3_lora/README.md` — one concise diagnostic-only invocation and non-production warning.

### Task 1: Isolated three-rank runner

**Files:**

- Create: `cosyvoice/finetune/balalaika/smoke.py`
- Create: `tests/finetune/balalaika/test_smoke.py`

**Interfaces:**

- Consumes: `CacheManifest`, `_load_memorization_cache`, `select_memorization_rows`, `build_memorization_dataloader`, `LoraSettings`, `load_base_llm`, `inject_lora`, `audit_trainable_parameters`, `atomic_write_json`, and `sha256_file`.
- Produces: `ThreeGpuSmokeRequest`, `ThreeGpuSmokeError`, `run_three_gpu_smoke(request) -> dict[str, object]`.

- [ ] **Step 1: Write failing request-isolation tests**

```python
def test_request_accepts_only_three_bf16_ranks_and_distinct_smoke_root(self):
    request = ThreeGpuSmokeRequest(cache=cache, output_root=root / "three_gpu_smoke", base_model_dir=base)
    self.assertEqual(request.world_size, 3)
    self.assertEqual(request.steps, 2)
    with self.assertRaisesRegex(ValueError, "world_size=3"):
        ThreeGpuSmokeRequest(cache=cache, output_root=root / "three_gpu_smoke", base_model_dir=base, world_size=8)
    with self.assertRaisesRegex(ValueError, "three_gpu_smoke"):
        ThreeGpuSmokeRequest(cache=cache, output_root=root / "memorization", base_model_dir=base)
```

- [ ] **Step 2: Run the new request test and confirm it fails because the module is absent**

Run: `rtk python -m unittest tests.finetune.balalaika.test_smoke.ThreeGpuSmokeTests.test_request_accepts_only_three_bf16_ranks_and_distinct_smoke_root -v`

Expected: import failure for `cosyvoice.finetune.balalaika.smoke`.

- [ ] **Step 3: Implement the immutable request**

```python
@dataclass(frozen=True)
class ThreeGpuSmokeRequest:
    cache: CacheManifest
    output_root: Path
    base_model_dir: Path
    split_plan: Path | None = None
    steps: int = 2
    world_size: int = 3
    mixed_precision: str = "bf16"
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    accelerator_factory: Callable[..., Any] | None = None
```

Require `world_size == 3`, `mixed_precision == "bf16"`, `steps == 2`, positive optimization values, and output directory name exactly `three_gpu_smoke`. Resolve the default split plan as `cache.root / "split_plan"`. Reject an existing target or `.three_gpu_smoke.incomplete` before work.

- [ ] **Step 4: Write a failing runner test with a three-rank fixture**

```python
def test_runner_gathers_two_finite_losses_and_publishes_only_smoke_manifest(self):
    report = run_three_gpu_smoke(request_with_fixture_accelerator_and_model())
    manifest = json.loads((root / "three_gpu_smoke" / "manifest.json").read_text())
    self.assertEqual(manifest["world_size"], 3)
    self.assertEqual(manifest["steps"], 2)
    self.assertEqual(len(manifest["losses_by_step"]), 2)
    self.assertTrue(all(len(values) == 3 for values in manifest["losses_by_step"]))
    self.assertFalse((root / "workflow_stages").exists())
    self.assertFalse((root / "memorization").exists())
```

- [ ] **Step 5: Run the runner test and confirm it fails because the runner is absent**

Run: `rtk python -m unittest tests.finetune.balalaika.test_smoke.ThreeGpuSmokeTests.test_runner_gathers_two_finite_losses_and_publishes_only_smoke_manifest -v`

Expected: FAIL because `run_three_gpu_smoke` is not defined.

- [ ] **Step 6: Implement the bounded runner**

Use `select_memorization_rows(request.split_plan, request.cache)` and a deterministic `build_memorization_dataloader(rows, batch_size=1)`. Create Accelerate with BF16 and no `log_with`. Require `accelerator.num_processes == 3`, load the base model, inject `LoraSettings()`, audit trainables, prepare model/optimizer/loader, and cycle local batches for precisely two steps. On every step require a finite scalar loss, call backward, clip, step, and gather one scalar loss per rank. Rank zero atomically writes `manifest.json` only after every rank succeeds; include world size, steps, cache manifest checksum, base `llm.pt` checksum, validated audit payload, and loss lists. Wrap in a target temporary directory and remove it on every exception after a barrier. Do not call any W&B, evaluation, checkpoint, or production-stage function.

- [ ] **Step 7: Write and run a failing cleanup test**

```python
def test_nonfinite_loss_removes_temporary_output_and_never_publishes_manifest(self):
    with self.assertRaisesRegex(ThreeGpuSmokeError, "finite"):
        run_three_gpu_smoke(request_with_nonfinite_fixture_model())
    self.assertFalse((root / "three_gpu_smoke").exists())
    self.assertFalse((root / ".three_gpu_smoke.incomplete").exists())
```

Run: `rtk python -m unittest tests.finetune.balalaika.test_smoke -v`

Expected before the cleanup implementation: FAIL on the missing cleanup behavior.

- [ ] **Step 8: Add synchronized failure handling and run the focused smoke suite**

Each rank captures its exception as a serializable status, gathers statuses, waits at the same collective boundary, removes the temporary directory only on rank zero, and raises one `ThreeGpuSmokeError` describing the first failing rank. Rerun:

`rtk python -m unittest tests.finetune.balalaika.test_smoke -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
rtk git add cosyvoice/finetune/balalaika/smoke.py tests/finetune/balalaika/test_smoke.py
rtk git commit -m "feat: add isolated three-GPU LoRA smoke"
rtk git push
```

### Task 2: Internal CLI, explicit launcher mode, and operator guide

**Files:**

- Modify: `cosyvoice/finetune/balalaika/workflow.py`
- Modify: `examples/balalaika/cosyvoice3_lora/run_phase1.sh`
- Modify: `examples/balalaika/cosyvoice3_lora/run_phase2.sh`
- Modify: `examples/balalaika/cosyvoice3_lora/README.md`
- Modify: `tests/finetune/balalaika/test_workflow.py`

**Interfaces:**

- Consumes: `ThreeGpuSmokeRequest` and `run_three_gpu_smoke` from Task 1.
- Produces: internal CLI command `three-gpu-smoke`; phase-1 smoke launcher path; phase-2 fail-closed behavior.

- [ ] **Step 1: Write failing CLI and launcher tests**

```python
def test_internal_smoke_command_does_not_construct_production_backend(self):
    with mock.patch("cosyvoice.finetune.balalaika.workflow.ProductionBackend") as production, \
         mock.patch("cosyvoice.finetune.balalaika.smoke.run_three_gpu_smoke", return_value={"world_size": 3}):
        self.assertEqual(main(["three-gpu-smoke", "--run-root", str(root)]), ExitCode.SUCCESS)
    production.assert_not_called()

def test_phase1_smoke_launcher_uses_only_three_visible_devices(self):
    result = subprocess.run(
        ["bash", str(phase1)], env={**os.environ, "BALALAIKA_THREE_GPU_SMOKE": "1", "CUDA_VISIBLE_DEVICES": "0,3,7", "PATH": fake_path},
        text=True, capture_output=True, check=True,
    )
    self.assertIn("--num_processes 3", captured_arguments())

def test_phase2_smoke_mode_is_rejected(self):
    result = subprocess.run(["bash", str(phase2)], env={**os.environ, "BALALAIKA_THREE_GPU_SMOKE": "1"}, text=True, capture_output=True)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("phase 1", result.stderr)
```

- [ ] **Step 2: Run the focused tests and confirm their expected failures**

Run: `rtk python -m unittest tests.finetune.balalaika.test_workflow.WorkflowTests.test_internal_smoke_command_does_not_construct_production_backend tests.finetune.balalaika.test_workflow.WorkflowTests.test_phase1_smoke_launcher_uses_only_three_visible_devices tests.finetune.balalaika.test_workflow.WorkflowTests.test_phase2_smoke_mode_is_rejected -v`

Expected: FAIL because the internal command and launcher branches do not exist.

- [ ] **Step 3: Add the internal CLI with no production backend path**

Add `three-gpu-smoke` to `_parser()`. Its handler parses the normal non-secret path options, creates `RunPaths` with exactly three logical devices, loads only `memorization_cache` through `_load_memorization_cache`, constructs `ThreeGpuSmokeRequest(output_root=run_root / "three_gpu_smoke")`, calls the runner, prints redacted JSON evidence on rank zero, and returns success. Do not call `_resolve`, `run_phase1`, `run_phase2`, `_stage_store`, or `ProductionBackend`.

- [ ] **Step 4: Add the phase-1 shell branch and phase-2 rejection**

At the beginning of `run_phase1.sh`, before the eight-device count check, recognize only `BALALAIKA_THREE_GPU_SMOKE=1`. Validate exactly three unique numeric IDs, set `BALALAIKA_VISIBLE_DEVICES=0,1,2`, preserve the repository/Matcha `PYTHONPATH`, and execute:

```bash
accelerate launch --num_processes 3 --mixed_precision bf16 \
  -m cosyvoice.finetune.balalaika.workflow three-gpu-smoke "$@"
```

For every other value (including unset), keep the existing literal eight-device branch. In `run_phase2.sh`, fail before device validation if the smoke variable equals `1`, explaining that the diagnostic is phase-1-only.

- [ ] **Step 5: Document the diagnostic with a non-production warning**

Add this operator example without credentials:

```bash
CUDA_VISIBLE_DEVICES=0,3,7 BALALAIKA_THREE_GPU_SMOKE=1 \
  bash examples/balalaika/cosyvoice3_lora/run_phase1.sh
```

State that it runs precisely two optimizer updates on the four cached clips, writes only `three_gpu_smoke/manifest.json`, and cannot satisfy the mandatory eight-GPU memorization gate or start training.

- [ ] **Step 6: Run focused verification**

Run:

```bash
rtk python -m unittest tests.finetune.balalaika.test_smoke tests.finetune.balalaika.test_workflow -v
rtk bash -n examples/balalaika/cosyvoice3_lora/run_phase1.sh
rtk bash -n examples/balalaika/cosyvoice3_lora/run_phase2.sh
rtk python -m py_compile cosyvoice/finetune/balalaika/smoke.py cosyvoice/finetune/balalaika/workflow.py
rtk git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 7: Commit and push**

```bash
rtk git add cosyvoice/finetune/balalaika/workflow.py examples/balalaika/cosyvoice3_lora/run_phase1.sh examples/balalaika/cosyvoice3_lora/run_phase2.sh examples/balalaika/cosyvoice3_lora/README.md tests/finetune/balalaika/test_workflow.py
rtk git commit -m "feat: expose three-GPU smoke launcher"
rtk git push
```

### Task 3: Real idle-GPU smoke execution

**Files:**

- Runtime output only: `/workspace/cosyvoice3-balalaika-lora/three_gpu_smoke/manifest.json`
- Modify: `implementation-notes.md`

**Interfaces:**

- Consumes: tested phase-1 launcher and real `memorization_cache`.
- Produces: checksum-bound diagnostic report; no repository code artifacts beyond a dated implementation note.

- [ ] **Step 1: Confirm the three GPUs are idle and production stages remain unchanged**

Run:

```bash
rtk nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
rtk ls -1 /workspace/cosyvoice3-balalaika-lora/workflow_stages
```

Require GPUs 0, 3, and 7 to have no active compute utilization before launch.

- [ ] **Step 2: Run the real bounded diagnostic**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES=0,3,7 BALALAIKA_THREE_GPU_SMOKE=1 \
  bash examples/balalaika/cosyvoice3_lora/run_phase1.sh
```

Expected: exactly three Accelerate ranks complete two optimizer steps and publish only `three_gpu_smoke/manifest.json`.

- [ ] **Step 3: Independently verify its manifest and isolation**

Run a read-only Python assertion that checks `world_size == 3`, `steps == 2`, three finite per-rank losses at each step, valid audit payload, and absence of all additional production stage files. Re-run `rtk nvidia-smi` afterward to confirm resources were released.

- [ ] **Step 4: Record the evidence and commit/push**

Append only public diagnostic facts to `implementation-notes.md`; do not include paths to credentials or their values. Then:

```bash
rtk git diff --check
rtk git add implementation-notes.md
rtk git commit -m "docs: record three-GPU LoRA smoke evidence"
rtk git push
```

## Plan self-review

- Spec coverage: Task 1 covers request validation, genuine model/data/LoRA/optimizer work, atomic evidence, cross-rank finite losses, and cleanup. Task 2 covers the sole explicit launcher mode, phase-2 rejection, preservation of eight-rank production, and documentation. Task 3 covers the requested real run and evidence.
- Placeholder scan: this plan contains no deferred implementation markers or unspecified validation steps.
- Type consistency: Task 2 imports precisely `ThreeGpuSmokeRequest` and `run_three_gpu_smoke` from Task 1; Task 3 invokes the same phase-1 mode tested in Task 2.
