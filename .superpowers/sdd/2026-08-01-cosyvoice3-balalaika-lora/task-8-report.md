# Task 8 Report: Accelerate Training, Global Progress, and Resume

## Status

Implemented the eight-rank Accelerate phase trainer, exact global sample boundaries, adapter-only optimizer gate, complete atomic checkpoints, pending-validation recovery, and bounded token-limit qualification.

## TDD evidence

### RED

The initial boundary and resume tests were written before `training.py` existed:

```text
rtk python -m unittest tests.finetune.balalaika.test_training -v
Ran 3 tests in 0.008s
FAILED (failures=3)
```

All three failures reported the expected missing training module. After the first GREEN slice, qualification tests were added after removing the untested draft implementation:

```text
Ran 10 tests in 1.638s
FAILED (errors=2)
```

Both errors were the expected absent `qualify_token_limit`. Pending-validation recovery then failed because no atomic checkpoint existed after the callback exception:

```text
Ran 1 test in 1.458s
FAILED (errors=1)
```

The final edge-case RED run caught all three intended defects: float-rounded large sample thresholds, same-number phase schedule drift accepted on resume, and a remote-rank qualification OOM not propagated collectively:

```text
Ran 13 tests in 1.774s
FAILED (failures=3)
```

The required smoke initially failed because Accelerate 1.12's `--cpu` launcher ran one process despite `--num_processes 2`. The fixture script gained a CPU-only two-worker Accelerate fallback and then passed with Gloo world size two.

### GREEN

```text
rtk python -m unittest tests.finetune.balalaika.test_training -v
Ran 13 tests in 1.221s
OK

rtk accelerate launch --cpu --num_processes 2 -m tests.finetune.balalaika.accelerate_smoke
exit 0
```

The smoke artifact contains `world_size: 2` and eight ordered records with ordinals 1 through 8, derived from collective real-sample reductions.

## Verification

```text
rtk python -m unittest tests.finetune.balalaika.test_artifacts tests.finetune.balalaika.test_cache tests.finetune.balalaika.test_data tests.finetune.balalaika.test_metrics tests.finetune.balalaika.test_model tests.finetune.balalaika.test_sources tests.finetune.balalaika.test_tokenizer tests.finetune.balalaika.test_training tests.finetune.balalaika.test_validation_data -v
Ran 106 tests in 6.291s
OK

rtk python -m py_compile cosyvoice/finetune/balalaika/training.py tests/finetune/balalaika/test_training.py tests/finetune/balalaika/accelerate_smoke.py
rtk git diff --check
```

Compilation and whitespace checks exited successfully.

## Self-review

- `Accelerator` is constructed with BF16, the requested accumulation count, and W&B logging. Training requires exactly eight ranks and uses `prepare`, `accumulate`, `backward`, Accelerate clipping, and Accelerate checkpoint state.
- AdamW receives only `requires_grad` parameters, and its parameter IDs must exactly equal Task 6's audited adapter names.
- Each local real count is summed collectively. Fraction targets use integer ceiling arithmetic, events wait for `sync_gradients`, and overshoot releases every due ordinal once. Phase 1 owns indices 1–16 and phase 2 owns 17–40.
- `ProgressState` checkpoints phase, epoch, batch offset, per-epoch/global samples, next fraction, validation index, and optimizer steps. Resume validates cache, complete phase spec, base checksum, LoRA settings, token limit, accumulation, and world size before state load and batch skipping.
- Accelerate state, adapter export, RNG files, all state checksums, optimizer/scheduler identity, fraction, progress, and provenance are staged in one directory and atomically renamed. A pending checkpoint is resumed at validation before any more optimizer work; succeeded checkpoints do not duplicate validation.
- Qualification tries only 2,000–6,000 tokens in 1,000-token increments, gathers OOM status and memory from all eight ranks, and selects the largest candidate retaining at least two GiB everywhere. Production OOM raises without changing the fixed limit.

## Concerns

Per instruction, no model checkpoint, production data, real GPU training, credential, installation, or upload was used. Real eight-GPU BF16 execution—including DDP synchronization of Task 7's explicit empty padding batches—remains a required hardware smoke. Accelerate 1.12 ignores `--num_processes` with its CPU launcher in this environment, so the committed smoke fixture starts two CPU Accelerate workers itself after detecting the launcher's single-process behavior.
