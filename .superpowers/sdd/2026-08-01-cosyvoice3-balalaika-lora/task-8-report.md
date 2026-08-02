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

## Fix Round 1

### Findings addressed

- Empty local ranks no longer create a loss directly from parameters. Every rank gathers real-count evidence; when any rank is empty, all ranks gather one CPU single-sample template, and empty ranks execute the prepared DDP forward on that real-shaped dummy with its loss multiplied by zero. Globally empty work bypasses forward, backward, clipping, optimizer, scheduler, and real-sample progress.
- The already rank-partitioned Task 7 loader is wrapped as an Accelerate `DataLoaderShard` with `num_processes=1`. This retains rank ownership while exposing `end_of_dataloader`, so an incomplete accumulation remainder synchronizes once. Samples and physical batch offsets become checkpoint-visible only after that optimizer step commits.
- A stale derived `.incomplete` staging path is safely removed before retry. The interrupted-staging regression proves the preceding completed checkpoint remains byte-identical and usable.
- `TrainRequest` accepts only the literal Task 1 PhaseSpecs. Resume identity additionally binds eligible samples, clipping, scheduler class/initial state, cache-manifest checksum, sampler seed/window, token budget, explicit dataloader identity, accumulation, world size, base checksum, LoRA settings, and full phase spec. Accelerate file checksums retain optimizer/scheduler/RNG state integrity.
- Pending recovery now drains every already-due fractional event before requesting another batch. This matters when one committed optimizer step overshoots several small-dataset thresholds and validation interrupts partway through the due sequence.
- The fake Accelerator now alternates `sync_gradients`, signals the final loader batch, provides explicit global reductions/rank counts, and supports shared dummy templates. The distributed smoke now executes `train_phase` through real two-rank CPU DDP with uneven batches and pending recovery.

### TDD evidence

Strict schedule validation was added first:

```text
rtk python -m unittest tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_train_request_rejects_modified_phase_schedules -v
Ran 1 test in 0.015s
FAILED (failures=4)
```

The expanded identity test then failed because sampler identity fields did not exist:

```text
Ran 1 test in 1.109s
FAILED (errors=1)
TypeError: TrainRequest.__init__() got an unexpected keyword argument 'sampler_seed'
```

Non-divisible accumulation and global-zero tests exposed three instead of four committed remainder steps and four instead of two optimizer steps:

```text
Ran 2 tests in 1.847s
FAILED (failures=2)
```

The empty-rank regression proved the prepared forward was bypassed (`forward_calls` was 0 instead of 2), and interrupted staging reproduced the stale-path refusal:

```text
Ran 1 test in 1.665s
FAILED (failures=1)

Ran 1 test in 1.547s
FAILED (errors=1)
FileExistsError: checkpoint publication target already exists
```

After correcting two smoke-fixture setup issues (mixed-precision initialization order and a batch-size-none loader unsupported by `skip_first_batches`), the behavior-level smoke RED showed both ranks replaying their real samples after pending validation resume. The resumed code was incorrectly resetting the epoch before draining already-due fraction ordinals.

### GREEN and verification

```text
rtk python -m unittest tests.finetune.balalaika.test_training -v
Ran 18 tests in 2.052s
OK

rtk python -m unittest tests.finetune.balalaika.test_artifacts tests.finetune.balalaika.test_cache tests.finetune.balalaika.test_data tests.finetune.balalaika.test_metrics tests.finetune.balalaika.test_model tests.finetune.balalaika.test_sources tests.finetune.balalaika.test_tokenizer tests.finetune.balalaika.test_training tests.finetune.balalaika.test_validation_data -v
Ran 111 tests in 5.814s
OK

rtk accelerate launch --cpu --num_processes 2 -m tests.finetune.balalaika.accelerate_smoke
exit 0
```

The smoke artifact records global samples `3`, optimizer steps `1`, boundary ordinals `1..8`, and identical adapter weights on both ranks. It covers two rank-local steps with accumulation `3`, rank sample counts `[1,1]` then `[1,0]`, a validation-index-3 interruption, pending resume, and no repeated model forward after resume.

`rtk python -m py_compile ...` and `rtk git diff --check` also exited successfully.

### Remaining qualification boundary

No real eight-GPU, BF16 model, production corpus, credential, installation, or upload was used. The passing smoke is real two-process CPU DDP exercised through Accelerate; it is not evidence of eight-GPU hardware qualification.

## Fix Round 2

### Finding addressed

Scheduler identity no longer assumes `state_dict()` captures future behavior. For schedulers exposing `lr_lambdas`, each callable now receives a SHA-256 fingerprint over stable Python semantics: bytecode, code constants and names, argument metadata, defaults, keyword defaults, closure cells, and referenced global values. Partials, bound methods, builtins, and inspectable callable objects have explicit representations. Recursive, non-finite, or unsupported configuration is rejected rather than assigned an ambiguous identity. Source paths and line numbers are excluded.

Standard scheduler class and initial state remain in the identity, so constructor-relevant immutable state stays bound alongside any callable semantics. Resume compares this complete identity before Accelerate state loading.

### TDD evidence

Two functions returned the same multiplier at initialization but different multipliers after step two. Their LambdaLR `state_dict()` values were deliberately equal. Before the fix, both the direct identity comparison and resume rejection failed:

```text
rtk python -m unittest tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_lambda_scheduler_identity_includes_future_callable_semantics tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_resume_rejects_changed_lambda_scheduler_semantics -v
Ran 2 tests in 1.555s
FAILED (failures=2)
```

After semantic fingerprinting:

```text
Ran 2 tests in 1.488s
OK
```

### Verification

```text
rtk python -m unittest tests.finetune.balalaika.test_training -v
Ran 20 tests in 2.281s
OK

rtk python -m unittest tests.finetune.balalaika.test_artifacts tests.finetune.balalaika.test_cache tests.finetune.balalaika.test_data tests.finetune.balalaika.test_metrics tests.finetune.balalaika.test_model tests.finetune.balalaika.test_sources tests.finetune.balalaika.test_tokenizer tests.finetune.balalaika.test_training tests.finetune.balalaika.test_validation_data -v
Ran 113 tests in 6.182s
OK

rtk accelerate launch --cpu --num_processes 2 -m tests.finetune.balalaika.accelerate_smoke
exit 0
```

`rtk python -m py_compile ...` and `rtk git diff --check` also exited successfully. No external work or eight-GPU hardware claim was made.

## Fix Round 3

### Finding addressed

Scheduler callable identity no longer represents a referenced module only by its module name. Python bytecode is inspected for actual `LOAD_GLOBAL` operations, including those inside nested code objects. Each name is resolved against the callable's real globals or builtins namespace and passed through the existing recursive semantic serializer. Supported immutable values, functions, tuples, frozensets, and string-keyed mappings remain fingerprintable.

Module namespaces, dynamic name resolution, missing globals, and other values that cannot be deterministically and completely serialized are rejected fail closed. This prevents a mutable module attribute from changing future LR behavior while retaining the same checkpoint identity. The production/default constant lambda remains stable across equivalent instances.

### TDD evidence

Before the fix, the module-backed schedule was accepted and a future multiplier referenced only from nested bytecode was omitted. The equivalent constant-lambda control already passed:

```text
rtk python -m unittest tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_fingerprints_globals_in_nested_code tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_module_backed_mutable_attribute tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_default_lambda_scheduler_identity_is_stable_across_instances -v
Ran 3 tests in 0.051s
FAILED (failures=2)
```

After recursive global resolution and fail-closed module handling:

```text
Ran 3 tests in 0.069s
OK
```

### Verification

```text
rtk python -m unittest tests.finetune.balalaika.test_training -v
Ran 23 tests in 1.225s
OK

rtk python -m unittest tests.finetune.balalaika.test_artifacts tests.finetune.balalaika.test_cache tests.finetune.balalaika.test_data tests.finetune.balalaika.test_metrics tests.finetune.balalaika.test_model tests.finetune.balalaika.test_sources tests.finetune.balalaika.test_tokenizer tests.finetune.balalaika.test_training tests.finetune.balalaika.test_validation_data -v
Ran 116 tests in 6.452s
OK

rtk accelerate launch --cpu --num_processes 2 -m tests.finetune.balalaika.accelerate_smoke
exit 0
```

`rtk python -m py_compile cosyvoice/finetune/balalaika/training.py tests/finetune/balalaika/test_training.py tests/finetune/balalaika/accelerate_smoke.py` and `rtk git diff --check` exited successfully.

### Self-review and qualification boundary

The new resolver keys identity from semantic global loads rather than every `co_names` entry, so attribute names are not mistaken for independent globals. Nested code is traversed, actual builtin bindings are fingerprinted when supported, and unresolved or dynamic lookup is rejected. No source path, line number, object `repr`, or module-name-only fallback is used.

No real eight-GPU, BF16 model, production corpus, credential, installation, or upload was used. The passing smoke remains real two-process CPU DDP through Accelerate, not eight-GPU hardware qualification.

## Fix Round 4

### Finding addressed

Scheduler callable identity now rejects dynamic namespace and name discovery instead of fingerprinting only the lookup builtin. Direct and aliased `globals`, `locals`, `vars`, and `getattr` paths fail closed, as do direct `__dict__` reads and the closely equivalent dynamic builtins `__import__`, `delattr`, `dir`, `eval`, `exec`, `hasattr`, and `setattr`.

Builtin validation happens in the recursive semantic serializer, so the rule also covers aliases stored in defaults, closures, nested functions, tuples, and mappings. Bytecode validation rejects `__dict__` loads in the root function or any nested code object. Statically referenced immutable globals, closures, defaults, and nested code retain their existing semantic fingerprints, and equivalent default constant schedules remain stable.

### TDD evidence

The production change that makes the regressions pass is rejection of dynamic lookup at the builtin-semantic and `__dict__` bytecode boundaries. Before that change, direct `globals()[dynamic_name]`, an aliased `globals` default, `locals`, `vars`, `getattr`, and `__dict__` were all accepted:

```text
rtk python -m unittest tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_globals_dynamic_name_lookup tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_equivalent_dynamic_name_lookups -v
Ran 2 tests in 0.050s
FAILED (failures=6)
```

After the minimal fail-closed validation:

```text
rtk python -m unittest tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_globals_dynamic_name_lookup tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_equivalent_dynamic_name_lookups -v
Ran 2 tests in 0.077s
OK
```

### Files and commit

- `cosyvoice/finetune/balalaika/training.py`: rejects dynamic lookup builtins by resolved identity and rejects `__dict__` load bytecode.
- `tests/finetune/balalaika/test_training.py`: covers direct and aliased `globals`, `locals`, `vars`, `getattr`, and `__dict__` lookup paths.
- Code/tests commit: `d94e64d69f84f0fbb25e943230139299d38f72ee` (`fix: reject dynamic scheduler lookups`).

### Verification

```text
rtk python -m unittest tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_globals_dynamic_name_lookup tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_equivalent_dynamic_name_lookups tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_fingerprints_globals_in_nested_code tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_module_backed_mutable_attribute tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_default_lambda_scheduler_identity_is_stable_across_instances tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_lambda_scheduler_identity_includes_future_callable_semantics tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_resume_rejects_changed_lambda_scheduler_semantics -v
Ran 7 tests in 1.074s
OK

rtk python -m unittest tests.finetune.balalaika.test_training -v
Ran 25 tests in 2.162s
OK

rtk python -m unittest tests.finetune.balalaika.test_artifacts tests.finetune.balalaika.test_cache tests.finetune.balalaika.test_data tests.finetune.balalaika.test_metrics tests.finetune.balalaika.test_model tests.finetune.balalaika.test_sources tests.finetune.balalaika.test_tokenizer tests.finetune.balalaika.test_training tests.finetune.balalaika.test_validation_data -v
Ran 118 tests in 5.484s
OK

rtk accelerate launch --cpu --num_processes 2 -m tests.finetune.balalaika.accelerate_smoke
exit 0

rtk python -m py_compile cosyvoice/finetune/balalaika/training.py tests/finetune/balalaika/test_training.py tests/finetune/balalaika/accelerate_smoke.py
exit 0

rtk git diff --check
exit 0
```

### Self-review

- Dynamic lookup is rejected by the resolved builtin object, not merely its source-level name. A shadowed pure-Python function named `globals` can still be fingerprinted, while an actual `builtins.globals` alias cannot evade validation.
- The same recursive callable serializer handles direct globals, defaults, closures, nested code, and values inside supported immutable containers, so there is no alias-specific partial fingerprint.
- `__dict__` access is rejected recursively at bytecode inspection. No attempt is made to infer the runtime stack target or capture only part of a mutable namespace.
- Existing nested immutable-global and equivalent default-scheduler controls passed unchanged.
- The conservative rule also rejects some statically resolvable uses of `getattr`; this is intentional because the implementation does not claim a sound bytecode data-flow proof for attribute targets.

### Concerns and qualification boundary

No model checkpoint, production data, credential, installation, upload, or other external work was used. The smoke remains real two-process CPU DDP through Accelerate; real eight-GPU BF16 behavior remains a hardware qualification boundary rather than a claim of this fix round.

## Fix Round 5

### Finding addressed

Scheduler callable identity now rejects `IMPORT_NAME`, `IMPORT_FROM`, and `IMPORT_STAR` anywhere in the root callable or recursively nested code. This closes the runtime-local namespace bypass where `import builtins; builtins.globals()[name]` or `from builtins import globals` could discover mutable global values that were absent from the fingerprint.

Direct `__import__`, `eval`, and `exec` rejection remains covered, and `compile` is now rejected by the same resolved-builtin identity check. Direct function namespace paths through `__globals__`, `__builtins__`, or `__getattribute__` are also rejected alongside `__dict__`. The production default lambda, statically resolved immutable globals, closures, defaults, and nested code retain their existing stable fingerprints.

### TDD evidence

Before the implementation change, root, from-import, nested, and import-star bytecode were all accepted. Direct `compile` and the three equivalent function-namespace paths were also accepted; direct `__import__`, `eval`, and `exec` already failed closed:

```text
rtk python -m unittest tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_runtime_imports_in_nested_code tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_dynamic_import_and_evaluation tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_unresolved_namespace_attributes -v
Ran 3 tests in 0.055s
FAILED (failures=8)
```

After the recursive opcode, builtin, and namespace-attribute validation:

```text
rtk python -m unittest tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_runtime_imports_in_nested_code tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_dynamic_import_and_evaluation tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_unresolved_namespace_attributes -v
Ran 3 tests in 0.063s
OK
```

### Files and commit

- `cosyvoice/finetune/balalaika/training.py`: rejects runtime import opcodes, `compile`, and direct unresolved function-namespace attributes.
- `tests/finetune/balalaika/test_training.py`: covers inline and nested imports, from-import and import-star forms, `__import__`, `eval`, `exec`, `compile`, and direct function namespace paths.
- Code/tests commit: `b196cf6bdc2c9ef9ba5828520cf059f16fc13039` (`fix: reject imported scheduler namespaces`).

### Verification

```text
rtk python -m unittest tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_runtime_imports_in_nested_code tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_dynamic_import_and_evaluation tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_unresolved_namespace_attributes tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_globals_dynamic_name_lookup tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_equivalent_dynamic_name_lookups tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_fingerprints_globals_in_nested_code tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_scheduler_identity_rejects_module_backed_mutable_attribute tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_default_lambda_scheduler_identity_is_stable_across_instances tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_lambda_scheduler_identity_includes_future_callable_semantics tests.finetune.balalaika.test_training.AccelerateTrainingTests.test_resume_rejects_changed_lambda_scheduler_semantics -v
Ran 10 tests in 1.513s
OK

rtk python -m unittest tests.finetune.balalaika.test_training -v
Ran 28 tests in 2.510s
OK

rtk python -m unittest tests.finetune.balalaika.test_artifacts tests.finetune.balalaika.test_cache tests.finetune.balalaika.test_data tests.finetune.balalaika.test_metrics tests.finetune.balalaika.test_model tests.finetune.balalaika.test_sources tests.finetune.balalaika.test_tokenizer tests.finetune.balalaika.test_training tests.finetune.balalaika.test_validation_data -v
Ran 121 tests in 6.438s
OK

rtk accelerate launch --cpu --num_processes 2 -m tests.finetune.balalaika.accelerate_smoke
exit 0

rtk python -m py_compile cosyvoice/finetune/balalaika/training.py tests/finetune/balalaika/test_training.py tests/finetune/balalaika/accelerate_smoke.py
exit 0

rtk git diff --check
exit 0
```

### Self-review and qualification boundary

The import check is opcode-based rather than spelling-based, so aliases and local bindings cannot bypass it, and the existing recursive code walk applies the same rule to nested functions. Forbidden dynamic builtins are checked by resolved object identity, preserving shadowed pure Python functions with the same source-level name. No model checkpoint, production data, credential, installation, upload, or other external work was used. The smoke remains real two-process CPU DDP through Accelerate; real eight-GPU BF16 behavior remains a hardware qualification boundary.

## Architectural resolution after review breaker

### Decision implemented

Arbitrary scheduler callables and factories are no longer part of the training API. `TrainRequest.scheduler_factory` was replaced by a frozen `SchedulerSpec`; its closed `SchedulerKind` enum contains only `constant-v1`, the fixed learning-rate schedule required by both approved Balalaika phases. The trainer constructs CosyVoice's `ConstantLR` internally.

The declarative spec validates unknown kinds, rejects non-`SchedulerSpec` request values, serializes to `{"kind": "constant-v1"}`, and is included verbatim under `identity.scheduler`. Resume identity comparison remains exact, so an added, removed, or changed scheduler-spec field rejects the checkpoint before Accelerate state loading. The callable bytecode, globals, closure, import, and reflection fingerprint machinery and all bypass-focused tests were removed.

### TDD evidence

The behavior tests were changed first. Against the callable API, the focused suite produced the expected missing-spec errors and old-identity failure:

```text
rtk python -m unittest tests.finetune.balalaika.test_training -v
Ran 21 tests in 1.551s
FAILED (failures=1, errors=3)
```

The three errors were the absent `SchedulerSpec`, absent `TrainRequest.scheduler_spec`, and the same absent spec in unknown-kind validation. The failure showed the checkpoint still stored a `LambdaLR` class/state/callable fingerprint instead of the literal declarative identity.

After the minimal closed-spec implementation:

```text
rtk python -m unittest tests.finetune.balalaika.test_training -v
Ran 21 tests in 2.196s
OK
```

The replacement tests cover JSON serialization and the real constant learning-rate trajectory, unknown scheduler kinds, non-spec request values, the exact manifest identity, and resume rejection when that identity contains an unrecognized field. No test injects a scheduler callable.

### Files and commit

- `cosyvoice/finetune/balalaika/training.py`: defines the closed immutable scheduler schema, constructs `ConstantLR` internally, records the literal spec identity, and removes callable fingerprint/reflection code.
- `tests/finetune/balalaika/test_training.py`: replaces bypass-oriented callable tests with declarative behavior and resume-identity regressions.
- `tests/finetune/balalaika/accelerate_smoke.py`: supplies the production declarative spec explicitly.
- Code/tests commit: `7ec547e688a8b393e4f77e00330afc6f5a1bb246` (`fix: replace scheduler callables with closed spec`).

### Verification

```text
rtk python -m unittest tests.finetune.balalaika.test_artifacts tests.finetune.balalaika.test_cache tests.finetune.balalaika.test_data tests.finetune.balalaika.test_metrics tests.finetune.balalaika.test_model tests.finetune.balalaika.test_sources tests.finetune.balalaika.test_tokenizer tests.finetune.balalaika.test_training tests.finetune.balalaika.test_validation_data -v
Ran 114 tests in 5.474s
OK

rtk accelerate launch --cpu --num_processes 2 -m tests.finetune.balalaika.accelerate_smoke
exit 0

rtk python -m py_compile cosyvoice/finetune/balalaika/training.py tests/finetune/balalaika/test_training.py tests/finetune/balalaika/accelerate_smoke.py
exit 0

rtk git diff --check
exit 0
```

The smoke artifact records `world_size: 2`, boundary ordinals `1..8`, `global_samples: 3`, `optimizer_steps: 1`, and identical adapter weights on both ranks. This reconfirms pending resume, empty-rank collective behavior, and final accumulation commit on the declarative scheduler path.

### Concerns and qualification boundary

The supported scheduler set is intentionally limited to the versioned constant schedule required by this recipe. Adding a new schedule requires a new explicit enum/schema member, construction branch, trajectory test, and identity behavior; arbitrary Python injection is not available. No model checkpoint, production data, credential, installation, upload, or other external work was used. The passing smoke remains real two-process CPU DDP through Accelerate, not evidence of real eight-GPU BF16 hardware qualification.
