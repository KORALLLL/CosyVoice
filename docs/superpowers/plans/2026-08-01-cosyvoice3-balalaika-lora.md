# CosyVoice3 Balalaika Two-Phase LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restart-safe, eight-GPU Accelerate/PEFT workflow that tokenizes the Balalaika corpus once, runs mandatory pilot and memorization gates, LoRA-trains CosyVoice3 in two ASR-agreement phases, validates hard-number speech with GigaAM v3 RNN-T and W&B, and exports a standalone `llm.pt`.

**Architecture:** Add a focused `cosyvoice.finetune.balalaika` package whose modules isolate provenance/state, corpus joining, token caching, model adaptation, batching, training, evaluation, and workflow orchestration. Two shell launchers under a new example recipe call one internal Python CLI. Original WebDataset shards remain read-only; training consumes compact per-phase Parquet caches containing text and CosyVoice3 speech tokens.

**Tech Stack:** Python 3.11, PyTorch 2.8.0+cu128, Transformers 4.57.3, Accelerate 1.12.0, PEFT 0.20.0, PyArrow 25.0.0, ONNX Runtime GPU for CUDA 12, `onnx-asr` 0.12.0, W&B 0.28.1, HyperPyYAML, unittest.

## Global Constraints

- Use base/non-RL `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`; never use `_RL`.
- Use exactly eight RTX 5090 GPUs (`0,1,2,3,4,5,6,7`) through `accelerate launch --multi_gpu --num_processes 8` for production work.
- Keep `/workspace/balalaika_proprietary_v2/train/shard_*.tar` and all canonical sidecars read-only.
- Training text is `rover_punctuated_accented`; instruction is `You are a helpful assistant.<|endofprompt|>`.
- Phase 1 is non-null `asr_agreement_mean < 0.95` for 2 epochs at `1e-4`.
- Phase 2 is non-null `asr_agreement_mean >= 0.95` for 3 epochs at `5e-5`.
- Exclude all 309 null-agreement rows and the 20 deterministic phase-2 prompt clips.
- LoRA defaults are rank 64, alpha 128, dropout 0.05, bias `none`.
- Target every active Qwen2 linear layer, Qwen2 text embeddings, CosyVoice3 speech embeddings, and CosyVoice3 `llm_decoder`; exclude internal Qwen `lm_head`.
- Do not train flow, HiFT, CampPlus, dense base weights, or the internal Qwen `lm_head`.
- Require manual checksum-bound tokenization-pilot approval before corpus tokenization.
- Require 100% per-sample teacher-forced speech-token accuracy on four samples for three consecutive checks before production work.
- Validate exactly 2,000 generations at baseline and every one-eighth epoch: 41 validation points and 82,000 total generations.
- Use `bitmanagerai/hard_number_eval_for_tts`, input `stressed`, reference `normalized_gold`, and `onnx_asr.load_model("gigaam-v3-rnnt")`.
- Produce `utt-wer`, `utt-cer`, `num-wer`, and `num-cer` as micro metrics and log them to one resumable W&B run.
- Read `HF_TOKEN` and `WANDB_API_KEY` only from the environment; never persist or print them.
- Provide exactly two user-facing scripts: `run_phase1.sh` and `run_phase2.sh`.
- Do not upload any cache, adapter, model, dataset, or Hub artifact.
- Prefix every shell command with `rtk` per repository instructions.

---

## File Structure

Create these focused implementation units:

- `cosyvoice/finetune/__init__.py`: finetuning namespace.
- `cosyvoice/finetune/balalaika/__init__.py`: recipe package exports.
- `cosyvoice/finetune/balalaika/config.py`: immutable paths, phase specs, and defaults.
- `cosyvoice/finetune/balalaika/artifacts.py`: checksums, atomic JSON, stage manifests, and dependency identity.
- `cosyvoice/finetune/balalaika/sources.py`: canonical input inventory, streaming joins, split plan, and 20-prompt reservation.
- `cosyvoice/finetune/balalaika/metrics.py`: normalization, edit counts, number-span extraction/projection, and aggregation.
- `cosyvoice/finetune/balalaika/validation_data.py`: authenticated private benchmark fetch and schema preflight.
- `cosyvoice/finetune/balalaika/cache.py`: tar reading, tokenizer protocol, atomic phase Parquet publication, and prompt WAV extraction.
- `cosyvoice/finetune/balalaika/tokenizer.py`: CUDA ONNX speech-token backend, reconstruction pilot, and approval binding.
- `cosyvoice/finetune/balalaika/model.py`: base LLM loading, broad LoRA injection/audit, adapter I/O, and merge.
- `cosyvoice/finetune/balalaika/data.py`: memory-mapped cache dataset, deterministic dynamic batches, and collator.
- `cosyvoice/finetune/balalaika/training.py`: Accelerate loop, global progress, fractional checkpoints, and phase resume.
- `cosyvoice/finetune/balalaika/memorization.py`: four-sample overfit gate using the production trainer path.
- `cosyvoice/finetune/balalaika/evaluation.py`: distributed synthesis, GigaAM ASR, result JSONL, W&B, and listening panel.
- `cosyvoice/finetune/balalaika/workflow.py`: phase state machine and internal CLI.
- `examples/balalaika/cosyvoice3_lora/conf/accelerate.yaml`: fixed one-node/eight-process BF16 Accelerate config.
- `examples/balalaika/cosyvoice3_lora/requirements-cu128.txt`: recipe dependency delta for the verified Blackwell stack.
- `examples/balalaika/cosyvoice3_lora/run_phase1.sh`: phase-1 operator entry point.
- `examples/balalaika/cosyvoice3_lora/run_phase2.sh`: phase-2 operator entry point.
- `examples/balalaika/cosyvoice3_lora/README.md`: complete operator guide.
- `tests/finetune/balalaika/`: one unittest module per implementation unit plus fixtures.

Modify only these existing files:

- `cosyvoice/llm/llm.py`: expose stable input-embedding access and per-sample token statistics needed by PEFT and memorization.
- `README.md`: add one concise link to the new recipe.

---

### Task 1: Runtime Configuration, Atomic Artifacts, and Preflight

**Files:**
- Create: `cosyvoice/finetune/__init__.py`
- Create: `cosyvoice/finetune/balalaika/__init__.py`
- Create: `cosyvoice/finetune/balalaika/config.py`
- Create: `cosyvoice/finetune/balalaika/artifacts.py`
- Create: `examples/balalaika/cosyvoice3_lora/conf/accelerate.yaml`
- Create: `examples/balalaika/cosyvoice3_lora/requirements-cu128.txt`
- Test: `tests/finetune/balalaika/test_artifacts.py`

**Interfaces:**
- Produces: `RunPaths.from_env() -> RunPaths`, `PhaseSpec.for_phase(number: int) -> PhaseSpec`, `sha256_file(path: Path) -> str`, `atomic_write_json(path: Path, value: Mapping[str, object]) -> None`, `StageRecord`, `StageStore.require(name: str) -> StageRecord`, `StageStore.publish(name: str, payload: Mapping[str, object]) -> StageRecord`, and `collect_environment() -> dict[str, object]`.
- Consumes: no earlier task interfaces.

- [ ] **Step 1: Write failing configuration and atomic-publication tests**

```python
class ArtifactTests(unittest.TestCase):
    def test_phase_boundary_is_literal(self):
        self.assertEqual(PhaseSpec.for_phase(1).predicate, "asr_agreement_mean < 0.95")
        self.assertEqual(PhaseSpec.for_phase(1).epochs, 2)
        self.assertEqual(PhaseSpec.for_phase(2).predicate, "asr_agreement_mean >= 0.95")
        self.assertEqual(PhaseSpec.for_phase(2).epochs, 3)

    def test_stage_payload_is_checksum_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StageStore(Path(tmp))
            published = store.publish("pilot", {"rows": 4, "seed": 1986})
            loaded = store.require("pilot")
            self.assertEqual(loaded.payload["rows"], 4)
            self.assertEqual(loaded.manifest_sha256, sha256_file(published.path))
```

- [ ] **Step 2: Run the tests and confirm imports fail**

Run: `rtk python -m unittest tests.finetune.balalaika.test_artifacts -v`

Expected: FAIL because `cosyvoice.finetune.balalaika` does not exist.

- [ ] **Step 3: Implement immutable defaults and atomic stage manifests**

```python
@dataclass(frozen=True)
class PhaseSpec:
    number: int
    agreement_min: float | None
    agreement_max: float | None
    epochs: int
    learning_rate: float
    predicate: str

    @classmethod
    def for_phase(cls, number: int) -> "PhaseSpec":
        specs = {
            1: cls(1, None, 0.95, 2, 1e-4, "asr_agreement_mean < 0.95"),
            2: cls(2, 0.95, None, 3, 5e-5, "asr_agreement_mean >= 0.95"),
        }
        if number not in specs:
            raise ValueError(f"phase must be 1 or 2, got {number}")
        return specs[number]
```

Implement atomic JSON with a same-directory temporary file, `fsync`, and `os.replace`. `StageRecord.manifest_sha256` is computed from the completed file rather than stored self-referentially inside it. `StageStore.require` must reject a missing stage, changed payload, changed dependency lock, or changed input provenance.

- [ ] **Step 4: Add exact Accelerate and dependency files**

`accelerate.yaml` must specify `LOCAL_MACHINE`, `MULTI_GPU`, BF16, one machine, machine rank 0, and eight processes. `requirements-cu128.txt` must pin the already verified installed versions and add `onnx-asr[gpu,hub]==0.12.0` with `onnxruntime-gpu[cuda,cudnn]<1.27`; do not pin or downgrade Torch.

- [ ] **Step 5: Add environment preflight assertions**

`collect_environment` must record Python, Torch, CUDA runtime, GPU names/capabilities, Accelerate, PEFT, Transformers, PyArrow, W&B, ONNX Runtime, and onnx-asr versions. Reject fewer than eight GPUs, any non-RTX-5090 device, CUDA unavailability, capability below `(12, 0)`, missing BF16, missing `CUDAExecutionProvider`, or an ONNX Runtime version at or above 1.27 on this CUDA-12 recipe.

- [ ] **Step 6: Run tests and lint the new modules**

Run: `rtk python -m unittest tests.finetune.balalaika.test_artifacts -v`

Run: `rtk flake8 --max-line-length 180 --ignore B006,B008,B905,C408,E402,E731,E741,W503,W504,F401,F403,F405,F722,F841 cosyvoice/finetune/balalaika/config.py cosyvoice/finetune/balalaika/artifacts.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add cosyvoice/finetune examples/balalaika/cosyvoice3_lora/conf/accelerate.yaml examples/balalaika/cosyvoice3_lora/requirements-cu128.txt tests/finetune/balalaika/test_artifacts.py
rtk git commit -m "feat: add Balalaika finetune runtime preflight"
```

### Task 2: Canonical Corpus Join, 0.95 Split, and Prompt Reservation

**Files:**
- Create: `cosyvoice/finetune/balalaika/sources.py`
- Test: `tests/finetune/balalaika/test_sources.py`
- Test fixture: `tests/finetune/balalaika/fixtures/combined.jsonl`
- Test fixture: `tests/finetune/balalaika/fixtures/rover.jsonl`

**Interfaces:**
- Consumes: `RunPaths`, `sha256_file`, and `atomic_write_json` from Task 1.
- Produces: `JoinedRow`, `SplitCounts`, `inventory_sources(paths: RunPaths) -> SourceInventory`, `build_split_plan(paths: RunPaths, seed: int = 1986) -> SplitCounts`, and `iter_split_rows(plan_dir: Path, shard: int) -> Iterator[JoinedRow]`.

- [ ] **Step 1: Write failing boundary, null, join, and reservation tests**

```python
def test_split_boundary_and_null(self):
    rows = [
        joined("000000/a.mp3", 0.949999),
        joined("000000/b.mp3", 0.95),
        joined("000000/c.mp3", None),
    ]
    assigned = assign_phases(rows, reserved=set())
    self.assertEqual([row.phase for row in assigned], [1, 2, None])

def test_reservation_is_stable_and_high_agreement_only(self):
    selected_a = reserve_prompt_ids(self.rows, count=20, seed=1986)
    selected_b = reserve_prompt_ids(reversed(self.rows), count=20, seed=1986)
    self.assertEqual(selected_a, selected_b)
    self.assertTrue(all(self.by_id[key].agreement >= 0.95 for key in selected_a))
```

Also test duplicate combined IDs, duplicate ROVER IDs, unequal join keys, 518/520 tar inventories, and reconciliation of phase/null/reserved counts.

- [ ] **Step 2: Run tests and confirm missing interfaces**

Run: `rtk python -m unittest tests.finetune.balalaika.test_sources -v`

Expected: FAIL with missing `sources` module.

- [ ] **Step 3: Implement streaming canonical readers**

Start `unzstd -c <archive>` with `subprocess.Popen`, pass its stdout to `tarfile.open(fileobj=stdout, mode="r|")`, and require both the decompressor and tar reader to finish successfully. Group rows by the six-digit shard component in `source_relative_path`. Stream the 1.3 GB combined sidecar line-by-line. Keep at most one 8,000-row shard dictionary in memory. Reject any schema version other than the documented version and any non-finite agreement.

- [ ] **Step 4: Implement order-independent deterministic prompt selection**

```python
def reservation_score(source_relative_path: str, seed: int) -> bytes:
    material = f"{seed}\0{source_relative_path}".encode("utf-8")
    return hashlib.sha256(material).digest()
```

Choose the 20 lowest scores among eligible phase-2 rows after text/model-limit preflight. Store the exact IDs and scores. A second streaming pass writes per-shard split-plan JSONLs with `phase`, `reserved`, `agreement`, `text`, and `instruct`.

- [ ] **Step 5: Implement exact source reconciliation**

Require 519 source tars, 4,075,032 combined rows, 4,075,032 ROVER rows, unique IDs, 309 null agreement rows, and exactly 20 reserved rows. Publish `split_plan/manifest.json` only when `phase1 + phase2 + null + reserved + model_limit_exclusions == 4_075_032`.

- [ ] **Step 6: Run focused and full fixture tests**

Run: `rtk python -m unittest tests.finetune.balalaika.test_sources -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add cosyvoice/finetune/balalaika/sources.py tests/finetune/balalaika/test_sources.py tests/finetune/balalaika/fixtures
rtk git commit -m "feat: add canonical Balalaika split planning"
```

### Task 3: Private Validation Dataset and Exact Error Metrics

**Files:**
- Create: `cosyvoice/finetune/balalaika/metrics.py`
- Create: `cosyvoice/finetune/balalaika/validation_data.py`
- Test: `tests/finetune/balalaika/test_metrics.py`
- Test: `tests/finetune/balalaika/test_validation_data.py`

**Interfaces:**
- Consumes: atomic artifact helpers from Task 1.
- Produces: `normalize_asr_text(text: str) -> str`, `EditCounts`, `NumberSpan`, `extract_number_span(row: Mapping[str, object]) -> NumberSpan`, `score_row(reference: str, hypothesis: str, number_span: NumberSpan) -> RowScores`, `aggregate_scores(rows: Iterable[RowScores]) -> MetricSummary`, and `fetch_validation_rows(cache_dir: Path, token: str) -> list[ValidationRow]`.

- [ ] **Step 1: Write failing normalization and edit-count tests**

```python
def test_normalization_matches_asr_policy(self):
    self.assertEqual(normalize_asr_text("Ёлка,  ПЯТЬ!"), "елка пять")

def test_number_only_error_ignores_outside_word(self):
    row = validation_row(
        text="Оплатите 25 рублей завтра.",
        hard_number="25",
        normalized_gold="Оплатите двадцать пять рублей завтра.",
    )
    span = extract_number_span(row)
    scores = score_row(row.normalized_gold, "внесите двадцать пять рублей завтра", span)
    self.assertGreater(scores.utterance.word.distance, 0)
    self.assertEqual(scores.number.word.distance, 0)
```

Add phone, decimal, date, range, repeated-digit, ordinal, and ambiguous-prefix cases. Assert ambiguous anchoring raises `NumberSpanError` rather than guessing.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `rtk python -m unittest tests.finetune.balalaika.test_metrics tests.finetune.balalaika.test_validation_data -v`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement deterministic Levenshtein alignment with boundary projection**

Represent alignment operations as `(reference_index | None, hypothesis_index | None, op)` tuples. Extract the reference number phrase by anchoring normalized raw-text prefix/suffix tokens in normalized gold. Project its first/last reference tokens through the full-sentence alignment to obtain the hypothesis span. Compute word edits on token arrays and character edits on whitespace-free strings.

- [ ] **Step 4: Implement micro and macro aggregation**

```python
@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def rate(self) -> float:
        return (self.substitutions + self.deletions + self.insertions) / self.reference_units
```

`MetricSummary` must expose primary keys `utt-wer`, `utt-cer`, `num-wer`, and `num-cer`, macro equivalents, and the same metrics grouped by `category`.

- [ ] **Step 5: Implement authenticated Dataset Viewer fetch without persisting credentials**

Read `HF_TOKEN` from the caller, send it only as an `Authorization` header, resolve the private Parquet URL through the Dataset Viewer API, download it atomically, and load it with PyArrow. Require config `default`, split `train`, 2,000 rows, ten columns, unique IDs 1–2000, 12 categories, and nonempty required strings. Never include headers in exceptions or manifests.

- [ ] **Step 6: Add all-2,000 span preflight fixture behavior**

The public function must attempt `extract_number_span` for every row, collect failures, and refuse publication if any row is ambiguous. The published benchmark manifest contains the dataset revision/ETag, Parquet checksum, schema, category counts, and 2,000 extracted span boundaries.

- [ ] **Step 7: Run tests**

Run: `rtk python -m unittest tests.finetune.balalaika.test_metrics tests.finetune.balalaika.test_validation_data -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
rtk git add cosyvoice/finetune/balalaika/metrics.py cosyvoice/finetune/balalaika/validation_data.py tests/finetune/balalaika/test_metrics.py tests/finetune/balalaika/test_validation_data.py
rtk git commit -m "feat: add hard-number validation metrics"
```

### Task 4: Tar Streaming and Atomic Compact Cache Builder

**Files:**
- Create: `cosyvoice/finetune/balalaika/cache.py`
- Test: `tests/finetune/balalaika/test_cache.py`
- Test fixture: `tests/finetune/balalaika/fixtures/shard_000000.tar`

**Interfaces:**
- Consumes: `JoinedRow`/`iter_split_rows` from Task 2 and artifact helpers from Task 1.
- Produces: `SpeechTokenizer` protocol, `TarAudioSample`, `iter_tar_audio(path: Path) -> Iterator[TarAudioSample]`, `build_cache_shard(request: CacheShardRequest, tokenizer: SpeechTokenizer) -> CacheShardResult`, and `verify_cache(root: Path) -> CacheManifest`.

- [ ] **Step 1: Write failing synthetic-tar and atomic-cache tests**

```python
class FakeTokenizer:
    def extract(self, audio: list[AudioInput]) -> list[list[int]]:
        return [[sample.frames % 6561, 7, 9] for sample in audio]

def test_cache_has_no_audio_column(self):
    result = build_cache_shard(self.request, FakeTokenizer())
    phase1 = pq.read_table(result.phase1_path)
    self.assertEqual(
        phase1.column_names,
        ["source_relative_path", "text", "instruct", "agreement", "speech_token", "speech_token_len"],
    )
    self.assertNotIn("audio_data", phase1.column_names)
```

Also test JSON/MP3 pairing in either tar member order, missing partners, unexpected member suffixes, decode failure, invalid token `6561`, partial-file recovery, and phase/reserved routing.

- [ ] **Step 2: Run tests and confirm failure**

Run: `rtk python -m unittest tests.finetune.balalaika.test_cache -v`

Expected: FAIL with missing cache module.

- [ ] **Step 3: Implement bounded tar pairing**

Read tar members sequentially and retain only unmatched members for the current key. Validate that embedded JSON `source_relative_path` matches `<shard>/<stem>.mp3`. Yield compressed audio bytes without extracting the source archive to disk.

- [ ] **Step 4: Implement phase Parquet publication**

Decode/resample a bounded batch, call the injected tokenizer, validate token IDs in `[0, 6560]`, and write one phase-1 and one phase-2 Parquet per source shard. Use list-of-int32 Arrow columns and Zstandard compression. Write `.partial`, close/fsync, reopen and verify, then `os.replace`.

- [ ] **Step 5: Extract only the 20 reserved prompt WAVs**

When a split-plan row is reserved, decode it to 24 kHz mono PCM and atomically write `cache/eval_prompts/voice_XX.wav`; write its metadata to `eval_prompts.parquet`. Do not include that row in either phase Parquet.

- [ ] **Step 6: Add cache reconciliation and dynamic worker requests**

`verify_cache` must compare every source-shard checksum, split-plan checksum, row count, token min/max, and Parquet checksum. Define serializable `CacheShardRequest`/`CacheShardResult` so Task 5 can distribute shard requests to persistent per-GPU workers.

- [ ] **Step 7: Run tests and inspect Parquet schema**

Run: `rtk python -m unittest tests.finetune.balalaika.test_cache -v`

Expected: PASS and no audio-bearing training column.

- [ ] **Step 8: Commit**

```bash
rtk git add cosyvoice/finetune/balalaika/cache.py tests/finetune/balalaika/test_cache.py tests/finetune/balalaika/fixtures/shard_000000.tar
rtk git commit -m "feat: add compact speech-token cache builder"
```

### Task 5: CUDA Speech Tokenizer, Reconstruction Pilot, and Approval Gate

**Files:**
- Create: `cosyvoice/finetune/balalaika/tokenizer.py`
- Test: `tests/finetune/balalaika/test_tokenizer.py`

**Interfaces:**
- Consumes: `SpeechTokenizer`, cache requests, `StageStore`, `RunPaths`, and base CosyVoice3 model paths.
- Produces: `OnnxSpeechTokenizer`, `run_tokenizer_qualification(paths: RunPaths) -> dict`, `build_pilot(paths: RunPaths) -> PilotManifest`, `approve_pilot(paths: RunPaths, checksum: str) -> Path`, and `run_cache_workers(paths: RunPaths) -> CacheManifest`.

- [ ] **Step 1: Write failing token shape, retry, and approval tests**

```python
def test_approval_is_bound_to_exact_pilot(self):
    manifest = self.build_fake_pilot()
    with self.assertRaises(PilotApprovalError):
        approve_pilot(self.paths, "0" * 64)
    approval = approve_pilot(self.paths, manifest.manifest_sha256)
    self.assertTrue(approval.exists())

def test_inference_batch_retries_same_items(self):
    session = OomOnceSession()
    backend = OnnxSpeechTokenizer(session=session, max_batch_size=8)
    result = backend.extract(self.audio)
    self.assertEqual(session.item_history, [self.ids, self.ids[:4], self.ids[4:]])
    self.assertEqual(len(result), len(self.audio))
```

- [ ] **Step 2: Run tests and verify missing module**

Run: `rtk python -m unittest tests.finetune.balalaika.test_tokenizer -v`

Expected: FAIL.

- [ ] **Step 3: Implement one CUDA ONNX session per worker**

Bind `CUDAExecutionProvider` to `LOCAL_RANK`, use `speech_tokenizer_v3.batch.onnx`, build 128-bin Whisper log-mel inputs, pad lengths explicitly, and return per-item int32 token arrays. On OOM only, bisect the exact batch until size one; a size-one OOM is fatal. Non-OOM ONNX errors are fatal.

- [ ] **Step 4: Implement tokenizer qualification**

Run fixed audio twice on every GPU, require identical tokens, legal token range, approximately 25 Hz token rate, and matching CPU-side feature lengths. Record providers and peak VRAM. Do not fall back to CPU.

- [ ] **Step 5: Implement duration-stratified pilot and reconstruction**

Select deterministic short/median/long real clips. Save original 24 kHz WAVs, extracted token arrays, token statistics, and reconstructed WAVs generated through frozen base flow/HiFT with the source prompt conditioning. Publish `pilot/index.md` with paired links and a checksum-bound manifest, then return a dedicated `PilotReviewRequired` exit status.

- [ ] **Step 6: Implement explicit approval and persistent GPU workers**

Approval requires `--approve-pilot-sha256 <64-hex>` equal to the current pilot manifest. `run_cache_workers` starts exactly eight spawned processes, assigns each a fixed CUDA device, leases missing shards from a multiprocessing queue, validates published results, and terminates all workers on the first fatal error.

- [ ] **Step 7: Run tests**

Run: `rtk python -m unittest tests.finetune.balalaika.test_tokenizer -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
rtk git add cosyvoice/finetune/balalaika/tokenizer.py tests/finetune/balalaika/test_tokenizer.py
rtk git commit -m "feat: add tokenization pilot and approval gate"
```

### Task 6: Stable Embedding Access and Broad PEFT LoRA Injection

**Files:**
- Modify: `cosyvoice/llm/llm.py:226`
- Modify: `cosyvoice/llm/llm.py:365`
- Create: `cosyvoice/finetune/balalaika/model.py`
- Test: `tests/finetune/balalaika/test_model.py`

**Interfaces:**
- Consumes: base model paths/config and LoRA defaults from Task 1.
- Produces: `Qwen2Encoder.get_input_embeddings()`, `load_base_llm(model_dir: Path) -> CosyVoice3LM`, `inject_lora(model: CosyVoice3LM, settings: LoraSettings) -> AdapterModel`, `audit_trainable_parameters(model: nn.Module) -> TrainableAudit`, `save_adapter(model: nn.Module, path: Path) -> AdapterManifest`, and `merge_adapter(base_dir: Path, adapter_dir: Path, output: Path) -> MergeReport`.

- [ ] **Step 1: Write failing embedding-access and trainable-inventory tests**

```python
def test_active_modules_receive_lora_and_qwen_head_does_not(self):
    model = tiny_cosyvoice3_llm()
    adapted = inject_lora(model, LoraSettings())
    audit = audit_trainable_parameters(adapted)
    self.assertIn("llm.model.model.layers.0.self_attn.q_proj", audit.target_modules)
    self.assertIn("llm.model.model.embed_tokens", audit.target_modules)
    self.assertIn("speech_embedding", audit.target_modules)
    self.assertIn("llm_decoder", audit.target_modules)
    self.assertNotIn("llm.model.lm_head", audit.target_modules)
    self.assertEqual(audit.unexpected_dense_parameters, ())
```

Add a no-op initialization test that logits match before/after injection, and a save/load/merge test that merged logits match adapter-active logits within BF16 tolerance.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk python -m unittest tests.finetune.balalaika.test_model -v`

Expected: FAIL.

- [ ] **Step 3: Replace hard-coded nested embedding access**

Add `Qwen2Encoder.get_input_embeddings()` delegating to `self.model.get_input_embeddings()`. Replace each direct nested embedding call with `self.llm.get_input_embeddings()(text_token)` or `self.llm.get_input_embeddings()(instruct_token)` as appropriate. Preserve state-dict keys and add a regression test for unadapted inference.

- [ ] **Step 4: Implement explicit active-module discovery**

Traverse `named_modules()`. Select every `nn.Linear` and `nn.Embedding` under the active CosyVoice3 LLM, explicitly exclude names ending in `llm.model.lm_head`, and require `speech_embedding`, `llm_decoder`, and Qwen input embeddings. Pass the exact names to PEFT in-place adapter injection so CosyVoice's outer module API remains unchanged.

- [ ] **Step 5: Implement adapter-only serialization and merge**

Save adapter tensors with `safetensors`, a JSON target inventory, base checkpoint checksum, PEFT settings, and code revision. Merge into a freshly loaded base, remove adapter wrappers, restore original state-dict names, and atomically write a complete `llm.pt`.

- [ ] **Step 6: Run tests and existing CosyVoice tests**

Run: `rtk python -m unittest tests.finetune.balalaika.test_model tests.test_generate_homographs -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add cosyvoice/llm/llm.py cosyvoice/finetune/balalaika/model.py tests/finetune/balalaika/test_model.py
rtk git commit -m "feat: add broad CosyVoice3 LoRA adaptation"
```

### Task 7: Memory-Mapped Cached Dataset and Deterministic Dynamic Batching

**Files:**
- Create: `cosyvoice/finetune/balalaika/data.py`
- Test: `tests/finetune/balalaika/test_data.py`

**Interfaces:**
- Consumes: verified phase Parquets from Task 4 and CosyVoice3 tokenizer from the base model.
- Produces: `CachedSpeechDataset`, `TokenBatchSampler`, `CosyVoice3Collator`, `build_phase_dataloader(cache: CacheManifest, phase: PhaseSpec, accelerator: Accelerator, epoch: int) -> DataLoader`, and `build_memorization_dataloader(rows: Sequence[CachedRow]) -> DataLoader`.

- [ ] **Step 1: Write failing no-audio and deterministic-batch tests**

```python
def test_collator_emits_only_llm_fields(self):
    batch = self.collator([self.row_a, self.row_b])
    self.assertEqual(
        set(batch),
        {"utts", "text", "text_token", "text_token_len", "instruct_token", "instruct_token_len", "speech_token", "speech_token_len"},
    )

def test_batches_are_rank_disjoint_and_epoch_complete(self):
    rank_batches = [list(self.make_sampler(rank)) for rank in range(8)]
    flattened = [index for batches in rank_batches for batch in batches for index in batch]
    self.assertEqual(sorted(flattened), list(range(len(self.dataset))))
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `rtk python -m unittest tests.finetune.balalaika.test_data -v`

Expected: FAIL.

- [ ] **Step 3: Implement memory-mapped Arrow dataset**

Open verified phase Parquets through `pyarrow.dataset`, retain row-group offsets, and materialize only requested rows. Validate text/instruction/speech-token length on access. Do not import torchaudio, Whisper, CampPlus, or ONNX Runtime in this module.

- [ ] **Step 4: Implement deterministic length-window batching**

Shuffle row indices from `seed + epoch`, form bounded windows, sort each window by `text_token_len + speech_token_len`, and pack until `max_tokens_per_gpu`. Partition complete batches across eight ranks without duplicating samples; pad the number of optimizer steps with explicit empty synchronization batches only if required by Accelerate.

- [ ] **Step 5: Implement CosyVoice3-only collator**

Tokenize `text` and `instruct` with `allowed_special="all"`, pad token tensors, and emit no acoustic feature or speaker-embedding fields. Assert the end-of-prompt token is present in every instruction.

- [ ] **Step 6: Run tests**

Run: `rtk python -m unittest tests.finetune.balalaika.test_data -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add cosyvoice/finetune/balalaika/data.py tests/finetune/balalaika/test_data.py
rtk git commit -m "feat: add cached CosyVoice3 training batches"
```

### Task 8: Accelerate Training Loop, Global Progress, and Resume

**Files:**
- Create: `cosyvoice/finetune/balalaika/training.py`
- Test: `tests/finetune/balalaika/test_training.py`
- Test: `tests/finetune/balalaika/accelerate_smoke.py`

**Interfaces:**
- Consumes: adapted model from Task 6, dataloaders from Task 7, artifact state from Task 1, and `PhaseSpec`.
- Produces: `ProgressState`, `FractionBoundary`, `TrainingCallbacks`, `qualify_token_limit(request: QualificationRequest) -> BatchQualification`, and `train_phase(request: TrainRequest, callbacks: TrainingCallbacks) -> PhaseResult`.

- [ ] **Step 1: Write failing boundary and resume tests**

```python
def test_fraction_boundaries_are_exact(self):
    boundaries = FractionBoundary.for_epoch(eligible_samples=80)
    self.assertEqual([item.sample_target for item in boundaries], [10, 20, 30, 40, 50, 60, 70, 80])

def test_resume_does_not_repeat_optimizer_steps(self):
    first = run_tiny_training(stop_after_boundary=3)
    resumed = run_tiny_training(resume=first.checkpoint)
    self.assertEqual(resumed.seen_sample_ids, self.all_ids)
    self.assertEqual(len(resumed.validation_boundaries), 8)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `rtk python -m unittest tests.finetune.balalaika.test_training -v`

Expected: FAIL.

- [ ] **Step 3: Implement Accelerate initialization and optimizer scope**

Create `Accelerator(mixed_precision="bf16", gradient_accumulation_steps=request.accumulation_steps, log_with="wandb")`. Build AdamW from parameters with `requires_grad=True` only, use `accelerator.prepare`, `accelerator.accumulate`, `accelerator.backward`, and `accelerator.clip_grad_norm_`. Assert optimizer parameter IDs equal the audited adapter parameter IDs.

- [ ] **Step 4: Implement global sample accounting and fractional callbacks**

All-reduce each batch's real sample count. Trigger a boundary only after an accumulation boundary, barrier all ranks, save state, call `callbacks.validate`, record success, and resume. Carry over any sample overshoot to the next target; never generate duplicate boundary events.

- [ ] **Step 5: Implement complete Accelerate resume state**

Register `ProgressState` for checkpointing. Save through `accelerator.save_state(temp_dir)`, write the adapter and manifest, then atomically rename. Resume with `accelerator.load_state` and `accelerator.skip_first_batches` using the recorded epoch batch offset. Refuse changed cache, phase, base checksum, LoRA config, token limit, accumulation, or world size.

- [ ] **Step 6: Implement bounded memory qualification**

Try per-GPU token limits `[2000, 3000, 4000, 5000, 6000]` on representative long batches, recording peak allocation. Select the largest candidate with at least 2 GiB free headroom on every rank. A production OOM raises `TrainingCapacityError` and does not adjust the limit automatically.

- [ ] **Step 7: Run CPU single-process tests and two-process smoke**

Run: `rtk python -m unittest tests.finetune.balalaika.test_training -v`

Run: `rtk accelerate launch --cpu --num_processes 2 -m tests.finetune.balalaika.accelerate_smoke`

Expected: PASS with eight boundary records in the smoke artifact.

- [ ] **Step 8: Commit**

```bash
rtk git add cosyvoice/finetune/balalaika/training.py tests/finetune/balalaika/test_training.py tests/finetune/balalaika/accelerate_smoke.py
rtk git commit -m "feat: add resumable Accelerate phase trainer"
```

### Task 9: Four-Sample 100%-Accuracy Memorization Gate

**Files:**
- Modify: `cosyvoice/llm/llm.py:400`
- Create: `cosyvoice/finetune/balalaika/memorization.py`
- Test: `tests/finetune/balalaika/test_memorization.py`

**Interfaces:**
- Consumes: production adapter/model path from Task 6, memorization dataloader from Task 7, and trainer primitives from Task 8.
- Produces: `CosyVoice3LM.forward` result keys `correct_tokens_per_sample` and `target_tokens_per_sample`, `select_memorization_rows(split_plan: Path, cache: CacheManifest, seed: int = 1986) -> tuple[CachedRow, CachedRow, CachedRow, CachedRow]`, and `run_memorization_gate(request: MemorizationRequest) -> MemorizationReport`.

- [ ] **Step 1: Write failing per-sample gate tests**

```python
def test_aggregate_100_percent_cannot_hide_sample_failure(self):
    checks = [SampleAccuracy("a", 10, 10), SampleAccuracy("b", 9, 10)]
    self.assertFalse(memorization_passed([checks, checks, checks]))

def test_requires_three_consecutive_exact_checks(self):
    exact = [SampleAccuracy(key, 10, 10) for key in "abcd"]
    almost = [SampleAccuracy("a", 9, 10), *exact[1:]]
    self.assertFalse(memorization_passed([exact, almost, exact]))
    self.assertTrue(memorization_passed([exact, exact, exact]))
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `rtk python -m unittest tests.finetune.balalaika.test_memorization -v`

Expected: FAIL.

- [ ] **Step 3: Add per-sample teacher-forced counts to CosyVoice3 forward**

Compute argmax only at non-`IGNORE_ID` targets and return int64 correct/total tensors per sample alongside `loss` and aggregate `acc`. Preserve existing callers by only adding keys.

- [ ] **Step 4: Implement deterministic four-row selection**

Choose two valid rows below 0.95 and two at or above 0.95 using seed-bound hash ordering, with one short and one long row per phase. Record identities and cache checksums. The rows remain eligible for later production training.

- [ ] **Step 5: Implement isolated production-path overfit**

Start a fresh base and production LoRA adapter, repeat only the four rows, and use the same Accelerate forward/backward/optimizer path for at most 2,000 steps. Evaluate all four without dropout every ten steps. Pass after three consecutive evaluations where `correct == total` for each sample. Save cross-entropy, predictions, targets, generated WAVs, and trainable inventory under `memorization/`; never publish its adapter as a phase input.

- [ ] **Step 6: Run tests**

Run: `rtk python -m unittest tests.finetune.balalaika.test_memorization tests.finetune.balalaika.test_model -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add cosyvoice/llm/llm.py cosyvoice/finetune/balalaika/memorization.py tests/finetune/balalaika/test_memorization.py
rtk git commit -m "feat: add four-sample memorization gate"
```

### Task 10: Distributed Synthesis, GigaAM Validation, and W&B

**Files:**
- Create: `cosyvoice/finetune/balalaika/evaluation.py`
- Test: `tests/finetune/balalaika/test_evaluation.py`

**Interfaces:**
- Consumes: 2,000 benchmark rows/spans from Task 3, 20 prompt WAVs from Task 4, current adapter/model from Task 6, metrics from Task 3, and `Accelerator` from Task 8.
- Produces: `build_voice_assignment(rows, prompts) -> list[EvaluationItem]`, `GigaAmRecognizer`, `evaluate_checkpoint(request: EvaluationRequest) -> EvaluationReport`, and `WandbValidationLogger.log(report: EvaluationReport, validation_index: int) -> None`.

- [ ] **Step 1: Write failing assignment, exact-count, and durability tests**

```python
def test_each_benchmark_row_is_generated_once(self):
    items = build_voice_assignment(self.rows_2000, self.prompts_20)
    self.assertEqual(len(items), 2000)
    self.assertEqual(len({item.benchmark_id for item in items}), 2000)
    self.assertEqual(Counter(item.voice_id for item in items), {f"voice_{i:02d}": 100 for i in range(20)})

def test_results_publish_before_wandb_failure(self):
    with self.assertRaises(WandbSyncError):
        evaluate_checkpoint(self.request_with_failing_wandb)
    self.assertTrue(self.request.output_jsonl.exists())
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `rtk python -m unittest tests.finetune.balalaika.test_evaluation -v`

Expected: FAIL.

- [ ] **Step 3: Implement stable 20-voice round-robin assignment**

Sort benchmark rows by integer `id` and prompt clips by `voice_id`; assign `rows[index]` to `prompts[index % 20]`. Persist the mapping checksum and reuse it for all 41 validation points.

- [ ] **Step 4: Implement distributed generation and ASR**

Shard exactly 2,000 items with `accelerator.split_between_processes`, synthesize stressed input with current LoRA plus frozen flow/HiFT, and write temporary 24 kHz WAVs. Load `gigaam-v3-rnnt` with local-rank CUDA provider, transcribe in bounded batches, retry identical ASR items on OOM, and gather row records. Refuse missing/duplicate IDs or anything other than 250 local items at world size eight.

- [ ] **Step 5: Implement local results and listening panel**

Atomically write one JSONL row per benchmark item with inputs, references, hypotheses, projected number spans, and edit counts. Write summary JSON and copy one fixed item per voice into a 20-audio panel. Delete other temporary WAVs only after successful publication unless `KEEP_EVAL_AUDIO=1`.

- [ ] **Step 6: Implement one resumable W&B run**

Use the W&B tracker already initialized by Task 8's `Accelerator`; do not create a second run. Persist its run ID after baseline and require `resume="must"` on later launches. Log scalar values through `accelerator.log` and use the tracker's underlying run only for the worst-error table and 20 panel audio files. If online logging fails, preserve a syncable local W&B directory and raise before training resumes.

- [ ] **Step 7: Run tests**

Run: `rtk python -m unittest tests.finetune.balalaika.test_evaluation tests.finetune.balalaika.test_metrics -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
rtk git add cosyvoice/finetune/balalaika/evaluation.py tests/finetune/balalaika/test_evaluation.py
rtk git commit -m "feat: add distributed hard-number evaluation"
```

### Task 11: Final Adapter Merge and Strict Inference Verification

**Files:**
- Modify: `cosyvoice/finetune/balalaika/model.py`
- Test: `tests/finetune/balalaika/test_merge.py`

**Interfaces:**
- Consumes: `merge_adapter` foundation from Task 6, final phase-2 adapter, base CosyVoice3 model directory, and GigaAM recognizer from Task 10.
- Produces: `export_final_llm(request: ExportRequest) -> FinalModelManifest` and `strict_verify_final_model(request: VerifyRequest) -> VerificationReport`.

- [ ] **Step 1: Write failing strict-load and key-layout tests**

```python
def test_merged_checkpoint_has_original_keys_only(self):
    manifest = export_final_llm(self.export_request)
    state = torch.load(manifest.llm_path, map_location="cpu", weights_only=True)
    self.assertEqual(set(state), set(self.base_state))
    self.assertFalse(any("lora_" in key for key in state))

def test_strict_loader_accepts_merged_checkpoint(self):
    report = strict_verify_final_model(self.verify_request)
    self.assertTrue(report.strict_load)
    self.assertEqual(report.smoke_utterances, 4)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `rtk python -m unittest tests.finetune.balalaika.test_merge -v`

Expected: FAIL.

- [ ] **Step 3: Implement fresh-base merge and atomic `llm.pt`**

Refuse any adapter whose base checksum, target inventory, rank/alpha, or phase lineage differs. Load a fresh base on CPU, inject the identical adapter structure, load adapter tensors strictly, merge/unload, compare original state-dict keys, and save `{parameter_name: tensor}` without training metadata.

- [ ] **Step 4: Implement normal-path strict verification**

Copy the base model directory's YAML/tokenizer/flow/HiFT references into a temporary verification view that substitutes only the merged LLM path. Instantiate normal `CosyVoice3`, require `strict=True` load, synthesize four fixed Russian prompts with four reserved voices, and transcribe them with GigaAM. Publish audio, ASR, checksums, and library identities.

- [ ] **Step 5: Run tests**

Run: `rtk python -m unittest tests.finetune.balalaika.test_merge tests.finetune.balalaika.test_model -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add cosyvoice/finetune/balalaika/model.py tests/finetune/balalaika/test_merge.py
rtk git commit -m "feat: export standalone CosyVoice3 llm checkpoint"
```

### Task 12: Phase State Machine, Internal CLI, and Exactly Two Launchers

**Files:**
- Create: `cosyvoice/finetune/balalaika/workflow.py`
- Create: `examples/balalaika/cosyvoice3_lora/run_phase1.sh`
- Create: `examples/balalaika/cosyvoice3_lora/run_phase2.sh`
- Test: `tests/finetune/balalaika/test_workflow.py`

**Interfaces:**
- Consumes: all Tasks 1–11.
- Produces: `main(argv: Sequence[str] | None = None) -> int`, `run_phase1(args: Namespace) -> int`, and `run_phase2(args: Namespace) -> int`.

- [ ] **Step 1: Write failing gate-order and launcher-count tests**

```python
def test_phase1_stops_for_manual_pilot_review(self):
    result = run_phase1(self.args_without_approval)
    self.assertEqual(result, ExitCode.PILOT_REVIEW_REQUIRED)
    self.assertFalse(self.paths.stage("memorization_complete").exists())

def test_phase2_refuses_incomplete_phase1(self):
    with self.assertRaises(StageRequirementError):
        run_phase2(self.phase2_args)

def test_recipe_has_exactly_two_shell_launchers(self):
    launchers = sorted(self.recipe.glob("*.sh"))
    self.assertEqual([path.name for path in launchers], ["run_phase1.sh", "run_phase2.sh"])
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `rtk python -m unittest tests.finetune.balalaika.test_workflow -v`

Expected: FAIL.

- [ ] **Step 3: Implement internal CLI arguments and secure defaults**

Support `phase1`, `phase2`, and `status` internal subcommands; both shell scripts translate operator flags to the appropriate subcommand. Expose dataset/run/model paths, pilot approval checksum, batch limit, accumulation, W&B project/name, retention, and resume checkpoint. Redact environment variables whose names contain `TOKEN`, `KEY`, or `SECRET` from printed config.

- [ ] **Step 4: Implement exact phase-1 gate sequence**

Run environment/input/benchmark preflight; create or validate pilot; stop without approval; validate checksum approval; run memorization; build/verify cache; run eight-GPU smoke; evaluate untouched base at validation index 0; train two low-agreement epochs with indices 1–16; publish `phase1_complete` only after validation 16 succeeds.

- [ ] **Step 5: Implement exact phase-2 gate sequence**

Require phase-1 stage and adapter checksum; start fresh optimizer at `5e-5`; train three high-agreement epochs with validation indices 17–40; merge/export/strict-verify; publish `complete`. Never call a Hub upload API.

- [ ] **Step 6: Implement robust shell launchers**

Both scripts use `set -euo pipefail`, resolve repository-relative paths without repurposing `HOME`, require eight visible devices, export `PYTHONPATH`, and call the pinned Accelerate config. `run_phase1.sh --status` and `run_phase2.sh --status` call the same read-only internal status command.

- [ ] **Step 7: Run tests and shell syntax checks**

Run: `rtk python -m unittest tests.finetune.balalaika.test_workflow -v`

Run: `rtk bash -n examples/balalaika/cosyvoice3_lora/run_phase1.sh`

Run: `rtk bash -n examples/balalaika/cosyvoice3_lora/run_phase2.sh`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
rtk git add cosyvoice/finetune/balalaika/workflow.py examples/balalaika/cosyvoice3_lora/run_phase1.sh examples/balalaika/cosyvoice3_lora/run_phase2.sh tests/finetune/balalaika/test_workflow.py
rtk git commit -m "feat: add two-phase CosyVoice3 launch workflow"
```

### Task 13: End-to-End Fixtures, Documentation, and Repository Verification

**Files:**
- Create: `tests/finetune/balalaika/test_integration.py`
- Create: `examples/balalaika/cosyvoice3_lora/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: complete workflow from Task 12.
- Produces: documented operator contract and end-to-end verification evidence.

- [ ] **Step 1: Write the failing tiny end-to-end test**

```python
def test_tiny_workflow_reaches_verified_export(self):
    result = run_tiny_workflow(
        source_shards=2,
        phase1_rows=8,
        phase2_rows=8,
        reserved_prompts=2,
        benchmark_rows=8,
    )
    self.assertEqual(result.stage, "complete")
    self.assertTrue(result.final_llm.exists())
    self.assertEqual(result.validation_indices, list(range(41)))
    self.assertEqual(result.upload_calls, 0)
```

The fixture scales counts while preserving the 2/3 epochs and eight boundaries per epoch; fake synthesis/ASR make the test CPU-only.

- [ ] **Step 2: Run the integration test and verify missing docs/fixture behavior**

Run: `rtk python -m unittest tests.finetune.balalaika.test_integration -v`

Expected: FAIL until the fixture harness and final status assertions are wired.

- [ ] **Step 3: Complete the tiny workflow harness**

Use real join/cache/data/model/training state interfaces with fake tokenizer, tiny model, fake synthesis, fake ASR, and offline W&B logger. Exercise pilot review exit, checksum approval, memorization pass, interrupted cache resume, interrupted phase resume, 41 validation indices, merge, and strict tiny-model load.

- [ ] **Step 4: Write the complete recipe README**

Document exact install commands for the current CUDA-12.8 environment, secure token exports using non-secret example values, base model download, paths, pilot invocation and listening checklist, approval rerun, memorization evidence, cache status, both phase commands, Accelerate settings, LoRA inventory, W&B series, resume examples, artifacts tree, final `llm.pt` installation, troubleshooting, verification, and deferred upload. State that the user must rotate the token exposed during design.

- [ ] **Step 5: Add a concise root README link**

Add one bullet under training/documentation pointing to `examples/balalaika/cosyvoice3_lora/README.md`; do not alter unrelated upstream sections.

- [ ] **Step 6: Run the full unit/integration suite**

Run: `rtk python -m unittest discover -s tests -p 'test_*.py' -v`

Expected: PASS.

- [ ] **Step 7: Run lint and repository checks**

Run: `rtk flake8 --max-line-length 180 --ignore B006,B008,B905,C408,E402,E731,E741,W503,W504,F401,F403,F405,F722,F841 --exclude ./third_party/,./runtime/python/grpc/cosyvoice_pb2*py cosyvoice/finetune tests/finetune`

Run: `rtk git diff --check`

Run: `rtk bash -n examples/balalaika/cosyvoice3_lora/run_phase1.sh`

Run: `rtk bash -n examples/balalaika/cosyvoice3_lora/run_phase2.sh`

Expected: PASS with no whitespace errors and exactly two recipe shell scripts.

- [ ] **Step 8: Commit**

```bash
rtk git add tests/finetune/balalaika/test_integration.py examples/balalaika/cosyvoice3_lora/README.md README.md
rtk git commit -m "docs: complete CosyVoice3 Balalaika training recipe"
```

### Task 14: Real Hardware Qualification and User Tokenization Review Gate

**Files:**
- No source changes expected; retain evidence under `/workspace/cosyvoice3-balalaika-lora`.

**Interfaces:**
- Consumes: completed implementation and real model/dataset credentials.
- Produces: environment qualification, tokenization pilot, and a user-review pause. Production cache extraction and phase training remain unstarted.

- [ ] **Step 1: Install only the recipe dependency delta and freeze the qualified environment**

Run: `rtk python -m pip install -r examples/balalaika/cosyvoice3_lora/requirements-cu128.txt`

Run: `rtk python -m pip freeze > /workspace/cosyvoice3-balalaika-lora/dependency-lock.txt`

Expected: Torch remains `2.8.0+cu128`; ONNX Runtime reports `CUDAExecutionProvider`; all eight GPUs report capability 12.0.

- [ ] **Step 2: Run non-mutating status/preflight**

Run: `rtk bash examples/balalaika/cosyvoice3_lora/run_phase1.sh --status`

Expected: exact model/input inventories, private benchmark access, W&B credentials, and no secret values in output.

- [ ] **Step 3: Generate only the real tokenization pilot**

Run: `rtk bash examples/balalaika/cosyvoice3_lora/run_phase1.sh`

Expected: exit code/documented status `PILOT_REVIEW_REQUIRED`; original/reconstructed audio and `pilot/index.md` exist; memorization, full cache, baseline, and training stages do not exist.

- [ ] **Step 4: Verify the pause and hand the audio to the user**

Run: `rtk bash examples/balalaika/cosyvoice3_lora/run_phase1.sh --status`

Expected: the status prints the pilot manifest checksum and listening bundle path. Ask the user to listen and explicitly approve or reject. Do not pass `--approve-pilot-sha256` on their behalf.

- [ ] **Step 5: Stop without production training**

Record the implementation commit, clean-worktree status, pilot path, and checksum in the handoff. The next turn, only after user approval, begins with the checksum-bound approval rerun; it then runs memorization before any corpus-wide cache or production phase.

---

## Final Review Checklist

- [ ] Every design-spec requirement maps to a task above.
- [ ] Tests cover the exact 0.95 equality boundary and all 309 null rows.
- [ ] No training Parquet contains audio bytes or speaker/mel features.
- [ ] The tokenizer pilot physically blocks memorization/cache/training until user approval.
- [ ] Memorization requires 100% on each of four samples for three checks.
- [ ] Trainable audit includes all active linear/embedding targets and excludes Qwen `lm_head` plus all dense base weights.
- [ ] Accelerate state resumes without repeated samples or validation indices.
- [ ] The benchmark preflight extracts all 2,000 number spans without guessing.
- [ ] Exactly 2,000 generations occur at each of 41 validation points.
- [ ] Local JSONL metrics publish before W&B and survive tracking failures.
- [ ] Final `llm.pt` contains original keys only and strict-loads normally.
- [ ] Exactly two user-facing shell scripts exist.
- [ ] README documents secure credentials and no upload path is callable.
- [ ] Real execution stops at the manual tokenization review gate.
