# CosyVoice3 Balalaika Two-Phase LoRA Design

Date: 2026-08-01

## Summary

Adapt this repository to supervised-fine-tune the non-RL
`Fun-CosyVoice3-0.5B-2512` LLM for general Russian multi-speaker and zero-shot
voice-cloning use. Training runs on eight RTX 5090 GPUs through Hugging Face
Accelerate and PEFT LoRA. It uses the original Balalaika WebDataset audio and
the punctuated, stress-marked combined ROVER transcripts without duplicating
the 373 GB audio corpus.

The workflow has exactly two user-facing launch scripts. Phase 1 trains for two
epochs on samples whose mean ASR agreement is below 0.95. Phase 2 resumes the
phase-1 adapter and trains for three epochs on samples whose mean ASR agreement
is at least 0.95. The final adapter is merged into a standalone `llm.pt` that
strict-loads through the repository's normal CosyVoice3 inference path.

Before production training, the workflow requires a manually approved audio
tokenization pilot and a four-sample memorization test that reaches 100% target
speech-token accuracy on every sample.

## Goals

- Preserve general multi-speaker and zero-shot voice-cloning behavior while
  adapting the CosyVoice3 LLM to Russian punctuated and stress-marked text.
- Use supervised learning only. Do not introduce RL, GRPO, DPO, preference
  data, or a reference policy model.
- Use all eight local RTX 5090 GPUs through Accelerate.
- Use broad LoRA coverage over the active CosyVoice3 LLM computation path.
- Avoid rereading and retokenizing audio during five production epochs.
- Make preprocessing, training, evaluation, and recovery deterministic and
  auditable.
- Validate intelligibility and hard-number pronunciation through GigaAM v3
  RNN-T and log results to Weights & Biases every one-eighth epoch.
- Produce a final drop-in `llm.pt` plus the unmerged final LoRA adapter.

## Non-goals

- Training the flow model, HiFT vocoder, or speaker encoder.
- Adapting to one fixed speaker.
- Using the corpus's `qwen_training_codes`; those codes use a different codec
  and are not CosyVoice3 speech tokens.
- Copying audio into CosyVoice's existing LibriTTS-style Parquet format.
- Publishing data, caches, adapters, models, W&B artifacts, or Hub repositories.
  Upload is a separate, later operation requiring explicit user authorization.
- Replacing the upstream root README. The root README receives a concise recipe
  link; the new recipe receives complete documentation.

## Verified environment and inputs

### Hardware and runtime

- Eight NVIDIA GeForce RTX 5090 GPUs, each with 32,607 MiB VRAM.
- NVIDIA driver 570.153.02 and CUDA 12.8.
- Installed PyTorch 2.8.0+cu128 with CUDA capability 12.0 support.
- Installed Accelerate 1.12.0, PEFT 0.20.0, W&B 0.28.1,
  Transformers 4.57.3, and PyArrow 25.0.0.
- `onnx-asr` and a CUDA ONNX Runtime must be installed and qualified without
  downgrading the working PyTorch/CUDA 12.8 stack.

The repository's older pinned Torch/CUDA requirements are not authoritative for
this machine. The recipe must document and verify a Blackwell-compatible
environment before any GPU work. Qualification writes an exact dependency lock
and environment manifest; all later launches require those recorded package
versions unless the operator starts a new qualified run root.

### Model

- Base model: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`.
- Use the base/non-RL checkpoint, never the `_RL` checkpoint.
- CosyVoice3's repository training path supports LLM training; flow and HiFT
  remain frozen base-model components.

### Training corpus

- Root: `/workspace/balalaika_proprietary_v2`.
- Audio: exactly 519 archives,
  `train/shard_000000.tar` through `train/shard_000518.tar`.
- Corpus size: 4,075,032 MP3/JSON sample pairs, approximately 373 GB.
- Training text:
  `combined_sidecars/rover-punctuation-stress-v1/rover-punctuation-stress.jsonl`.
- Text field: `rover_punctuated_accented`.
- Join key: `source_relative_path`.
- Agreement source:
  `punctuation_artifacts/20260729T135419Z/balalaika-rover-results-20260729T135419Z.tar.zst`.
- Agreement field: `asr_agreement_mean`.
- The 309 rows with null agreement are excluded and counted explicitly.

### Validation corpus

- Private Hugging Face dataset:
  `bitmanagerai/hard_number_eval_for_tts`, config `default`, split `train`.
- Exactly 2,000 rows and ten columns.
- Required fields: `id`, `category`, `hard_number`, `text`,
  `normalized_gold`, and `stressed`.
- There are 12 validation categories.
- Synthesis input is `stressed`, not raw digit-bearing `text`.
- Metric reference is `normalized_gold`.
- Access uses `HF_TOKEN` from the environment. No token may be persisted.

## User-facing entry points

The recipe exposes exactly two executable shell scripts:

1. `run_phase1.sh`
   - Verify environment, model, corpus, sidecars, and credentials.
   - Build and stop at the audio-tokenization pilot on the first run.
   - After explicit pilot approval, run the four-sample memorization gate.
   - Build or resume the eight-GPU compact token cache.
   - Run the untouched-base benchmark.
   - Train phase 1 for two epochs on agreement `< 0.95`.

2. `run_phase2.sh`
   - Require verified cache and phase-1 completion manifests.
   - Resume the completed phase-1 adapter with a fresh phase-2 optimizer.
   - Train phase 2 for three epochs on agreement `>= 0.95`.
   - Merge the final adapter into the original base LLM.
   - Export, strict-load, synthesize with, and verify the final `llm.pt`.

Both scripts call shared importable Python modules. Those modules are internal
implementation units, not additional operator-facing launch scripts.

## Default paths and configuration

- Dataset root: `/workspace/balalaika_proprietary_v2`.
- Repository root: `/workspace/CosyVoice`.
- Run root: `/workspace/cosyvoice3-balalaika-lora`.
- Base model directory:
  `/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512`.
- Visible devices: `0,1,2,3,4,5,6,7`.
- Random seed: `1986` for split assignment, prompt selection, batching, and
  model training.

Paths, batch limits, learning rates, epochs, W&B names, and retention settings
are command-line or environment overrides. Defaults are printed and stored in
every run manifest before work begins.

## Workflow and gates

The workflow is a strict state machine:

1. Environment and dependency preflight.
2. Immutable-input inventory and checksum verification.
3. Validation-dataset schema and number-span preflight.
4. Audio-tokenization pilot generation.
5. Manual pilot approval.
6. Four-sample memorization test.
7. Corpus-wide compact-cache extraction.
8. Eight-GPU production-path smoke training.
9. Untouched-base W&B validation.
10. Phase-1 training and validation.
11. Phase-2 training and validation.
12. Adapter merge, strict load, synthesis smoke test, and ASR verification.
13. Local completion manifest.

No later state may run if an earlier gate is incomplete. A rerun validates and
reuses completed work rather than assuming its presence is sufficient.

## Tokenization pilot and manual approval

The first phase-1 invocation selects a small, deterministic,
duration-stratified set of real corpus examples. It extracts
`speech_tokenizer_v3.batch.onnx` tokens and creates a listening bundle with:

- source identity and metadata;
- token count, token rate, minimum/maximum token ID, and duration;
- original WAV;
- token-to-wave reconstructed WAV;
- a machine-readable verification report; and
- an HTML or Markdown A/B listening index.

The script stops after atomic publication of this bundle. The assistant asks
the user to listen to the examples. Corpus-wide extraction is permitted only
when the same `run_phase1.sh` is rerun with an explicit approval option bound to
the pilot manifest checksum. Approval for one pilot cannot approve a changed
model, tokenizer, dataset, seed, or pilot result.

## Four-sample memorization gate

After pilot approval and before corpus-wide preprocessing or production
training, select four deterministic examples: two from each agreement phase,
with varied durations and valid stress-marked text. Tokenize only those four and
run the exact production Accelerate, PEFT, batching, loss, and checkpoint path.

The temporary adapter trains for at most 2,000 optimizer steps. It passes only
after every one of the four examples achieves 100% teacher-forced target
speech-token accuracy for three consecutive checks. Aggregate accuracy is not
sufficient. Cross-entropy, predictions, targets, generated audio, step counts,
and trainable-parameter inventory are retained as diagnostic evidence.

The test checks token classification, not waveform MSE. CosyVoice3 trains its
LLM with cross-entropy over discrete speech-token targets; flow/vocoder output
is not sample-aligned with the source waveform. The temporary adapter is
isolated from and never loaded by either production phase. Failure blocks all
later work. After the temporary adapter is discarded, the four samples remain
eligible for their normal production phase; the gate tests the training path
and does not create an additional corpus holdout.

## Compact cache design

### Why cache speech tokens

The current LibriTTS recipe can extract features online or store audio bytes in
Parquet. Online extraction would repeat speech-token inference during all five
epochs and compete with training for GPU resources. Audio-bearing Parquet would
duplicate approximately 373 GB. The chosen design reads each MP3 once and
stores only compact model inputs.

### Join and split

For every sample:

1. Require an exact unique `source_relative_path` in the source tar metadata,
   combined transcript sidecar, and canonical ROVER result.
2. Set `text` to `rover_punctuated_accented`.
3. Set `instruct` to
   `You are a helpful assistant.<|endofprompt|>`.
4. Exclude null `asr_agreement_mean`.
5. Assign `< 0.95` to phase 1 and `>= 0.95` to phase 2.
6. Before finalizing phase 2, reserve 20 eligible high-agreement clips using
   seed 1986 and remove them from training.

The source lacks reliable speaker identities, so these are 20 reproducible
voice-cloning prompt clips, not a claim of 20 unique people. Their identities,
audio, text, agreement, duration, and selection evidence are stored.

### Cache layout

The default layout under the run root is:

```text
cache/
  manifest.json
  audit.json
  phase1/shard_NNNNNN.parquet
  phase2/shard_NNNNNN.parquet
  eval_prompts.parquet
  eval_prompts/voice_00.wav ... voice_19.wav
  rejected_rows.jsonl
```

Each phase row contains only source identity, text, instruction, agreement,
speech-token IDs, speech-token length, and required audit metadata. Audio bytes,
speaker embeddings, mel features, and the source JSON payload are not copied.
`eval_prompts.parquet` stores the 20 prompt identities, texts, and selection
evidence, while `eval_prompts/` stores only their 20 decoded prompt WAVs. This
small explicit exception avoids repeatedly scanning 373 GB of source tars for
validation and does not duplicate training audio.

The builder uses eight persistent GPU workers and a dynamic shard queue.
Workers batch MP3 decoding, resampling, Whisper feature construction, and
CosyVoice3 ONNX token extraction. A shard is first written to a temporary path,
then checked for identities, phase assignment, token range, token length, row
count, and checksum, and finally published with an atomic rename.

Model duration and text-token limits are applied explicitly. Every intentional
exclusion is listed by identity and reason. Decode, ONNX, join, missing-row,
duplicate-row, and unexpected-token errors are fatal rather than silently
filtered. Final phase counts are computed from the canonical inputs and stored
in the audit; their union plus null rows, reserved prompts, and documented model
limit exclusions must reconcile to all 4,075,032 source samples.

## Training data path

Training reads the metadata-only phase Parquets. It tokenizes text and
instructions, checks cached speech tokens, shuffles deterministically, sorts
within bounded windows by combined sequence length, forms dynamic batches, and
pads only fields consumed by CosyVoice3LM.

The LLM-only path must not decode audio, compute mel features, run CampPlus,
compute speaker embeddings, or invoke the speech tokenizer. Existing processor
functions may be reused where they match this contract; focused new processors
must keep cached-LLM training independent from the audio-bearing LibriTTS path.

## Accelerate training runtime

Both scripts invoke the same trainer with:

```bash
accelerate launch --multi_gpu --num_processes 8 ...
```

Accelerate owns device placement, distributed preparation, BF16 autocast,
gradient accumulation, backward synchronization, gradient clipping, metric
gathering, and checkpoint state. A startup report records the distributed type,
rank/device map, precision, trainable parameters, effective batch settings, and
library versions.

Dynamic batches vary in token count. Loss accounting and accumulation must use
the repository's length-normalized token loss consistently across workers.
Before production, a short qualification sweep chooses the largest safe
per-GPU token limit from a bounded candidate list. The selected value and peak
VRAM are stored; all production workers use the same value. Gradient
accumulation remains configurable and is recorded as part of checkpoint
identity.

## LoRA scope and defaults

Use PEFT LoRA over every compatible module on the active CosyVoice3 LLM loss
path:

- every active Qwen2 linear projection;
- Qwen2 `embed_tokens`;
- CosyVoice3 `speech_embedding`; and
- CosyVoice3 `llm_decoder`, the external speech-token output head.

Exclude the internal Qwen `lm_head`. CosyVoice's Qwen wrapper consumes only the
last hidden state, and the training loss is computed through the separate
CosyVoice3 `llm_decoder`; the internal Qwen logits do not contribute to loss.

Defaults:

- rank `r = 64`;
- alpha `128`;
- dropout `0.05`;
- bias mode `none`;
- no dense modules-to-save; and
- all non-adapter base parameters frozen.

Startup enumerates matched modules and trainable parameters. It fails if any
required module is absent, the internal Qwen head is trainable, an unexpected
dense parameter is trainable, or the matched inventory differs on any rank.

## Phase schedules

### Phase 1

- Data: non-null agreement `< 0.95`.
- Epochs: 2.
- Optimizer: AdamW over trainable adapter parameters only.
- Learning rate: `1e-4`.
- Precision: BF16.
- Start: fresh no-op-initialized LoRA on the base non-RL checkpoint.

### Phase 2

- Data: non-null agreement `>= 0.95`, excluding the 20 reserved prompts.
- Epochs: 3.
- Optimizer: fresh AdamW over the resumed adapter parameters.
- Learning rate: `5e-5`.
- Precision: BF16.
- Start: verified completed phase-1 adapter.

Within a phase, resume restores the Accelerate state, optimizer, scheduler, RNG,
dataloader position, global sample count, and W&B identity. At the phase
boundary, only adapter weights and lineage are carried forward; optimizer and
scheduler reset intentionally.

## One-eighth-epoch checkpoints

Progress uses the globally processed sample count, not an approximate local
batch count. Each epoch defines synchronized boundaries at 12.5%, 25%, 37.5%,
50%, 62.5%, 75%, 87.5%, and 100% of its exact eligible sample count.

At each boundary:

1. finish the current optimizer accumulation boundary;
2. synchronize ranks;
3. atomically save a complete Accelerate resume checkpoint and adapter export;
4. run the 2,000-generation validation;
5. persist and log metrics; and
6. resume training only after validation succeeds.

This yields 16 phase-1 validations and 24 phase-2 validations. Together with
the untouched-base baseline, the experiment contains 41 validation points.

## Validation generation

The 2,000 validation rows are assigned once, round-robin, across the fixed 20
reserved prompt clips. Each validation checkpoint therefore produces exactly
2,000 utterances, not a 40,000-item cross-product. The mapping is stable across
the base model and every checkpoint.

At a validation boundary, all eight workers switch to distributed inference.
Each worker handles 250 rows, applies the current adapter without requiring a
full merge, and uses the frozen base flow and vocoder. Generated audio is
transcribed locally in batches with:

```python
onnx_asr.load_model("gigaam-v3-rnnt")
```

The CUDA execution provider is bound to the worker's local GPU. A qualification
run verifies model loading, batch recognition, deterministic output for fixed
audio, and compatibility with the CUDA 12.8 environment.

## Metric definitions

Reference and ASR text receive the same normalization before scoring:

- Unicode NFC;
- lowercase;
- `ё` mapped to `е`;
- punctuation removed;
- whitespace collapsed; and
- no transliteration or fuzzy lexical correction.

Primary metrics use corpus-level micro aggregation:

- `utt-WER`: summed word edit distance divided by summed reference words over
  full utterances;
- `utt-CER`: summed character edit distance divided by summed reference
  characters over full utterances;
- `num-WER`: summed word edit distance divided by summed reference words over
  the spoken hard-number spans; and
- `num-CER`: summed character edit distance divided by summed reference
  characters over the spoken hard-number spans.

Number-span extraction is validated for all 2,000 rows before training. Locate
`hard_number` in raw `text`, use the unchanged sentence prefix and suffix to
anchor the corresponding spoken span in `normalized_gold`, and reject ambiguous
or missing anchors. During scoring, align the full normalized reference and ASR
hypothesis, then project the gold number-span boundaries into the hypothesis.
No hand correction or category-specific metric shortcut is allowed.

Also log macro per-row averages and metrics for each of the 12 `category`
values, but keep the four micro metrics above as the primary model-selection
series.

## W&B design

Baseline, phase 1, and phase 2 form one resumable W&B run whose ID is stored in
the run manifest. Global validation index increases monotonically from 0 through
40. Training logs include loss, token accuracy, learning rate, gradient norm,
throughput, global samples, token counts, peak GPU memory, and timing.

Validation logs include the four primary metrics, macro diagnostics,
per-category metrics, ASR/generation latency, a fixed 20-audio listening panel,
and a worst-error table. Do not upload all 2,000 audio files to W&B.

Every validation first writes a local result JSONL containing benchmark ID,
voice-prompt identity, stressed input, normalized reference, ASR hypothesis,
reference and hypothesis number spans, and edit counts. Local files are the
audit source of truth. Missing W&B credentials fail preflight. A transient W&B
failure cannot erase metrics or checkpoints; it preserves a local syncable
record and stops before further optimizer updates unless the operator explicitly
selects documented offline mode.

## Checkpoint, merge, and inference artifacts

Intra-phase checkpoints use Accelerate's state format and are tied to the same
shared trainer implementation. Each checkpoint includes a manifest with phase,
epoch fraction, samples, adapter checksum, optimizer identity, cache checksum,
base-model checksum, RNG state, and validation status.

After phase 2:

1. Load a fresh base `Fun-CosyVoice3-0.5B-2512` LLM.
2. Load and validate the final adapter inventory.
3. Merge all linear and embedding LoRA updates into base tensors.
4. Reassemble the original CosyVoice3 LLM state-dict key layout.
5. Save standalone `llm.pt` atomically.
6. Strict-load it through normal CosyVoice3 initialization.
7. Synthesize fixed Russian smoke prompts with reserved voices.
8. Transcribe them with GigaAM v3 RNN-T and write verification evidence.

The final directory retains the merged `llm.pt`, unmerged adapter, adapter
config, base-model identity, merge report, strict-load report, smoke audio,
smoke ASR results, and checksums. Flow, HiFT, tokenizer, ONNX files, and YAML
remain base-model artifacts and are not duplicated unless required for a
self-contained inference bundle requested later.

## Failure handling and recovery

- Source tars and canonical sidecars are read-only.
- Temporary cache/checkpoint outputs publish only through atomic rename.
- Completed cache shards are checksum-validated on restart.
- Invalid partial outputs may be replaced; valid published artifacts are never
  silently overwritten.
- Phase 2 refuses missing or incomplete phase-1 lineage.
- Join mismatch, missing row, duplicate identity, invalid token, decode failure,
  ONNX failure, non-finite loss/gradient, and metric-span ambiguity are fatal.
- Tokenization and evaluation inference may retry the identical content at a
  smaller inference batch because content and optimization do not change.
- Production training OOM is fatal and reports a smaller token-limit override;
  it does not silently alter effective batching.
- Validation failure stops before the next optimizer step and preserves its
  triggering checkpoint and evidence.
- Temporary validation audio is removed only after local results and the fixed
  listening panel publish successfully. `KEEP_EVAL_AUDIO=1` retains it.

## Security and publication

- `HF_TOKEN` and `WANDB_API_KEY` are environment-only secrets.
- Never store tokens in shell scripts, README examples, configs, manifests,
  logs, process arguments, W&B config, or checkpoints.
- The token shared during design must be rotated by the user because it was
  exposed in conversation.
- Neither launch script performs Hub login, repository creation, upload, push,
  model publication, or W&B artifact publication beyond configured run logging.
- After local computation and verification, wait for explicit user direction
  before designing or executing upload.

## Tests

### Unit tests

- Agreement values below, equal to, and above 0.95.
- Null-agreement exclusion and exact corpus reconciliation.
- Deterministic 20-prompt selection and training exclusion.
- Exact sidecar/archive/tar joins, duplicates, missing rows, and ordering.
- Cache row schema, token ranges, checksums, and rejection accounting.
- Number-span extraction for every validation category and ambiguous anchors.
- Known full-utterance and number-span CER/WER cases.
- Broad LoRA module matching, frozen-parameter audit, and internal Qwen-head
  exclusion.
- Adapter save/load, phase transition, merge, and state-dict key restoration.
- One-eighth progress boundary and resume counters.
- Atomic artifact recovery from interrupted writes.

### Integration tests

- Tiny synthetic tar/sidecar/ROVER fixture through cache construction.
- Cached-token batch through CosyVoice3 forward and loss.
- Four-item memorization harness with a reduced test model.
- Accelerate save/resume with dataloader position and RNG restoration.
- Validation mapping, generation stub, ASR stub, metrics, and W&B payload.
- Final merged checkpoint strict-load test.

### Required real-system qualifications

- Blackwell CUDA/PyTorch import and kernel smoke test on all eight GPUs.
- Real audio tokenization/reconstruction pilot and manual approval.
- Four-real-sample 100%-accuracy memorization gate.
- Short eight-GPU Accelerate production-path smoke run.
- Real `gigaam-v3-rnnt` CUDA batch transcription.
- Final merged-model synthesis and ASR smoke test.

## Documentation

Create a complete recipe README covering:

- purpose, architecture, and limitations;
- exact source datasets and model;
- Blackwell-compatible installation and dependency qualification;
- secure Hugging Face and W&B credentials;
- artifact paths and storage expectations;
- tokenizer pilot generation, listening checklist, and approval command;
- memorization gate and interpretation of token accuracy versus waveform MSE;
- phase-1 and phase-2 commands;
- every supported override and default;
- Accelerate configuration and eight-GPU verification;
- LoRA target inventory and parameter counts;
- W&B metric definitions and expected 41 validation points;
- status, restart, checkpoint, and recovery procedures;
- final merge, `llm.pt` installation, and inference example;
- common failure diagnosis; and
- local completion and deferred-upload policy.

Add a concise root README entry linking to this recipe. Do not rewrite unrelated
upstream documentation.

## Acceptance criteria

The implementation is complete only when all of the following hold:

- Exactly two user-facing launch scripts implement the workflow.
- The pilot stops and cannot be bypassed without checksum-bound user approval.
- The four-sample test reaches 100% per-sample target-token accuracy for three
  consecutive checks before production work.
- Cache audit reconciles every source identity and records every exclusion.
- Production training uses Accelerate on eight GPUs and only intended LoRA
  parameters are trainable.
- Phase 1 uses `< 0.95` for two epochs; phase 2 uses `>= 0.95` for three epochs.
- Twenty prompt clips are fixed, audited, and absent from training.
- Each validation generates exactly 2,000 utterances and records all four
  primary metrics.
- Baseline plus 40 fractional-epoch validations appear locally and in the
  resumable W&B run.
- Interrupted cache, training, and validation work resumes without data loss or
  silent configuration changes.
- The final adapter merges into a standalone `llm.pt` that strict-loads and
  synthesizes through normal CosyVoice3 inference.
- Tests, the complete recipe README, and the root README link pass review.
- Nothing is uploaded without a later explicit authorization.
