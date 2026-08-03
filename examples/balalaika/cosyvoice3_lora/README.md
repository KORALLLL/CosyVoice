# CosyVoice3 Balalaika two-phase SFT LoRA

This recipe fine-tunes the non-RL `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
SFT language model for multi-speaker voice cloning on eight RTX 5090 GPUs. It
uses Accelerate, broad PEFT LoRA, checksum-bound resumable artifacts, and two
separate operator scripts. It never uploads a dataset, cache, adapter, model,
or evaluation artifact.

The workflow is intentionally gated. The first phase invocation creates a
three-clip speech-token reconstruction pilot and exits with status 20. A human
must listen and approve that exact pilot checksum. After approval, a four-audio
memorization test must reach 100% teacher-forced speech-token accuracy for
three consecutive checks before the full corpus cache, baseline evaluation, or
training can begin.

## Fixed experiment contract

| Item | Value |
| --- | --- |
| Base | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (base/SFT, never `_RL`) |
| Data | `/workspace/balalaika_proprietary_v2` |
| Sidecars | `/workspace/balalaika_proprietary_v2/combined_sidecars` |
| Training text | normalized, punctuated, stress-marked `rover_punctuated_accented` |
| Instruction | `You are a helpful assistant.<|endofprompt|>` |
| Phase 1 | non-null `asr_agreement_mean < 0.95`, 2 epochs, `1e-4` |
| Phase 2 | non-null `asr_agreement_mean >= 0.95`, 3 epochs, `5e-5` |
| Validation | baseline plus every 1/8 epoch: indices 0 through 40 |
| Validation size | exactly 2,000 generations per point; 82,000 total |
| Voices | exactly 20 phase-2 clips, deterministically sampled with seed 1986 and excluded from training |
| Benchmark | `bitmanagerai/hard_number_eval_for_tts`: `stressed` input, `normalized_gold` reference |
| ASR | `onnx-asr` GigaAM v3 RNN-T (`onnx_asr.load_model("gigaam-v3-rnnt")`) |
| Metrics | micro `utt-cer`, `utt-wer`, `num-cer`, `num-wer` in W&B |

The canonical reconciliation accounts for every one of the 4,075,032 source
rows: 2,469,768 phase-1 rows, 1,578,206 phase-2 rows, 20 reserved prompt clips,
141 true null-agreement rows, 168 empty transcripts, and 26,729 transcripts
above the model's 200-token limit. Empty and over-limit transcripts remain in
the split audit but are excluded from training and prompt reservation. Agreement
`0.95` belongs to phase 2.
Stress marks and the already normalized sidecar text are preserved; this recipe
does not substitute raw transcripts.

For validation, synthesis always uses `stressed`. The raw digit-bearing `text`
is used only to locate the ordered numeric groups and unchanged sentence
context that bound the spoken number span in `normalized_gold`. The Dataset
Viewer Parquet is downloaded through the immutable commit currently referenced
by `refs/convert/parquet`, then independently bound by its local SHA-256.

## CUDA 12.8 environment

Use a clean Python 3.12 environment on a host with the CUDA 12.8 driver stack
and eight RTX 5090 cards. Install the upstream dependencies first, then upgrade
to the qualified Blackwell versions so the upstream CUDA 12.1 pins do not win.

```bash
cd /workspace/CosyVoice
python3.12 -m venv .venv-cu128
source .venv-cu128/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --upgrade torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade \
  -r examples/balalaika/cosyvoice3_lora/requirements-cu128.txt
```

Do not downgrade Torch after this step. Confirm the qualified versions and
providers before any run. The recipe overrides upstream `openai-whisper`
20231117 with 20250625: the older release caps Triton below 3, while Torch 2.8
requires Triton 3.4. Audio tokenization imports the package's `whisper` module
directly, so preflight records its version and fails before corpus hashing if
it is absent.

```bash
python - <<'PY'
import torch
import onnxruntime as ort

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
print("gpus", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("capabilities", [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])
print("onnxruntime", ort.__version__, ort.get_available_providers())
assert torch.cuda.device_count() == 8
assert all("RTX 5090" in torch.cuda.get_device_name(i) for i in range(8))
assert all(torch.cuda.get_device_capability(i) >= (12, 0) for i in range(8))
assert torch.cuda.is_bf16_supported()
assert "CUDAExecutionProvider" in ort.get_available_providers()
PY
```

ONNX Runtime normally reports the created speech-tokenizer session as
`["CUDAExecutionProvider", "CPUExecutionProvider"]`: CUDA is primary, while
shape/control nodes may use the standard CPU fallback. Qualification rejects a
CPU-only session or any session where CUDA is not first, and runs the real model
twice on every GPU to verify deterministic outputs and expected token lengths.
Under Accelerate this is a true collective: each of the eight ranks qualifies
only its local GPU, then rank 0 validates and publishes the gathered device set.

The Accelerate configuration is fixed at one machine, eight processes, GPU IDs
0-7, and BF16 in [`conf/accelerate.yaml`](conf/accelerate.yaml). Both launchers
also reject anything other than eight unique visible device IDs.

## Credentials and base model

An HF token was exposed during this recipe's design. Revoke/rotate it before
running these commands. Never put real tokens in a command, README, config,
checkpoint, shell history, or W&B metadata. The workflow reads only `HF_TOKEN`
and `WANDB_API_KEY` from its environment and redacts secret-looking status
fields.

Non-secret placeholders look like this:

```bash
export HF_TOKEN='hf_REPLACE_WITH_A_NEW_ROTATED_TOKEN'
export WANDB_API_KEY='REPLACE_WITH_YOUR_WANDB_KEY'
```

For interactive use, avoid shell history:

```bash
read -rsp 'Rotated HF token: ' HF_TOKEN; echo; export HF_TOKEN
read -rsp 'W&B API key: ' WANDB_API_KEY; echo; export WANDB_API_KEY
```

Download the base/non-RL model to the exact default location. Do not download
or rename the `_RL` checkpoint.

```bash
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --exclude llm.rl.pt \
  --local-dir /workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512
test -f /workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512/llm.pt
test ! -e /workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512/llm.rl.pt
```

Verify the canonical inputs are present and keep them read-only:

```bash
test -d /workspace/balalaika_proprietary_v2/combined_sidecars
test -f /workspace/balalaika_proprietary_v2/train/shard_000000.tar
test -f /workspace/balalaika_proprietary_v2/train/shard_000518.tar
```

Defaults place all generated state under
`/workspace/cosyvoice3-balalaika-lora`. Override paths only with the CLI options
shown by `python -m cosyvoice.finetune.balalaika.workflow phase1 --help` or the
equivalent `BALALAIKA_*` environment variables.

## 1. Status and tokenization pilot

Status is read-only and never launches Accelerate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash examples/balalaika/cosyvoice3_lora/run_phase1.sh --status
```

Generate the pilot. The command performs preflight and tokenizer qualification,
then deliberately exits 20 after writing the listening bundle. It must not run
memorization, corpus-wide cache extraction, baseline validation, or training.
A retry still rechecks the immutable source inventory; when a complete split
manifest already matches it, the workflow verifies every recorded shard
checksum and reuses the plan instead of repeating both four-million-row
tokenizer scans. Any changed, missing, extra, partial, or corrupt shard forces a
fresh plan build.

Pilot selection does not stream the 373 GB audio corpus a second time. It
verifies the published split-plan shard checksums, selects the 33 lowest seeded
source IDs from that approximately 1.9 GB plan, verifies only the source tars
containing those IDs, and extracts only those 33 MP3 members. On the reference
corpus, plan selection took about 43 seconds and selected 33 distinct tars; tar
verification and targeted extraction took about 68 seconds.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash examples/balalaika/cosyvoice3_lora/run_phase1.sh
```

Open `/workspace/cosyvoice3-balalaika-lora/pilot/index.md` and compare all three
duration-stratified A/B pairs (`short`, `median`, and `long`). For every pair:

- confirm the original is the intended speaker and contains no clipping;
- confirm the reconstruction preserves identity, words, stress, tempo, pitch,
  and pauses closely enough for voice-cloning training;
- reject metallic noise, dropouts, repeated/missing speech, speaker changes,
  severe timing drift, or obvious bandwidth collapse;
- confirm each WAV is audible, 24 kHz mono PCM, and matched to its label.

Get the exact current checksum and bundle path from the JSON printed by the
first run. If that output was lost, rerun phase 1 without an approval argument;
the authenticated pilot stage is reused and the same review-required JSON is
printed again. Do not approve from memory and do not approve a checksum after
any pilot artifact changes. The separate status command confirms which stage
envelopes are present and checksum-valid.

```bash
bash examples/balalaika/cosyvoice3_lora/run_phase1.sh
```

Only after listening, rerun phase 1 with the exact lowercase checksum:

```bash
PILOT_SHA256='REPLACE_WITH_64_HEX_FROM_REVIEW_OUTPUT'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash examples/balalaika/cosyvoice3_lora/run_phase1.sh \
  --approve-pilot-sha256 "${PILOT_SHA256}" \
  --wandb-project cosyvoice3-balalaika-lora \
  --wandb-name cosyvoice3-balalaika
```

The real-hardware qualification step for this implementation stops after pilot
generation and asks the user to approve the audio. It must not supply
`--approve-pilot-sha256` on the user's behalf.

## 2. Mandatory memorization and phase 1

The approved rerun first builds a compact four-row cache: short and long clips
from each agreement phase. The production LoRA/Accelerate loss path must report
100% teacher-forced speech-token accuracy for every one of the four real audios
at three consecutive checks. Failed or stale evidence blocks the full cache.
The gate is exact token accuracy, not an MSE proxy.

After that gate passes, the same invocation builds/verifies the full compact
Parquet token cache, runs the eight-GPU capacity smoke, evaluates the untouched
base at index 0, and trains phase 1. Phase 1 has 16 validations: eight boundaries
per epoch for two epochs. W&B is mandatory; every point must durably log metrics,
the worst-error table, and the 20-voice listening panel before training proceeds.

The broad LoRA inventory is discovered and audited from the active CosyVoice3
LLM. It includes every active Qwen2 linear projection (attention and MLP), Qwen2
text embeddings, `speech_embedding`, and the outer CosyVoice3 `llm_decoder`.
It explicitly excludes Qwen's internal `llm.model.lm_head`. Flow, HiFT,
CampPlus, all dense base parameters, and every other non-adapter parameter stay
frozen. Defaults are rank 64, alpha 128, dropout 0.05, and bias `none`.

## 3. Phase 2 and local export

Run phase 2 only after `phase1_complete` is sealed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash examples/balalaika/cosyvoice3_lora/run_phase2.sh \
  --wandb-project cosyvoice3-balalaika-lora \
  --wandb-name cosyvoice3-balalaika
```

Phase 2 authenticates and loads only the sealed phase-1 adapter, starts a fresh
optimizer/scheduler at `5e-5`, and emits validation indices 17-40 across three
epochs. Together with baseline 0 and phase-1 indices 1-16, this is exactly 41
points. Each point assigns all 2,000 benchmark rows round-robin over the same 20
reserved voices and computes micro utterance/number CER and WER from GigaAM v3
RNN-T hypotheses.

After validation 40, the workflow merges the adapter into a fresh base model,
strict-loads the standalone state, synthesizes four smoke utterances, checks
them with GigaAM, and seals `/workspace/cosyvoice3-balalaika-lora/final/llm.pt`.
This export remains local and contains no Hub operation.

## Resume, status, and retention

Completed stages are authenticated and skipped when the same command is rerun.
For an interrupted phase, point `--resume-checkpoint` at the exact sealed or
pending checkpoint directory under the corresponding `checkpoints/` tree (the
validation index also appears in training logs):

```bash
bash examples/balalaika/cosyvoice3_lora/run_phase1.sh \
  --approve-pilot-sha256 "${PILOT_SHA256}" \
  --resume-checkpoint /workspace/cosyvoice3-balalaika-lora/phase1/checkpoints/phase-1-validation-07

bash examples/balalaika/cosyvoice3_lora/run_phase2.sh \
  --resume-checkpoint /workspace/cosyvoice3-balalaika-lora/phase2/checkpoints/phase-2-validation-29
```

Resume refuses changed cache/model/LoRA/schedule/sampler identities and
re-authenticates prior local evaluation plus live W&B commits. Phase 2 also
requires the original phase-1 adapter checksum. Do not move or edit a checkpoint.

Use either launcher's identical read-only status command:

```bash
bash examples/balalaika/cosyvoice3_lora/run_phase2.sh --status
```

The default retains the current checkpoint and the two most recent older
checkpoints (`--keep-checkpoints 3`). Evaluation WAV workspaces are deleted
after checksummed results, summaries, W&B media, and listening panels commit;
use `--keep-eval-audio` only when disk capacity permits. Increase retention with
`--keep-checkpoints N`.

## Artifacts

The principal default tree is:

```text
/workspace/cosyvoice3-balalaika-lora/
├── split_plan/                 # immutable joined rows and 0.95 assignments
├── validation_data/            # authenticated hard-number rows/manifest
├── stages/tokenizer_qualification.json  # ONNX/CUDA tokenizer evidence
├── pilot/                      # A/B WAVs, token arrays, index.md
├── memorization_cache/         # exactly four tokenized real audios
├── memorization/               # 3x exact-accuracy evidence and seal
├── cache/
│   ├── phase1/                 # compact low-agreement Parquet
│   ├── phase2/                 # compact high-agreement Parquet
│   └── eval_prompts/           # exactly 20 reserved voice WAVs
├── evaluation/validation-00/ … validation-40/
├── wandb-run.json
├── wandb-validation-commits/
├── phase1/checkpoints/
├── phase2/checkpoints/
├── workflow_stages/            # checksum-bound state machine envelopes
└── final/
    ├── llm.pt                  # merged, adapter-free CosyVoice3 LLM
    ├── adapter/                # retained lineage copy, not needed at runtime
    ├── final_model_manifest.json
    ├── final-success.json
    └── strict-verification/
```

The source tar files and canonical sidecars are read-only inputs. Training
Parquet contains text, instructions, agreement, and speech-token IDs, not audio.

## Install and verify the merged `llm.pt`

Keep the base directory immutable and make a complete runtime copy. Then replace
only its LLM state with the sealed local export:

```bash
FINAL_ROOT=/workspace/cosyvoice3-balalaika-lora/final
INSTALL_ROOT=/workspace/cosyvoice3-balalaika-lora/installed/Fun-CosyVoice3-0.5B-2512
mkdir -p "$(dirname "${INSTALL_ROOT}")"
cp -a /workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512 "${INSTALL_ROOT}"
cp "${FINAL_ROOT}/llm.pt" "${INSTALL_ROOT}/llm.pt"
python - <<'PY'
from pathlib import Path
from cosyvoice.finetune.balalaika.model import load_base_llm, require_committed_final

base = Path('/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512')
final = Path('/workspace/cosyvoice3-balalaika-lora/final')
installed = Path('/workspace/cosyvoice3-balalaika-lora/installed/Fun-CosyVoice3-0.5B-2512')
require_committed_final(final, base, expected_mode='production')
model = load_base_llm(installed)
print(type(model).__name__, 'strict merged llm.pt load OK')
PY
```

`require_committed_final` rechecks the final seal, base/adapter inventories,
merged-logit evidence, strict synthesis evidence, and checksums. Keep the entire
`final/` directory for provenance even when deployment consumes only the copied
model directory.

## Troubleshooting

- **Pilot exits 20:** this is the required listening pause, not a crash. Open
  `pilot/index.md`, listen, then rerun with the printed checksum.
- **Approval checksum mismatch:** do not bypass it. Rerun phase 1 without an
  approval argument, inspect the current pilot again, and approve only the
  checksum printed in that review-required response.
- **Preflight rejects GPUs/BF16:** confirm exactly eight RTX 5090 devices are
  visible, capability is at least 12.0, and no scheduler/container remaps them.
- **NCCL timeout during preflight or cache preparation:** rank zero may spend
  hours checksumming the roughly 373 GB corpus while the other ranks wait at a
  collective. The workflow configures a 24-hour process-group timeout for these
  main-only operations. Preflight reuses that single checksummed inventory when
  constructing the split plan; it does not hash all source archives twice. If
  the timeout is still reached, treat it as a storage
  or process hang and inspect all rank logs; do not mask it with launcher-only
  timeout environment variables.
- **ONNX CUDA provider missing:** reinstall the recipe requirements in the CUDA
  12.8 environment and verify `CUDAExecutionProvider`; do not proceed on CPU.
- **GigaAM or private benchmark access fails:** confirm the rotated `HF_TOKEN`,
  network access, `bitmanagerai/hard_number_eval_for_tts` permission, and the
  `onnx-asr==0.12.0` model cache. Validation cannot be skipped.
- **W&B fails or is offline:** restore connectivity and the same run ID/API key,
  then resume. Production requires remotely verified commits; local-only W&B is
  used solely by the CPU integration test.
- **CUDA OOM during qualification:** lower `--token-limit`; the smoke refuses a
  requested value above the measured safe limit. Do not change batch semantics
  after checkpoints exist.
- **Resume identity changed:** use the original data, cache, base, seed, LoRA,
  schedule, token limit, and checkpoint. Start a new run root for a new contract.
- **Disk pressure:** leave evaluation audio retention off, reduce checkpoint
  retention, and preserve all sealed manifests/results before moving artifacts.

## Upload policy

There is no upload command or upload stage in either script. The final model,
adapter lineage, caches, private benchmark rows, generated speech, and W&B-local
evidence stay under the run root. Upload is explicitly deferred until the user
reviews the completed artifacts and separately authorizes a destination and
scope. Do not add a Hub token to any artifact in anticipation of that decision.
