# CosyVoice3 TTS Server — Quick Deploy Guide

This guide reproduces the exact steps used to deploy `Fun-CosyVoice3-0.5B-2512` as a FastAPI inference endpoint on a CUDA-capable Linux host.

## Prerequisites

- Linux with NVIDIA GPU (tested on Blackwell / sm_120 with CUDA 12.8)
- `uv` installed (https://docs.astral.sh/uv/)
- `git` with submodule support
- ~10 GB free disk space (model + env + TRT cache)

## 1. Clone & prepare repo

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive
```

## 2. Create uv venv (Python 3.12)

```bash
uv venv .venv --python 3.12 --system-site-packages
```

## 3. Install PyTorch (CUDA 12.8 build for Blackwell)

If you are on an older GPU (Ampere/Hopper with CUDA 12.1), you can skip to the next step and let `uv` install `torch==2.3.1+cu121` from requirements.  
For **Blackwell (sm_120)** you **must** install PyTorch 2.8.0+cu128 manually:

```bash
# Download wheels once
pip download torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -d /tmp/torch_wheels

# Install into venv
uv pip install --python .venv/bin/python /tmp/torch_wheels/*.whl --force-reinstall
```

## 4. Install remaining dependencies

```bash
# Filter out the broken [IP_ADDRESS] placeholders and the old torch pins
sed -e '/tensorrt-cu12/d' \
    -e '/torch==/d' \
    -e '/torchaudio==/d' \
    -e '/triton==/d' \
    requirements.txt > requirements_filtered.txt

cat >> requirements_filtered.txt <<EOF
prometheus-client
httpx
python-multipart
EOF

uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  -r requirements_filtered.txt

uv pip install --python .venv/bin/python \
  vllm==0.11.0 transformers==4.57.1 numpy==1.26.4

uv pip install --python .venv/bin/python -e ./vllm_cosyvoice_plugin
```

If `openai-whisper==20231117` fails to build, install it separately:

```bash
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --no-build-isolation openai-whisper==20231117
```

## 5. Download model

```bash
.venv/bin/python -c "
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  cache_dir='./pretrained_models')
"
```

The model lands under `./pretrained_models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512`.

## 6. Start the server

Create a launcher script `run_server.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="$(pwd):$(pwd)/third_party/Matcha-TTS:$(pwd)/runtime/python/fastapi"

export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-$(pwd)/pretrained_models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512}"
export COSYVOICE_BACKEND="${COSYVOICE_BACKEND:-vllm}"
export COSYVOICE_FP16="${COSYVOICE_FP16:-true}"
export COSYVOICE_LOAD_TRT="${COSYVOICE_LOAD_TRT:-false}"
export COSYVOICE_DEFAULT_SPEED="${COSYVOICE_DEFAULT_SPEED:-1.0}"
export COSYVOICE_GPU_MEM="${COSYVOICE_GPU_MEM:-0.4}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-cosyvoice}"
export TTS_PORT="${TTS_PORT:-8003}"
export TTS_HOST="${TTS_HOST:-0.0.0.0}"
export TTS_WARMUP_ENABLED="${TTS_WARMUP_ENABLED:-false}"
export MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-5000}"
export ENABLE_TEXT_NORMALIZATION="${ENABLE_TEXT_NORMALIZATION:-true}"
export GENERATED_AUDIO_DIR="${GENERATED_AUDIO_DIR:-$(pwd)/generated}"

mkdir -p "$GENERATED_AUDIO_DIR"

exec .venv/bin/python runtime/python/fastapi/server_cosyvoice3.py
```

The checked-in launchers default to `COSYVOICE_BACKEND=vllm`, `COSYVOICE_DEFAULT_SPEED=1.0`, and `VLLM_PLUGINS=cosyvoice`; set these before launching to override them. `COSYVOICE_GPU_MEM` is passed to vLLM as `gpu_memory_utilization`.

### Run with vLLM

Use this mode when you want FastAPI to host the HTTP API and run the CosyVoice3 LLM through the local vLLM plugin in the same server process. No separate OpenAI/vLLM HTTP server is needed.

```bash
export PYTHONPATH="$(pwd):$(pwd)/third_party/Matcha-TTS:$(pwd)/runtime/python/fastapi"
export COSYVOICE_MODEL_DIR="$(pwd)/pretrained_models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512"
export COSYVOICE_BACKEND=vllm
export VLLM_PLUGINS=cosyvoice
export COSYVOICE_GPU_MEM="${COSYVOICE_GPU_MEM:-0.4}"
export COSYVOICE_FP16="${COSYVOICE_FP16:-true}"
export COSYVOICE_LOAD_TRT="${COSYVOICE_LOAD_TRT:-false}"
export COSYVOICE_DEFAULT_SPEED="${COSYVOICE_DEFAULT_SPEED:-1.0}"
export TTS_HOST="${TTS_HOST:-0.0.0.0}"
export TTS_PORT="${TTS_PORT:-8003}"
export TTS_WARMUP_ENABLED="${TTS_WARMUP_ENABLED:-false}"

.venv/bin/python runtime/python/fastapi/server_cosyvoice3.py
```

Or run the checked-in launcher, which sets the same vLLM defaults:

```bash
COSYVOICE_BACKEND=vllm TTS_PORT=8003 ./run_server.sh
```

Validate vLLM mode:

```bash
curl -s http://localhost:8003/health | python3 -m json.tool
curl -s http://localhost:8003/v1/audio/voices | python3 -m json.tool
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"optimized_short","input":"Привет, это тест.","response_format":"pcm","stream":true,"lang":"ru"}' \
  -o output.raw
```

The health response should show `"backend": "vllm"`. If startup fails because the vLLM plugin is missing, reinstall it with:

```bash
uv pip install --python .venv/bin/python -e ./vllm_cosyvoice_plugin
```

To use a TensorRT-LLM Serve process for the LLM stage, keep it on its own port and point FastAPI at it:

```bash
export COSYVOICE_BACKEND=trtllm-serve
export COSYVOICE_TRT_SERVE_URL=http://127.0.0.1:8000
export COSYVOICE_TRT_SERVE_MODEL_NAME="${COSYVOICE_TRT_SERVE_MODEL_NAME:-trt_engines_bfloat16}"
export TTS_PORT=8003
```

The FastAPI server still uses `COSYVOICE_DEFAULT_REF_AUDIO` for the prompt audio reference. `trtllm-serve` should listen on `8000`; the FastAPI TTS API should listen on `8003`.

Or use the dedicated launcher, which starts only the FastAPI side on `8003` and defaults to `COSYVOICE_TRT_SERVE_URL=http://127.0.0.1:8000`, `COSYVOICE_TRT_SERVE_MODEL_NAME=trt_engines_bfloat16`, `COSYVOICE_DEFAULT_REF_AUDIO=asset/qwen_ref_4.wav`, `COSYVOICE_COMPAT_MODE=cross_lingual`, `COSYVOICE_LOAD_TRT=false`, and warmup disabled:

```bash
./start_trtllm_serve_fastapi.sh
```

Run it:

```bash
chmod +x run_server.sh
nohup ./run_server.sh > server.log 2>&1 &
```

Or use the wrapper that hard-codes `0.0.0.0` to avoid hostname resolution issues:

```bash
nohup .venv/bin/python run_uvicorn.py > server.log 2>&1 &
```

(see `run_uvicorn.py` in this repo for the exact wrapper)

## 7. Verify

```bash
curl -s http://localhost:8003/health | python3 -m json.tool
```

## 8. Generate TTS (example — Russian cross-lingual)

```bash
curl -X POST http://localhost:8003/tts-stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Привет, это тест.","lang":"ru"}' \
  -o output.raw
```

Reference audio and TRT-specific settings are configured by environment variables such as
`COSYVOICE_DEFAULT_REF_AUDIO`, `COSYVOICE_PROMPT_TEXT`, `COSYVOICE_COMPAT_MODE`,
`COSYVOICE_TRT_SERVE_URL`, and `COSYVOICE_HF_MODEL_DIR`; they are not request fields.

Convert raw int16 PCM @ 24 kHz to WAV:

```bash
python3 -c "
import wave, sys
with open('output.raw','rb') as f: pcm=f.read()
with wave.open('output.wav','wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes(pcm)
"
```

## 9. Batch generation (Python script)

Use the provided `generate_homographs.py` as a template. Key points:
- Send JSON requests matching the Qwen-compatible endpoint schema
- Configure prompt/reference audio and mode through environment variables
- Response is raw int16 PCM, 24 kHz, mono

## Known issues & fixes applied

| Issue | Fix |
|-------|-----|
| `python-multipart` missing | `uv pip install python-multipart` |
| BytesIO cannot be re-read by torchaudio | Save uploaded audio to `tempfile.NamedTemporaryFile` first |
| Warmup text too short → hifigan kernel error | Set `TTS_WARMUP_ENABLED=false` |
| Host `[IP_ADDRESS]` not resolvable | Use `0.0.0.0` or `127.0.0.1` in `run_uvicorn.py` |
| Blackwell GPU (sm_120) incompatible with torch 2.3.1 | Upgrade to `torch==2.8.0+cu128` |
| ONNX Runtime missing `libcudnn.so.8` | Falls back to CPU automatically (adds latency but works) |
| `openai-whisper` build fails | Install with `--no-build-isolation` |

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check + model status |
| `POST /tts-stream` | Streaming TTS (form data) |
| `POST /v1/audio/speech` | OpenAI-compatible streaming |
| `POST /v1/audio/speech/wav` | OpenAI-compatible WAV |
| `GET /v1/audio/voices` | List available speakers |
| `GET /demo` | Web UI |

## File reference

- `run_server.sh` — bash launcher
- `run_uvicorn.py` — Python launcher (binds `0.0.0.0:8003` by default)
- `start_trtllm_serve_fastapi.sh` — FastAPI launcher for an existing `trtllm-serve` process
- `generate_homographs.py` — batch generation example
- `server.log` — runtime logs
