#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="$(pwd):$(pwd)/third_party/Matcha-TTS:$(pwd)/runtime/python/fastapi"

export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-$(pwd)/pretrained_models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512}"
export COSYVOICE_BACKEND="${COSYVOICE_BACKEND:-vllm}"
export COSYVOICE_FP16="${COSYVOICE_FP16:-true}"
export COSYVOICE_LOAD_TRT="${COSYVOICE_LOAD_TRT:-false}"
export COSYVOICE_DEFAULT_SPEED="${COSYVOICE_DEFAULT_SPEED:-1.0}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-cosyvoice}"
export TTS_PORT="${TTS_PORT:-8003}"
export TTS_HOST="${TTS_HOST:-0.0.0.0}"
export TTS_WARMUP_ENABLED="${TTS_WARMUP_ENABLED:-false}"
export MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-5000}"
export ENABLE_TEXT_NORMALIZATION="${ENABLE_TEXT_NORMALIZATION:-true}"
export GENERATED_AUDIO_DIR="${GENERATED_AUDIO_DIR:-$(pwd)/generated}"

mkdir -p "$GENERATED_AUDIO_DIR"

echo "Starting CosyVoice3 TTS Server (${COSYVOICE_BACKEND} backend) on port ${TTS_PORT}..."
echo "Model: $COSYVOICE_MODEL_DIR"
echo "Host: $TTS_HOST"
echo "Port: $TTS_PORT"

exec .venv/bin/python run_uvicorn.py
