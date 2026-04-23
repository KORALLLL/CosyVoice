#!/usr/bin/env bash
set -euo pipefail

cd /workspace/CosyVoice

export PYTHONPATH=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS:/workspace/CosyVoice/runtime/python/fastapi:${PYTHONPATH:-}

export COSYVOICE_MODEL_DIR="/workspace/CosyVoice/pretrained_models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512"
export COSYVOICE_BACKEND="native"
export COSYVOICE_FP16="true"
export COSYVOICE_LOAD_TRT="true"
export TTS_PORT="8000"
export TTS_HOST="0.0.0.0"
export TTS_WARMUP_ENABLED="true"
export MAX_TEXT_LENGTH="5000"
export ENABLE_TEXT_NORMALIZATION="true"
export GENERATED_AUDIO_DIR="/workspace/CosyVoice/generated"

mkdir -p "$GENERATED_AUDIO_DIR"

echo "Starting CosyVoice3 TTS Server with TRT engine..."
echo "Model: $COSYVOICE_MODEL_DIR"
echo "Backend: $COSYVOICE_BACKEND (load_trt=$COSYVOICE_LOAD_TRT)"
echo "Port: $TTS_PORT"

exec /workspace/CosyVoice/.venv/bin/python /workspace/CosyVoice/runtime/python/fastapi/server_cosyvoice3.py
