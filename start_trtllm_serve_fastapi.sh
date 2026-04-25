#!/usr/bin/env bash
set -euo pipefail

cd /workspace/CosyVoice

export PYTHONPATH=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS:/workspace/CosyVoice/runtime/python/fastapi:${PYTHONPATH:-}

export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-/workspace/CosyVoice/pretrained_models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512}"
export COSYVOICE_BACKEND="trtllm-serve"
export COSYVOICE_TRT_SERVE_URL="${COSYVOICE_TRT_SERVE_URL:-http://127.0.0.1:8000}"
export COSYVOICE_TRT_SERVE_MODEL_NAME="${COSYVOICE_TRT_SERVE_MODEL_NAME:-trt_engines_bfloat16}"
export COSYVOICE_HF_MODEL_DIR="${COSYVOICE_HF_MODEL_DIR:-/workspace/CosyVoice/runtime/triton_trtllm/hf_cosyvoice3_llm}"
export COSYVOICE_DEFAULT_REF_AUDIO="${COSYVOICE_DEFAULT_REF_AUDIO:-/workspace/CosyVoice/asset/qwen_ref_4.wav}"
export COSYVOICE_COMPAT_MODE="${COSYVOICE_COMPAT_MODE:-cross_lingual}"
export COSYVOICE_DEFAULT_SPEED="${COSYVOICE_DEFAULT_SPEED:-1.0}"
export COSYVOICE_FP16="${COSYVOICE_FP16:-true}"
export COSYVOICE_LOAD_TRT="${COSYVOICE_LOAD_TRT:-false}"
export TTS_PORT="${TTS_PORT:-8003}"
export TTS_HOST="${TTS_HOST:-0.0.0.0}"
export TTS_WARMUP_ENABLED="${TTS_WARMUP_ENABLED:-false}"
export MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-5000}"
export ENABLE_TEXT_NORMALIZATION="${ENABLE_TEXT_NORMALIZATION:-true}"
export GENERATED_AUDIO_DIR="${GENERATED_AUDIO_DIR:-/workspace/CosyVoice/generated}"

mkdir -p "$GENERATED_AUDIO_DIR"

echo "Starting CosyVoice3 FastAPI server with trtllm-serve backend..."
echo "Model: $COSYVOICE_MODEL_DIR"
echo "trtllm-serve URL: $COSYVOICE_TRT_SERVE_URL"
echo "trtllm-serve model: $COSYVOICE_TRT_SERVE_MODEL_NAME"
echo "HF tokenizer/model dir: $COSYVOICE_HF_MODEL_DIR"
echo "Reference audio: $COSYVOICE_DEFAULT_REF_AUDIO"
echo "Compat mode: $COSYVOICE_COMPAT_MODE"
echo "FastAPI: $TTS_HOST:$TTS_PORT"

exec /workspace/CosyVoice/.venv/bin/python /workspace/CosyVoice/runtime/python/fastapi/server_cosyvoice3.py
