#!/usr/bin/env bash
set -euo pipefail

COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-}"
COSYVOICE_BACKEND="${COSYVOICE_BACKEND:-vllm}"
COSYVOICE_DEFAULT_SPEED="${COSYVOICE_DEFAULT_SPEED:-1.0}"
VLLM_PLUGINS="${VLLM_PLUGINS:-cosyvoice}"
COSYVOICE_TRT_ENGINE_DIR="${COSYVOICE_TRT_ENGINE_DIR:-}"
COSYVOICE_TRT_SERVE_URL="${COSYVOICE_TRT_SERVE_URL:-}"
COSYVOICE_TRT_SERVE_MODEL_NAME="${COSYVOICE_TRT_SERVE_MODEL_NAME:-}"
COSYVOICE_HF_MODEL_DIR="${COSYVOICE_HF_MODEL_DIR:-}"
TTS_PORT="${TTS_PORT:-8003}"
TTS_HOST="${TTS_HOST:-0.0.0.0}"
TRTLLM_SERVE_HOST="${TRTLLM_SERVE_HOST:-0.0.0.0}"
TRTLLM_SERVE_PORT="${TRTLLM_SERVE_PORT:-8000}"
TLLM_WORKER_USE_SINGLE_PROCESS="${TLLM_WORKER_USE_SINGLE_PROCESS:-1}"

export COSYVOICE_BACKEND
export COSYVOICE_DEFAULT_SPEED
export COSYVOICE_MODEL_DIR
export COSYVOICE_TRT_ENGINE_DIR
export COSYVOICE_TRT_SERVE_URL
export COSYVOICE_TRT_SERVE_MODEL_NAME
export COSYVOICE_HF_MODEL_DIR
export VLLM_PLUGINS
export TTS_PORT
export TTS_HOST
export TLLM_WORKER_USE_SINGLE_PROCESS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ENTRYPOINT="${SCRIPT_DIR}/server_cosyvoice3.py"

TRTLLM_SERVE_PID=""

cleanup() {
    if [[ -n "${TRTLLM_SERVE_PID}" ]]; then
        echo "Stopping trtllm-serve (pid=${TRTLLM_SERVE_PID})"
        kill "${TRTLLM_SERVE_PID}" >/dev/null 2>&1 || true
        wait "${TRTLLM_SERVE_PID}" >/dev/null 2>&1 || true
        TRTLLM_SERVE_PID=""
    fi
}

trap cleanup EXIT INT TERM

if [[ -z "${COSYVOICE_MODEL_DIR}" ]]; then
    echo "ERROR: COSYVOICE_MODEL_DIR is required" >&2
    exit 1
fi

if [[ "${COSYVOICE_BACKEND}" == "trtllm" ]]; then
    if [[ -z "${COSYVOICE_TRT_ENGINE_DIR}" ]]; then
        echo "ERROR: COSYVOICE_TRT_ENGINE_DIR is required for trtllm backend" >&2
        exit 1
    fi

    if [[ ! -d "${COSYVOICE_TRT_ENGINE_DIR}" ]]; then
        echo "TRT-LLM engine not found at ${COSYVOICE_TRT_ENGINE_DIR}, building..."
        HF_OUTPUT_DIR="${COSYVOICE_HF_MODEL_DIR:-${COSYVOICE_MODEL_DIR}/hf_merged}"

        if [[ ! -d "${HF_OUTPUT_DIR}" ]]; then
            echo "Converting CosyVoice3 LLM to HF format..."
            python3 "${SCRIPT_DIR}/../../../runtime/triton_trtllm/scripts/convert_cosyvoice3_to_hf.py" \
                --model-dir "${COSYVOICE_MODEL_DIR}" \
                --output-dir "${HF_OUTPUT_DIR}"
        fi

        echo "Building TRT-LLM engine (this may take a while)..."
        # Minimal trtllm-build example; adjust to your TRT-LLM version and requirements
        trtllm-build \
            --checkpoint_dir "${HF_OUTPUT_DIR}" \
            --output_dir "${COSYVOICE_TRT_ENGINE_DIR}" \
            --gemm_plugin auto \
            --max_batch_size 1 \
            --max_input_len 512 \
            --max_seq_len 2048
    fi
fi

if [[ "${COSYVOICE_BACKEND}" == "trtllm-serve" ]]; then
    if [[ -z "${COSYVOICE_TRT_SERVE_URL}" ]]; then
        COSYVOICE_TRT_SERVE_URL="http://127.0.0.1:${TRTLLM_SERVE_PORT}"
        export COSYVOICE_TRT_SERVE_URL
    fi

    # Check if trtllm-serve is already running at the URL
    if ! curl -sf "${COSYVOICE_TRT_SERVE_URL}/v1/models" >/dev/null 2>&1; then
        if [[ -z "${COSYVOICE_TRT_ENGINE_DIR}" ]]; then
            echo "ERROR: COSYVOICE_TRT_ENGINE_DIR is required to auto-start trtllm-serve" >&2
            echo "Set COSYVOICE_TRT_SERVE_URL to use an already-running service instead." >&2
            exit 1
        fi

        echo "Starting trtllm-serve in background..."
        HF_OUTPUT_DIR="${COSYVOICE_HF_MODEL_DIR:-${COSYVOICE_MODEL_DIR}/hf_merged}"

        if [[ ! -d "${HF_OUTPUT_DIR}" ]]; then
            echo "Converting CosyVoice3 LLM to HF format..."
            python3 "${SCRIPT_DIR}/../../../runtime/triton_trtllm/scripts/convert_cosyvoice3_to_hf.py" \
                --model-dir "${COSYVOICE_MODEL_DIR}" \
                --output-dir "${HF_OUTPUT_DIR}"
        fi

        trtllm-serve serve \
            --backend tensorrt \
            --tokenizer "${HF_OUTPUT_DIR}" \
            "${COSYVOICE_TRT_ENGINE_DIR}" \
            --host "${TRTLLM_SERVE_HOST}" \
            --port "${TRTLLM_SERVE_PORT}" \
            --max_batch_size 1 \
            --max_num_tokens 4096 \
            --kv_cache_free_gpu_memory_fraction "${TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION:-0.3}" &
        TRTLLM_SERVE_PID="$!"

        echo "Waiting for trtllm-serve readiness..."
        for i in $(seq 1 60); do
            if curl -sf "${COSYVOICE_TRT_SERVE_URL}/v1/models" >/dev/null 2>&1; then
                echo "trtllm-serve is ready"
                break
            fi
            if ! kill -0 "${TRTLLM_SERVE_PID}" >/dev/null 2>&1; then
                echo "ERROR: trtllm-serve exited before becoming ready" >&2
                exit 1
            fi
            sleep 2
        done
        if ! curl -sf "${COSYVOICE_TRT_SERVE_URL}/v1/models" >/dev/null 2>&1; then
            echo "ERROR: trtllm-serve did not become ready at ${COSYVOICE_TRT_SERVE_URL}" >&2
            exit 1
        fi
    else
        echo "trtllm-serve already running at ${COSYVOICE_TRT_SERVE_URL}"
    fi
fi

echo "Starting CosyVoice3 TTS server (backend=${COSYVOICE_BACKEND}) on ${TTS_HOST}:${TTS_PORT}..."
if [[ "${COSYVOICE_BACKEND}" == "trtllm-serve" ]]; then
    echo "Using trtllm-serve at ${COSYVOICE_TRT_SERVE_URL} (default port ${TRTLLM_SERVE_PORT})"
fi
exec python3 "${SERVER_ENTRYPOINT}"
