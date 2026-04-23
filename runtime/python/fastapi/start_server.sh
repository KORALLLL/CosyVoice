#!/usr/bin/env bash
set -euo pipefail

COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-}"
COSYVOICE_BACKEND="${COSYVOICE_BACKEND:-native}"
COSYVOICE_TRT_ENGINE_DIR="${COSYVOICE_TRT_ENGINE_DIR:-}"
COSYVOICE_TRT_SERVE_URL="${COSYVOICE_TRT_SERVE_URL:-}"
COSYVOICE_HF_MODEL_DIR="${COSYVOICE_HF_MODEL_DIR:-}"
TTS_PORT="${TTS_PORT:-8000}"
TTS_HOST="${TTS_HOST:-0.0.0.0}"

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
        echo "ERROR: COSYVOICE_TRT_SERVE_URL is required for trtllm-serve backend" >&2
        exit 1
    fi

    # Check if trtllm-serve is already running at the URL
    if ! curl -sf "${COSYVOICE_TRT_SERVE_URL}/v1/models" >/dev/null 2>&1; then
        echo "Starting trtllm-serve in background..."
        HF_OUTPUT_DIR="${COSYVOICE_HF_MODEL_DIR:-${COSYVOICE_MODEL_DIR}/hf_merged}"

        if [[ ! -d "${HF_OUTPUT_DIR}" ]]; then
            echo "Converting CosyVoice3 LLM to HF format..."
            python3 "${SCRIPT_DIR}/../../../runtime/triton_trtllm/scripts/convert_cosyvoice3_to_hf.py" \
                --model-dir "${COSYVOICE_MODEL_DIR}" \
                --output-dir "${HF_OUTPUT_DIR}"
        fi

        trtllm-serve "${HF_OUTPUT_DIR}" \
            --host "${TTS_HOST}" \
            --port "${TTS_PORT}" \
            --max_batch_size 1 \
            --max_num_tokens 2048 &
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
    else
        echo "trtllm-serve already running at ${COSYVOICE_TRT_SERVE_URL}"
    fi
fi

echo "Starting CosyVoice3 TTS server (backend=${COSYVOICE_BACKEND})..."
exec python3 "${SERVER_ENTRYPOINT}"
