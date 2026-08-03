#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd -P)"

if [[ "${BALALAIKA_THREE_GPU_SMOKE:-}" == "1" ]]; then
  visible_devices="${CUDA_VISIBLE_DEVICES:-}"
  IFS=',' read -r -a device_ids <<< "${visible_devices}"
  if [[ ${#device_ids[@]} -ne 3 ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain exactly three device IDs for the three-GPU smoke" >&2
    exit 2
  fi
  declare -A seen_devices=()
  for device_id in "${device_ids[@]}"; do
    if [[ ! "${device_id}" =~ ^[0-9]+$ ]]; then
      echo "CUDA_VISIBLE_DEVICES must contain three unique non-negative integer IDs for the three-GPU smoke" >&2
      exit 2
    fi
    normalized_device_id="${device_id#"${device_id%%[!0]*}"}"
    normalized_device_id="${normalized_device_id:-0}"
    if [[ -n "${seen_devices[${normalized_device_id}]:-}" ]]; then
      echo "CUDA_VISIBLE_DEVICES must contain three unique non-negative integer IDs for the three-GPU smoke" >&2
      exit 2
    fi
    seen_devices["${normalized_device_id}"]=1
  done

  export CUDA_VISIBLE_DEVICES="${visible_devices}"
  export BALALAIKA_VISIBLE_DEVICES="0,1,2"
  export PYTHONPATH="${repo_root}:${repo_root}/third_party/Matcha-TTS${PYTHONPATH:+:${PYTHONPATH}}"
  exec accelerate launch --multi_gpu --gpu_ids all --num_machines 1 --num_processes 3 --mixed_precision bf16 \
    -m cosyvoice.finetune.balalaika.workflow three-gpu-smoke "$@"
fi

visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

IFS=',' read -r -a device_ids <<< "${visible_devices}"
if [[ ${#device_ids[@]} -ne 8 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly eight device IDs" >&2
  exit 2
fi
declare -A seen_devices=()
for device_id in "${device_ids[@]}"; do
  if [[ ! "${device_id}" =~ ^[0-9]+$ ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain eight unique non-negative integer IDs" >&2
    exit 2
  fi
  normalized_device_id="${device_id#"${device_id%%[!0]*}"}"
  normalized_device_id="${normalized_device_id:-0}"
  if [[ -n "${seen_devices[${normalized_device_id}]:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain eight unique non-negative integer IDs" >&2
    exit 2
  fi
  seen_devices["${normalized_device_id}"]=1
done

export CUDA_VISIBLE_DEVICES="${visible_devices}"
export BALALAIKA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="${repo_root}:${repo_root}/third_party/Matcha-TTS${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${1:-}" == "--status" ]]; then
  shift
  exec python -m cosyvoice.finetune.balalaika.workflow status "$@"
fi

exec accelerate launch \
  --config_file "${repo_root}/examples/balalaika/cosyvoice3_lora/conf/accelerate.yaml" \
  --num_processes 8 \
  --mixed_precision bf16 \
  -m cosyvoice.finetune.balalaika.workflow phase1 "$@"
