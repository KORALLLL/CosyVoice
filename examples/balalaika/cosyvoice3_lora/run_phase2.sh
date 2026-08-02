#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd -P)"
visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

IFS=',' read -r -a device_ids <<< "${visible_devices}"
if [[ ${#device_ids[@]} -ne 8 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly eight device IDs" >&2
  exit 2
fi
declare -A seen_devices=()
for device_id in "${device_ids[@]}"; do
  if [[ ! "${device_id}" =~ ^[0-9]+$ ]] || [[ -n "${seen_devices[${device_id}]:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain eight unique non-negative integer IDs" >&2
    exit 2
  fi
  seen_devices["${device_id}"]=1
done

export CUDA_VISIBLE_DEVICES="${visible_devices}"
export BALALAIKA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${1:-}" == "--status" ]]; then
  shift
  exec python -m cosyvoice.finetune.balalaika.workflow status "$@"
fi

exec accelerate launch \
  --config_file "${repo_root}/examples/balalaika/cosyvoice3_lora/conf/accelerate.yaml" \
  --num_processes 8 \
  --mixed_precision bf16 \
  -m cosyvoice.finetune.balalaika.workflow phase2 "$@"
