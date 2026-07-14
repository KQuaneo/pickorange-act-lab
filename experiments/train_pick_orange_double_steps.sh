#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source ./activate.sh

COUNT="${1:?Usage: $0 30|50 a0|a1_001|a1_002|a1_003}"
SELECTION="${2:?Usage: $0 30|50 a0|a1_001|a1_002|a1_003}"
if [[ "${COUNT}" != "30" && "${COUNT}" != "50" ]]; then
  echo "COUNT must be 30 or 50, got ${COUNT}" >&2
  exit 2
fi

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PROJECT_ROOT}/data/lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/.cache/huggingface/datasets}"
export PYTHONUNBUFFERED=1

BATCH_SIZE=64
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT="${PROJECT:-leisaac-act-pick-orange-gate-double}"
TAG="gate_exp${COUNT}"
RUN_TAG="${TAG}_double"

episode_count() {
  python experiments/gate_runs/lerobot_episode_count.py "$1" --format count
}

validate_checkpoints() {
  local output_dir="$1"
  local target="$2"
  local interval="$3"
  local expected=(
    "$(printf '%06d' "$((target - 2 * interval))")"
    "$(printf '%06d' "$((target - interval))")"
    "$(printf '%06d' "${target}")"
  )
  local actual=()
  while IFS= read -r name; do actual+=("${name}"); done < <(
    find "${output_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
      | awk '/^[0-9]+$/' | sort
  )
  if [[ "${actual[*]}" != "${expected[*]}" ]]; then
    echo "Checkpoint validation failed for ${output_dir}" >&2
    echo "expected: ${expected[*]}" >&2
    echo "actual:   ${actual[*]}" >&2
    return 1
  fi
  [[ -f "${output_dir}/checkpoints/$(printf '%06d' "${target}")/pretrained_model/model.safetensors" ]]
}

prune_recovery_checkpoints() {
  local output_dir="$1"
  local target="$2"
  local interval="$3"
  local keep_a keep_b keep_c path name
  keep_a="$(printf '%06d' "$((target - 2 * interval))")"
  keep_b="$(printf '%06d' "$((target - interval))")"
  keep_c="$(printf '%06d' "${target}")"
  for path in "${output_dir}"/checkpoints/[0-9]*; do
    [[ -e "${path}" ]] || continue
    name="$(basename "${path}")"
    case "${name}" in
      "${keep_a}"|"${keep_b}"|"${keep_c}") ;;
      *)
        echo "Prune transient recovery checkpoint: ${path}"
        rm -rf -- "${path}"
        ;;
    esac
  done
}

train_one() {
  local group="$1"
  local repo_name="$2"
  local steps="$3"
  local save_freq="$4"
  local dataset_root="data/lerobot/local/${repo_name}"
  local episodes
  episodes="$(episode_count "${dataset_root}")"
  local job_name="${RUN_TAG}_${group}_act_ch100_b${BATCH_SIZE}_s${steps}_train${episodes}_noval"
  local output_dir="outputs/train/${job_name}"
  local final_step
  final_step="$(printf '%06d' "${steps}")"

  if [[ ! -f "${dataset_root}/meta/info.json" ]]; then
    echo "Missing dataset: ${dataset_root}" >&2
    exit 1
  fi
  if [[ -f "${output_dir}/TRAINING_DONE" ]] && validate_checkpoints "${output_dir}" "${steps}" "${save_freq}"; then
    echo "Skip completed training: ${job_name}"
    return
  fi
  if [[ -f "${output_dir}/checkpoints/${final_step}/pretrained_model/model.safetensors" ]]; then
    prune_recovery_checkpoints "${output_dir}" "${steps}" "${save_freq}"
    validate_checkpoints "${output_dir}" "${steps}" "${save_freq}"
    touch "${output_dir}/TRAINING_DONE"
    echo "Recovered completed training marker: ${job_name}"
    return
  fi

  if find "${output_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit 2>/dev/null | grep -q .; then
    echo "Resume existing run: ${job_name}"
    lerobot-train --resume=true --output_dir="${output_dir}"
  else
    if [[ -e "${output_dir}" ]]; then
      backup="${output_dir}.precheckpoint_$(date +%Y%m%d_%H%M%S)"
      echo "Move pre-checkpoint failed run aside: ${output_dir} -> ${backup}"
      mv "${output_dir}" "${backup}"
    fi
    echo "Start ${job_name}: episodes=${episodes} steps=${steps} save=${save_freq} batch=${BATCH_SIZE}"
    lerobot-train \
      --dataset.repo_id="local/${repo_name}" \
      --dataset.root="${dataset_root}" \
      --dataset.episodes="$(python experiments/gate_runs/lerobot_episode_count.py "${dataset_root}" --format json)" \
      --dataset.video_backend=pyav \
      --policy.type=act \
      --policy.device=cuda \
      --policy.use_amp=true \
      --policy.push_to_hub=false \
      --policy.chunk_size=100 \
      --policy.n_action_steps=100 \
      --output_dir="${output_dir}" \
      --job_name="${job_name}" \
      --steps="${steps}" \
      --batch_size="${BATCH_SIZE}" \
      --num_workers="${NUM_WORKERS}" \
      --log_freq=50 \
      --eval_freq=0 \
      --save_checkpoint=true \
      --save_freq="${save_freq}" \
      --wandb.enable=true \
      --wandb.mode="${WANDB_MODE}" \
      --wandb.project="${PROJECT}"
  fi

  prune_recovery_checkpoints "${output_dir}" "${steps}" "${save_freq}"
  validate_checkpoints "${output_dir}" "${steps}" "${save_freq}"
  touch "${output_dir}/TRAINING_DONE"
  echo "Training complete: ${job_name}"
}

case "${SELECTION}" in
  a0)
    train_one a0_full "so101_pick_orange_${TAG}_a0_joint6_v0" 42000 6000
    ;;
  a1_001)
    train_one a1_prefixstrict_orange001 "so101_pick_orange_${TAG}_a1_event_prefixstrict_orange001_joint6_v0" 14000 2000
    ;;
  a1_002)
    train_one a1_prefixstrict_orange002 "so101_pick_orange_${TAG}_a1_event_prefixstrict_orange002_joint6_v0" 14000 2000
    ;;
  a1_003)
    train_one a1_prefixstrict_orange003 "so101_pick_orange_${TAG}_a1_event_prefixstrict_orange003_joint6_v0" 14000 2000
    ;;
  *)
    echo "Unknown selection: ${SELECTION}" >&2
    exit 2
    ;;
esac
