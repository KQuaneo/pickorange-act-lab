#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source ./activate.sh

COUNT="${1:?Usage: $0 30|50}"
MAX_PARALLEL="${TRAIN_MAX_PARALLEL:-2}"
LOG_ROOT="outputs/logs/pick_orange_gate_exp${COUNT}_double/train"
DONE="outputs/train/pick_orange_gate_exp${COUNT}_double/TRAINING_DONE"
mkdir -p "${LOG_ROOT}" "$(dirname "${DONE}")"

run_job() {
  local name="$1"
  echo "========== train ${name} =========="
  bash experiments/train_pick_orange_double_steps.sh "${COUNT}" "${name}" \
    > >(tee "${LOG_ROOT}/${name}.log") 2>&1
}

if [[ "${MAX_PARALLEL}" -le 1 ]]; then
  run_job a0
  run_job a1_001
  run_job a1_002
  run_job a1_003
else
  # Historical stable layout: one long A0 lane and one sequential A1 lane.
  run_job a0 &
  a0_pid=$!
  (
    run_job a1_001
    run_job a1_002
    run_job a1_003
  ) &
  a1_pid=$!
  status=0
  wait "${a0_pid}" || status=1
  wait "${a1_pid}" || status=1
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
fi

cat > "${DONE}" <<EOF
count=${COUNT}
batch_size=64
a0_steps=42000
a0_save_freq=6000
a1_steps=14000
a1_save_freq=2000
retained_checkpoints=3
parallel=${MAX_PARALLEL}
finished_at=$(date --iso-8601=seconds)
EOF
echo "Double-step training complete: ${DONE}"
