#!/usr/bin/env bash
set -euo pipefail

# (optional) tame CPU thread oversubscription
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

STUDY="latticebench-once"
TR=480
EP=300
BOOST=1200
ART_BASE="runs/optuna"
LOG="${ART_BASE}/run_$(date +'%Y%m%d-%H%M%S').log"

mkdir -p "${ART_BASE}"

python -u -m src.optuna_search \
  --study "${STUDY}" \
  --storage "sqlite:///runs/optuna/${STUDY}.db" \
  --trials ${TR} \
  --epochs ${EP} \
  --teacher_seed 20250923 \
  --n_jobs 1 \
  --artifacts_dir "${ART_BASE}" \
  --boost_to ${BOOST} \
  --boost_topk_ratio 0.15 \
  --pruner_warmup_ratio 0.5 \
  --auto_decide \
  --lr_min 3e-3 --lr_max 8e-3 --lr_log \
  --w_plaq_min 0.02 --w_plaq_max 0.06 \
  --w_unitary_min 0.02 --w_unitary_max 0.20 \
  --w_phi_smooth_min 0.03 --w_phi_smooth_max 0.06 \
  --w_theta_smooth_min 0.015 --w_theta_smooth_max 0.045 \
  --w_phi_l2_min 5e-4 --w_phi_l2_max 1e-2 \
  --huber_delta_wil_min 0.002 --huber_delta_wil_max 0.010 \
  --huber_delta_cr_min  0.004 --huber_delta_cr_max  0.016 \
  --search_use_huber True False \
  2>&1 | tee "${LOG}"
