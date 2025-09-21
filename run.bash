#!/usr/bin/env bash
set -euo pipefail

# EN: Grid sweep launcher for LatticeBench training runs
# JA: LatticeBench 学習のグリッド実験を起動するスクリプト

# ---------- Common base (shared across runs) / 共通設定 ----------
BASE="python -m src.train --epochs 900 --print_every 50 \
  --w_unitary 0.06 \
  --w_phi_smooth 0.04 --w_theta_smooth 0.02 --w_phi_l2 0.002 \
  --w_wil11 0.10 --w_wil22 0.28 --w_wil13 0.22 --w_wil23 0.18"

# EN: Lightweight anchor on plaquette weight / JA: plaq 重みの軽いアンカー
PLAQS=(0.02 0.04 0.06)

# EN: Emphasize W(1x2) and Creutz / JA: W(1x2) と χ を強めに
W12S=(0.32 0.44 0.56)
CRS=(0.45 0.55 0.65)

# EN: Learning rates (2 choices) / JA: 学習率（2値）
LRS=(5e-3 4e-3)

# EN: Fixed seed(s) / JA: シード（固定）
SEEDS=(42)

mkdir -p runs
i=0

for lr in "${LRS[@]}"; do
  for w12 in "${W12S[@]}"; do
    for wcr in "${CRS[@]}"; do
      for wpl in "${PLAQS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          i=$((i+1))
          tag="lr${lr}_w12${w12}_cr${wcr}_pl${wpl}_s${seed}"

          # -------- Pattern A: plain MSE / 素直な MSE --------
          # Logs are tee'd to runs/A_*.log / ログは runs/A_*.log に保存
          eval ${BASE} --lr ${lr} --w_wil12 ${w12} --w_cr ${wcr} --w_plaq ${wpl} --seed ${seed} \
            | tee "runs/A_${tag}.log"

          # -------- Pattern B: Huber (sharper deltas) / Huber（デルタ小さめで鋭く） --------
          eval ${BASE} --lr ${lr} --w_wil12 ${w12} --w_cr ${wcr} --w_plaq ${wpl} --seed ${seed} \
            --use_huber --huber_delta_wil 0.003 --huber_delta_cr 0.008 \
            | tee "runs/B_${tag}.log"
        done
      done
    done
  done
done

echo "Jobs launched: $i"
