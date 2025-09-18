# src/train.py
"""
LatticeBench: Training script for a discrete PINN on a small 2D SU(2) lattice
============================================================================
EN: Train the structured PINN to reproduce gauge-invariant observables from a
    randomized teacher SU(2) configuration. Prints progress with ETA/time.
JA: 乱数で作った教師SU(2)場のゲージ不変量を、構造制約付きPINNで再現する
    トレーニングスクリプト。進捗とETA/時間を表示します。

Usage:
  python -m src.train [--epochs 800 --lr 1e-2 ...]
"""

import argparse
import sys
import time
from datetime import timedelta
from typing import Dict
import sys, shlex

import torch
import torch.nn.functional as F

# Local modules / ローカルモジュール
from .groups import get_device
from .lattice import Lattice, average_plaquette_traces, wilson_loop_trace, creutz_ratio
from .gauge_data import SU2GaugeConfig
from .model_structured import StructuredSU2Model
from .gauge_align import gauge_aligned_link_error
from .utils import Timer, format_td  # 進捗表示用（t+=, ETA）


# ------------------------------
# Small helpers / 補助関数
# ------------------------------
def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    """
    EN: Smooth L1 (Huber) penalty. Quadratic near zero, linear in tails.
    JA: スムーズL1（ハッバー）損失。ゼロ近傍は二乗、外側は線形。
    """
    absx = x.abs()
    quad = 0.5 * (absx ** 2) / delta
    lin = absx - 0.5 * delta
    return torch.where(absx <= delta, quad, lin).mean()


def complex_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    EN: MSE for complex tensors by splitting real/imag.
    JA: 複素テンソル用のMSE（実部・虚部で計算）。
    """
    return F.mse_loss(a.real, b.real) + F.mse_loss(a.imag, b.imag)


def build_teacher_targets(U_true: torch.Tensor, lat: Lattice) -> Dict[str, torch.Tensor]:
    """
    EN: Precompute teacher observables used in the loss (W11, W12, W22, W13, W23, chi2).
    JA: 損失で使う教師側の観測量を前計算（W11, W12, W22, W13, W23, χ22）。
    """
    return {
        "W11": wilson_loop_trace(U_true, lat, 1, 1),
        "W12": wilson_loop_trace(U_true, lat, 1, 2),
        "W22": wilson_loop_trace(U_true, lat, 2, 2),
        "W13": wilson_loop_trace(U_true, lat, 1, 3),
        "W23": wilson_loop_trace(U_true, lat, 2, 3),
        "chi2": creutz_ratio(U_true, lat, 2),
        "avgP": average_plaquette_traces(U_true, lat),
    }


def compute_pred_observables(U_pred: torch.Tensor, lat: Lattice) -> Dict[str, torch.Tensor]:
    """
    EN: Compute the same set of observables on prediction.
    JA: 予測側でも同じ観測量を計算。
    """
    return {
        "W11": wilson_loop_trace(U_pred, lat, 1, 1),
        "W12": wilson_loop_trace(U_pred, lat, 1, 2),
        "W22": wilson_loop_trace(U_pred, lat, 2, 2),
        "W13": wilson_loop_trace(U_pred, lat, 1, 3),
        "W23": wilson_loop_trace(U_pred, lat, 2, 3),
        "chi2": creutz_ratio(U_pred, lat, 2),
        "avgP": average_plaquette_traces(U_pred, lat),
    }


# ------------------------------
# Argument parsing / 引数
# ------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train discrete PINN for SU(2) lattice (LatticeBench)")

    # Lattice & model / 格子・モデル
    ap.add_argument("--L", type=int, default=4, help="lattice size (2D) / 格子サイズ(2次元)")
    ap.add_argument("--hidden", type=int, default=32, help="hidden dim for node embeddings / ノード埋め込み次元")

    # Optim / 最適化
    ap.add_argument("--epochs", type=int, default=800, help="training epochs / 学習エポック数")
    ap.add_argument("--lr", type=float, default=1e-2, help="learning rate / 学習率")
    ap.add_argument("--seed", type=int, default=42, help="random seed / 乱数シード")
    ap.add_argument("--print_every", type=int, default=25, help="print interval / ログ表示間隔")

    # Loss weights / 損失の重み
    ap.add_argument("--w_plaq", type=float, default=0.0, help="weight for plaquette-trace MSE (monitor-only if 0)")
    ap.add_argument("--w_wil11", type=float, default=0.10, help="weight for Wilson loop W(1x1)")
    ap.add_argument("--w_wil12", type=float, default=0.28, help="weight for Wilson loop W(1x2)")
    ap.add_argument("--w_wil22", type=float, default=0.20, help="weight for Wilson loop W(2x2)")
    ap.add_argument("--w_wil13", type=float, default=0.22, help="weight for Wilson loop W(1x3)")
    ap.add_argument("--w_wil23", type=float, default=0.18, help="weight for Wilson loop W(2x3)")
    ap.add_argument("--w_cr", type=float, default=0.20, help="weight for Creutz ratio χ(2,2)")
    ap.add_argument("--w_unitary", type=float, default=0.05, help="weight for unitarity penalty / ユニタリティ罰則")
    ap.add_argument("--w_phi_smooth", type=float, default=0.06, help="weight for node-embedding smoothness")
    ap.add_argument("--w_theta_smooth", type=float, default=0.03, help="weight for link-algebra smoothness")
    ap.add_argument("--w_phi_l2", type=float, default=0.003, help="weight for L2 on node embeddings")

    # Huber options / ハッバー損失
    ap.add_argument("--use_huber", action="store_true", help="use Huber instead of MSE for W and χ")
    ap.add_argument("--huber_delta_wil", type=float, default=0.010, help="Huber delta for Wilson loops")
    ap.add_argument("--huber_delta_cr", type=float, default=0.040, help="Huber delta for Creutz ratio")

    return ap


# ------------------------------
# Main train / メイントレイン
# ------------------------------
def main():
    ap = build_argparser()
    cfg = ap.parse_args()

    # --- Log the exact command line for downstream analysis ---
    print("CMD:", " ".join(shlex.quote(a) for a in sys.argv))

    # Device & seed / デバイスとシード
    device = get_device()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Lattice, teacher, targets / 格子・教師データ・教師観測量
    lat = Lattice(cfg.L)
    teacher = SU2GaugeConfig(lat)
    teacher.randomize(scale=0.6)  # EN: random su(2) algebra params / JA: su(2)代数パラメタを乱数初期化
    with torch.no_grad():
        target_traces = teacher.plaquette_traces().detach()  # for monitoring / 監視用
        U_true = teacher.links()
        T = build_teacher_targets(U_true, lat)

    # Model & optimizer / モデルと最適化
    model = StructuredSU2Model(lat, hidden_dim=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Timer for ETA / ETA用タイマー
    timer = Timer()
    start_time = time.time()

    for ep in range(1, cfg.epochs + 1):
        opt.zero_grad(set_to_none=True)

        # Forward prediction / 予測
        U_pred, a_pred = model.forward_links()  # U_pred: (L,L,2,2,2), a_pred: (L,L,2,3)

        # Observables / 観測量
        P_pred = model.plaquette_traces()  # vector of Tr(P) over lattice / 格子全体のTr(P)
        P_true = target_traces
        O = compute_pred_observables(U_pred, lat)

        # Loss: Wilson loops / Wilsonループ損失
        if cfg.use_huber:
            loss_wil = (
                cfg.w_wil11 * huber(O["W11"].real - T["W11"].real, cfg.huber_delta_wil) +
                cfg.w_wil12 * huber(O["W12"].real - T["W12"].real, cfg.huber_delta_wil) +
                cfg.w_wil22 * huber(O["W22"].real - T["W22"].real, cfg.huber_delta_wil) +
                cfg.w_wil13 * huber(O["W13"].real - T["W13"].real, cfg.huber_delta_wil) +
                cfg.w_wil23 * huber(O["W23"].real - T["W23"].real, cfg.huber_delta_wil)
            )
            loss_cr = cfg.w_cr * huber(O["chi2"] - T["chi2"], cfg.huber_delta_cr)
        else:
            loss_wil = (
                cfg.w_wil11 * F.mse_loss(O["W11"].real, T["W11"].real) +
                cfg.w_wil12 * F.mse_loss(O["W12"].real, T["W12"].real) +
                cfg.w_wil22 * F.mse_loss(O["W22"].real, T["W22"].real) +
                cfg.w_wil13 * F.mse_loss(O["W13"].real, T["W13"].real) +
                cfg.w_wil23 * F.mse_loss(O["W23"].real, T["W23"].real)
            )
            loss_cr = cfg.w_cr * F.mse_loss(O["chi2"], T["chi2"])

        # Optional: plaquette-trace vector MSE (monitor or weak guidance)
        loss_plaq = cfg.w_plaq * complex_mse(P_pred, P_true)

        # Physics-structure penalties / 物理・構造ペナルティ
        loss_unit = cfg.w_unitary * model.unitarity_loss()
        loss_phi_s = cfg.w_phi_smooth * model.phi_smooth_loss()
        loss_theta_s = cfg.w_theta_smooth * model.theta_smooth_loss(a_pred)
        loss_phi_l2 = cfg.w_phi_l2 * (model.node_embed.pow(2).mean())

        loss = loss_wil + loss_cr + loss_plaq + loss_unit + loss_phi_s + loss_theta_s + loss_phi_l2
        loss.backward()
        opt.step()

        # Logging / ログ表示
        if ep % cfg.print_every == 1 or ep == cfg.epochs:
            # progress time / 経過時間とETA
            t_elapsed = timer.lap()
            avg_per_ep = t_elapsed / cfg.print_every if ep > 1 else t_elapsed
            remain = avg_per_ep * (cfg.epochs - ep)
            eta = format_td(timedelta(seconds=int(remain.total_seconds()))) if hasattr(remain, "total_seconds") else format_td(timedelta(seconds=int(remain)))

            # Quick scalar monitor / 手早いモニタ
            mean_abs = (P_pred - P_true).abs().mean().item()

            print(
                f"[ep{ep:03d}] total={loss.item():.4e} plaq={loss_plaq.item():.4e} "
                f"|ΔTr|={mean_abs:.3e} "
                f"W11={O['W11']:+.4f}/{T['W11']:+.4f} "
                f"W12={O['W12']:+.4f}/{T['W12']:+.4f} "
                f"W22={O['W22']:+.4f}/{T['W22']:+.4f} "
                f"χ22={O['chi2']:+.3f}/{T['chi2']:+.3f} "
                f"lr={cfg.lr:.3e}  t+={format_td(timedelta(seconds=int(t_elapsed.total_seconds())))}  ETA={eta}"
            )

    # ------------------------------
    # Evaluation / 検証
    # ------------------------------
    with torch.no_grad():
        U_pred, a_pred = model.forward_links()
        O = compute_pred_observables(U_pred, lat)
        avg_pred = O["avgP"]
        avg_true = T["avgP"]
        W11_p, W12_p, W22_p = O["W11"], O["W12"], O["W22"]
        W11_t, W12_t, W22_t = T["W11"], T["W12"], T["W22"]
        chi2_p, chi2_t = O["chi2"], T["chi2"]

    ga_err = gauge_aligned_link_error(U_pred, teacher.links(), lat, steps=200, lr=0.05)

    print("\n=== Invariants & Alignment ===")
    print(f"mean Tr P  pred={avg_pred:+.4f}, true={avg_true:+.4f}, |Δ|={(avg_pred-avg_true).abs().item():.3e}")
    print(f"Wilson(1x1) pred={W11_p:+.4f}, true={W11_t:+.4f}")
    print(f"Wilson(1x2) pred={W12_p:+.4f}, true={W12_t:+.4f}")
    print(f"Wilson(2x2) pred={W22_p:+.4f}, true={W22_t:+.4f}")
    print(f"Creutz χ(2,2) pred={chi2_p:+.4f}, true={chi2_t:+.4f}")
    print(f"Gauge-aligned link RMSE (≈Frob): {ga_err:.4e}")
    total_time = time.time() - start_time
    print(f"Total time: {format_td(timedelta(seconds=int(total_time)))}")
    print("Training finished")


if __name__ == "__main__":
    main()
