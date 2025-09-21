"""
LatticeBench: Training script for a discrete PINN on a small 2D SU(2) lattice
============================================================================
EN: Train the structured PINN to reproduce gauge-invariant observables from a
    randomized teacher SU(2) configuration. Prints progress with ETA/time.
JA: 乱数で作った教師SU(2)場のゲージ不変量を、構造制約付きPINNで再現する
    トレーニングスクリプト。進捗とETA/時間を表示します。

-------------------------------------------------------------------------------
Flow overview (処理の流れ)
-------------------------------------------------------------------------------

   ┌───────────────────────────────────────────┐
   │ 1. Setup                                  │
   │   - Parse args / config                   │
   │   - Set random seed, device               │
   └───────────────────────────────────────────┘
                        │
                        ▼
   ┌───────────────────────────────────────────┐
   │ 2. Teacher gauge field (教師場)            │
   │   - Randomize su(2) algebra params        │
   │   - Build links U_true                    │
   │   - Precompute observables (plaquette,    │
   │     Wilson loops, Creutz ratio)           │
   └───────────────────────────────────────────┘
                        │
                        ▼
   ┌───────────────────────────────────────────┐
   │ 3. Model setup                            │
   │   - StructuredSU2Model (PINN-like net)    │
   │   - Optimizer (Adam)                      │
   │   - Logger/CSV logger                     │
   └───────────────────────────────────────────┘
                        │
                        ▼
   ┌───────────────────────────────────────────┐
   │ 4. Training loop (for each epoch)         │
   │   - Forward: model predicts links U_pred  │
   │   - Compute predicted observables         │
   │   - Loss = weighted sum of:               │
   │        · Wilson loops / Creutz ratio      │
   │        · Plaquette trace                  │
   │        · Regularizers (unitarity, smooth) │
   │   - Backpropagation, optimizer.step()     │
   │   - Log progress                          │
   └───────────────────────────────────────────┘
                        │
                        ▼
   ┌───────────────────────────────────────────┐
   │ 5. Evaluation                             │
   │   - Gauge-aligned RMSE                    │
   │   - |Δ avg plaquette|                     │
   │   - Save results to log/CSV               │
   └───────────────────────────────────────────┘
"""

import os
import argparse
import time
from typing import Dict, Any

import torch
import torch.nn.functional as F

# Local modules
from .groups import get_device
from .lattice import Lattice, average_plaquette_traces, wilson_loop_trace, creutz_ratio
from .gauge_data import SU2GaugeConfig
from .model_structured import StructuredSU2Model
from .gauge_align import gauge_aligned_link_error
from .utils import Timer, SimpleLogger, CSVLogger


# ----------------------------------------------------------------------
# Loss helper functions
# ----------------------------------------------------------------------
def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    """
    EN: Huber loss: quadratic for |x|<=δ, linear outside.
    JA: Huber損失。小さい誤差は二乗、大きい誤差は線形で扱う。
    """
    absx = x.abs()
    quad = 0.5 * (absx ** 2) / delta   # |x|<=δ の場合
    lin = absx - 0.5 * delta           # |x|>δ の場合
    return torch.where(absx <= delta, quad, lin).mean()


def complex_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    EN: MSE for complex tensors (real+imag separately).
    JA: 複素数テンソルに対するMSE（実部と虚部を分けて比較）。
    """
    return F.mse_loss(a.real, b.real) + F.mse_loss(a.imag, b.imag)


# ----------------------------------------------------------------------
# Observable helpers
# ----------------------------------------------------------------------
def build_teacher_targets(U_true: torch.Tensor, lat: Lattice) -> Dict[str, torch.Tensor]:
    """
    EN: Precompute teacher-side observables (Wilson loops, Creutz ratio, avg plaquette).
    JA: 教師側の観測量（ウィルソンループ、クルーツ比、平均プラークエット）を事前計算。
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
    EN: Compute the same set of observables for model predictions.
    JA: モデル予測に対して同じ観測量を計算する。
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


# ----------------------------------------------------------------------
# Config dataclass-like container
# ----------------------------------------------------------------------
class TrainConfig:
    """
    EN: Holds all hyperparameters (learning rate, weights, epochs, etc.).
    JA: すべてのハイパーパラメータを格納するコンテナ（学習率、重み、エポック数など）。
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# ----------------------------------------------------------------------
# Core training loop (can be reused by Optuna)
# ----------------------------------------------------------------------
def run_training(cfg: TrainConfig, quiet: bool = False) -> Dict[str, Any]:
    """
    EN: Train the model given a TrainConfig, return metrics dict.
    JA: TrainConfigをもとにモデルを学習し、指標を辞書形式で返す。
    """

    # --- setup random seed & device ---
    device = get_device()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # --- Teacher gauge configuration ---
    lat = Lattice(cfg.L)
    teacher = SU2GaugeConfig(lat)

    # 教師の乱数制御（指定があれば固定）
    if getattr(cfg, "teacher_seed", None) is not None:
        torch.manual_seed(cfg.teacher_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.teacher_seed)

    # ここで一度だけ初期化（scaleは引数から取得）
    teacher.randomize(scale=float(getattr(cfg, "teacher_scale", 0.6)))

    with torch.no_grad():
        target_traces = teacher.plaquette_traces().detach()  # 教師の平均プラークエット
        U_true = teacher.links()                            # 教師リンク
        T = build_teacher_targets(U_true, lat)              # 教師観測量

    # --- Model & optimizer ---
    model = StructuredSU2Model(lat, hidden_dim=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # --- Logging setup ---
    logger = None
    if not quiet and getattr(cfg, "logfile", None):
        logger = SimpleLogger(cfg.logfile, mirror_stdout=True)
    csvlogger = None
    if getattr(cfg, "csv", None):
        csvlogger = CSVLogger(cfg.csv)

    timer = Timer()
    start_time = time.time()
    losses = []

    # ======================================================================
    # Training loop
    # ======================================================================
    for ep in range(1, cfg.epochs + 1):
        opt.zero_grad(set_to_none=True)

        # Forward pass: model predicts link variables
        U_pred, a_pred = model.forward()

        # Compute predicted & teacher observables
        P_pred = model.plaquette_traces()
        P_true = target_traces
        O = compute_pred_observables(U_pred, lat)

        # --------------------------------------------------------------
        # Loss terms
        # --------------------------------------------------------------
        if cfg.use_huber:
            # Huber loss version
            loss_wil = (
                cfg.w_wil11 * huber(O["W11"].real - T["W11"].real, cfg.huber_delta_wil) +
                cfg.w_wil12 * huber(O["W12"].real - T["W12"].real, cfg.huber_delta_wil) +
                cfg.w_wil22 * huber(O["W22"].real - T["W22"].real, cfg.huber_delta_wil) +
                cfg.w_wil13 * huber(O["W13"].real - T["W13"].real, cfg.huber_delta_wil) +
                cfg.w_wil23 * huber(O["W23"].real - T["W23"].real, cfg.huber_delta_wil)
            )
            loss_cr = cfg.w_cr * huber(O["chi2"] - T["chi2"], cfg.huber_delta_cr)
        else:
            # Simple MSE version
            loss_wil = (
                cfg.w_wil11 * F.mse_loss(O["W11"].real, T["W11"].real) +
                cfg.w_wil12 * F.mse_loss(O["W12"].real, T["W12"].real) +
                cfg.w_wil22 * F.mse_loss(O["W22"].real, T["W22"].real) +
                cfg.w_wil13 * F.mse_loss(O["W13"].real, T["W13"].real) +
                cfg.w_wil23 * F.mse_loss(O["W23"].real, T["W23"].real)
            )
            loss_cr = cfg.w_cr * F.mse_loss(O["chi2"], T["chi2"])

        # Plaquette trace match
        loss_plaq = cfg.w_plaq * complex_mse(P_pred, P_true)

        # Regularization / smoothness terms
        loss_unit = cfg.w_unitary * model.unitarity_loss()
        loss_phi_s = cfg.w_phi_smooth * model.phi_smooth_loss()
        loss_theta_s = cfg.w_theta_smooth * model.theta_smooth_loss(a_pred)
        loss_phi_l2 = cfg.w_phi_l2 * (model.node_embed.pow(2).mean())

        # Total loss
        loss = loss_wil + loss_cr + loss_plaq + loss_unit + loss_phi_s + loss_theta_s + loss_phi_l2
        loss.backward()
        opt.step()
        losses.append(loss.item())

        # --- Logging per epoch ---
        if not quiet and (ep % cfg.print_every == 1 or ep == cfg.epochs):
            t_elapsed = timer.lap()
            mean_abs = (P_pred - P_true).abs().mean().item()
            msg = (
                f"[ep{ep:03d}] total={loss.item():.4e} "
                f"|ΔTr|={mean_abs:.3e} "
                f"W11={O['W11']:+.4f}/{T['W11']:+.4f} "
                f"W12={O['W12']:+.4f}/{T['W12']:+.4f} "
                f"W22={O['W22']:+.4f}/{T['W22']:+.4f} "
                f"χ22={O['chi2']:+.3f}/{T['chi2']:+.3f} "
            )
            if logger:
                logger.info(msg)
            else:
                print(msg)

    # ======================================================================
    # Evaluation after training
    # ======================================================================
    with torch.no_grad():
        U_pred, _ = model.forward()
        O = compute_pred_observables(U_pred, lat)

    # Gauge-aligned error
    ga_err = gauge_aligned_link_error(U_pred, teacher.links(), lat, steps=200, lr=0.05)

    # Average plaquette difference
    avg_pred, avg_true = O["avgP"], T["avgP"]
    avg_diff = (avg_pred - avg_true).abs().item()

    # Pack results
    total_time = time.time() - start_time
    results = {
        "ga_rmse": ga_err,
        "avgTrP_pred": float(avg_pred.real.item()),
        "avgTrP_true": float(avg_true.real.item()),
        "avgTrP_absdiff": avg_diff,
        "loss_curve": losses,
        "cfg": cfg.__dict__,
        "total_time": total_time,
    }

    # Save checkpoints (latest + best by ga_rmse)
    if getattr(cfg, "save_ckpt", None):
        os.makedirs(cfg.save_ckpt, exist_ok=True)

        # latest
        latest_path = os.path.join(cfg.save_ckpt, "latest.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "metrics": results}, latest_path)

        # best (ga_rmse)
        best_path = os.path.join(cfg.save_ckpt, "best.pt")
        if not os.path.exists(best_path):
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "metrics": results}, best_path)
        else:
            prev = torch.load(best_path, map_location="cpu")
            if results["ga_rmse"] < prev.get("metrics", {}).get("ga_rmse", float("inf")):
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "metrics": results}, best_path)

    # --- Final logging ---
    if logger:
        logger.info(f"Finished: ga_rmse={ga_err:.4e}, |Δ avgTrP|={avg_diff:.4e}")
        logger.close()

    if csvlogger:
        csvlogger.write({
            "ga_rmse": ga_err,
            "avgTrP_absdiff": avg_diff,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
        })

    return results


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    """
    EN: Build argument parser for CLI training run.
    JA: CLI用の引数パーサを構築する。
    """
    ap = argparse.ArgumentParser(
        description=(
            "Train discrete PINN for SU(2) lattice (LatticeBench). "
            "Teacher SU(2) field is randomized (optionally fixed by --teacher_seed)."
        )
    )

    # ---------------- Lattice & model ----------------
    ap.add_argument(
        "--L", type=int, default=4,
        help="EN: 2D lattice linear size (LxL). JA: 2次元格子の一辺のサイズ（L×L）。"
    )
    ap.add_argument(
        "--hidden", type=int, default=32,
        help="EN: Node-embedding dimension φ(x) size. JA: ノード埋め込み φ(x) の次元。"
    )

    # ---------------- Optimization ----------------
    ap.add_argument(
        "--epochs", type=int, default=800,
        help="EN: Number of training epochs. JA: 学習エポック数。"
    )
    ap.add_argument(
        "--lr", type=float, default=1e-2,
        help="EN: Learning rate for Adam. JA: Adam の学習率。"
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="EN: Global RNG seed for model/training. JA: モデル/学習用の乱数シード。"
    )
    ap.add_argument(
        "--print_every", type=int, default=25,
        help="EN: Interval of progress logging (epochs). JA: 進捗ログを出すエポック間隔。"
    )

    # ---------------- Loss weights ----------------
    ap.add_argument(
        "--w_plaq", type=float, default=0.0,
        help="EN: Weight for plaquette-trace vector MSE (complex). "
             "JA: プラークエット・トレース（複素）のベクトルMSEの重み。0なら監視のみ。"
    )
    ap.add_argument(
        "--w_wil11", type=float, default=0.10,
        help="EN: Weight for Wilson loop W(1×1). JA: W(1×1) の重み。"
    )
    ap.add_argument(
        "--w_wil12", type=float, default=0.28,
        help="EN: Weight for Wilson loop W(1×2). JA: W(1×2) の重み。"
    )
    ap.add_argument(
        "--w_wil22", type=float, default=0.20,
        help="EN: Weight for Wilson loop W(2×2). JA: W(2×2) の重み。"
    )
    ap.add_argument(
        "--w_wil13", type=float, default=0.22,
        help="EN: Weight for Wilson loop W(1×3). JA: W(1×3) の重み。"
    )
    ap.add_argument(
        "--w_wil23", type=float, default=0.18,
        help="EN: Weight for Wilson loop W(2×3). JA: W(2×3) の重み。"
    )
    ap.add_argument(
        "--w_cr", type=float, default=0.20,
        help="EN: Weight for Creutz ratio χ(2,2). JA: クルーツ比 χ(2,2) の重み。"
    )
    ap.add_argument(
        "--w_unitary", type=float, default=0.05,
        help="EN: Weight for soft unitarity (U†U≈I) and det=1 penalty. "
             "JA: ユニタリティ（U†U≈I）および det=1 のソフト罰則の重み。"
    )
    ap.add_argument(
        "--w_phi_smooth", type=float, default=0.06,
        help="EN: Weight for φ(x) neighbor-smoothness. JA: ノード埋め込み φ の近傍平滑化の重み。"
    )
    ap.add_argument(
        "--w_theta_smooth", type=float, default=0.03,
        help="EN: Weight for a(x,μ) (su(2) algebra) smoothness across neighbors. "
             "JA: a(x,μ) の近傍間平滑化の重み。"
    )
    ap.add_argument(
        "--w_phi_l2", type=float, default=0.003,
        help="EN: L2 regularization weight on φ embeddings. JA: φ 埋め込みのL2正則化の重み。"
    )

    # ---------------- Huber options ----------------
    ap.add_argument(
        "--use_huber", action="store_true",
        help="EN: Use Huber loss for Wilson loops & Creutz ratio instead of MSE. "
             "JA: Wilson/Creutz に MSE の代わりに Huber 損失を使う。"
    )
    ap.add_argument(
        "--huber_delta_wil", type=float, default=0.010,
        help="EN: Huber δ for Wilson-loop terms. JA: Wilson 項の Huber δ。"
    )
    ap.add_argument(
        "--huber_delta_cr", type=float, default=0.040,
        help="EN: Huber δ for Creutz-ratio term. JA: Creutz 項の Huber δ。"
    )

    # ---------------- Logging / Outputs ----------------
    ap.add_argument(
        "--logfile", type=str, default=None,
        help="EN: Path to append a human-readable text log. None = disable. "
             "JA: 人間可読なテキストログの出力先。未指定で無効化。"
    )
    ap.add_argument(
        "--csv", type=str, default=None,
        help="EN: Path to append a one-line CSV summary. None = disable. "
             "JA: 1行CSVサマリの追記先。未指定で無効化。"
    )

    # ---------------- Teacher field control ----------------
    ap.add_argument(
        "--teacher_seed", type=int, default=None,
        help="EN: Fix RNG for teacher SU(2) field. None = randomized per run. "
             "JA: 教師SU(2)場の乱数シード固定。未指定なら実行ごとにランダム生成。"
    )
    ap.add_argument(
        "--teacher_scale", type=float, default=0.6,
        help="EN: Std-dev of Normal(0,scale) for teacher algebra init. "
             "JA: 教師場 su(2) 代数初期化の標準偏差（正規分布 N(0,scale)）。"
    )

    # ---------------- Checkpointing ----------------
    ap.add_argument(
        "--save_ckpt", type=str, default=None,
        help="EN: Directory to save 'latest.pt' and the best-by-ga_rmse 'best.pt'. "
             "JA: 'latest.pt' と ga_rmse が最良の 'best.pt' を保存するディレクトリ。"
    )

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()
    cfg = TrainConfig(**vars(args))
    run_training(cfg, quiet=False)


if __name__ == "__main__":
    main()
