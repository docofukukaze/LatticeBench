"""
Optuna search for LatticeBench hyperparameters
==============================================

特徴:
- ほぼ全ハイパラを CLI から「固定 or 探索レンジ」で指定可能
- 対数スケール探索 (--*_log) やカテゴリ選択 (use_huber) に対応
- 途中再開 (RDB storage) / 並列 (--n_jobs) / スタディ名指定
- すべての試行とパレート前線を CSV/PNG/JSON に保存

使い方（例）:
  # 1) 代表的な探索（学習率・各重みなど）
  python -m src.optuna_search \
    --trials 120 --epochs 900 --study latticebench-wide \
    --lr_min 3e-3 --lr_max 8e-3 --lr_log \
    --w_plaq_min 0.0 --w_plaq_max 0.20 \
    --w_wil12_min 0.20 --w_wil12_max 0.70 \
    --w_cr_min 0.30 --w_cr_max 0.80 \
    --huber_delta_wil_min 0.001 --huber_delta_wil_max 0.02 \
    --huber_delta_cr_min  0.002 --huber_delta_cr_max  0.05 \
    --search_use_huber True False \
    --n_jobs 1

  # 2) 一部は固定、一部は探索（w_wil11等を固定）
  python -m src.optuna_search \
    --trials 80 --epochs 900 --study latticebench-fixsome \
    --lr_min 3e-3 --lr_max 6e-3 --lr_log \
    --w_plaq_min 0.0 --w_plaq_max 0.10 \
    --w_wil11 0.10 --w_wil22 0.28 --w_wil13 0.22 --w_wil23 0.18 \
    --w_wil12_min 0.25 --w_wil12_max 0.60 \
    --w_cr_min 0.40 --w_cr_max 0.75 \
    --search_use_huber True False

  # 3) 多数の正則化重みも探索に含める
  python -m src.optuna_search \
    --trials 150 --epochs 900 --study latticebench-regs \
    --lr_min 3e-3 --lr_max 8e-3 --lr_log \
    --w_unitary_min 0.03 --w_unitary_max 0.12 \
    --w_phi_smooth_min 0.02 --w_phi_smooth_max 0.08 \
    --w_theta_smooth_min 0.01 --w_theta_smooth_max 0.06 \
    --w_phi_l2_min 0.0005 --w_phi_l2_max 0.01 \
    --search_use_huber True False
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import pandas as pd
import matplotlib.pyplot as plt

from .train import TrainConfig, run_training


# -----------------------------------------------------------------------------
# 汎用サジェストヘルパ
# -----------------------------------------------------------------------------
def suggest_or_fixed_float(
    trial: optuna.trial.Trial,
    name: str,
    fixed: Optional[float],
    lo: Optional[float],
    hi: Optional[float],
    log: bool = False,
    default_if_missing: Optional[float] = None,
) -> float:
    """
    fixed が与えられれば固定値を返す。
    lo/hi が両方与えられれば suggest_float。
    どちらも無ければ default_if_missing を返す（Noneの場合はエラー）。
    """
    if fixed is not None:
        return float(fixed)
    if lo is not None and hi is not None:
        if log:
            return trial.suggest_float(name, float(lo), float(hi), log=True)
        return trial.suggest_float(name, float(lo), float(hi))
    if default_if_missing is not None:
        return float(default_if_missing)
    raise ValueError(f"{name}: neither fixed nor (lo,hi) provided.")


def suggest_or_fixed_categorical(
    trial: optuna.trial.Trial,
    name: str,
    fixed: Optional[Any],
    choices: Optional[List[Any]],
    default_if_missing: Optional[Any] = None,
) -> Any:
    """
    fixed が与えられれば固定値、なければ choices から suggest_categorical。
    両方無ければ default_if_missing。なければエラー。
    """
    if fixed is not None:
        return fixed
    if choices:
        return trial.suggest_categorical(name, choices)
    if default_if_missing is not None:
        return default_if_missing
    raise ValueError(f"{name}: neither fixed nor choices provided.")


# ---------------------------------------------------------------------
# 検索空間の構築（CLIの fixed / min-max / log 指定を反映）
# ---------------------------------------------------------------------
def build_search_space(trial: optuna.trial.Trial, A: argparse.Namespace) -> TrainConfig:
    # --- 学習率 ---
    lr = suggest_or_fixed_float(
        trial, "lr",
        fixed=A.lr,
        lo=A.lr_min, hi=A.lr_max,
        log=bool(A.lr_log),
        default_if_missing=1e-2,  # どれも未指定なら既定値
    )

    # --- Huber 使用（固定 or カテゴリ探索）---
    use_huber = suggest_or_fixed_categorical(
        trial, "use_huber",
        fixed=A.use_huber,
        choices=A.search_use_huber,      # 例: [True, False]
        default_if_missing=False,
    )

    # まとめ関数: 各パラに対して fixed / min-max を使い分け
    def S(name: str, *, log: bool = False, default: Optional[float] = None) -> float:
        return suggest_or_fixed_float(
            trial, name,
            fixed=getattr(A, name, None),
            lo=getattr(A, f"{name}_min", None),
            hi=getattr(A, f"{name}_max", None),
            log=log,
            default_if_missing=default,
        )

    return TrainConfig(
        # --- ベース設定（固定値） ---
        L=getattr(A, "L", 4),
        hidden=getattr(A, "hidden", 32),
        epochs=A.epochs,
        seed=getattr(A, "seed", 42),
        print_every=getattr(A, "print_every", 200),
        logfile=None, csv=None,

        # --- 探索パラ ---
        lr=lr,
        w_plaq=S("w_plaq", default=0.0),
        w_wil11=S("w_wil11", default=0.10),
        w_wil12=S("w_wil12", default=0.28),
        w_wil22=S("w_wil22", default=0.20),
        w_wil13=S("w_wil13", default=0.22),
        w_wil23=S("w_wil23", default=0.18),
        w_cr=S("w_cr", default=0.20),

        w_unitary=S("w_unitary", default=0.05),
        w_phi_smooth=S("w_phi_smooth", default=0.06),
        w_theta_smooth=S("w_theta_smooth", default=0.03),
        w_phi_l2=S("w_phi_l2", log=True, default=3e-3),

        use_huber=use_huber,
        huber_delta_wil=S("huber_delta_wil", log=True, default=1e-2),
        huber_delta_cr=S("huber_delta_cr",  log=True, default=4e-2),

        # 教師場の強さも（必要なら）探索
        teacher_scale=S("teacher_scale", default=0.6) if hasattr(A, "teacher_scale") or hasattr(A, "teacher_scale_min") else 0.6,
    )


# -----------------------------------------------------------------------------
# 可視化と成果物の保存
# -----------------------------------------------------------------------------
def save_artifacts(study: optuna.Study, outdir: Path, top_k: int = 5) -> None:
    outdir.mkdir(exist_ok=True)

    # --- すべての trial を表形式で保存 ---
    rows_all: List[Dict[str, Any]] = []
    for t in study.trials:
        if t.values is None:
            continue
        cfg = t.user_attrs.get("cfg", {})
        rows_all.append({
            "number": t.number,
            "ga_rmse": t.values[0],
            "avgTrP_absdiff": t.values[1],
            "state": str(t.state),
            **{f"cfg_{k}": v for k, v in cfg.items()},
            **{f"param_{k}": v for k, v in t.params.items()},
        })
    df_all = pd.DataFrame(rows_all)
    df_all.to_csv(outdir / "all_trials.csv", index=False)

    # --- パレート前線だけ保存 ---
    pareto = study.best_trials
    rows_p: List[Dict[str, Any]] = []
    for t in pareto:
        cfg = t.user_attrs.get("cfg", {})
        rows_p.append({
            "number": t.number,
            "ga_rmse": t.values[0],
            "avgTrP_absdiff": t.values[1],
            **{f"cfg_{k}": v for k, v in cfg.items()},
        })
    df_p = pd.DataFrame(rows_p).sort_values(["ga_rmse", "avgTrP_absdiff"], ascending=[True, True])
    df_p.to_csv(outdir / "pareto_trials.csv", index=False)

    # --- Top-k by objectives ---
    if not df_all.empty:
        df_all.sort_values("ga_rmse", ascending=True).head(top_k).to_json(outdir / "topk_by_ga_rmse.json", orient="records", indent=2)
        df_all.sort_values("avgTrP_absdiff", ascending=True).head(top_k).to_json(outdir / "topk_by_dAvgTrP.json", orient="records", indent=2)

    # --- 散布図（パレートを強調） ---
    if not df_all.empty:
        plt.figure(figsize=(7.5, 6))
        plt.scatter(df_all["ga_rmse"], df_all["avgTrP_absdiff"], s=16, alpha=0.3, label="All trials")
        if not df_p.empty:
            plt.plot(df_p["ga_rmse"], df_p["avgTrP_absdiff"], "o-", label="Pareto front", ms=5)
        plt.xlabel("Gauge-aligned RMSE (lower is better)")
        plt.ylabel("|Δ avgTrP| (lower is better)")
        plt.title("Optuna search: ga_rmse vs |Δ avgTrP|")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "pareto_scatter.png", dpi=150)
        plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Optuna search over LatticeBench hyperparameters")

    # ベース構成（固定）
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print_every", type=int, default=200)

    # Optuna 実行設定
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--storage", type=str, default=None, help="e.g., sqlite:///optuna.db")
    ap.add_argument("--study", type=str, default="latticebench")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--artifacts_dir", type=str, default="optuna_artifacts")
    ap.add_argument("--topk", type=int, default=5)

    # ======== 探索 or 固定: 学習率 ========
    ap.add_argument("--lr", type=float, default=None, help="固定学習率。指定があればレンジ無視")
    ap.add_argument("--lr_min", type=float, default=None)
    ap.add_argument("--lr_max", type=float, default=None)
    ap.add_argument("--lr_log", action="store_true", help="学習率を対数スケールで探索")

    # ======== 探索 or 固定: Huber 使用 ========
    ap.add_argument("--use_huber", type=lambda s: s.lower() in {"1","true","t","yes","y"}, default=None,
                    help="固定: True/False。指定が無ければ --search_use_huber でカテゴリ探索")
    ap.add_argument("--search_use_huber", nargs="*", default=None,
                    help="例: --search_use_huber True False")

    # ======== 探索 or 固定: 個別重み（固定値 or レンジ） ========
    def add_float_param(name, default=None):
        ap.add_argument(f"--{name}", type=float, default=default)
        ap.add_argument(f"--{name}_min", type=float, default=None)
        ap.add_argument(f"--{name}_max", type=float, default=None)

    for p in [
        "w_plaq",
        "w_wil11", "w_wil12", "w_wil22", "w_wil13", "w_wil23",
        "w_cr",
        "w_unitary", "w_phi_smooth", "w_theta_smooth", "w_phi_l2",
        "huber_delta_wil", "huber_delta_cr",
    ]:
        add_float_param(p)

    A = ap.parse_args()

    # 文字列 True/False を bool に変換（--search_use_huber）
    if A.search_use_huber is not None:
        parsed_choices: List[bool] = []
        for s in A.search_use_huber:
            if isinstance(s, bool):
                parsed_choices.append(s)
            else:
                parsed_choices.append(str(s).lower() in {"1","true","t","yes","y"})
        A.search_use_huber = parsed_choices

    # スタディ作成
    study = optuna.create_study(
        study_name=A.study,
        storage=A.storage,
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )

    # 目的関数
    def objective(trial):
        cfg = build_search_space(trial, A)  # ← trial を渡す
        print(f"[trial {trial.number}] params={trial.params}")
        
        results = run_training(cfg, quiet=True)
        # 便利な情報を残す
        trial.set_user_attr("cfg", cfg.__dict__)
        trial.set_user_attr("ga_rmse", results["ga_rmse"])
        trial.set_user_attr("avgTrP_absdiff", results["avgTrP_absdiff"])
        # 2目的最小化
        # （create_studyで directions=["minimize","minimize"] を指定済みであること）
        return results["ga_rmse"], results["avgTrP_absdiff"]

    # 最適化
    study.optimize(objective, n_trials=A.trials, n_jobs=A.n_jobs)

    # 成果物出力
    outdir = Path(A.artifacts_dir)
    save_artifacts(study, outdir, top_k=A.topk)
    print(f"[INFO] Artifacts saved to: {outdir}")


if __name__ == "__main__":
    main()
