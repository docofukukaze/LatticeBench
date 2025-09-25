# src/optuna_search.py
"""
EN: Optuna search for LatticeBench hyperparameters with a modular structure:
    - Flexible CLI with EN/JA helps
    - Multi-objective optimization (NSGA-II + optional pruner)
    - Two-phase execution (Wide → Boost)
    - Per-trial artifact saving with SimpleLogger progress
    - Clean run directory + manifest + (optional) automatic decision

JA: モジュール構成の LatticeBench ハイパラ探索:
    - EN/JA 併記の柔軟な CLI
    - 2目的最適化（NSGA-II + 任意のプルーナー）
    - 2段階実行（Wide → Boost）
    - 各 trial 完了ごとの成果物保存と SimpleLogger による進捗出力
    - 実行フォルダ＋マニフェスト＋（任意）自動判定
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import json
import subprocess
import sys
from datetime import datetime

import optuna
import pandas as pd
import matplotlib.pyplot as plt
from optuna.samplers import NSGAIISampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

from .train import TrainConfig, run_training
from .utils import SimpleLogger, kv_str


# =============================================================================
# Small data holder
# =============================================================================
@dataclass
class RunContext:
    """
    EN: Small container for run-wide context (paths, ids, args, logger).
    JA: 実行全体の文脈（パス/ID/引数/ロガー）を持つ小さなコンテナ。
    """
    args: argparse.Namespace
    run_id: str
    outdir: Path
    logger: SimpleLogger


# =============================================================================
# Suggest helpers
# =============================================================================
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
    EN: Suggest a float parameter or return fixed/default value.
    JA: 浮動小数パラメータを提案、または固定/既定値を返す。
    """
    if fixed is not None:
        return float(fixed)
    if lo is not None and hi is not None:
        return trial.suggest_float(name, float(lo), float(hi), log=bool(log))
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
    EN: Suggest a categorical parameter or return fixed/default value.
    JA: カテゴリパラメータを提案、または固定/既定値を返す。
    """
    if fixed is not None:
        return fixed
    if choices:
        return trial.suggest_categorical(name, choices)
    if default_if_missing is not None:
        return default_if_missing
    raise ValueError(f"{name}: neither fixed nor choices provided.")


# =============================================================================
# Search-space builder
# =============================================================================
def build_search_space(
    trial: optuna.trial.Trial,
    A: argparse.Namespace,
    epochs: int,
    seed: int,
) -> TrainConfig:
    """
    EN: Construct TrainConfig from CLI + trial suggestions.
    JA: CLI と trial 提案から TrainConfig を構築。
    """
    lr = suggest_or_fixed_float(
        trial, "lr", A.lr, A.lr_min, A.lr_max, log=A.lr_log, default_if_missing=5e-3
    )
    use_huber = suggest_or_fixed_categorical(
        trial, "use_huber", A.use_huber, A.search_use_huber, default_if_missing=False
    )

    def S(name: str, *, log: bool = False, default: Optional[float] = None) -> float:
        return suggest_or_fixed_float(
            trial, name,
            fixed=getattr(A, name, None),
            lo=getattr(A, f"{name}_min", None),
            hi=getattr(A, f"{name}_max", None),
            log=log,
            default_if_missing=default,
        )

    cfg = TrainConfig(
        L=A.L, hidden=A.hidden, epochs=epochs, seed=seed, print_every=A.print_every,
        logfile=None, csv=None,
        lr=lr,
        w_plaq=S("w_plaq", default=0.04),
        w_wil11=S("w_wil11", default=0.10),
        w_wil12=S("w_wil12", default=0.28),
        w_wil22=S("w_wil22", default=0.20),
        w_wil13=S("w_wil13", default=0.22),
        w_wil23=S("w_wil23", default=0.18),
        w_cr=S("w_cr", default=0.20),
        w_unitary=S("w_unitary", default=0.06),
        w_phi_smooth=S("w_phi_smooth", default=0.04),
        w_theta_smooth=S("w_theta_smooth", default=0.02),
        w_phi_l2=S("w_phi_l2", log=True, default=3e-3),
        use_huber=use_huber,
        huber_delta_wil=S("huber_delta_wil", log=True, default=0.01),
        huber_delta_cr=S("huber_delta_cr", log=True, default=0.04),
    )
    # EN: Pass through teacher_seed if available.
    # JA: teacher_seed があれば透過的に渡す。
    if hasattr(cfg, "__dict__"):
        setattr(cfg, "teacher_seed", getattr(A, "teacher_seed", None))
    return cfg


def defaults_from_args(A: argparse.Namespace) -> Dict[str, Any]:
    """
    EN: Build a complete default-param dict from CLI args (fixed or fallback defaults).
    JA: CLI引数（固定指定があればそれ）と既定値から、全パラメータを埋めた辞書を作る。
    """
    def get(name: str, default: float) -> float:
        v = getattr(A, name, None)
        return float(v) if v is not None else float(default)

    return dict(
        lr=get("lr", 5e-3),
        w_plaq=get("w_plaq", 0.04),
        w_wil11=get("w_wil11", 0.10),
        w_wil12=get("w_wil12", 0.28),
        w_wil22=get("w_wil22", 0.20),
        w_wil13=get("w_wil13", 0.22),
        w_wil23=get("w_wil23", 0.18),
        w_cr=get("w_cr", 0.20),
        w_unitary=get("w_unitary", 0.06),
        w_phi_smooth=get("w_phi_smooth", 0.04),
        w_theta_smooth=get("w_theta_smooth", 0.02),
        w_phi_l2=get("w_phi_l2", 3e-3),
        use_huber=(getattr(A, "use_huber", None)
                   if getattr(A, "use_huber", None) is not None
                   else False),
        huber_delta_wil=get("huber_delta_wil", 0.01),
        huber_delta_cr=get("huber_delta_cr", 0.04),
    )


# =============================================================================
# Objective factory
# =============================================================================
def make_objective(
    A: argparse.Namespace,
    *,
    epochs: int,
    seed: int,
    phase: str,
    override_params: Optional[Dict[str, Any]] = None,
    parent_trial_id: Optional[int] = None,
) -> Callable[[optuna.trial.Trial], Tuple[float, float]]:
    """
    EN: Make an Optuna objective with optional param overrides (for Boost).
    JA: ブースト用の上書き設定にも対応した目的関数を作る。
    """
    def objective(trial: optuna.trial.Trial) -> Tuple[float, float]:
        if override_params is None:
            cfg = build_search_space(trial, A, epochs=epochs, seed=seed)
        else:
            # EN: Merge CLI defaults with override_params (from Pareto trial) to fill gaps.
            # JA: CLI由来の既定値に override_params を重ねて欠損を補完。
            base_defaults = defaults_from_args(A)
            merged = {**base_defaults, **override_params}
            cfg = TrainConfig(
                L=A.L, hidden=A.hidden, epochs=epochs, seed=seed, print_every=A.print_every,
                logfile=None, csv=None, **merged
            )
            if hasattr(cfg, "__dict__"):
                setattr(cfg, "teacher_seed", getattr(A, "teacher_seed", None))

        results = run_training(cfg, quiet=True)

        trial.set_user_attr("cfg", cfg.__dict__)
        trial.set_user_attr("ga_rmse", results["ga_rmse"])
        trial.set_user_attr("abs_dTrP", results["avgTrP_absdiff"])
        trial.set_user_attr("phase", phase)
        trial.set_user_attr("seed", seed)
        trial.set_user_attr("epochs", epochs)
        if parent_trial_id is not None:
            trial.set_user_attr("parent_trial_id", parent_trial_id)

        return results["ga_rmse"], results["avgTrP_absdiff"]

    return objective


# =============================================================================
# Study & runtime helpers
# =============================================================================
def create_study(A: argparse.Namespace) -> optuna.Study:
    """
    EN: Create or load an Optuna study with sampler/pruner.
    JA: サンプラー/プルーナー込みで Optuna スタディを作成/ロード。
    """
    sampler = NSGAIISampler(seed=A.teacher_seed or 0)
    pruner = None if A.no_pruner else MedianPruner(
        n_warmup_steps=max(1, int(A.epochs * max(0.0, min(1.0, A.pruner_warmup_ratio))))
    )
    return optuna.create_study(
        study_name=A.study,
        storage=A.storage,
        directions=["minimize", "minimize"],
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )


def select_boost_trials(
    study: optuna.Study,
    k_ratio: float,
) -> List[optuna.trial.FrozenTrial]:
    """
    EN: Select Boost candidates: start with Pareto, then fill by sum of objectives.
    JA: ブースト候補の選定：まず Pareto、足りなければ目的の和で補完。
    """
    complete = [tr for tr in study.get_trials(deepcopy=False)
                if tr.values is not None and tr.state == TrialState.COMPLETE]
    if not complete:
        return []
    k = max(1, int(len(complete) * max(0.0, min(1.0, k_ratio))))
    pareto = list(study.best_trials) if study.best_trials else []
    selected = pareto[:k]
    if len(selected) < k:
        remain = [tr for tr in complete if tr not in selected]
        remain.sort(key=lambda t: (t.values[0] + t.values[1]))
        selected += remain[: (k - len(selected))]
    return selected


# =============================================================================
# Artifacts (CSV/PNG/JSON)
# =============================================================================
def _filter_by_phase(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    if "phase" not in df.columns:
        return df
    return df[df["phase"] == phase].copy()


def save_artifacts(
    study: optuna.Study,
    outdir: Path,
    top_k: int,
    phase: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    EN: Save phase-specific all-trials CSV, Pareto CSV, scatter, and top-k JSON.
    JA: フェーズ毎の全試行CSV・Pareto CSV・散布図・Top-k JSON を保存。
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # collect all trials (with phase tagging)
    rows_all: List[Dict[str, Any]] = []
    for t in study.get_trials(deepcopy=False):
        if t.values is None:
            continue
        cfg = t.user_attrs.get("cfg", {})
        rows_all.append({
            "phase": t.user_attrs.get("phase", phase),
            "number": t.number,
            "ga_rmse": t.values[0],
            "abs_dTrP": t.values[1],
            "state": str(t.state),
            "parent_trial_id": t.user_attrs.get("parent_trial_id", None),
            "seed": t.user_attrs.get("seed", None),
            "epochs": t.user_attrs.get("epochs", None),
            **{f"cfg_{k}": v for k, v in cfg.items()},
            **{f"param_{k}": v for k, v in t.params.items()},
        })

    df_all_allphases = pd.DataFrame(rows_all)
    df_all = _filter_by_phase(df_all_allphases, phase).sort_values(
        ["ga_rmse", "abs_dTrP"], ascending=[True, True]
    )
    df_all.to_csv(outdir / f"all_trials_{phase}.csv", index=False)

    # Pareto front (study.best_trials are across all trials);
    # filter by phase after collecting their rows.
    rows_p: List[Dict[str, Any]] = []
    for t in study.best_trials or []:
        cfg = t.user_attrs.get("cfg", {})
        rows_p.append({
            "phase": t.user_attrs.get("phase", phase),
            "number": t.number,
            "ga_rmse": t.values[0],
            "abs_dTrP": t.values[1],
            "parent_trial_id": t.user_attrs.get("parent_trial_id", None),
            **{f"cfg_{k}": v for k, v in cfg.items()},
        })
    df_p_allphases = pd.DataFrame(rows_p)
    df_p = _filter_by_phase(df_p_allphases, phase).sort_values(
        ["ga_rmse", "abs_dTrP"], ascending=[True, True]
    )
    df_p.to_csv(outdir / f"pareto_trials_{phase}.csv", index=False)

    if not df_all.empty:
        df_all.sort_values("ga_rmse").head(top_k).to_json(outdir / f"topk_by_ga_rmse_{phase}.json", orient="records", indent=2)
        df_all.sort_values("abs_dTrP").head(top_k).to_json(outdir / f"topk_by_abs_dTrP_{phase}.json", orient="records", indent=2)

        plt.figure(figsize=(7.6, 6.2))
        plt.scatter(df_all["ga_rmse"], df_all["abs_dTrP"], s=16, alpha=0.35, label="All trials")
        if not df_p.empty:
            plt.plot(df_p["ga_rmse"], df_p["abs_dTrP"], "o-", label="Pareto front", ms=5)
        plt.xlabel("Gauge-aligned RMSE (lower is better)")
        plt.ylabel("|ΔTrP| (lower is better)")
        plt.title(f"Optuna search ({phase})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"pareto_scatter_{phase}.png", dpi=150)
        plt.close()

    return df_all, df_p


# =============================================================================
# Decision (moved from bash)
# =============================================================================
def make_decision_from_csvs(
    base_csv: Path,
    boost_csv: Path,
) -> Dict[str, Any]:
    """
    EN: Replicate the strict decision rule in Python.
    JA: bash の厳密な判定ロジックを Python 化。
    """
    if not base_csv.exists():
        return {"decision": "REJECT", "reason": "no_base_results"}
    if not boost_csv.exists():
        return {"decision": "REJECT", "reason": "no_boost_results"}

    base = pd.read_csv(base_csv)
    boost = pd.read_csv(boost_csv)
    if base.empty or boost.empty:
        return {"decision": "REJECT", "reason": "empty_results"}

    mb_g, mb_p = base["ga_rmse"].median(), base["abs_dTrP"].median()
    q25_g, q25_p = base["ga_rmse"].quantile(0.25), base["abs_dTrP"].quantile(0.25)

    eps = 1e-12
    def norm_to_base(col: str, df: pd.DataFrame) -> pd.Series:
        lo, hi = base[col].min(), base[col].max()
        return (df[col] - lo) / max(hi - lo, eps)

    b = boost.copy()
    b["ga_n"] = norm_to_base("ga_rmse", b)
    b["plaq_n"] = norm_to_base("abs_dTrP", b)
    b["prod_n"] = b["ga_n"] * b["plaq_n"]
    b["r2"] = b["ga_n"]**2 + b["plaq_n"]**2
    knee = b.sort_values("r2").iloc[0]

    # RULE A: Knee 30% improvement on both
    ruleA = (knee["ga_rmse"] <= 0.7 * mb_g) and (knee["abs_dTrP"] <= 0.7 * mb_p)

    # RULE B: Both under base Q25
    ruleB = ((boost["ga_rmse"] <= q25_g) & (boost["abs_dTrP"] <= q25_p)).any()

    # RULE C: Product improvement vs. base median of normalized product
    bb = base.copy()
    bb["ga_n"] = norm_to_base("ga_rmse", bb)
    bb["plaq_n"] = norm_to_base("abs_dTrP", bb)
    median_prod_base = (bb["ga_n"] * bb["plaq_n"]).median()
    ruleC = (b["prod_n"].min() <= 0.5 * median_prod_base)

    decision = "ACCEPT" if (ruleA or ruleB or ruleC) else "REJECT"
    return {
        "knee": {"ga_rmse": float(knee["ga_rmse"]), "abs_dTrP": float(knee["abs_dTrP"])},
        "base_stats": {
            "median_ga_rmse": float(mb_g),
            "median_abs_dTrP": float(mb_p),
            "q25_ga_rmse": float(q25_g),
            "q25_abs_dTrP": float(q25_p),
            "median_prod_n_base": float(median_prod_base),
        },
        "rules": {"knee_30pct_both": bool(ruleA), "both_Q25": bool(ruleB), "prod_gain_50pct": bool(ruleC)},
        "decision": decision,
    }


def write_decision(outdir: Path, logger: SimpleLogger) -> Dict[str, Any]:
    """
    EN: Run decision using CSVs in outdir and write decision.json (+ markers).
    JA: 出力フォルダ内のCSVから判定を実行し、decision.json とマーカーを出力。
    """
    basep = outdir / "all_trials_base.csv"
    boostp = outdir / "all_trials_boost.csv"
    result = make_decision_from_csvs(basep, boostp)
    (outdir / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    marker = "_ACCEPT" if result.get("decision") == "ACCEPT" else "_REJECT"
    (outdir / marker).touch()
    logger.info(f"[DECISION] {result.get('decision')}  → saved: {outdir/'decision.json'}  marker: {marker}")
    return result


# =============================================================================
# Run directory & manifest
# =============================================================================
def build_run_dir_and_manifest(A: argparse.Namespace) -> RunContext:
    """
    EN: Create run directory with timestamped ID, manifest, and a run-scoped logger.
    JA: タイムスタンプIDの実行フォルダとマニフェスト、ラン専用ロガーを作成。
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}__{A.study}__tr{A.trials}__ep{A.epochs}" + (f"__b{A.boost_to}" if A.boost_to else "")
    outdir = Path(A.artifacts_dir)
    if not str(outdir).endswith(run_id):
        outdir = outdir / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    def _sh(x: str) -> Optional[str]:
        try:
            return subprocess.check_output(x, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    manifest = {
        "run_id": run_id,
        "timestamp": ts,
        "cmdline": "python -m src.optuna_search " + " ".join(map(str, sys.argv[1:])),
        "args": vars(A),
        "git": {
            "rev": _sh("git rev-parse --short=12 HEAD"),
            "status_dirty": bool(_sh("git diff --quiet || echo DIRTY")),
            "branch": _sh("git rev-parse --abbrev-ref HEAD"),
        },
        "env": {"python": sys.version.split()[0]},
    }
    try:
        import torch
        manifest["env"].update({
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        })
    except Exception:
        pass

    (outdir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # mutate artifacts_dir so downstream uses the nested one; also announce RUN_DIR
    A.artifacts_dir = str(outdir)
    logger = SimpleLogger(str(outdir / "optuna.log"), mirror_stdout=True, append=True)
    logger.info(f"[RUN_DIR] {outdir.resolve()}")
    return RunContext(args=A, run_id=run_id, outdir=outdir, logger=logger)


# =============================================================================
# CLI parser (EN/JA help for all args)
# =============================================================================
def parse_args() -> argparse.Namespace:
    """
    EN: Parse CLI arguments with EN/JA help messages.
    JA: EN/JA 併記のヘルプメッセージ付きで CLI 引数を解析。
    """
    ap = argparse.ArgumentParser(
        description=(
            "EN: Optuna search over LatticeBench hyperparameters (Wide → Boost).\n"
            "JA: LatticeBench のハイパラ探索（Wide → Boost 対応）。"
        )
    )

    # Base
    ap.add_argument("--L", type=int, default=4, help="EN: Lattice size. JA: 格子サイズ。")
    ap.add_argument("--hidden", type=int, default=32, help="EN: Hidden dim. JA: 隠れ次元。")
    ap.add_argument("--epochs", type=int, default=900, help="EN: Wide epochs. JA: Wide のエポック数。")
    ap.add_argument("--seed", type=int, default=42, help="EN: Training seed. JA: 学習シード。")
    ap.add_argument("--teacher_seed", type=int, default=None, help="EN: Fixed teacher seed. JA: 教師シード固定。")
    ap.add_argument("--print_every", type=int, default=200, help="EN: Log interval (epochs). JA: ログ間隔（エポック）。")

    # Optuna runtime
    ap.add_argument("--trials", type=int, default=120, help="EN: #trials. JA: 試行数。")
    ap.add_argument("--storage", type=str, default=None, help="EN: Storage URL. JA: ストレージURL。")
    ap.add_argument("--study", type=str, default="latticebench", help="EN: Study name. JA: スタディ名。")
    ap.add_argument("--n_jobs", type=int, default=1, help="EN: Parallel jobs. JA: 並列ジョブ数。")
    ap.add_argument("--artifacts_dir", type=str, default="runs/optuna", help="EN: Artifacts base. JA: 成果物ベース。")
    ap.add_argument("--topk", type=int, default=5, help="EN: Top-k per objective. JA: 目的ごとの上位件数。")

    # Sampler / Pruner
    ap.add_argument("--no_pruner", action="store_true", help="EN: Disable pruner. JA: プルーナー無効。")
    ap.add_argument("--pruner_warmup_ratio", type=float, default=0.45, help="EN: Warmup ratio. JA: ウォームアップ比。")

    # Two-phase boost
    ap.add_argument("--boost_to", type=int, default=None,
                    help="EN: If set and > --epochs, re-run top trials up to this epochs. JA: --epochs より大きければ上位試行を再実行。")
    ap.add_argument("--boost_topk_ratio", type=float, default=0.2, help="EN: Ratio of trials to boost. JA: ブースト割合。")
    ap.add_argument("--boost_seed", type=int, default=None, help="EN: Seed for boost. JA: ブースト時シード。")

    # Learning rate
    ap.add_argument("--lr", type=float, default=None, help="EN: Fixed LR. JA: 学習率固定。")
    ap.add_argument("--lr_min", type=float, default=None, help="EN: LR min. JA: 学習率下限。")
    ap.add_argument("--lr_max", type=float, default=None, help="EN: LR max. JA: 学習率上限。")
    ap.add_argument("--lr_log", action="store_true", help="EN: Log-scale LR. JA: 対数スケール探索。")

    # Huber usage
    ap.add_argument("--use_huber", type=lambda s: str(s).lower() in {"1","true","t","yes","y"}, default=None,
                    help="EN: Fixed Huber usage. JA: Huber 使用の固定。")
    ap.add_argument("--search_use_huber", nargs="*", default=None,
                    help="EN: Choices for Huber usage (e.g., True False). JA: Huber 使用のカテゴリ探索。")

    # Float params (fixed or range)
    def add_float_param(name: str, en: str, ja: str):
        ap.add_argument(f"--{name}", type=float, default=None, help=f"EN: {en} JA: {ja}")
        ap.add_argument(f"--{name}_min", type=float, default=None, help=f"EN: Min for {name}. JA: {name} の下限。")
        ap.add_argument(f"--{name}_max", type=float, default=None, help=f"EN: Max for {name}. JA: {name} の上限。")

    add_float_param("w_plaq", "Weight for plaquette.", "プラーケット項の重み。")
    for p in ["w_wil11", "w_wil12", "w_wil22", "w_wil13", "w_wil23"]:
        add_float_param(p, f"Weight for {p} (Wilson/Wilson×).", f"{p}（Wilson/Wilson×）の重み。")
    add_float_param("w_cr", "Weight for Creutz ratio.", "Creutz 比の重み。")
    add_float_param("w_unitary", "Weight for unitarity reg.", "ユニタリティ正則化の重み。")
    add_float_param("w_phi_smooth", "Weight for phi smooth.", "φ 平滑化の重み。")
    add_float_param("w_theta_smooth", "Weight for theta smooth.", "θ 平滑化の重み。")
    add_float_param("w_phi_l2", "Weight for phi L2.", "φ の L2 正則化重み。")
    add_float_param("huber_delta_wil", "Huber δ for Wilson.", "Wilson 用 Huber δ。")
    add_float_param("huber_delta_cr", "Huber δ for Creutz.", "Creutz 用 Huber δ。")

    # Decision options
    ap.add_argument("--auto_decide", action="store_true",
                    help="EN: Write decision.json and markers at the end. JA: 終了時に decision.json とマーカーを書き出す。")

    A = ap.parse_args()

    # Normalize --search_use_huber
    if A.search_use_huber:
        A.search_use_huber = [str(s).lower() in {"1","true","t","yes","y"} for s in A.search_use_huber]
    else:
        A.search_use_huber = None

    return A


# =============================================================================
# Orchestrator (per-trial saving with callbacks)
# =============================================================================
def _make_trial_callback(
    ctx: RunContext,
    phase: str,
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """
    EN: Callback to run after each trial — save artifacts for the given phase and log progress.
    JA: 各 trial 完了後に呼ばれるコールバック ― 指定フェーズの成果物保存と進捗ログを行う。
    """
    # local counter for phase progress
    state = {"done": 0}
    total_hint = ctx.args.trials if phase == "base" else None  # set later for boost

    def cb(study: optuna.Study, frozen: optuna.trial.FrozenTrial):
        if frozen.state != TrialState.COMPLETE:
            return
        # save artifacts per trial
        save_artifacts(study, ctx.outdir, top_k=ctx.args.topk, phase=phase)

        # values
        vals = dict(ga_rmse=f"{frozen.values[0]:.6f}",
                    abs_dTrP=f"{frozen.values[1]:.6f}")

        # prefer sampler params; if empty (boost), fall back to cfg in user_attrs
        params = dict(frozen.params or {})
        if not params:
            cfg = frozen.user_attrs.get("cfg", {})  # set in make_objective()
            # keep only user-facing HPs
            keep = ["lr","use_huber","w_plaq","w_unitary","w_phi_smooth",
                    "w_theta_smooth","w_phi_l2","huber_delta_wil","huber_delta_cr",
                    "w_wil11","w_wil12","w_wil22","w_wil13","w_wil23","w_cr"]
            params = {k: cfg.get(k) for k in keep if k in cfg}

        # pretty order
        order = ["lr","use_huber","w_plaq","w_unitary","w_phi_smooth","w_theta_smooth",
                 "w_phi_l2","w_wil11","w_wil12","w_wil22","w_wil13","w_wil23",
                 "w_cr","huber_delta_wil","huber_delta_cr"]
        pick = {k: params[k] for k in order if k in params}

        # line
        if phase == "base":
            head = f"[{phase} {cb.done}/{ctx.args.trials}]"
        else:
            head = f"[{phase} {cb.done}/{total_hint or '?'}]"
        ctx.logger.info(f"{head} trial={frozen.number}  {kv_str(vals)}  :: {kv_str(pick)}")

        cb.done += 1
    cb.done = 1

    # attach a way to set total for boost
    cb.set_total = lambda n: None  # type: ignore[attr-defined]
    if phase == "boost":
        def set_total(n: int) -> None:
            nonlocal total_hint
            total_hint = int(n)
        cb.set_total = set_total  # type: ignore[attr-defined]

    return cb


def run_phase(
    study: optuna.Study,
    objective: Callable[[optuna.trial.Trial], Tuple[float, float]],
    n_trials: int,
    n_jobs: int,
    callback: Callable[[optuna.Study, optuna.trial.FrozenTrial], None],
) -> None:
    """
    EN: Run a phase (base or boost) with given objective and per-trial callback.
    JA: 指定目的関数でフェーズ（base/boost）を実行。各 trial ごとにコールバックを呼ぶ。
    """
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[callback])


def orchestrate(A: argparse.Namespace) -> None:
    """
    EN: High-level orchestration of Base → Boost inside an isolated run directory with logger.
    JA: 個別ランディレクトリ＆ロガーのもとで Base → Boost を高レベル実行。
    """
    ctx = build_run_dir_and_manifest(A)
    study = create_study(ctx.args)

    # ---- Base (Wide) ----
    ctx.logger.info(f"[INFO] Wide phase started. (trials={ctx.args.trials}, epochs={ctx.args.epochs})")
    base_obj = make_objective(
        ctx.args, epochs=ctx.args.epochs, seed=ctx.args.seed, phase="base",
        override_params=None, parent_trial_id=None
    )
    base_cb = _make_trial_callback(ctx, phase="base")
    run_phase(study, base_obj, n_trials=ctx.args.trials, n_jobs=ctx.args.n_jobs, callback=base_cb)
    save_artifacts(study, ctx.outdir, top_k=ctx.args.topk, phase="base")
    ctx.logger.info(f"[INFO] Wide phase finished. Artifacts under: {ctx.outdir}")

    # ---- Boost ----
    if ctx.args.boost_to and ctx.args.boost_to > ctx.args.epochs:
        candidates = select_boost_trials(study, ctx.args.boost_topk_ratio)
        n = len(candidates)
        if n == 0:
            ctx.logger.info("[WARN] No candidates for boost; skipping.")
        else:
            boost_seed = ctx.args.boost_seed if ctx.args.boost_seed is not None else ctx.args.seed
            ctx.logger.info(f"[INFO] Boost phase started. (candidates={n}, target_epochs={ctx.args.boost_to}, seed={boost_seed})")
            boost_cb = _make_trial_callback(ctx, phase="boost")
            boost_cb.set_total(n)  # type: ignore[attr-defined]

            for i, tr in enumerate(candidates, 1):
                override = dict(tr.params)
                # Respect CLI fixed overrides
                for name in [
                    "lr", "w_plaq", "w_wil11","w_wil12","w_wil22","w_wil13","w_wil23",
                    "w_cr","w_unitary","w_phi_smooth","w_theta_smooth","w_phi_l2",
                    "huber_delta_wil","huber_delta_cr","use_huber",
                ]:
                    v = getattr(ctx.args, name, None)
                    if v is not None:
                        override[name] = v

                ctx.logger.info(f"[boost {i}/{n}] parent={tr.number} -> epochs={ctx.args.boost_to}")
                boost_obj = make_objective(
                    ctx.args, epochs=ctx.args.boost_to, seed=boost_seed, phase="boost",
                    override_params=override, parent_trial_id=tr.number
                )
                run_phase(study, boost_obj, n_trials=1, n_jobs=1, callback=boost_cb)

            # final snapshot
            save_artifacts(study, ctx.outdir, top_k=ctx.args.topk, phase="boost")
            ctx.logger.info(f"[INFO] Boost phase finished. Artifacts under: {ctx.outdir}")

    # ---- Decision (optional) ----
    if ctx.args.auto_decide:
        result = write_decision(ctx.outdir, ctx.logger)
        ctx.logger.info(json.dumps(result, indent=2))

    ctx.logger.info(f"[DONE] Artifacts directory: {ctx.outdir.resolve()}")


# =============================================================================
# Main
# =============================================================================
def main():
    """
    EN: CLI entry point — parse args and orchestrate.
    JA: CLI エントリーポイント — 引数解析して実行をオーケストレーション。
    """
    A = parse_args()
    orchestrate(A)


if __name__ == "__main__":
    main()
