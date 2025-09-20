# src/optuna_search.py
# -----------------------------------------------------------------------------
# Multi-objective hyperparameter search with Optuna (NSGA-II).
# Objectives (minimize both):
#   1) GA-RMSE            -> "ga_rmse"
#   2) |Δ avgTrP|         -> "avgTrP_absdiff"
#
# Each trial calls your existing CLI:
#   python -m src.train ... | tee runs/OPT_*.log
# Then parses metrics via:
#   python -m src.analyze_log <log> --out <csv>
#
# Example usage (overnight):
#   pip install optuna
#   python -m src.optuna_search --trials 100 --epochs 900 \
#       --storage sqlite:///optuna.db --study latticebench-multiobj --n_jobs 1
#
# Tips:
# - For a single GPU/CPU box, keep --n_jobs=1 (subprocess per trial).
# - You can resume the same study later with the same --storage/--study.
# - If you want a quicker warm-up search, use --epochs 300 first.
# -----------------------------------------------------------------------------

import argparse
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import optuna
from optuna.samplers import NSGAIISampler

METRIC_GA = "ga_rmse"
METRIC_PL = "avgTrP_absdiff"


def _run_cmd(cmd: str) -> int:
    print(f"[CMD] {cmd}")
    return subprocess.call(cmd, shell=True)


def run_train_and_parse(args, log_path: Path):
    """Run one training config and parse metrics via analyze_log."""
    base = [
        "python", "-m", "src.train",
        "--epochs", str(args.epochs),
        "--print_every", str(max(50, args.epochs // 10)),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--w_wil11", "0.10",
        "--w_wil22", "0.28",
        "--w_wil13", "0.22",
        "--w_wil23", "0.18",
        "--w_cr", str(args.w_cr),
        "--w_plaq", str(args.w_plaq),
        "--w_unitary", "0.06",
        "--w_phi_smooth", "0.04",
        "--w_theta_smooth", "0.02",
        "--w_phi_l2", "0.002",
        "--w_wil12", str(args.w_wil12),
    ]

    if args.use_huber:
        base += [
            "--use_huber",
            "--huber_delta_wil", str(args.huber_delta_wil),
            "--huber_delta_cr", str(args.huber_delta_cr),
        ]

    Path("runs").mkdir(exist_ok=True)
    cmd = " ".join(shlex.quote(s) for s in base) + f" | tee {shlex.quote(str(log_path))}"
    rc = _run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"Training failed with exit code {rc}")

    out_csv = log_path.with_suffix(".csv")
    cmd2 = f"python -m src.analyze_log {shlex.quote(str(log_path))} --out {shlex.quote(str(out_csv))}"
    rc2 = _run_cmd(cmd2)
    if rc2 != 0 or not out_csv.exists():
        raise RuntimeError("analyze_log failed to produce a CSV.")

    import pandas as pd
    df = pd.read_csv(out_csv)
    if METRIC_GA not in df.columns or METRIC_PL not in df.columns:
        raise RuntimeError(f"CSV missing required columns: {METRIC_GA}, {METRIC_PL}")
    row = df.iloc[0].to_dict()
    return float(row[METRIC_GA]), float(row[METRIC_PL]), row  # (ga_rmse, avgTrP_absdiff, all_fields)


def build_trial_args(trial, base_epochs: int):
    """Sample hyperparameters from widened ranges and package as a simple object."""
    class Args:
        pass

    a = Args()
    a.epochs = base_epochs

    # Learning rate (log scale)
    a.lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)

    # Loss weights
    a.w_plaq = trial.suggest_float("w_plaq", 0.005, 0.08, log=True)
    a.w_wil12 = trial.suggest_float("w_wil12", 0.24, 0.72)
    a.w_cr = trial.suggest_float("w_cr", 0.35, 0.80)

    # Seed
    a.seed = trial.suggest_categorical("seed", [42, 777])

    # Loss type: MSE vs Huber (+ deltas)
    a.use_huber = trial.suggest_categorical("use_huber", [False, True])
    if a.use_huber:
        a.huber_delta_wil = trial.suggest_float("huber_delta_wil", 0.0015, 0.006, log=True)
        a.huber_delta_cr = trial.suggest_float("huber_delta_cr", 0.004, 0.016, log=True)
    else:
        a.huber_delta_wil = None
        a.huber_delta_cr = None

    return a


def objective_multi(trial: optuna.Trial, base_epochs: int, tag: str):
    targs = build_trial_args(trial, base_epochs)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_name = f"runs/OPT_{tag}_t{trial.number}_{ts}.log"
    ga, pl, fields = run_train_and_parse(targs, Path(log_name))

    trial.set_user_attr("log_path", log_name)
    trial.set_user_attr("metrics", fields)

    return ga, pl  # two objectives (minimize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100, help="Number of trials")
    ap.add_argument("--epochs", type=int, default=900, help="Epochs per trial")
    ap.add_argument("--timeout", type=int, default=None, help="Seconds to stop after (optional)")
    ap.add_argument("--storage", type=str, default=None, help='Optuna storage, e.g., \"sqlite:///optuna.db\"')
    ap.add_argument("--study", type=str, default="latticebench-multiobj", help="Study name")
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel workers (>=2 requires RDB storage)")
    args = ap.parse_args()

    sampler = NSGAIISampler(seed=2025)
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
        study_name=args.study,
        storage=args.storage,
        load_if_exists=True,
    )

    print(f"[INFO] Starting study '{args.study}' (trials={args.trials}, epochs={args.epochs}, n_jobs={args.n_jobs})")
    study.optimize(lambda tr: objective_multi(tr, args.epochs, tag=args.study),
                   n_trials=args.trials, timeout=args.timeout, n_jobs=args.n_jobs, gc_after_trial=True)

    # Save Pareto trials
    os.makedirs("optuna_artifacts", exist_ok=True)
    pareto = [t for t in study.best_trials]  # non-dominated set
    import pandas as pd
    rows = []
    for t in pareto:
        row = {"trial": t.number, METRIC_GA: t.values[0], METRIC_PL: t.values[1]}
        row.update(t.params)
        row["log_path"] = t.user_attrs.get("log_path", "")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values([METRIC_GA, METRIC_PL])
    csv_path = "optuna_artifacts/pareto_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Pareto trials saved -> {csv_path}")

    # Plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(df[METRIC_PL], df[METRIC_GA], s=20, alpha=0.8)
    plt.xlabel("|Δ avgTrP|")
    plt.ylabel("GA-RMSE")
    plt.title("Optuna Pareto Trials")
    plt.grid(True, linestyle="--", alpha=0.3)
    plot_path = "optuna_artifacts/pareto_trials.png"
    Path("optuna_artifacts").mkdir(exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"[INFO] Plot saved -> {plot_path}")

    # Top-5 JSON (quick glance)
    top_json = "optuna_artifacts/top5.json"
    df.head(5).to_json(top_json, orient="records", indent=2, force_ascii=False)
    print(f"[INFO] Top-5 JSON -> {top_json}")


if __name__ == "__main__":
    main()
