# src/plot_runs.py
"""
可視化ユーティリティ（runs_all.csv / runs.csv -> 図 & サマリ）
- ga_rmse vs |Δ avgTrP| の散布図（学習率で色分け、Huber/MSEでマーカー切替）
- ga_rmse と |Δ avgTrP| のヒスト
- 上位N (= --top) のCSV書き出し（ga_rmse, |Δ avgTrP| それぞれ）
- 簡易パレート前線（両指標を同時に小さくする解）抽出とCSV/図出力
- 可能ならファイル名からハイパラ推定（lr, w12, wcr, wpl, seed, huber/MSE）

使い方:
  python -m src.plot_runs runs_all.csv --outdir plots --top 20

備考:
- 列名のゆらぎに強くしています（'ga_rmse', 'avgTrP_absdiff' / 'dAvgTrP' などを探索）
- 学習率は列('lr' or 'arg_lr' or 'lr_seen')が無ければファイル名から推定
- Huber/MSEはファイル名の先頭 'A_'=MSE, 'B_'=Huber で推定
"""
import argparse
import re
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------- helpers ----------
def pick_col(df: pd.DataFrame, candidates: List[str], default: Optional[str]=None) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return default

def infer_from_filename(s: str, key: str) -> Optional[str]:
    # 例: A_lr5e-3_w12 0.44 _cr0.55_pl0.06_s777.log など
    # lr:  lr(?P<val>[0-9eE\.\-]+)
    # w12: _w12(?P<val>[0-9eE\.\-]+)
    # cr:  _cr(?P<val>[0-9eE\.\-]+)
    # pl:  _pl(?P<val>[0-9eE\.\-]+)
    # s:   _s(?P<val>\d+)
    pat = {
        "lr": r"lr(?P<val>[0-9eE\.\-]+)",
        "w12": r"_w12(?P<val>[0-9eE\.\-]+)",
        "cr": r"_cr(?P<val>[0-9eE\.\-]+)",
        "pl": r"_pl(?P<val>[0-9eE\.\-]+)",
        "seed": r"_s(?P<val>\d+)"
    }.get(key)
    if not pat:
        return None
    m = re.search(pat, s)
    return m.group("val") if m else None

def infer_huber_from_filename(s: str) -> Optional[str]:
    # 先頭 'A_' = MSE, 'B_' = Huber の想定
    base = os.path.basename(s)
    return "Huber" if base.startswith("B_") else ("MSE" if base.startswith("A_") else None)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def pareto_front(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    # 低いほど良い: x=ga_rmse, y=dAvgTrP（または abs diff）
    # 非支配解を抽出（単純・安定のためソート→線形スキャン）
    tmp = df[[x, y]].values
    order = np.lexsort((tmp[:,1], tmp[:,0]))  # x昇順→y昇順
    pts = tmp[order]
    pf_mask = np.zeros(len(pts), dtype=bool)
    best_y = np.inf
    for i, (_, yy) in enumerate(pts):
        if yy < best_y:
            best_y = yy
            pf_mask[i] = True
    pf_idx = [order[i] for i, m in enumerate(pf_mask) if m]
    return df.iloc[pf_idx].copy()

def fmt_lr(v: Optional[str]) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "?"
    return str(v)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Plot/summary for LatticeBench runs CSV")
    ap.add_argument("csv", help="runs_all.csv または runs.csv")
    ap.add_argument("--outdir", default="plots", help="出力ディレクトリ")
    ap.add_argument("--top", type=int, default=20, help="トップNを表示/保存")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = pd.read_csv(args.csv)

    # 列名の解決
    col_ga = pick_col(df, ["ga_rmse", "ga_rmse_best", "ga_err", "ga_error"])
    col_dP = pick_col(df, ["avgTrP_absdiff", "dAvgTrP", "avg_dTrP", "avgDeltaTrP"])
    col_file = pick_col(df, ["file", "filename"])
    col_cmd  = pick_col(df, ["cmdline", "cmd", "args"])
    col_lr   = pick_col(df, ["lr", "arg_lr", "lr_seen"])
    col_seed = pick_col(df, ["arg_seed", "seed"])

    # 必須列チェック
    miss = [n for n in [col_ga, col_dP] if n is None]
    if miss:
        raise SystemExit(f"必要列が見つかりません: ga_rmse系/avgTrP_absdiff系（見つかった: {df.columns.tolist()}）")

    # 欠損をファイル名から補完（lr/seed/ハブ）
    if col_file is None and "file" not in df.columns:
        # なければ cmdline からでも可
        col_file = col_cmd
    if col_lr is None:
        if col_file:
            df["_lr_infer"] = df[col_file].astype(str).apply(lambda s: infer_from_filename(s, "lr"))
            col_lr = "_lr_infer"
    if col_seed is None:
        if col_file:
            df["_seed_infer"] = df[col_file].astype(str).apply(lambda s: infer_from_filename(s, "seed"))
            col_seed = "_seed_infer"

    # Huber/MSE 推定（filename優先 / なければ cmdline ）
    if col_file:
        df["_loss_kind"] = df[col_file].astype(str).apply(infer_huber_from_filename)
    elif col_cmd:
        df["_loss_kind"] = df[col_cmd].astype(str).apply(infer_huber_from_filename)
    else:
        df["_loss_kind"] = None
    # もし audit_runs が loss_kind を既に埋めていれば優先
    if "loss_kind" in df.columns:
        df["_loss_kind"] = df["loss_kind"].fillna(df["_loss_kind"])

    # 数値化（安全に）
    df[col_ga] = pd.to_numeric(df[col_ga], errors="coerce")
    df[col_dP] = pd.to_numeric(df[col_dP], errors="coerce")
    if col_lr:
        df[col_lr] = df[col_lr].astype(str).str.strip()

    # 散布図
    plt.figure(figsize=(7.5, 6))
    # 色分け: lr、マーカー: Huber/MSE
    lrs = sorted(df[col_lr].dropna().unique().tolist()) if col_lr else []
    if lrs:
        colors = {lr: c for lr, c in zip(lrs, plt.rcParams['axes.prop_cycle'].by_key()['color'])}
    else:
        colors = {}

    for loss_kind, g in df.groupby(df["_loss_kind"].fillna("Unknown")):
        marker = "o" if loss_kind == "MSE" else ("^" if loss_kind == "Huber" else "s")
        if col_lr:
            for lr, g2 in g.groupby(g[col_lr]):
                plt.scatter(g2[col_ga], g2[col_dP], s=24, marker=marker,
                            label=f"{loss_kind} / lr={fmt_lr(lr)}",
                            alpha=0.75,
                            color=colors.get(lr, None))
        else:
            plt.scatter(g[col_ga], g[col_dP], s=24, marker=marker,
                        label=f"{loss_kind}",
                        alpha=0.75)

    plt.xlabel("Gauge-aligned RMSE (lower better)")
    plt.ylabel("|Δ avgTrP| (lower better)")
    plt.title("Tradeoff: ga_rmse vs |Δ avgTrP|")
    plt.grid(True, alpha=0.3)
    # 凡例が大きくなりすぎる場合もあるので、最大12個に制限
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 12:
        plt.legend(handles[:12], labels[:12], fontsize=8)
    else:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "scatter_rmse_vs_dAvgTrP.png", dpi=150)
    plt.close()

    # ヒスト：ga_rmse
    plt.figure(figsize=(7.5, 4.2))
    plt.hist(df[col_ga].dropna().values, bins=30, alpha=0.85)
    plt.xlabel("Gauge-aligned RMSE")
    plt.ylabel("count")
    plt.title("Distribution of ga_rmse")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "hist_ga_rmse.png", dpi=150)
    plt.close()

    # ヒスト：|Δ avgTrP|
    plt.figure(figsize=(7.5, 4.2))
    plt.hist(df[col_dP].dropna().values, bins=30, alpha=0.85)
    plt.xlabel("|Δ avgTrP|")
    plt.ylabel("count")
    plt.title("Distribution of |Δ avgTrP|")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "hist_dAvgTrP.png", dpi=150)
    plt.close()

    # トップN（ga_rmse）
    topN = args.top
    top_rmse = df.sort_values(col_ga, ascending=True).head(topN).copy()
    top_rmse.to_csv(outdir / "top_by_ga_rmse.csv", index=False)

    # トップN（|Δ avgTrP|）
    top_dP = df.sort_values(col_dP, ascending=True).head(topN).copy()
    top_dP.to_csv(outdir / "top_by_dAvgTrP.csv", index=False)

    # パレート前線
    pf = pareto_front(df, col_ga, col_dP)
    pf_sorted = pf.sort_values([col_ga, col_dP], ascending=[True, True]).copy()
    pf_sorted.to_csv(outdir / "pareto_front.csv", index=False)

    # パレート前線の重ね書き
    plt.figure(figsize=(7.5, 6))
    plt.scatter(df[col_ga], df[col_dP], s=16, alpha=0.25, label="all runs")
    plt.plot(pf_sorted[col_ga], pf_sorted[col_dP], "o-", label="Pareto front", ms=5)
    plt.xlabel("Gauge-aligned RMSE (lower better)")
    plt.ylabel("|Δ avgTrP| (lower better)")
    plt.title("Pareto front (ga_rmse vs |Δ avgTrP|)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pareto_front.png", dpi=150)
    plt.close()

    # ざっくり表
    # 学習率×Huber/MSE で平均・最良を出す（ある列がなければスキップ）
    summary_rows = []
    if col_lr:
        for lr, g in df.groupby(df[col_lr]):
            for lk, g2 in g.groupby(df["_loss_kind"].fillna("Unknown")):
                row = {
                    "lr": lr,
                    "loss_kind": lk,
                    "n": len(g2),
                    "ga_rmse_mean": float(np.nanmean(g2[col_ga])),
                    "ga_rmse_min": float(np.nanmin(g2[col_ga])),
                    "dAvgTrP_mean": float(np.nanmean(g2[col_dP])),
                    "dAvgTrP_min": float(np.nanmin(g2[col_dP])),
                }
                summary_rows.append(row)
    if summary_rows:
        pd.DataFrame(summary_rows).sort_values(["lr", "loss_kind"]).to_csv(outdir / "group_summary.csv", index=False)

    print(f"Saved plots and tables -> {outdir}")
    print(" - scatter_rmse_vs_dAvgTrP.png")
    print(" - hist_ga_rmse.png")
    print(" - hist_dAvgTrP.png")
    print(" - top_by_ga_rmse.csv")
    print(" - top_by_dAvgTrP.csv")
    print(" - pareto_front.csv / pareto_front.png")
    if summary_rows:
        print(" - group_summary.csv (lr × loss_kind の集計)")
    print("Done.")
    

if __name__ == "__main__":
    main()
