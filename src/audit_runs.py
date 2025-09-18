# src/audit_runs.py
"""
EN: Audit many run logs, aggregate into a single CSV with hyperparams filled from CMD lines.
JA: 複数のログを監査し、CMD の引数からハイパーパラメータを埋めた集計 CSV を作ります。

Usage:
  python -m src.audit_runs runs/*.log --out runs_all.csv --top 20
"""

import argparse
import glob
import os
from collections import defaultdict

from .analyze_log import parse_log, write_csv  # re-use robust parser

def hp_key(row):
    # define a grouping key that ignores seed (= duplicates across seeds)
    keys = ["arg_lr","arg_epochs","arg_L","w_plaq","w_wil11","w_wil12","w_wil22","w_wil13","w_wil23","w_cr",
            "w_unitary","w_phi_smooth","w_theta_smooth","w_phi_l2","use_huber","huber_delta_wil","huber_delta_cr"]
    return tuple(row.get(k,"") for k in keys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+", help="glob patterns for log files")
    ap.add_argument("--out", type=str, default="runs_all.csv")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    files = []
    for pat in args.patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    rows = []
    for p in files:
        try:
            r = parse_log(p)
            # どのログから来たかを入れておく（プロットで A/B 判定に使う）
            r["filename"] = os.path.basename(p)
            # 先頭 A_/B_ から損失種別（MSE/Huber）を推定（保険）
            base = os.path.basename(p)
            if   base.startswith("A_"): r["loss_kind"] = "MSE"
            elif base.startswith("B_"): r["loss_kind"] = "Huber"
            rows.append(r)
        except Exception as e:
            # record minimal info
            rows.append({"cmdline":"","parse_error":str(e),"filename":os.path.basename(p)})

    # Duplicate hyperparam groups (ignoring seed)
    groups = defaultdict(list)
    for i, r in enumerate(rows, 1):
        groups[hp_key(r)].append(i)

    # report
    print(f"Files seen: {len(files)}")
    print(f"Runs parsed: {len(rows)} (finished={len(rows)}, partial=0)")

    dups = [(k,v) for k,v in groups.items() if len(v) > 1]
    print(f"Duplicate hyperparam groups: {len(dups)}")
    if dups:
        print("  (showing up to 5 groups)")
        for k, idxs in dups[:5]:
            # example cmd from the first
            i0 = idxs[0]-1
            fname = os.path.basename(files[i0]) if i0 < len(files) else "?"
            print(f"  x{len(idxs)}  example cmd: ...")
            print("    files:", ", ".join(os.path.basename(files[j-1]) for j in idxs[:6]))

    # LR distribution quick view
    lr_hist = defaultdict(int)
    for r in rows:
        if r.get("arg_lr"):
            lr_hist[f"{float(r['arg_lr']):.3e}"] += 1
    print("LR distribution:", dict(lr_hist))

    # write csv
    write_csv(rows, args.out)
    print(f"Wrote: {args.out}  (rows={len(rows)})")

    # Rankings
    scored = [r for r in rows if r.get("ga_rmse")]
    scored.sort(key=lambda x: float(x["ga_rmse"]))
    print("\nTop by Gauge-aligned RMSE (lower is better)")
    for i, r in enumerate(scored[:args.top], 1):
        print(f"[{i:02d}] ga_rmse={float(r['ga_rmse']):.4e}  dAvgTrP={r.get('avgTrP_absdiff','')}  lr={r.get('arg_lr','')}  seed={r.get('arg_seed','')}  file={r.get('filename', '...') or '...'}")

    davg = [r for r in rows if r.get("avgTrP_absdiff")]
    davg.sort(key=lambda x: float(x["avgTrP_absdiff"]))
    print("\nTop by |Δ avgTrP| (lower is better)")
    for i, r in enumerate(davg[:args.top], 1):
        print(f"[{i:02d}] dAvgTrP={float(r['avgTrP_absdiff']):.4e}  ga_rmse={r.get('ga_rmse','')}  lr={r.get('arg_lr','')}  seed={r.get('arg_seed','')}  file={r.get('filename', '...') or '...'}")

if __name__ == "__main__":
    main()
