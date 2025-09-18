# src/analyze_log.py
"""
Parse a single training log that contains one run and write a CSV row.
English/Japanese logs supported. Robust to complex-like numbers (e.g. +0.2283-0.0000j).

Usage:
  python -m src.analyze_log LOGFILE --out runs.csv --top 15
"""

import argparse
import csv
import os
import re
import shlex
from typing import Dict, Optional

# --- regex helpers (both EN/JA) ---
RE_CMD = re.compile(r'^\s*CMD:\s+(.*)$')

# [ep***] line
RE_EPLINE = re.compile(r'^\[ep(\d+)\]\s+total=([0-9eE+.\-]+)\s+plaq=([0-9eE+.\-]+)')

# gauge-aligned RMSE
RE_GA = re.compile(r'Gauge-aligned link RMSE .*?:\s*([0-9eE+.\-]+)')

# total time: JA("総時間 / Total time:") と EN("Total time:") の両方
RE_TIME_BOTH = re.compile(r'(?:総時間\s*/\s*)?Total time:\s*([0-9:]+)')

# === Looser patterns: capture the whole token after "pred=" up to the next comma ===
# mean Tr P (EN/JA)
RE_AVG_EN = re.compile(r'mean\s*Tr\s*P\s*pred=([^,]+),\s*true=([^,\n]+)', re.I)
RE_AVG_JA = re.compile(r'平均Tr P\s*pred=([^,]+),\s*true=([^,\n]+)')

# Wilson loops (same label in EN/JA)
RE_W11 = re.compile(r'Wilson\(1x1\)\s*pred=([^,]+),\s*true=([^,\n]+)', re.I)
RE_W12 = re.compile(r'Wilson\(1x2\)\s*pred=([^,]+),\s*true=([^,\n]+)', re.I)
RE_W22 = re.compile(r'Wilson\(2x2\)\s*pred=([^,]+),\s*true=([^,\n]+)', re.I)

# Creutz ratio χ(2,2)
RE_CHI = re.compile(r'Creutz\s*χ?\(2,?2\)\s*pred=([^,]+),\s*true=([^,\n]+)', re.I)

# filename fallback: A_lr5e-3_w12 0.44 _cr0.55 _pl0.02 _s42.log
RE_FNAME = re.compile(
    r'(?:^|[_/])(?:A|B)_lr([0-9eE.\-]+)_w12([0-9.]+)_cr([0-9.]+)_pl([0-9.]+)_s(\d+)'
)

FIELDS = [
    "arg_seed","arg_lr","lr_seen","arg_epochs","arg_L",
    "avgTrP_absdiff","ga_rmse",
    "W1x1_pred","W1x1_true","W1x2_pred","W1x2_true","W2x2_pred","W2x2_true",
    "chi_2x2_pred","chi_2x2_true",
    "total_time_hms","total_time_sec",
    "last_ep","last_total","last_plaq",
    "cmdline","avgTrP_pred","avgTrP_true",
    # weights/flags
    "w_plaq","w_wil11","w_wil12","w_wil22","w_wil13","w_wil23","w_cr",
    "w_unitary","w_phi_smooth","w_theta_smooth","w_phi_l2",
    "use_huber","huber_delta_wil","huber_delta_cr",
]

# --- number helpers ----------------------------------------------------------

RE_COMPLEX_LIKE = re.compile(
    r'^\s*'                                      # leading space
    r'([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)'   # group1: leading real number
    r'(?:'                                       # optional imaginary tail like "+0.0000j" or "-1e-3j"
    r'[+\-]\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?j'     # imag part with 'j'
    r')?\s*$'                                    # trailing space
)

def complex_like_to_float(s: str) -> Optional[float]:
    """
    Convert strings like '+0.2283-0.0000j' to float(real_part).
    Falls back to stripping trailing 'j' and plain float, else None.
    """
    if s is None:
        return None
    s = s.strip()
    m = RE_COMPLEX_LIKE.match(s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # simple fallbacks
    try:
        return float(s.rstrip('j'))
    except Exception:
        return None

def to_float_str(s: str) -> str:
    try:
        return f"{float(s):.12g}"
    except Exception:
        return s

# --- parse helpers -----------------------------------------------------------

def parse_cmdline_from_text(text: str) -> Optional[str]:
    for line in text.splitlines():
        m = RE_CMD.match(line.strip())
        if m:
            return m.group(1).strip()
    return None

def parse_fallback_from_filename(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    m = RE_FNAME.search(path)
    if not m:
        return out
    out["arg_lr"] = to_float_str(m.group(1))
    out["w_wil12"] = to_float_str(m.group(2))
    out["w_cr"] = to_float_str(m.group(3))
    out["w_plaq"] = to_float_str(m.group(4))
    out["arg_seed"] = m.group(5)
    return out

def parse_args_from_cmd(cmd: str) -> Dict[str, str]:
    toks = shlex.split(cmd)
    d: Dict[str, str] = {}
    for i, t in enumerate(toks):
        if not t.startswith("--"):
            continue
        key = t[2:]
        val = None
        if i + 1 < len(toks) and not toks[i+1].startswith("--"):
            val = toks[i+1]
        d[key] = "1" if val is None else val
    norm = {}
    def copy(k, cast_float=False):
        if k in d:
            norm_key = {"L":"arg_L","lr":"arg_lr","epochs":"arg_epochs","seed":"arg_seed"}.get(k, k)
            norm[norm_key] = to_float_str(d[k]) if cast_float else d[k]
    copy("L")
    copy("lr", True)
    copy("epochs")
    copy("seed")
    for k in ["w_plaq","w_wil11","w_wil12","w_wil22","w_wil13","w_wil23","w_cr",
              "w_unitary","w_phi_smooth","w_theta_smooth","w_phi_l2",
              "huber_delta_wil","huber_delta_cr"]:
        if k in d:
            norm[k] = to_float_str(d[k])
    norm["use_huber"] = "1" if d.get("use_huber") is not None else "0"
    return norm

def hms_to_sec(hms: str) -> int:
    parts = [int(x) for x in hms.split(":")]
    if len(parts) == 3:
        h,m,s = parts
        return h*3600 + m*60 + s
    if len(parts) == 2:
        m,s = parts
        return m*60 + s
    return int(parts[0])

def _search_first(text: str, pats):
    if not isinstance(pats, (list, tuple)):
        pats = [pats]
    for p in pats:
        m = p.search(text)
        if m:
            return m
    return None

def parse_log(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    row: Dict[str, str] = {k:"" for k in FIELDS}

    # CMD
    cmd = parse_cmdline_from_text(text)
    if cmd:
        row["cmdline"] = cmd
        row.update(parse_args_from_cmd(cmd))
    else:
        row.update(parse_fallback_from_filename(os.path.basename(path)))

    # last [ep***] line
    last_ep, last_total, last_plaq = "", "", ""
    for line in text.splitlines():
        m = RE_EPLINE.search(line)
        if m:
            last_ep = m.group(1)
            last_total = m.group(2)
            last_plaq = m.group(3)
    row["last_ep"] = last_ep
    row["last_total"] = last_total
    row["last_plaq"] = last_plaq

    # metrics block
    m = RE_GA.search(text)
    if m: row["ga_rmse"] = m.group(1)

    m = RE_TIME_BOTH.search(text)
    if m:
        hms = m.group(1)
        row["total_time_hms"] = hms
        row["total_time_sec"] = str(hms_to_sec(hms))

    # mean Tr P (EN/JA)
    m = _search_first(text, [RE_AVG_EN, RE_AVG_JA])
    if m:
        pred_s, true_s = m.group(1).strip(), m.group(2).strip()
        row["avgTrP_pred"] = pred_s
        row["avgTrP_true"] = true_s
        pr = complex_like_to_float(pred_s)
        tr = complex_like_to_float(true_s)
        if pr is not None and tr is not None:
            row["avgTrP_absdiff"] = f"{abs(pr - tr):.6g}"

    # Wilson / Creutz (値自体はそのまま保存)
    for pat, keys in [
        (RE_W11, ("W1x1_pred","W1x1_true")),
        (RE_W12, ("W1x2_pred","W1x2_true")),
        (RE_W22, ("W2x2_pred","W2x2_true")),
        (RE_CHI,  ("chi_2x2_pred","chi_2x2_true")),
    ]:
        m = pat.search(text)
        if m:
            row[keys[0]] = m.group(1).strip()
            row[keys[1]] = m.group(2).strip()

    if row.get("arg_lr"):
        row["lr_seen"] = row["arg_lr"]

    return row

def write_csv(rows, out_path: str):
    all_fields = list(FIELDS)
    for r in rows:
        for k in r.keys():
            if k not in all_fields:
                all_fields.append(k)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=str)
    ap.add_argument("--out", type=str, default="runs.csv")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    row = parse_log(args.logfile)
    write_csv([row], args.out)

    print(f"Wrote: {args.out}  (rows=1)")
    if row.get("ga_rmse"):
        print("\nTop by Gauge-aligned RMSE (lower is better)")
        print(f"[01] ga_rmse={float(row['ga_rmse']):.4e}  avgΔTrP={row.get('avgTrP_absdiff','')}  lr={row.get('arg_lr','')}  seed={row.get('arg_seed','')}  cmd=......")

    if row.get("avgTrP_absdiff"):
        print("\nTop by |Δ avgTrP| (lower is better)")
        print(f"[01] avgΔTrP={float(row['avgTrP_absdiff']):.4e}  ga_rmse={row.get('ga_rmse','')}  lr={row.get('arg_lr','')}  seed={row.get('arg_seed','')}  cmd=......")

if __name__ == "__main__":
    main()
