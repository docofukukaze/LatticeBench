# LatticeBench — A Discrete Gauge Theory Playground (PoC)

**English / 日本語**

---

## 🧩 Overview | 概要

**English**  
**LatticeBench** is a **minimal proof-of-concept (PoC)** platform for experimenting with
**discrete space-time structures** using a
**Physics-Informed Neural Network (PINN) on a 2D SU(2) lattice**.

In conventional lattice QCD, exploring new physical structures requires
heavy modification of large-scale HPC frameworks (e.g., QUDA, Chroma, openQCD)
and massive computational resources. This makes **testing new theoretical structures
or principles very difficult** for individuals.

LatticeBench aims to be a **lightweight numerical playground** where one can
**embed physical constraints (unitarity, gauge symmetry) into the network structure**,
and **learn to reproduce gauge-invariant observables (plaquette, Wilson loops, etc.)**.

Unlike typical PINNs (which approximate **continuous PDEs**),
LatticeBench attempts to explore **discrete space-time structures themselves** —
a direction rarely explored so far.

This is **not a validated physical model**, but intended as an **idea prototype** for those
interested in **structure-informed neural approaches to lattice gauge theory and beyond**.

---

**日本語**  
**LatticeBench** は、**時空が本質的に離散的かもしれない**という仮定のもとに、
**2次元 SU(2) 格子上での物理インフォームド NN（PINN）** を構築する
**最小限の概念実証（PoC）** 実装です。

従来の格子 QCD では、新しい物理構造を検証するには
QUDA・Chroma・openQCD などの **大規模 HPC コード** を改造する必要があり、
膨大な計算資源と専門知識を要するため、**個人が理論構造を柔軟に試すことは極めて困難**でした。

LatticeBench は、**物理的制約（ユニタリティ・ゲージ対称性）をネットワーク構造に組み込み**、
**プラークエットや Wilson ループといったゲージ不変量** を再現するように学習させることで、
**新しい理論構造を小規模に数値検証できる実験的プラットフォーム** を目指します。

これは、**連続場方程式の近似器として使われてきた PINN を**
**あえて離散的な時空構造そのものの探索に応用する** という、
これまでにほとんど例のない試みです。

**物理的妥当性は未検証** であり、
**構造制約付き NN を格子ゲージ理論に応用する着想を提供する** ことを目的としています。

---

## 🚧 Limitations & Future Work | 制約と今後の展望

- **Scope (PoC):** This repository favors **experimentability over validity**. No claim of physical correctness.
- **Small lattice / SU(2):** All experiments use a **4×4 periodic lattice** with the **SU(2)** gauge group.
- **Model reuse & inference-only workflows:** *Not assumed* in the current PoC.  
  However, as lattice **dimensionality/complexity increases** (e.g., SU(3) or 3D/4D), saving and **reusing trained models** (for inference or as **warm starts / fine-tuning** bases) may become meaningful.

—

- **スコープ（PoC）:** 本リポジトリは **物理的な厳密性より実験容易性を重視** しています。
- **小規模格子 / SU(2):** 実験は **4×4 周期格子**・**SU(2)** を前提としています。
- **モデル再利用・推論専用:** 現段階では **前提としていません**。  
  ただし、**次元/複雑さの拡大**（例：SU(3) や 3D/4D）では、**学習済みモデルの保存・再利用**（推論や **ウォームスタート / 微調整**）の有効性が高まる可能性があります。

---

## ⚙️ Features | 特徴
- SU(2) gauge group on **4×4 periodic lattice**
- **Structure-aware** approach:
  - SU(2) exponential map for strict unitarity
  - *(Planned)* Gauge factorization: `U_{x,μ} = g_x · V_μ · g_{x+μ}†`
- **Loss components:**
  - Plaquette traces (complex MSE)
  - Unitarity penalty
  - Smoothness penalty
- **Evaluation metrics:**
  - Mean plaquette trace
  - Wilson loop (1×1, 1×2, 2×2)
  - Creutz ratio χ(2,2)
  - Gauge-aligned link RMSE

---

## 📈 Evaluation | 評価指標
- `|ΔTr P|`: 平均プラークエット誤差
- `Wilson(Rx,Ry)`: 汎化確認（未使用サイズ含む）
- `Creutz χ(2,2)`: 面張力推定
- `Gauge-aligned RMSE`: ゲージ変換自由度を除いたリンク RMSE

---

## 📦 Requirements | 必要環境

Minimal requirements (CPU execution):

```
numpy
torch
pandas
matplotlib
```

> Note: `requirements.txt` deliberately excludes GPU‑specific builds of PyTorch.

---

## ⚡ GPU / CUDA (optional but recommended) | GPU / CUDA（推奨）

Install PyTorch with CUDA support **before** installing `requirements.txt`.
Check the official PyTorch site for the correct build for your system.

### Linux / Windows (CUDA 12.1 example)

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install -r requirements.txt
```

### CPU only

```bash
pip install torch torchvision   # CPU build
pip install -r requirements.txt
```

### Apple Silicon (MPS)

```bash
pip install torch torchvision   # MPS build
pip install -r requirements.txt
```

### Quick check

```bash
python - <<'PY'
import torch
print("cuda? ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu  : ", torch.cuda.get_device_name(0))
print("mps?  ", torch.backends.mps.is_available())
PY
```

---

## 🚀 Usage | 使い方

Run a single training:

```bash
python -m src.train \
  --epochs 700 --print_every 100 \
  --lr 5e-3 --seed 777 \
  --use_huber --huber_delta_wil 0.003 --huber_delta_cr 0.008 \
  --w_wil11 0.10 --w_wil12 0.56 --w_wil22 0.28 --w_wil13 0.20 --w_wil23 0.16 \
  --w_cr 0.55 \
  --w_plaq 0.04 \
  --w_unitary 0.06 \
  --w_phi_smooth 0.04 --w_theta_smooth 0.02 --w_phi_l2 0.002
```

Outputs are written to `runs/*.log`.

---

## 🧪 Batch Sweep (run.bash) | 一括スイープ

**English**  
`run.bash` launches a grid search over several loss weights, learning rates, Huber on/off, and seeds.
Each run writes a human-readable log to `runs/*.log`. It can take many hours on CPU.

**日本語**  
`run.bash` は複数の損失重み・学習率・Huber 有無・シードのグリッド探索を実行します。
各実行のログは `runs/*.log` に保存されます。CPU では長時間（数時間〜）かかります。

### Run

```bash
bash run.bash
# logs -> runs/A_*.log, runs/B_*.log
```

**Notes | 注意**  
- Edit arrays in the header of `run.bash` to adjust the sweep.  
  （スクリプト先頭の配列を編集して探索範囲を調整できます）
- Logs and artifacts are ignored by git via `.gitignore` (`runs/`)。  
  （`runs/` は `.gitignore` 済み）

---

## 🧾 Single-Log Analysis (analyze_log.py) | 単発ログ解析

**English**  
Parses one training log and writes a single CSV row with metrics and recovered args.

**日本語**  
単一の学習ログを解析し、指標と推定引数を 1 行の CSV として出力します。

```bash
python -m src.analyze_log runs/A_lr5e-3_w12_0.44_cr0.55_pl0.04_s42.log --out runs.csv
```

**Output columns (subset) | 主な列**  
- `ga_rmse` — Gauge-aligned link RMSE
- `avgTrP_pred`, `avgTrP_true`, `avgTrP_absdiff`
- `W1x1_*`, `W1x2_*`, `W2x2_*`, `chi_2x2_*`
- recovered args (e.g., `arg_lr`, `arg_seed`, weights…)

> Tip: Keep the **“CMD:” line** in logs (printed by `train.py`) so arguments are recovered exactly.  
> （ログに **CMD 行** があると引数を完全復元できます）

---

## 📊 Aggregation & Plots | 集計と可視化

**English**  
Aggregate multiple logs into one CSV, then plot top runs.

**日本語**  
複数ログを 1 つの CSV に集計し、上位結果を可視化します。

```bash
# 1) Aggregate
python -m src.audit_runs runs/*.log --out runs_all.csv --top 20

# 2) Plot summaries
python -m src.plot_runs runs_all.csv --outdir plots --top 20

# (Optional) Visualize single-run CSV from analyze_log.py
python -m src.plot_runs runs.csv --outdir plots_runs --top 30
```

**Artifacts | 生成物**  
- `runs_all.csv` — one row per log/run（ログ 1 個 = 1 行）
- `plots/` — scatter/pareto & histograms; top-k CSV by `ga_rmse` and by `|Δ avgTrP|`
  （散布図・パレート・ヒスト、`ga_rmse` と `|Δ avgTrP|` の上位を CSV で出力）

---

## 📈 Results & Discussion | 結果と考察

This section summarizes the batch experiment driven by **`run.bash`**, the search space, and the **best settings** we observed under two evaluation priorities.
（このセクションでは **`run.bash`** による一括実験、その探索範囲、そして **評価指標ごとの最良設定** をまとめます。）

### 🔧 What was run (run.bash) | 実験の内容

We launched a grid over selected loss weights, learning rates, and loss types (MSE vs Huber). Each setting ran for **900 epochs** and wrote a human‑readable log under `runs/*.log`.
（損失重み・学習率・損失種別（MSE / Huber）に対してグリッド探索を実行。各設定は **900 エポック** 学習し、`runs/*.log` にログを保存。）

**Fixed base (common to all runs) | 共通設定**
```bash
python -m src.train --epochs 900 --print_every 50 \
  --w_unitary 0.06 \
  --w_phi_smooth 0.04 --w_theta_smooth 0.02 --w_phi_l2 0.002 \
  --w_wil11 0.10 --w_wil22 0.28 --w_wil13 0.22 --w_wil23 0.18
```
- These weights enforce **unitarity** and regularization (smoothness/L2), and include Wilson 1×1/2×2/1×3/2×3 components at fixed strengths.  
  （**ユニタリティ**と正則化（スムース/L2）を課し、Wilson 1×1/2×2/1×3/2×3 を固定重みで含めます。）

**Swept hyperparameters | 探索パラメータ**
- **Plaquette weight `w_plaq`**: `0.02`, `0.04`, `0.06`
- **Wilson (1×2) weight `w_wil12`**: `0.32`, `0.44`, `0.56`
- **Creutz ratio weight `w_cr`**: `0.45`, `0.55`, `0.65`
- **Learning rate `lr`**: `5e-3`, `4e-3`
- **Seed**: `42` (fixed)
- **Loss type**: **A=MSE**, **B=Huber** with `--huber_delta_wil 0.003 --huber_delta_cr 0.008`

> Total grid size = 3 (plaq) × 3 (w12) × 3 (cr) × 2 (lr) × 1 (seed) × 2 (loss) = **108 runs**.

**Launch script (excerpt) | 実行スクリプト（抜粋）**
```bash
## Pattern A: MSE
${BASE} --lr ${lr} --w_wil12 ${w12} --w_cr ${wcr} --w_plaq ${wpl} --seed ${seed} \
  | tee "runs/A_lr${lr}_w12${w12}_cr${wcr}_pl${wpl}_s${seed}.log"

## Pattern B: Huber (delta small, sharper)
${BASE} --lr ${lr} --w_wil12 ${w12} --w_cr ${wcr} --w_plaq ${wpl} --seed ${seed} \
  --use_huber --huber_delta_wil 0.003 --huber_delta_cr 0.008 \
  | tee "runs/B_lr${lr}_w12${w12}_cr${wcr}_pl${wpl}_s${seed}.log"
```

### 🧪 Metrics | 評価指標
- **Gauge‑aligned RMSE** (lower better): link‑wise error after gauge alignment.
- **|Δ avgTrP|** (lower better): absolute error of the **mean plaquette trace**.
- We treat them as **two objectives**; a single setting rarely minimizes both.  
  （両指標はしばしばトレードオフになります。）

### 🥇 Best Settings | 最良設定（指標別）

**A. Minimize Gauge‑aligned RMSE（GA‑RMSE 最小）**  
- **Observed**: `ga_rmse ≈ 0.356` (global best in this sweep)
- **Trade‑off**: `|Δ avgTrP| ≈ 0.545` (plaquette mismatch remains)
- **Tendencies**: **Huber** helps RMSE stability; smaller `w_plaq` avoids over‑fitting avgTrP.
- **Command to reproduce:**
```bash
python -m src.train \
  --epochs 900 --print_every 50 \
  --lr 5e-3 --seed 42 \
  --use_huber --huber_delta_wil 0.003 --huber_delta_cr 0.008 \
  --w_wil11 0.10 --w_wil12 0.32 --w_wil22 0.28 --w_wil13 0.22 --w_wil23 0.18 \
  --w_cr 0.45 \
  --w_plaq 0.02 \
  --w_unitary 0.06 \
  --w_phi_smooth 0.04 --w_theta_smooth 0.02 --w_phi_l2 0.002
```

**B. Minimize |Δ avgTrP|（平均プラークエット誤差 最小）**  
- **Observed**: `|Δ avgTrP| ≈ 0.0015` (near‑perfect match)
- **Trade‑off**: `ga_rmse ≈ 0.412` (slightly worse alignment)
- **Tendencies**: **MSE** + **higher `w_plaq`** helps pin the mean plaquette tightly.
- **Command to reproduce:**
```bash
python -m src.train \
  --epochs 900 --print_every 50 \
  --lr 5e-3 --seed 42 \
  --w_wil11 0.10 --w_wil12 0.32 --w_wil22 0.28 --w_wil13 0.22 --w_wil23 0.18 \
  --w_cr 0.45 \
  --w_plaq 0.06 \
  --w_unitary 0.06 \
  --w_phi_smooth 0.04 --w_theta_smooth 0.02 --w_phi_l2 0.002
```

> **Pareto view**: These two lie near the empirical **Pareto front** (improving one metric worsens the other).
> （経験的パレート前線上に近い位置にあります。）

### 💡 Practical Guidance | 実務の指針
- **If you care about local link quality (alignment)** → prefer **Huber** + lower `w_plaq`.
- **If you care about global average matching** → prefer **MSE** + higher `w_plaq`.
- Start with `lr=5e-3`; try `4e-3` if loss plateaus or oscillates.
- Keep `w_wil12` and `w_cr` moderate‑to‑high (`0.44–0.56`, `0.55–0.65`) when you want Wilson(1×2)/Creutz to generalize beyond plaquette.

### ⚠️ Caveats | 注意点
- This is a **PoC**, not a validated physical model.
- Metrics are computed on a fixed small lattice (4×4, SU(2)); scaling behavior may differ.
- Huber/MSE preference can invert if you substantially change the loss mix or schedule.

---

## 💬 Motivation | 動機

**English**  
- Lattice QCD focuses on precise verification rather than theory discovery; codes are large and HPC‑oriented.
- There is **no lightweight, open platform** for trying new discrete structures or NN‑based inductive biases.
- LatticeBench aims to serve as a **seed platform** for such explorations.

**日本語**  
- 格子 QCD は「理論発見」より「精密検証」に重点が置かれ、コードは大規模で HPC 依存です。
- **新しい離散構造や NN ベースの帰納的バイアスを試す軽量オープン基盤** は存在しません。
- LatticeBench はそのような探索のための **シードプラットフォーム** を目指しています。

---

## 📝 Note | 注記

**English**  
I myself studied lattice QCD about 20 years ago at university.
This code was originally generated while chatting with ChatGPT about old memories,
but I decided to publish it here just in case it may be useful.

**日本語**  
私は 20 年ほど前に大学で格子 QCD を研究していました。
このコードは、ChatGPT とのたわいない思い出話の中で生成されたものですが、
念のため公開しておきます。

---

## 📜 License | ライセンス
MIT
