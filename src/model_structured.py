# src/model_structured.py
r"""
EN: Structured SU(2) model for a small 2D lattice with built-in inductive bias.
    - Node embeddings φ(x) ∈ R^H
    - Shared MLP that emits su(2) algebra vectors a(x,μ) ∈ R^3
    - Exponential map su2_exp(a) → U(x,μ) ∈ SU(2)
    Provides helper losses: φ-smoothness, θ(a)-smoothness, and φ L2 regularization.
    Vectorized implementation (no Python loops over lattice).

JA: 小さな2次元格子向けの構造化SU(2)モデル（帰納的バイアス付き）。
    - ノード埋め込み φ(x) ∈ R^H
    - 共有MLPで su(2) 代数ベクトル a(x,μ) ∈ R^3 を生成
    - 指数写像 su2_exp(a) で U(x,μ) ∈ SU(2) を構成
    φの平滑化損失、a(=θ)の平滑化損失、φのL2正則化を提供。
    ループを排してベクトル化しています。
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lattice import Lattice
from .groups import su2_exp


class StructuredSU2Model(nn.Module):
    r"""EN: Lattice ↦ SU(2) links via node embeddings and a shared MLP.
        JA: ノード埋め込みと共有MLPで格子→SU(2)リンクを生成するモデル。
    """

    def __init__(
        self,
        lat: Lattice,
        hidden_dim: int = 32,
        mlp_width: int = 64,
        embed_init_scale: float = 1e-2,
        dtype_real: torch.dtype = torch.float32,
        dtype_cplx: torch.dtype = torch.complex64,
    ) -> None:
        """
        Args:
            lat: lattice description / 格子情報
            hidden_dim: node embedding size φ の次元
            mlp_width: shared MLP width / 共有MLPの幅
            embed_init_scale: φ の初期化スケール
            dtype_real: 実数テンソルの型（既定: float32）
            dtype_cplx: 複素数テンソルの型（既定: complex64）
        """
        super().__init__()
        self.lat = lat
        self.H = int(hidden_dim)
        self.dtype_real = dtype_real
        self.dtype_cplx = dtype_cplx

        # Node embeddings φ(x) ∈ R^H  / ノード埋め込み
        self.node_embed = nn.Parameter(
            torch.randn(lat.L, lat.L, self.H, dtype=dtype_real) * embed_init_scale
        )

        # Shared MLP: concat(φ(x), φ(x+μ)) → a(x,μ) ∈ R^3
        # 共有MLP: 近傍ノード埋め込みを結合して su(2) 代数ベクトルを出力
        self.link_mlp = nn.Sequential(
            nn.Linear(2 * self.H, mlp_width),
            nn.Tanh(),
            nn.Linear(mlp_width, mlp_width),
            nn.Tanh(),
            nn.Linear(mlp_width, 3),
        )

        # Identity buffer for unitarity check / 単位行列をバッファ登録
        I = torch.eye(2, dtype=self.dtype_cplx)
        self.register_buffer("I2", I, persistent=False)

    # ------------------------------------------------------------------ #
    # Core forward / コア計算（ベクトル化）
    # ------------------------------------------------------------------ #
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            U:  (L,L,2,2,2) complex SU(2) link matrices
            a:  (L,L,2,3)   real su(2) algebra vectors emitted by MLP
        """
        φ = self.node_embed                                   # (L,L,H)
        φ_xp = torch.roll(φ, shifts=-1, dims=0)               # (x+1,y)
        φ_yp = torch.roll(φ, shifts=-1, dims=1)               # (x,y+1)

        # μ=0: concat(φ(x), φ(x+êx)), μ=1: concat(φ(x), φ(x+êy))
        cat0 = torch.cat([φ, φ_xp], dim=-1)                   # (L,L,2H)
        cat1 = torch.cat([φ, φ_yp], dim=-1)                   # (L,L,2H)

        L = self.lat.L
        a0 = self.link_mlp(cat0.view(L * L, -1)).view(L, L, 1, 3)
        a1 = self.link_mlp(cat1.view(L * L, -1)).view(L, L, 1, 3)
        a  = torch.cat([a0, a1], dim=2).contiguous()          # (L,L,2,3) (real)

        # su(2) → SU(2) in batch
        U = su2_exp(a.view(L * L * 2, 3)).view(L, L, 2, 2, 2).to(self.dtype_cplx)
        return U, a

    # ------------------------------------------------------------------ #
    # Plaquette traces / プラークエットトレース（ベクトル化）
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def plaquette_traces(self) -> torch.Tensor:
        """EN: Tr(P) for all 1×1 plaquettes (complex).
           JA: 全 1×1 プラークエットのトレース（複素数）。
        """
        L = self.lat.L
        U, _ = self.forward()
        Ux, Uy = U[..., 0, :, :], U[..., 1, :, :]             # (L,L,2,2)
        Ux_yplus = torch.roll(Ux, shifts=-1, dims=1)          # U(x,y+1,0)
        Uy_xplus = torch.roll(Uy, shifts=-1, dims=0)          # U(x+1,y,1)
        # daggers
        Ux_dag = Ux.conj().transpose(-1, -2)
        Uy_dag = Uy.conj().transpose(-1, -2)
        # P = Ux * Uy(x+1,y) * Ux†(x,y+1) * Uy†
        P = Ux @ Uy_xplus @ torch.roll(Ux_dag, shifts=-1, dims=1) @ Uy_dag
        tr = torch.einsum("...ii->...", P)                    # trace
        return tr.reshape(L * L)                               # (L*L,)

    # ------------------------------------------------------------------ #
    # Optional penalties used by train.py / 学習で使う補助損失
    # ------------------------------------------------------------------ #
    def unitarity_loss(self, U: torch.Tensor | None = None) -> torch.Tensor:
        """
        EN: Soft penalty for unitarity: ||U†U - I||_F^2, computed in real domain
            to avoid complex MSE issues on CUDA/MPS.
        JA: ユニタリティ罰則 ||U†U - I||_F^2。CUDA/MPSで複素MSE未実装のため、
            実部・虚部を分けて実数で計算します。

        Args:
            U: Optional precomputed links (...,2,2). If None, calls self.forward().
        Returns:
            scalar real tensor
        """
        if U is None:
            U, _ = self.forward()  # (..., 2, 2) complex

        # U†U
        UhU = U.conj().transpose(-1, -2) @ U  # (..., 2, 2) complex
        I2 = self.I2.to(dtype=U.dtype, device=U.device).expand_as(UhU)

        diff = UhU - I2                         # complex
        # Frobenius^2 in real domain: sum(real^2 + imag^2) / N
        loss = (diff.real.pow(2) + diff.imag.pow(2)).mean()
        return loss


    def phi_smooth_loss(self) -> torch.Tensor:
        """EN: Encourage neighboring node embeddings φ to be close.
           JA: 近傍ノード埋め込み φ の差を抑制する平滑化項。
        """
        φ = self.node_embed
        dx = φ - torch.roll(φ, shifts=-1, dims=0)
        dy = φ - torch.roll(φ, shifts=-1, dims=1)
        return 0.5 * (dx.pow(2).mean() + dy.pow(2).mean())

    def theta_smooth_loss(self, a: torch.Tensor) -> torch.Tensor:
        """EN: Smoothness on a(x,μ) emitted by the MLP (no recomputation).
           JA: MLP 出力 a(x,μ) の平滑化（再計算しない）。
        """
        a0, a1 = a[..., 0, :], a[..., 1, :]                   # (L,L,3)
        dx0 = a0 - torch.roll(a0, shifts=-1, dims=0)          # 同一μ=0での隣接差分
        dy1 = a1 - torch.roll(a1, shifts=-1, dims=1)          # 同一μ=1での隣接差分
        return 0.5 * (dx0.pow(2).mean() + dy1.pow(2).mean())

    def reg_phi_l2(self) -> torch.Tensor:
        """EN: L2 regularization on node embeddings φ.
           JA: ノード埋め込み φ のL2正則化。
        """
        return self.node_embed.pow(2).mean()
