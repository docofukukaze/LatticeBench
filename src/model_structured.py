# src/model_structured.py
r"""
EN: Structured SU(2) model for a small 2D lattice with built-in inductive bias.
    - Node embeddings φ(x) ∈ R^H
    - Shared MLP that emits su(2) algebra vectors a(x,μ) ∈ R^3
    - Exponential map su2_exp(a) → U(x,μ) ∈ SU(2)
    Provides helper losses: φ-smoothness, θ(a)-smoothness, and φ L2 regularization.

JA: 小さな2次元格子向けの構造化SU(2)モデル（帰納的バイアス付き）。
    - ノード埋め込み φ(x) ∈ R^H
    - 共有MLPで su(2) 代数ベクトル a(x,μ) ∈ R^3 を生成
    - 指数写像 su2_exp(a) で U(x,μ) ∈ SU(2) を構成
    φの平滑化損失、a(=θ)の平滑化損失、φのL2正則化も提供します。
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn

from .lattice import Lattice
from .groups import su2_exp


def _unitarity_penalty(U: torch.Tensor) -> torch.Tensor:
    """EN: Penalize deviation from unitarity and det=1 (soft).
       JA: ユニタリティと det=1 からの乖離をソフトに罰則化。
    """
    I2 = torch.eye(2, dtype=U.dtype, device=U.device)
    Uh = U.conj().transpose(-1, -2)
    eye_dev = ((Uh @ U) - I2).abs().pow(2).mean()
    det = torch.linalg.det(U)
    det_dev = (det.real - 1.0).pow(2).mean() + (det.imag).pow(2).mean()
    return eye_dev + det_dev


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
    ) -> None:
        """
        Args:
            lat: lattice description / 格子情報
            hidden_dim: node embedding size φ の次元
            mlp_width: shared MLP width / 共有MLPの幅
            embed_init_scale: φ の初期化スケール
        """
        super().__init__()
        self.lat = lat
        self.H = int(hidden_dim)

        # Node embeddings φ(x) ∈ R^H  / ノード埋め込み
        self.node_embed = nn.Parameter(
            torch.randn(lat.L, lat.L, self.H, dtype=torch.float32) * embed_init_scale
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

    # ------------------------------------------------------------------ #
    # Core forward / コア計算
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _neighbors(self, x: int, y: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return (x+μ) for μ=0,1 with periodic BC. 周期境界の近傍座標を返す。"""
        xp, y0 = self.lat.neighbor(x, y, 0)
        x0, yp = self.lat.neighbor(x, y, 1)
        return (xp, y0), (x0, yp)

    def forward_links(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            U:  (L,L,2,2,2) complex SU(2) link matrices
            a:  (L,L,2,3)   real su(2) algebra vectors emitted by MLP
        """
        L = self.lat.L
        # a(x,μ) container (CPU/CUDAは .to() でモデル移動時に一緒に移動)
        a = self.node_embed.new_zeros(L, L, 2, 3)  # real32
        for x in range(L):
            for y in range(L):
                h = self.node_embed[x, y]  # φ(x)
                (xp, y0), (x0, yp) = self._neighbors(x, y)
                # μ=0 : (x → x+ê_x), μ=1 : (y → y+ê_y)
                a[x, y, 0] = self.link_mlp(torch.cat([h, self.node_embed[xp, y0]], dim=-1))
                a[x, y, 1] = self.link_mlp(torch.cat([h, self.node_embed[x0, yp]], dim=-1))

        # su(2) → SU(2)
        U = su2_exp(a.view(L * L * 2, 3)).view(L, L, 2, 2, 2)
        return U, a

    def plaquette_traces(self) -> torch.Tensor:
        """EN: Tr(P) for all 1×1 plaquettes.
           JA: 全 1×1 プラークエットのトレース（複素数）を返す。
        """
        L = self.lat.L
        U, _ = self.forward_links()
        traces = []
        for x in range(L):
            for y in range(L):
                xp, y0 = self.lat.neighbor(x, y, 0)
                x0, yp = self.lat.neighbor(x, y, 1)
                Ux = U[x, y, 0]
                Uy = U[xp, y, 1]
                Ux2 = U[x, yp, 0]
                Uy2 = U[x, y, 1]
                P = Ux @ Uy @ Ux2.conj().transpose(-1, -2) @ Uy2.conj().transpose(-1, -2)
                traces.append(torch.trace(P))
        return torch.stack(traces)

    # ------------------------------------------------------------------ #
    # Optional penalties used by train.py / 学習で使う補助損失
    # ------------------------------------------------------------------ #
    def unitarity_loss(self) -> torch.Tensor:
        """EN: Soft unitarity + det=1 penalty. JA: ユニタリティ+det=1罰則。"""
        U, _ = self.forward_links()
        return _unitarity_penalty(U)

    def phi_smooth_loss(self) -> torch.Tensor:
        """EN: Encourage neighboring node embeddings φ to be close.
           JA: 近傍ノード埋め込み φ の差を抑制する平滑化項。
        """
        L = self.lat.L
        diffs = []
        for x in range(L):
            for y in range(L):
                h = self.node_embed[x, y]
                xp, y0 = self.lat.neighbor(x, y, 0)
                x0, yp = self.lat.neighbor(x, y, 1)
                diffs.append((h - self.node_embed[xp, y0]).pow(2).mean())
                diffs.append((h - self.node_embed[x0, yp]).pow(2).mean())
        return torch.stack(diffs).mean()

    def theta_smooth_loss(self, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        """EN: Smoothness on a(x,μ) emitted by the MLP. If 'a' is provided
               (shape L×L×2×3), reuse it to avoid recomputation.
           JA: MLP 出力 a(x,μ) の平滑化。引数 a（形状 L×L×2×3）が渡されれば
               再計算せずそれを利用。
        """
        L = self.lat.L
        if a is None:
            _, a = self.forward_links()  # recompute if not provided

        diffs = []
        for x in range(L):
            for y in range(L):
                xp, y0 = self.lat.neighbor(x, y, 0)
                x0, yp = self.lat.neighbor(x, y, 1)
                # same μ across neighbors / 同じ向きμでの近傍差
                diffs.append((a[x, y, 0] - a[xp, y0, 0]).pow(2).mean())
                diffs.append((a[x, y, 1] - a[x0, yp, 1]).pow(2).mean())
        return torch.stack(diffs).mean()

    def reg_phi_l2(self) -> torch.Tensor:
        """EN: L2 regularization on node embeddings φ.
           JA: ノード埋め込み φ のL2正則化。
        """
        return self.node_embed.pow(2).mean()
