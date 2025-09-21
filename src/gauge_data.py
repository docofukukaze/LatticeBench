# src/gauge_data.py
# =============================================================================
# EN: Teacher SU(2) gauge configuration utilities on a 2D periodic lattice.
# JA: 2次元周期格子上の教師 SU(2) ゲージ構成ユーティリティ。
# =============================================================================

from __future__ import annotations
import torch

from .groups import su2_exp, get_device
from .lattice import Lattice


# -----------------------------------------------------------------------------
# Helpers / 補助関数
# -----------------------------------------------------------------------------
def dagger(M: torch.Tensor) -> torch.Tensor:
    """Conjugate transpose (adjoint)."""
    return M.conj().transpose(-1, -2)


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------
class SU2GaugeConfig:
    r"""
    EN: Teacher configuration: su(2) algebra params a[x,y,mu,3] → links U[x,y,mu,2,2].
    JA: 教師設定：su(2)代数パラメタ a[x,y,mu,3] をリンク U[x,y,mu,2,2] に写像。

    Attributes:
        lat: Lattice geometry
        a:   (L,L,2,3) float32 tensor of su(2) algebra parameters
    """

    def __init__(self, lat: Lattice, device: torch.device | None = None):
        self.lat = lat
        self.device = device if device is not None else get_device()
        # Algebra params θ(x,y,μ) ∈ ℝ³
        self.a = torch.zeros(lat.L, lat.L, 2, 3, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def randomize(self, scale: float = 0.6) -> None:
        """
        EN: Initialize su(2) algebra parameters with Normal(0,scale).
        JA: su(2) 代数パラメタを正規分布(0,scale)で初期化。
        """
        self.a.normal_(mean=0.0, std=scale)

    def links(self, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
        """
        EN: Map algebra params to SU(2) link matrices via exponential map.
        JA: 代数パラメタを SU(2) リンク行列に指数写像。

        Returns:
            U: (L,L,2,2,2) complex tensor (dtype指定可), unitary with det≈1
        """
        L = self.lat.L
        a = self.a.view(L * L * 2, 3)
        U = su2_exp(a)  # (L*L*2,2,2)
        return U.view(L, L, 2, 2, 2).to(dtype)

    def plaquette(self, x: int, y: int, U: torch.Tensor | None = None) -> torch.Tensor:
        """
        EN: Compute 1×1 Wilson loop (plaquette) at site (x,y).
        JA: サイト (x,y) を起点とする 1×1 プラークエットを計算。

        Args:
            x,y: site coordinates
            U:   optional precomputed links (L,L,2,2,2)
        Returns:
            (2,2) complex tensor
        """
        if U is None:
            U = self.links()

        xp, y0 = self.lat.neighbor(x, y, 0)  # +x
        x0, yp = self.lat.neighbor(x, y, 1)  # +y
        Ux  = U[x,  y,  0]
        Uy  = U[xp, y,  1]
        Ux2 = U[x,  yp, 0]
        Uy2 = U[x,  y,  1]
        return Ux @ Uy @ dagger(Ux2) @ dagger(Uy2)

    def plaquette_traces(self, U: torch.Tensor | None = None) -> torch.Tensor:
        """
        EN: Flattened vector of Tr(P) for all plaquettes.
        JA: 格子上すべてのプラークエットの Tr(P) を一次元ベクトルで返す。

        Args:
            U: optional precomputed links (L,L,2,2,2)
        Returns:
            (L*L,) complex tensor
        """
        if U is None:
            U = self.links()
        L = self.lat.L
        traces = []
        for x in range(L):
            for y in range(L):
                P = self.plaquette(x, y, U)
                traces.append(torch.trace(P))
        return torch.stack(traces)
