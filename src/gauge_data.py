# src/gauge_data.py
"""
Teacher SU(2) gauge configuration utilities
===========================================
EN: Provides a minimal "teacher" SU(2) gauge field on a 2D periodic lattice.
JA: 2次元周期格子上の最小限な教師SU(2)ゲージ場ユーティリティを提供。

This module defines:
- SU2GaugeConfig: holds algebra parameters a_{x,mu} ∈ R^3 and exposes:
    - randomize(scale): normal init in su(2) algebra
    - links(): map a -> U ∈ SU(2) via exponential map
    - plaquette(x,y): 1x1 Wilson loop at (x,y)
    - plaquette_traces(): vector of Tr(P) over lattice

Notes:
- Uses groups.su2_exp for the exponential map with unitary structure.
- Assumes periodic boundary via Lattice.neighbor.
"""

import torch

from .groups import su2_exp, get_device
from .lattice import Lattice

Device = get_device()
I2 = torch.eye(2, dtype=torch.complex64, device=Device)

class SU2GaugeConfig:
    """
    EN: Teacher configuration: su(2) algebra params a[x,y,mu,3] -> links U[x,y,mu,2,2].
    JA: 教師設定：su(2)代数パラメタ a[x,y,mu,3] をリンク U[x,y,mu,2,2] に写像。
    """
    def __init__(self, lat: Lattice):
        self.lat = lat
        # a[x,y,mu,3], mu ∈ {0(=x),1(=y)}
        self.a = torch.zeros(lat.L, lat.L, 2, 3, dtype=torch.float32, device=Device)

    @torch.no_grad()
    def randomize(self, scale: float = 0.6) -> None:
        """
        EN: Initialize su(2) algebra parameters with Normal(0,scale).
        JA: su(2) 代数パラメタを正規分布(0,scale)で初期化。
        """
        self.a.normal_(mean=0.0, std=scale)

    def links(self) -> torch.Tensor:
        """
        EN: Map a -> U via su2_exp for all links.
        JA: すべてのリンクについて a -> U に指数写像。
        Returns:
            U: (L,L,2,2,2) complex64, unitary with det=1 (up to num. error)
        """
        L = self.lat.L
        a = self.a.view(L * L * 2, 3)
        U = su2_exp(a)
        return U.view(L, L, 2, 2, 2)

    def plaquette(self, x: int, y: int) -> torch.Tensor:
        """
        EN: 1x1 Wilson loop at (x,y).
        JA: (x,y) を基点とする 1x1 Wilson ループ。
        """
        U = self.links()
        xp, y0 = self.lat.neighbor(x, y, 0)  # +x
        x0, yp = self.lat.neighbor(x, y, 1)  # +y
        Ux  = U[x,  y,  0]
        Uy  = U[xp, y,  1]
        Ux2 = U[x,  yp, 0]
        Uy2 = U[x,  y,  1]
        return Ux @ Uy @ Ux2.conj().transpose(-1, -2) @ Uy2.conj().transpose(-1, -2)

    def plaquette_traces(self) -> torch.Tensor:
        """
        EN: Flattened vector of Tr(P) over all 1x1 plaquettes on the lattice.
        JA: 格子上の全 1x1 プラークエットの Tr(P) を一次元ベクトルで返す。
        Shape: (L*L,), dtype: complex64
        """
        L = self.lat.L
        traces = []
        for x in range(L):
            for y in range(L):
                P = self.plaquette(x, y)
                traces.append(torch.trace(P))
        return torch.stack(traces)
