# src/lattice.py
# =============================================================================
# EN: 2D periodic lattice utilities and gauge-invariant observables.
# JA: 2次元周期格子ユーティリティとゲージ不変量計算。
# =============================================================================

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import torch


# -----------------------------------------------------------------------------
# Lattice geometry / 格子幾何
# -----------------------------------------------------------------------------
@dataclass
class Lattice:
    r"""EN: Simple 2D periodic lattice.
        JA: 2次元の周期境界格子の簡易実装。
    """
    L: int = 4  # EN: linear size / JA: 一辺のサイズ

    # ----- coordinate helpers / 座標補助 -----
    def coords(self) -> List[Tuple[int, int]]:
        """Enumerate all (x,y) coordinates."""
        return [(x, y) for x in range(self.L) for y in range(self.L)]

    def in_bounds(self, x: int, y: int) -> bool:
        """Check if (x,y) ∈ [0,L)×[0,L)."""
        return 0 <= x < self.L and 0 <= y < self.L

    # ----- neighbors / 近傍 -----
    def neighbor(self, x: int, y: int, mu: int) -> Tuple[int, int]:
        """Periodic forward neighbor (+x if mu=0, +y if mu=1)."""
        if mu == 0:
            return ((x + 1) % self.L, y)
        else:
            return (x, (y + 1) % self.L)

    def prev_neighbor(self, x: int, y: int, mu: int) -> Tuple[int, int]:
        """Periodic backward neighbor (−x if mu=0, −y if mu=1)."""
        if mu == 0:
            return ((x - 1) % self.L, y)
        else:
            return (x, (y - 1) % self.L)

    # ----- iterators / 反復子 -----
    def iter_sites(self) -> Iterator[Tuple[int, int]]:
        """Iterate all sites (x,y)."""
        for x in range(self.L):
            for y in range(self.L):
                yield (x, y)

    def iter_links(self) -> Iterator[Tuple[int, int, int]]:
        """Iterate all oriented links (x,y,mu) with mu∈{0,1}."""
        for x, y in self.iter_sites():
            yield (x, y, 0)
            yield (x, y, 1)


# -----------------------------------------------------------------------------
# Helpers / 補助関数
# -----------------------------------------------------------------------------
def eye2_like(U: torch.Tensor) -> torch.Tensor:
    """2x2 identity on same device/dtype as U."""
    return torch.eye(2, dtype=U.dtype, device=U.device)


def _adjoint(M: torch.Tensor) -> torch.Tensor:
    """Matrix conjugate transpose (dagger)."""
    return M.conj().transpose(-1, -2)


# -----------------------------------------------------------------------------
# Gauge-invariant observables / ゲージ不変量
# -----------------------------------------------------------------------------
def average_plaquette_traces(U: torch.Tensor, lat: Lattice) -> torch.Tensor:
    r"""
    EN: Mean Tr(P) over all 1×1 plaquettes.
    JA: 全 1×1 プラークエットのトレース平均。

    Args:
        U: (L,L,2,2,2) complex link variables
        lat: Lattice
    Returns:
        scalar complex tensor
    """
    L = lat.L
    traces = []
    for x in range(L):
        for y in range(L):
            xp, y0 = lat.neighbor(x, y, 0)
            x0, yp = lat.neighbor(x, y, 1)
            Ux = U[x, y, 0]
            Uy = U[xp, y, 1]
            Ux2 = U[x, yp, 0]
            Uy2 = U[x, y, 1]
            P = Ux @ Uy @ _adjoint(Ux2) @ _adjoint(Uy2)
            traces.append(torch.trace(P))
    return torch.stack(traces).mean()


def wilson_loop_trace(U: torch.Tensor, lat: Lattice, Rx: int, Ry: int) -> torch.Tensor:
    r"""
    EN: Average trace of an Rx×Ry Wilson loop (averaged over all base points).
    JA: Rx×Ry ウィルソンループのトレースの格子平均。

    Args:
        U: (L,L,2,2,2) complex link variables
        lat: Lattice
        Rx, Ry: rectangle size
    Returns:
        scalar complex tensor
    """
    L = lat.L
    loops = []
    I2 = eye2_like(U)

    for x in range(L):
        for y in range(L):
            X, Y = x, y
            M = I2

            # +x × Rx
            for _ in range(Rx):
                M = M @ U[X, Y, 0]
                X, Y = lat.neighbor(X, Y, 0)

            # +y × Ry
            for _ in range(Ry):
                M = M @ U[X, Y, 1]
                X, Y = lat.neighbor(X, Y, 1)

            # −x × Rx
            for _ in range(Rx):
                Xp, Yp = lat.prev_neighbor(X, Y, 0)
                M = M @ _adjoint(U[Xp, Yp, 0])
                X, Y = Xp, Yp

            # −y × Ry
            for _ in range(Ry):
                Xp, Yp = lat.prev_neighbor(X, Y, 1)
                M = M @ _adjoint(U[Xp, Yp, 1])
                X, Y = Xp, Yp

            loops.append(torch.trace(M))

    return torch.stack(loops).mean()


def creutz_ratio(U: torch.Tensor, lat: Lattice, R: int) -> torch.Tensor:
    r"""
    EN: Creutz ratio χ(R,R):
        χ(R,R) = -log( W(R,R) W(R-1,R-1) / ( W(R,R-1) W(R-1,R) ) )

    JA: Creutz 比 χ(R,R)：
        χ(R,R) = -log( W(R,R) W(R-1,R-1) / ( W(R,R-1) W(R-1,R) ) )

    Args:
        U: (L,L,2,2,2) complex link variables
        lat: Lattice
        R: loop linear size (must be ≥2)
    Returns:
        real scalar tensor
    """
    if R < 2:
        raise ValueError("Creutz ratio requires R ≥ 2")

    WRR   = wilson_loop_trace(U, lat, R, R)
    WmRmR = wilson_loop_trace(U, lat, R - 1, R - 1)
    WRm   = wilson_loop_trace(U, lat, R, R - 1)
    WmR   = wilson_loop_trace(U, lat, R - 1, R)

    eps = torch.finfo(torch.float32).eps
    val = -torch.log(
        (WRR.abs() + eps) * (WmRmR.abs() + eps) /
        ((WRm.abs() + eps) * (WmR.abs() + eps))
    )
    return val.real
