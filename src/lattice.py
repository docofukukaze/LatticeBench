# src/lattice.py
r"""
EN: 2D periodic lattice utilities and gauge-invariant helpers.
JA: 2次元・周期境界の格子ユーティリティとゲージ不変量の補助関数。

This module provides:
- Lattice: minimal 2D periodic lattice abstraction.
- plaquette_indices: indices of the 1x1 plaquette edges (indexing only).
- wilson_loop_path: closed rectangular loop path from (x,y) with size Rx×Ry.
- average_plaquette_traces(U, lat): mean Tr(plaquette) over all sites.
- wilson_loop_trace(U, lat, Rx, Ry): average trace of Rx×Ry Wilson loop.
- creutz_ratio(U, lat, R): Creutz ratio χ(R,R) from Wilson loops.

Tensor conventions:
- U[x,y,mu] ∈ C^{2×2}, with mu ∈ {0,1} (0: +x direction, 1: +y direction).
- Reverse links are implemented by the conjugate transpose U^\dagger.
- Periodic boundary conditions via Lattice.neighbor / prev_neighbor.

Note:
- Functions here do **not** depend on model internals; they only consume
  link variables U and the lattice geometry, and compute gauge-invariant
  observables in a straightforward way.
"""

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import torch


# ---------------------------------------------------------------------
# Lattice geometry / 格子幾何
# ---------------------------------------------------------------------
@dataclass
class Lattice:
    r"""EN: Simple 2D periodic lattice.
        JA: 2次元の周期境界格子の簡易実装。
    """
    L: int = 4  # EN: linear size / JA: 一辺のサイズ

    # ----- coordinate helpers / 座標補助 -----
    def coords(self) -> List[Tuple[int, int]]:
        r"""EN: Enumerate all (x,y) coordinates.
            JA: 全ての (x,y) を列挙。
        """
        return [(x, y) for x in range(self.L) for y in range(self.L)]

    def in_bounds(self, x: int, y: int) -> bool:
        r"""EN: Check (x,y) ∈ [0,L)×[0,L).
            JA: (x,y) が [0,L)×[0,L) にあるかを判定。
        """
        return 0 <= x < self.L and 0 <= y < self.L

    # ----- neighbors / 近傍 -----
    def neighbor(self, x: int, y: int, mu: int) -> Tuple[int, int]:
        r"""EN: Periodic forward neighbor (+x if mu=0, +y if mu=1).
            JA: 周期境界の前方近傍（mu=0 なら +x, mu=1 なら +y）。
        """
        if mu == 0:
            return ((x + 1) % self.L, y)
        else:
            return (x, (y + 1) % self.L)

    def prev_neighbor(self, x: int, y: int, mu: int) -> Tuple[int, int]:
        r"""EN: Periodic backward neighbor (−x if mu=0, −y if mu=1).
            JA: 周期境界の後方近傍（mu=0 なら −x, mu=1 なら −y）。
        """
        if mu == 0:
            return ((x - 1) % self.L, y)
        else:
            return (x, (y - 1) % self.L)

    # ----- iterators / 反復子 -----
    def iter_sites(self) -> Iterator[Tuple[int, int]]:
        r"""EN: Iterate over sites (x,y).
            JA: サイト (x,y) を順に返す。
        """
        for x in range(self.L):
            for y in range(self.L):
                yield (x, y)

    def iter_links(self) -> Iterator[Tuple[int, int, int]]:
        r"""EN: Iterate oriented links (x,y,mu) with mu∈{0,1}.
            JA: 有向リンク (x,y,mu) を列挙（mu∈{0,1}）。
        """
        for x, y in self.iter_sites():
            yield (x, y, 0)
            yield (x, y, 1)


# ---------------------------------------------------------------------
# Plaquette / Wilson-loop indexing helpers（添字計算のみ）
# ---------------------------------------------------------------------
def plaquette_indices(x: int, y: int, lat: Lattice) -> Tuple[Tuple[int, int, int], ...]:
    r"""EN: Return the 4 oriented links forming the unit plaquette at (x,y), ordered:
            (x,y,+x), (x+1,y,+y), (x,y+1,−x), (x,y,−y)
        JA: (x,y) を起点とする 1×1 プラークエットの4辺リンク添字を返す（上記順）。
    """
    xp, y0 = lat.neighbor(x, y, 0)  # +x
    x0, yp = lat.neighbor(x, y, 1)  # +y
    # forward +x, +y, then reverse −x, −y（逆辺は後で随伴を取る想定）
    return ((x, y, 0), (xp, y, 1), (x, yp, 0), (x, y, 1))


def wilson_loop_path(x: int, y: int, lat: Lattice, Rx: int, Ry: int) -> List[Tuple[int, int, int, int]]:
    r"""EN: Build a closed rectangular path starting at (x,y) with size Rx×Ry.
           Returns a list of (x,y,mu,sign) where sign=+1 means U(x,y,mu),
           and sign=−1 means U(x',y',mu)^\dagger for the backward edge.
        JA: 起点 (x,y) から Rx×Ry の長方形ウィルソンループ経路を返す。
           返り値は (x,y,mu,sign) のリスト。sign=+1 で順方向、−1 で逆方向（随伴）。
    """
    path: List[Tuple[int, int, int, int]] = []
    X, Y = x, y
    # +x * Rx
    for _ in range(Rx):
        path.append((X, Y, 0, +1))
        X, Y = lat.neighbor(X, Y, 0)
    # +y * Ry
    for _ in range(Ry):
        path.append((X, Y, 1, +1))
        X, Y = lat.neighbor(X, Y, 1)
    # −x * Rx
    for _ in range(Rx):
        Xp, Yp = lat.prev_neighbor(X, Y, 0)
        path.append((Xp, Yp, 0, -1))
        X, Y = Xp, Yp
    # −y * Ry
    for _ in range(Ry):
        Xp, Yp = lat.prev_neighbor(X, Y, 1)
        path.append((Xp, Yp, 1, -1))
        X, Y = Xp, Yp
    return path


# ---------------------------------------------------------------------
# Gauge-invariant observables / ゲージ不変量
# ---------------------------------------------------------------------
def _eye_like_2x2(U: torch.Tensor) -> torch.Tensor:
    """EN: 2x2 identity on same device/dtype as U.
       JA: U と同じ device/dtype の 2×2 単位行列を返す。
    """
    return torch.eye(2, dtype=U.dtype, device=U.device)


def average_plaquette_traces(U: torch.Tensor, lat: Lattice) -> torch.Tensor:
    r"""EN: Mean Tr(P) over all unit plaquettes.
        JA: 全 1×1 プラークエットの Tr(P) の平均を返す。
    Args:
        U: (L,L,2,2,2) complex link variables
        lat: Lattice
    Returns:
        scalar complex tensor (mean of traces)
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
            P = Ux @ Uy @ Ux2.conj().transpose(-1, -2) @ Uy2.conj().transpose(-1, -2)
            traces.append(torch.trace(P))
    return torch.stack(traces).mean()


def wilson_loop_trace(U: torch.Tensor, lat: Lattice, Rx: int, Ry: int) -> torch.Tensor:
    r"""EN: Average trace of an Rx×Ry Wilson loop (averaged over all base points).
        JA: Rx×Ry ウィルソンループのトレースの格子平均（全起点平均）。
    Args:
        U: (L,L,2,2,2) complex link variables
        lat: Lattice
        Rx, Ry: rectangle size along +x,+y
    Returns:
        scalar complex tensor
    """
    L = lat.L
    I2 = _eye_like_2x2(U)
    loops = []
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
            # −x × Rx (use adjoint of forward link)
            for _ in range(Rx):
                Xp, Yp = lat.prev_neighbor(X, Y, 0)
                M = M @ U[Xp, Yp, 0].conj().transpose(-1, -2)
                X, Y = Xp, Yp
            # −y × Ry
            for _ in range(Ry):
                Xp, Yp = lat.prev_neighbor(X, Y, 1)
                M = M @ U[Xp, Yp, 1].conj().transpose(-1, -2)
                X, Y = Xp, Yp
            loops.append(torch.trace(M))
    return torch.stack(loops).mean()


def creutz_ratio(U: torch.Tensor, lat: Lattice, R: int) -> torch.Tensor:
    r"""EN: Creutz ratio χ(R,R) using Wilson loops:
           χ(R,R) = -log( W(R,R) * W(R-1,R-1) / ( W(R,R-1) * W(R-1,R) ) )
        JA: Creutz 比 χ(R,R) をウィルソンループから計算：
           χ(R,R) = -log( W(R,R) * W(R-1,R-1) / ( W(R,R-1) * W(R-1,R) ) )
    Args:
        U: (L,L,2,2,2) complex link variables
        lat: Lattice
        R: loop linear size
    Returns:
        real scalar tensor (float)  ※複素誤差を避けるため実部を採用
    """
    WRR   = wilson_loop_trace(U, lat, R, R)
    WmRmR = wilson_loop_trace(U, lat, R - 1, R - 1)
    WRm   = wilson_loop_trace(U, lat, R, R - 1)
    WmR   = wilson_loop_trace(U, lat, R - 1, R)

    eps = 1e-8
    val = -torch.log(
        (WRR.abs() + eps) * (WmRmR.abs() + eps) / ((WRm.abs() + eps) * (WmR.abs() + eps))
    )
    return val.real
