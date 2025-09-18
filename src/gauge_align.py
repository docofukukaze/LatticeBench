# src/gauge_align.py
r"""
EN: Gauge alignment utilities: optimize local gauge transforms g_x ∈ SU(2)
    to align a predicted link field U_pred with a reference field U_true.
JA: ゲージ整合ユーティリティ。局所ゲージ変換 g_x ∈ SU(2) を最適化し、
    予測リンク U_pred を参照リンク U_true にできるだけ一致させます。

Usage:
    from .gauge_align import gauge_aligned_link_error
    err = gauge_aligned_link_error(U_pred, U_true, lat, steps=200, lr=0.05)

Notes:
- We parameterize g_x via su(2) algebra θ_x ∈ R^3 and map with su2_exp(θ_x).
- The alignment is computed by minimizing the Frobenius RMSE of links after
  applying the gauge transform to U_true: U_true^g(x,μ) = g_x U_true(x,μ) g_{x+μ}^\dagger.
"""

import torch
import torch.nn as nn

from .lattice import Lattice
from .groups import su2_exp


class GaugeAligner(nn.Module):
    r"""EN: Optimize local gauge transforms g_x ∈ SU(2) on a 2D lattice.
        JA: 2次元格子上の局所ゲージ変換 g_x ∈ SU(2) を最適化するクラス。
    """
    def __init__(self, lat: Lattice, device: torch.device | None = None):
        super().__init__()
        self.lat = lat
        self.device = device if device is not None else torch.device("cpu")
        # θ_x ∈ R^3 (algebra); initialized at zero → g_x ≈ I
        # θ_x は R^3 のsu(2)代数パラメタ。初期値ゼロで単位行列近傍から開始。
        self.theta = nn.Parameter(torch.zeros(lat.L, lat.L, 3, dtype=torch.float32, device=self.device))

    def g(self) -> torch.Tensor:
        """EN: Map θ to SU(2) matrices g via su2_exp.
           JA: θ を su2_exp で SU(2) 行列 g に写像。
        """
        return su2_exp(self.theta.to(self.device))

    def loss(self, U_pred: torch.Tensor, U_true: torch.Tensor) -> torch.Tensor:
        r"""EN: Mean squared Frobenius distance after gauge-transforming U_true.
            JA: U_true にゲージ変換を施した後の Frobenius 距離（二乗平均）を損失に。
        """
        L = self.lat.L
        g = self.g()
        s = U_pred.new_tensor(0.0, dtype=torch.float32)
        for x in range(L):
            for y in range(L):
                for mu in (0, 1):
                    xn, yn = self.lat.neighbor(x, y, mu)
                    # U_true^g = g_x U_true g_{x+μ}^†
                    U_t = g[x, y] @ U_true[x, y, mu] @ g[xn, yn].conj().transpose(-1, -2)
                    diff = U_pred[x, y, mu] - U_t
                    s = s + (diff.abs() ** 2).mean()
        return s / (L * L * 2)


@torch.no_grad()
def _final_rmse(aligner: GaugeAligner, U_pred: torch.Tensor, U_true: torch.Tensor) -> float:
    """EN/JA: Compute sqrt(MSE) of link differences after alignment."""
    L = aligner.lat.L
    g = aligner.g()
    s = U_pred.new_tensor(0.0, dtype=torch.float32)
    for x in range(L):
        for y in range(L):
            for mu in (0, 1):
                xn, yn = aligner.lat.neighbor(x, y, mu)
                U_t = g[x, y] @ U_true[x, y, mu] @ g[xn, yn].conj().transpose(-1, -2)
                diff = U_pred[x, y, mu] - U_t
                s = s + (diff.abs() ** 2).mean()
    mse = s / (L * L * 2)
    return float(torch.sqrt(mse).item())


def gauge_aligned_link_error(
    U_pred: torch.Tensor,
    U_true: torch.Tensor,
    lat: Lattice,
    steps: int = 200,
    lr: float = 0.05,
    verbose: bool = False,
) -> float:
    r"""EN: Run a small inner optimization to align gauges and return RMSE.
        JA: 簡単な内側最適化でゲージ整合を行い、RMSE を返します。
    Args:
        U_pred, U_true: (L,L,2,2,2) complex64/complex128 (same dtype/device).
        lat: Lattice
        steps: alignment steps
        lr: Adam learning rate
        verbose: print debug stats during alignment
    """
    assert U_pred.shape == U_true.shape, "U_pred and U_true must have the same shape."
    device = U_pred.device
    aligner = GaugeAligner(lat, device=device).to(device)
    opt = torch.optim.Adam(aligner.parameters(), lr=lr)

    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        Lval = aligner.loss(U_pred, U_true)
        Lval.backward()
        opt.step()
        if verbose and (it % max(1, steps // 5) == 0 or it == steps - 1):
            print(f"[align {it:04d}/{steps}] loss={float(Lval.item()):.4e}")

    return _final_rmse(aligner, U_pred, U_true)
