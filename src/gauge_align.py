# src/gauge_align.py
# =============================================================================
# EN: Gauge alignment utilities: optimize local gauge transforms g_x ∈ SU(2)
#     to align a predicted link field U_pred with a reference field U_true.
# JA: ゲージ整合ユーティリティ。局所ゲージ変換 g_x ∈ SU(2) を最適化し、
#     予測リンク U_pred を参照リンク U_true にできるだけ一致させます。
# =============================================================================

import torch
import torch.nn as nn

from .lattice import Lattice
from .groups import su2_exp


class GaugeAligner(nn.Module):
    r"""
    EN: Optimize local gauge transforms g_x ∈ SU(2) on a 2D lattice.
    JA: 2次元格子上の局所ゲージ変換 g_x ∈ SU(2) を最適化するクラス。

    - Each site (x,y) has parameter θ(x,y) ∈ ℝ³ (su(2) algebra).
    - Mapped via su2_exp(θ) to g(x,y) ∈ SU(2).
    """
    def __init__(self, lat: Lattice, device: torch.device | None = None):
        super().__init__()
        self.lat = lat
        self.device = device if device is not None else torch.device("cpu")
        # θ_x initialized at zero → g_x ≈ I
        self.theta = nn.Parameter(
            torch.zeros(lat.L, lat.L, 3, dtype=torch.float32, device=self.device)
        )

    @property
    def g(self) -> torch.Tensor:
        """EN: Current gauge transforms g_x ∈ SU(2).  
           JA: 現在のゲージ変換行列 g_x ∈ SU(2) を返す。
        """
        return su2_exp(self.theta.to(self.device))

    def loss(self, U_pred: torch.Tensor, U_true: torch.Tensor) -> torch.Tensor:
        """
        EN: Frobenius MSE of links after gauge-transforming U_true.
        JA: U_true にゲージ変換を適用後、U_pred との差をフロベニウス二乗平均で測る。

        Args:
            U_pred, U_true: (L,L,2,2,2) complex SU(2) link fields
        Returns:
            scalar loss (float32 tensor)
        """
        L = self.lat.L
        g = self.g
        diffs = []
        for x in range(L):
            for y in range(L):
                for mu in (0, 1):
                    xn, yn = self.lat.neighbor(x, y, mu)
                    # U_true^g(x,μ) = g_x U_true g_{x+μ}^†
                    U_t = g[x, y] @ U_true[x, y, mu] @ g[xn, yn].conj().transpose(-1, -2)
                    diff = U_pred[x, y, mu] - U_t
                    diffs.append(diff)
        diffs = torch.stack(diffs)  # shape: (L*L*2,2,2)
        return (diffs.abs() ** 2).mean()


@torch.no_grad()
def _final_rmse(aligner: GaugeAligner, U_pred: torch.Tensor, U_true: torch.Tensor) -> float:
    """EN/JA: Compute sqrt(MSE) of link differences after alignment."""
    L = aligner.lat.L
    g = aligner.g
    diffs = []
    for x in range(L):
        for y in range(L):
            for mu in (0, 1):
                xn, yn = aligner.lat.neighbor(x, y, mu)
                U_t = g[x, y] @ U_true[x, y, mu] @ g[xn, yn].conj().transpose(-1, -2)
                diff = U_pred[x, y, mu] - U_t
                diffs.append(diff)
    diffs = torch.stack(diffs)
    mse = (diffs.abs() ** 2).mean()
    return float(torch.sqrt(mse).item())


def gauge_aligned_link_error(
    U_pred: torch.Tensor,
    U_true: torch.Tensor,
    lat: Lattice,
    steps: int = 200,
    lr: float = 0.05,
    verbose: bool = False,
) -> float:
    r"""
    EN: Optimize gauge transforms and return RMSE between U_pred and aligned U_true.
    JA: ゲージ変換を最適化し、U_pred と整合した U_true の RMSE を返す。

    Args:
        U_pred, U_true: (L,L,2,2,2) complex link fields (same shape/dtype/device)
        lat: Lattice
        steps: inner optimization steps
        lr: Adam learning rate
        verbose: print progress every ~20% of steps
    Returns:
        float: gauge-aligned RMSE
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
