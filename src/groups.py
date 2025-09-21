# src/groups.py
# =============================================================================
# EN: SU(2) utilities (device helpers, Pauli matrices, exponential map, penalty)
# JA: SU(2) ユーティリティ（デバイス補助、パウリ行列、指数写像、ユニタリ罰則）
# =============================================================================

from __future__ import annotations
from functools import lru_cache
from typing import Optional

import math
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Device helpers / デバイス補助
# -----------------------------------------------------------------------------
def get_device() -> torch.device:
    """
    EN: Prefer CUDA if available, then Apple MPS, else CPU.
    JA: CUDA → MPS → CPU の優先でデバイスを返す。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple MPS (Metal, Apple Silicon)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Pauli matrices / パウリ行列
# -----------------------------------------------------------------------------
def _pauli_matrices(dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    EN: Return (σ1, σ2, σ3) on CPU as a tensor of shape (3, 2, 2).
    JA: (σ1, σ2, σ3) を CPU 上の (3,2,2) テンソルで返す。
    NOTE:
      - We keep them on CPU and move to the correct device lazily with `.to(device)`.
      - Using complex dtype here avoids later dtype promotions/mismatches.
        （最初から complex にしておくと後の dtype 衝突を避けられます）
    """
    i = 1j
    s1 = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    s2 = torch.tensor([[0, -i], [i, 0]], dtype=dtype)
    s3 = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
    return torch.stack([s1, s2, s3], dim=0)


@lru_cache(maxsize=4)
def pauli(dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    EN: Cached Pauli matrices on CPU (by dtype). Call `.to(device)` at the use site.
    JA: CPU 上に dtype ごとにキャッシュしたパウリ行列。使用時に `.to(device)` で移動。
    """
    return _pauli_matrices(dtype=dtype)


# -----------------------------------------------------------------------------
# SU(2) exponential map / SU(2) 指数写像
# -----------------------------------------------------------------------------
def su2_exp(
    a: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    r"""
    EN:
      Exponential map from su(2) algebra vector to SU(2) group element.
      For a ∈ ℝ³, define θ = ||a|| and A = a·σ (2×2 complex).
      Then
          exp(i a·σ) = cos(θ) I + i * sinc(θ) * (a·σ),
      where sinc(θ) = sin(θ)/θ (not PyTorch's π-sinc).
      This form is numerically stable near θ → 0 (no explicit normalization a/||a||).

    JA:
      su(2) 代数ベクトル a から SU(2) 群要素への指数写像。
      a ∈ ℝ³, θ = ||a||, A = a·σ（2×2複素）として
          exp(i a·σ) = cos(θ) I + i * sinc(θ) * (a·σ),
      ここで sinc(θ) = sin(θ)/θ（PyTorch の π-sinc とは定義が異なる点に注意）。
      θ→0 でも a/||a|| の正規化を避けられ、数値的に安定です。

    Args:
      a: (..., 3) real tensor (float32 推奨) / 実数テンソル
      sigma: optional Pauli matrices (3,2,2) complex on the same device as `a`.
             None の場合は CPU キャッシュから取得し `a.device` に移動します。
      out_dtype: output dtype (complex64 推奨)

    Returns:
      U: (..., 2, 2) complex SU(2) matrices (unitary up to FP error).
    """
    # shape checks
    assert a.shape[-1] == 3, "Expected 'a' to have last dimension = 3 (su(2) algebra coords)"

    device = a.device
    a = a.to(torch.float32)  # keep algebra parameters in real32

    # Prepare Pauli matrices on the same device (complex)
    if sigma is None:
        sigma = pauli().to(device)

    # θ = ||a||  (shape: (...,))
    a_real = a.to(torch.float32)
    theta = torch.linalg.norm(a, dim=-1)

    # A = a_x σ1 + a_y σ2 + a_z σ3  (shape: (..., 2, 2))
    # einsum result is real * complex => complex（ただし明示キャストで安全側に）
    a_c = a_real.to(out_dtype)                         # (...,3) -> complex
    A = torch.einsum("...k,kij->...ij", a_c, sigma)    # (...,2,2) complex

    # sinc(θ) = sin(θ)/θ を計算。torch.sinc は sin(πx)/(πx) なので注意。
    # ⇒ sinc(θ) = torch.sinc(θ / π)
    # Shape を (...,1,1) にしてブロードキャストを明示。
    cos_th = torch.cos(theta).to(out_dtype)[..., None, None]
    sinc_th = torch.sinc(theta / math.pi).to(out_dtype)[..., None, None]

    # 単位行列 I_2
    I2 = torch.eye(2, dtype=out_dtype, device=device)

    # exp(i A) = cos θ I + i * sinc θ * A
    # NOTE: ここで out_dtype を complex に固定しているため、dtype 衝突は起きません。
    U = cos_th * I2 + (1j) * sinc_th * A
    return U


# -----------------------------------------------------------------------------
# (Optional) Unitarity penalty / ユニタリティ罰則
# -----------------------------------------------------------------------------
def su2_unitarity_penalty(U: torch.Tensor) -> torch.Tensor:
    """
    EN:
      Soft penalty for deviations from unitarity:
          minimize || U†U - I ||_F^2
      (det(U)=1 is analytically true for exact SU(2), but small FP drift can occur;
       this focuses on U†U≈I which is sufficient in practice.)

    JA:
      ユニタリティ（U†U ≈ I）からの逸脱に対する軽い罰則。
      厳密SU(2)では det(U)=1 だが、FP 誤差で少しズレる可能性があるため、
      実務上は U†U≈I のみを重視すれば十分なことが多い。
    """
    I2 = torch.eye(2, dtype=U.dtype, device=U.device)
    UhU = U.conj().transpose(-1, -2) @ U
    return F.mse_loss(UhU, I2)
