# src/groups.py
# EN: SU(2) utilities (Pauli matrices, exponential map, unitary penalty).
# JA: SU(2) のユーティリティ（パウリ行列、指数写像、ユニタリ罰則）。

import torch

# Device helper / デバイス補助
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pauli(device=None) -> torch.Tensor:
    """EN: Return Pauli matrices σ1,σ2,σ3 as (3,2,2) complex tensor.
       JA: パウリ行列 σ1,σ2,σ3 を (3,2,2) 複素テンソルで返す。
    """
    if device is None: device = get_device()
    i = torch.complex(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    s1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    s2 = torch.tensor([[0, -i], [i, 0]], dtype=torch.complex64, device=device)
    s3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    return torch.stack([s1, s2, s3], dim=0)

def su2_exp(a: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
    """EN: Exponential map from su(2) algebra vector a (R^3) to SU(2) matrix.
       JA: su(2) 代数ベクトル a (R^3) を SU(2) 行列に指数写像で変換。
       Args:
         a: (..., 3) float32
         sigma: (3,2,2) complex Pauli matrices; if None, built on the fly.
    """
    device = a.device
    if sigma is None:
        sigma = pauli(device)
    I2 = torch.eye(2, dtype=torch.complex64, device=device)
    a = a.to(torch.float32)
    norm = torch.clamp(a.norm(dim=-1, keepdim=True), min=1e-8)
    ahat = a / norm
    theta = norm[..., 0]
    c = torch.cos(theta); s = torch.sin(theta)
    comp = (ahat[..., 0][..., None, None] * sigma[0]
          + ahat[..., 1][..., None, None] * sigma[1]
          + ahat[..., 2][..., None, None] * sigma[2])
    i = torch.complex(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    return c[..., None, None] * I2 + i * s[..., None, None] * comp

def su2_unitarity_penalty(U: torch.Tensor) -> torch.Tensor:
    """EN: Penalty for deviation from unitarity and det=1.
       JA: ユニタリティおよび det=1 からのずれに対する罰則。
    """
    device = U.device
    I2 = torch.eye(2, dtype=torch.complex64, device=device)
    Uh = U.conj().transpose(-1, -2)
    eye_dev = ((Uh @ U) - I2).abs().pow(2).mean()
    det = torch.linalg.det(U)
    det_dev = (det.real - 1.0).pow(2).mean() + (det.imag).pow(2).mean()
    return eye_dev + det_dev
