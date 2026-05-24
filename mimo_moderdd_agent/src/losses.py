from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    d = x2 + y2 - 2.0 * x @ y.T
    return d.clamp_min(0.0)


def median_bandwidth(x: torch.Tensor, min_sigma: float = 0.05, max_points: int = 512) -> torch.Tensor:
    """Median heuristic bandwidth for Gaussian kernels."""
    with torch.no_grad():
        if x.shape[0] > max_points:
            idx = torch.randperm(x.shape[0], device=x.device)[:max_points]
            x = x[idx]
        d = pairwise_sq_dists(x, x)
        vals = d[d > 1e-12]
        if vals.numel() == 0:
            return torch.tensor(min_sigma, device=x.device)
        sigma = torch.sqrt(torch.median(vals) + 1e-12)
        return sigma.clamp_min(min_sigma)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=x.device, dtype=x.dtype)
    d = pairwise_sq_dists(x, y)
    return torch.exp(-d / (2.0 * sigma.pow(2).clamp_min(1e-12)))


def cauchy_schwarz_kernel_divergence(
    real: torch.Tensor,
    syn: torch.Tensor,
    sigma: torch.Tensor | float | None = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Kernelized Cauchy-Schwarz divergence.

    D_CS(p, q) = -log( <p,q>^2 / (<p,p><q,q>) ).
    The inner products are estimated by Gaussian kernel averages.
    """
    if sigma is None:
        sigma = median_bandwidth(real.detach())
    k_rr = gaussian_kernel(real, real, sigma).mean()
    k_ss = gaussian_kernel(syn, syn, sigma).mean()
    k_rs = gaussian_kernel(real, syn, sigma).mean()
    ratio = (k_rs.pow(2) + eps) / (k_rr * k_ss + eps)
    loss = -torch.log(ratio.clamp_min(eps))
    stats = {
        "k_rr": float(k_rr.detach().cpu()),
        "k_ss": float(k_ss.detach().cpu()),
        "k_rs": float(k_rs.detach().cpu()),
        "sigma": float(sigma.detach().cpu()) if torch.is_tensor(sigma) else float(sigma),
    }
    return loss, stats


def matrix_renyi2_entropy(x: torch.Tensor, sigma: torch.Tensor | float | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Second-order matrix Rényi entropy from a normalized Gram matrix.

    Given a positive Gram matrix K, A=K/tr(K), H_2(A)=-log(tr(A^2)).
    This scalar increases when samples are more diverse in kernel space.
    """
    if x.shape[0] <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    if sigma is None:
        sigma = median_bandwidth(x.detach())
    k = gaussian_kernel(x, x, sigma)
    a = k / (torch.trace(k) + eps)
    return -torch.log((a * a).sum().clamp_min(eps))


def entropy_matching_loss(real: torch.Tensor, syn: torch.Tensor, sigma: torch.Tensor | float | None = None) -> torch.Tensor:
    h_r = matrix_renyi2_entropy(real.detach(), sigma=sigma)
    h_s = matrix_renyi2_entropy(syn, sigma=sigma)
    return (h_s - h_r).pow(2)


def mode_coverage_loss(
    syn: torch.Tensor,
    centers: torch.Tensor,
    target_probs: torch.Tensor,
    tau: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match synthetic soft occupancy over real modes.

    centers: [K, D], target_probs: [K]. The synthetic distribution induces a
    soft assignment q_s over K modes. We use symmetric KL to avoid ignoring tail modes.
    """
    if centers.shape[0] == 0:
        z = torch.zeros((), device=syn.device, dtype=syn.dtype)
        return z, torch.empty(0, device=syn.device)
    d = pairwise_sq_dists(syn, centers)
    logits = -d / max(tau, 1e-6)
    assign = torch.softmax(logits, dim=1)
    q = assign.mean(dim=0).clamp_min(eps)
    q = q / q.sum()
    p = target_probs.to(syn.device).to(syn.dtype).clamp_min(eps)
    p = p / p.sum()
    kl_pq = (p * (p.log() - q.log())).sum()
    kl_qp = (q * (q.log() - p.log())).sum()
    return 0.5 * (kl_pq + kl_qp), q
