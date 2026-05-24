from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .agent import ResearchAgent
from .data import init_synthetic_from_real, make_toy_dataset
from .encoders import FrozenRandomEncoderBank
from .evaluate import evaluate_synthetic_set
from .losses import gaussian_kernel, matrix_renyi2_entropy, median_bandwidth, mode_coverage_loss
from .mode_bank import build_mode_bank
from .visualize import save_distribution_plot, save_loss_curve


def precompute_real_cache(
    real_features: Dict[str, torch.Tensor],
    real_y: torch.Tensor,
    num_classes: int,
) -> Dict[Tuple[str, int], Dict[str, torch.Tensor]]:
    """Cache real-side terms that do not depend on synthetic samples."""
    cache: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}
    for layer, feat in real_features.items():
        for c in range(num_classes):
            r = feat[real_y == c].detach()
            sigma = median_bandwidth(r)
            k_rr = gaussian_kernel(r, r, sigma).mean().detach()
            h_r = matrix_renyi2_entropy(r, sigma=sigma).detach()
            cache[(layer, c)] = {"real": r, "sigma": sigma.detach(), "k_rr": k_rr, "h_r": h_r}
    return cache


def cached_cs_divergence(real: torch.Tensor, syn: torch.Tensor, sigma: torch.Tensor, k_rr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    k_ss = gaussian_kernel(syn, syn, sigma).mean()
    k_rs = gaussian_kernel(real, syn, sigma).mean()
    ratio = (k_rs.pow(2) + eps) / (k_rr * k_ss + eps)
    return -torch.log(ratio.clamp_min(eps))


def cached_entropy_loss(syn: torch.Tensor, sigma: torch.Tensor, h_r: torch.Tensor) -> torch.Tensor:
    h_s = matrix_renyi2_entropy(syn, sigma=sigma)
    return (h_s - h_r).pow(2)


def compute_distillation_loss(
    encoder: FrozenRandomEncoderBank,
    real_cache: Dict[Tuple[str, int], Dict[str, torch.Tensor]],
    mode_bank,
    syn_x: torch.Tensor,
    syn_y: torch.Tensor,
    num_classes: int,
    w_cs: float,
    w_cov: float,
    w_ent: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    syn_feats = encoder.forward_features(syn_x)
    total = torch.zeros((), device=syn_x.device)
    cs_total = torch.zeros((), device=syn_x.device)
    cov_total = torch.zeros((), device=syn_x.device)
    ent_total = torch.zeros((), device=syn_x.device)
    terms = 0

    for layer in syn_feats.keys():
        for c in range(num_classes):
            s = syn_feats[layer][syn_y == c]
            if s.shape[0] < 2:
                continue
            cached = real_cache[(layer, c)]
            cs = cached_cs_divergence(cached["real"], s, cached["sigma"], cached["k_rr"])
            ent = cached_entropy_loss(s, cached["sigma"], cached["h_r"])
            entry = mode_bank.get(layer, c)
            cov, _ = mode_coverage_loss(s, entry.centers, entry.probs, tau=entry.tau)
            cs_total = cs_total + cs
            cov_total = cov_total + cov
            ent_total = ent_total + ent
            terms += 1

    denom = max(terms, 1)
    cs_total = cs_total / denom
    cov_total = cov_total / denom
    ent_total = ent_total / denom
    total = w_cs * cs_total + w_cov * cov_total + w_ent * ent_total
    stats = {
        "total": float(total.detach().cpu()),
        "cs": float(cs_total.detach().cpu()),
        "coverage": float(cov_total.detach().cpu()),
        "entropy": float(ent_total.detach().cpu()),
    }
    return total, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Mode-aware Rényi dataset distillation toy demo")
    parser.add_argument("--dataset", type=str, default="moons", choices=["moons", "circles", "blobs3", "blobs4"])
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--ipc", type=int, default=8, help="images/instances per class")
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--w-cs", type=float, default=1.0)
    parser.add_argument("--w-cov", type=float, default=0.25)
    parser.add_argument("--w-ent", type=float, default=0.05)
    parser.add_argument("--out", type=str, default="outputs/demo")
    parser.add_argument("--use-llm-agent", action="store_true")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads for small-matrix demo")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_toy_dataset(args.dataset, n_samples=args.n_samples, seed=args.seed, device=args.device)
    encoder = FrozenRandomEncoderBank(seed=args.seed).to(args.device)
    real_features = encoder.forward_features(dataset.x_train)
    mode_bank = build_mode_bank(real_features, dataset.y_train, dataset.num_classes, modes_per_class=args.modes, seed=args.seed)
    real_cache = precompute_real_cache(real_features, dataset.y_train, dataset.num_classes)

    syn_x, syn_y = init_synthetic_from_real(dataset.x_train, dataset.y_train, dataset.num_classes, args.ipc, seed=args.seed)
    syn_x0 = syn_x.detach().clone()
    save_distribution_plot(dataset.x_train, dataset.y_train, syn_x0, syn_y, str(out_dir / "synthetic_before.png"), "Before distillation")

    opt = torch.optim.Adam([syn_x], lr=args.lr)
    history: Dict[str, List[float]] = {"total": [], "cs": [], "coverage": [], "entropy": []}
    for epoch in range(args.epochs):
        loss, stats = compute_distillation_loss(
            encoder, real_cache, mode_bank, syn_x, syn_y, dataset.num_classes,
            w_cs=args.w_cs, w_cov=args.w_cov, w_ent=args.w_ent,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([syn_x], 5.0)
        opt.step()
        for k in history:
            history[k].append(stats[k])
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch:04d} total={stats['total']:.4f} cs={stats['cs']:.4f} cov={stats['coverage']:.4f} ent={stats['entropy']:.4f}")

    save_distribution_plot(dataset.x_train, dataset.y_train, syn_x, syn_y, str(out_dir / "synthetic_after.png"), "After mode-aware Rényi distillation")
    save_loss_curve(history, str(out_dir / "loss_curve.png"))
    np.savez(
        out_dir / "synthetic_data.npz",
        syn_x=syn_x.detach().cpu().numpy(),
        syn_y=syn_y.detach().cpu().numpy(),
        before_x=syn_x0.detach().cpu().numpy(),
    )
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    eval_result = evaluate_synthetic_set(syn_x, syn_y, dataset.x_test, dataset.y_test, dataset.num_classes, seed=args.seed)
    metrics = {"logistic_acc_on_real_test": eval_result.logistic_acc, "mlp_acc_on_real_test": eval_result.mlp_acc}
    print("metrics=", json.dumps(metrics, ensure_ascii=False, indent=2))

    config = vars(args)
    final_losses = {k: v[-1] for k, v in history.items()}
    agent = ResearchAgent(use_llm=args.use_llm_agent)
    report = agent.make_report(config=config, metrics=metrics, final_losses=final_losses)
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report)
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "metrics": metrics, "final_losses": final_losses}, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
