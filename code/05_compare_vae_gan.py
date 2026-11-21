"""Section: comparison with VAEs and GANs.
Trains a normalizing flow, a VAE and a GAN on the same data with similar
network sizes, and measures what the paper's comparison section discusses:
  * sample quality (fig_compare_samples)
  * training stability: loss curves over several seeds (fig_compare_training)
  * likelihood: exact for the flow, an upper bound (IWAE) for the VAE, none for the GAN
  * training and sampling time (results/comparison.tex)"""
import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nf.common import set_seed, train_flow, test_nll, plot_samples, plot_density, save_fig, save_results
from nf.datasets import DATASETS
from nf.flows import build_flow, mlp

LOG2PI = math.log(2 * math.pi)


def gauss_logpdf(x, mu, logvar):
    return -0.5 * (LOG2PI + logvar + (x - mu) ** 2 / logvar.exp()).sum(-1)


class VAE(nn.Module):
    def __init__(self, dim=2, latent=2, hidden=128):
        super().__init__()
        self.latent = latent
        self.enc = mlp(dim, 2 * latent, hidden, 3)
        self.dec = mlp(latent, 2 * dim, hidden, 3)

    def decode(self, z):
        mu, logvar = self.dec(z).chunk(2, -1)
        return mu, logvar.clamp(-7, 2)

    def elbo_terms(self, x):
        mu, logvar = self.enc(x).chunk(2, -1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        rec = gauss_logpdf(x, *self.decode(z)).mean()
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(-1).mean()
        return kl, rec

    @torch.no_grad()
    def iwae_nll(self, x, k=256):
        """Importance-weighted bound: an UPPER bound on the true NLL (the flow needs no bound)."""
        mu, logvar = self.enc(x).chunk(2, -1)
        std = (0.5 * logvar).exp()
        z = mu + torch.randn(k, *mu.shape) * std
        log_w = (gauss_logpdf(x, *self.decode(z)) + gauss_logpdf(z, torch.zeros_like(z), torch.zeros_like(z))
                 - gauss_logpdf(z, mu, logvar))
        return -(torch.logsumexp(log_w, 0) - math.log(k)).mean().item()

    @torch.no_grad()
    def sample(self, n):
        mu, logvar = self.decode(torch.randn(n, self.latent))
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


def train_vae(sampler, steps, batch=512):
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses, t0 = [], time.perf_counter()
    for step in range(steps):
        kl, rec = model.elbo_terms(sampler(batch))
        # KL warm-up: ramp the KL weight from 0 to 1 over the first half of training.
        # Without it the VAE often ignores a latent dimension ("posterior collapse").
        beta = min(1.0, step / (0.5 * steps))
        loss = beta * kl - rec
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append((kl - rec).item())  # always log the true negative ELBO
    return model, losses, time.perf_counter() - t0


class GAN(nn.Module):
    def __init__(self, dim=2, latent=2, hidden=128):
        super().__init__()
        self.latent = latent
        self.G = mlp(latent, dim, hidden, 3)
        self.D = mlp(dim, 1, hidden, 3)

    @torch.no_grad()
    def sample(self, n):
        return self.G(torch.randn(n, self.latent))


def train_gan(sampler, steps, batch=512):
    model = GAN()
    opt_g = torch.optim.Adam(model.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    losses, t0 = [], time.perf_counter()
    ones, zeros = torch.ones(batch, 1), torch.zeros(batch, 1)
    for _ in range(steps):
        real = sampler(batch)
        fake = model.G(torch.randn(batch, model.latent))
        d_loss = (F.binary_cross_entropy_with_logits(model.D(real), ones)
                  + F.binary_cross_entropy_with_logits(model.D(fake.detach()), zeros))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        g_loss = F.binary_cross_entropy_with_logits(model.D(fake), ones)  # non-saturating loss
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
        losses.append(g_loss.item())
    return model, losses, time.perf_counter() - t0


def time_sampling(model, n=10_000):
    t0 = time.perf_counter()
    model.sample(n)
    return (time.perf_counter() - t0) * 1e3


def smooth(y, w=50):
    return np.convolve(y, np.ones(w) / w, mode="valid")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dataset", default="eight_gaussians", choices=DATASETS)
    args = p.parse_args()
    sampler = DATASETS[args.dataset]

    curves = {"Normalizing flow": [], "VAE": [], "GAN": []}
    stats = {k: dict(nll=[], train_s=[], sample_ms=[]) for k in curves}
    first = {}
    for seed in range(args.seeds):
        print(f"  seed {seed}")
        set_seed(seed)
        flow = build_flow("realnvp", n_layers=12, hidden=128)
        loss, secs = train_flow(flow, sampler, steps=args.steps, lr=2e-3)
        curves["Normalizing flow"].append(loss)
        stats["Normalizing flow"]["nll"].append(test_nll(flow, sampler))
        stats["Normalizing flow"]["train_s"].append(secs)
        stats["Normalizing flow"]["sample_ms"].append(time_sampling(flow))

        set_seed(seed)
        vae, loss, secs = train_vae(sampler, args.steps)
        curves["VAE"].append(loss)
        stats["VAE"]["nll"].append(vae.iwae_nll(sampler(2000)))
        stats["VAE"]["train_s"].append(secs)
        stats["VAE"]["sample_ms"].append(time_sampling(vae))

        set_seed(seed)
        gan, loss, secs = train_gan(sampler, args.steps)
        curves["GAN"].append(loss)
        stats["GAN"]["nll"].append(float("nan"))
        stats["GAN"]["train_s"].append(secs)
        stats["GAN"]["sample_ms"].append(time_sampling(gan))
        if seed == 0:
            first = {"Normalizing flow": flow, "VAE": vae, "GAN": gan}

    # Samples (seed 0) plus the flow's exact density.
    fig, axes = plt.subplots(1, 5, figsize=(7, 1.75))
    plot_samples(axes[0], sampler(3000), title="data")
    for ax, (name, m) in zip(axes[1:4], first.items()):
        plot_samples(ax, m.sample(3000), title=f"{name} samples")
    plot_density(axes[4], first["Normalizing flow"].log_prob, title="flow: exact density")
    fig.tight_layout()
    save_fig(fig, "fig_compare_samples")

    # Training curves across seeds.
    fig, axes = plt.subplots(1, 3, figsize=(7, 1.9))
    ylabels = ["NLL (exact)", "negative ELBO", "generator loss"]
    for ax, (name, runs), yl in zip(axes, curves.items(), ylabels):
        for run in runs:
            ax.plot(smooth(run), lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig_compare_training")

    def fmt(vals, digits=2):
        v = np.array(vals, dtype=float)
        if np.isnan(v).all():
            return "n/a"
        return f"{v.mean():.{digits}f} $\\pm$ {v.std():.{digits}f}"

    tex = ("\\begin{tabular}{lccc}\n\\hline\nModel & NLL & Train (s) & Sample (ms) \\\\\n\\hline\n")
    labels = {"Normalizing flow": "NF (exact)", "VAE": "VAE (bound)", "GAN": "GAN"}
    for name, s in stats.items():
        tex += f"{labels[name]} & {fmt(s['nll'])} & {fmt(s['train_s'], 0)} & {fmt(s['sample_ms'], 1)} \\\\\n"
    tex += "\\hline\n\\end{tabular}\n"
    save_results("comparison", {"dataset": args.dataset, "stats": stats}, tex)
