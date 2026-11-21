"""Training, plotting and bookkeeping helpers shared by the experiment scripts."""
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]          # nf-paper/
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150})


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_flow(model, sampler, steps=4000, batch=512, lr=1e-3):
    """Maximum likelihood: minimise the negative log-likelihood -log p(x).
    Because the flow gives p(x) exactly, this is the true objective, not a bound."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    start = time.perf_counter()
    for step in range(steps):
        x = sampler(batch)
        loss = -model.log_prob(x).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"    step {step + 1:>5}/{steps}  NLL {np.mean(losses[-100:]):.3f}")
    return losses, time.perf_counter() - start


@torch.no_grad()
def test_nll(model, sampler, n=20000):
    return -model.log_prob(sampler(n)).mean().item()


@torch.no_grad()
def plot_density(ax, log_prob_fn, lim=3.5, n=200, title=""):
    g = torch.linspace(-lim, lim, n)
    xx, yy = torch.meshgrid(g, g, indexing="xy")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)
    dens = log_prob_fn(pts).exp().reshape(n, n)
    ax.imshow(dens.numpy(), extent=[-lim, lim, -lim, lim], origin="lower", cmap="magma")
    tidy(ax, lim, title)


def plot_samples(ax, x, lim=3.5, title="", color="#1f4e79", c=None, cmap=None):
    x = x.detach().numpy()
    ax.scatter(x[:, 0], x[:, 1], s=1, alpha=0.5, color=None if c is not None else color,
               c=c, cmap=cmap, rasterized=True)
    tidy(ax, lim, title)


def tidy(ax, lim, title):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def save_fig(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> figures/{name}.pdf")


def save_results(name, data, table_tex=None):
    RES_DIR.mkdir(exist_ok=True)
    (RES_DIR / f"{name}.json").write_text(json.dumps(data, indent=2))
    if table_tex is not None:
        (RES_DIR / f"{name}.tex").write_text(table_tex)
        print(f"  -> results/{name}.tex")
