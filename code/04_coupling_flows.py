"""Section: coupling and autoregressive flows.
Trains RealNVP (affine coupling), MAF (autoregressive) and a Glow-style stack
(ActNorm + 1x1 conv + coupling) on three datasets, then
  * plots the learned densities (fig_coupling_flows)
  * shows the bijection in both directions (fig_flow_mapping)
  * records NLL, training time and sampling time (results/coupling_flows.tex)."""
import argparse
import time

import matplotlib.pyplot as plt
import torch

from nf.common import set_seed, train_flow, test_nll, plot_density, plot_samples, save_fig, save_results
from nf.datasets import DATASETS
from nf.flows import build_flow

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--layers", type=int, default=12)
p.add_argument("--hidden", type=int, default=128)
args = p.parse_args()

models = {"realnvp": "RealNVP (coupling)", "maf": "MAF (autoregressive)", "glow": "Glow-style"}
datasets = ["two_moons", "eight_gaussians", "checkerboard"]

fig, axes = plt.subplots(len(datasets), len(models) + 1, figsize=(7, 5.4))
rows, trained = [], {}
for r, ds in enumerate(datasets):
    sampler = DATASETS[ds]
    set_seed(r)
    plot_samples(axes[r, 0], sampler(4000), title="data" if r == 0 else "")
    axes[r, 0].set_ylabel(ds.replace("_", " "))
    for c, (kind, label) in enumerate(models.items(), start=1):
        print(f"  training {kind} on {ds}")
        model = build_flow(kind, n_layers=args.layers, hidden=args.hidden)
        _, secs = train_flow(model, sampler, steps=args.steps, lr=2e-3)
        nll = test_nll(model, sampler)
        t0 = time.perf_counter()
        model.sample(10_000)
        sample_ms = (time.perf_counter() - t0) * 1e3
        plot_density(axes[r, c], model.log_prob, title=(label + "\n" if r == 0 else "") + f"NLL {nll:.2f}")
        rows.append(dict(dataset=ds, model=label, nll=nll, train_s=secs, sample_ms=sample_ms,
                         params=sum(p.numel() for p in model.parameters())))
        trained[(kind, ds)] = model
for ax in axes[:, 0]:
    ax.yaxis.label.set_visible(True)
fig.tight_layout()
save_fig(fig, "fig_coupling_flows")

# The same trained model, used in both directions.
model, sampler = trained[("realnvp", "two_moons")], DATASETS["two_moons"]
x = sampler(4000)
with torch.no_grad():
    z, _ = model(x)
z0 = torch.randn(4000, 2)
fig, axes = plt.subplots(1, 4, figsize=(7, 1.95))
plot_samples(axes[0], x, title=r"data $x$", c=x[:, 0].numpy(), cmap="coolwarm")
plot_samples(axes[1], z, title=r"$z=f(x)$ (normalizing)", c=x[:, 0].numpy(), cmap="coolwarm")
plot_samples(axes[2], z0, title=r"$z\sim\mathcal{N}(0,I)$", c=z0[:, 0].numpy(), cmap="viridis")
with torch.no_grad():
    x_gen = z0
    for f in reversed(model.flows):
        x_gen = f.inverse(x_gen)
plot_samples(axes[3], x_gen, title=r"$x=f^{-1}(z)$ (generating)", c=z0[:, 0].numpy(), cmap="viridis")
fig.tight_layout()
save_fig(fig, "fig_flow_mapping")

tex = "\\begin{tabular}{llrrr}\n\\hline\nData & Model & NLL & Train (s) & Sample (ms) \\\\\n\\hline\n"
for row in rows:
    tex += (f"{row['dataset'].replace('_', ' ')} & {row['model'].split(' (')[0]} & {row['nll']:.3f} & "
            f"{row['train_s']:.0f} & {row['sample_ms']:.1f} \\\\\n")
tex += "\\hline\n\\end{tabular}\n"
save_results("coupling_flows", rows, tex)
