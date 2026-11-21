"""Section: linear flows (diagonal, triangular, 1x1 convolution).
Trains each linear flow on two moons. Because a linear map of a Gaussian is
still a Gaussian, every one of them can only fit an ellipse - which is the
motivating limitation for coupling flows."""
import argparse

import matplotlib.pyplot as plt

from nf.common import set_seed, train_flow, test_nll, plot_density, plot_samples, save_fig, save_results
from nf.datasets import DATASETS
from nf.flows import build_flow

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--dataset", default="two_moons", choices=DATASETS)
args = p.parse_args()

set_seed(0)
sampler = DATASETS[args.dataset]
kinds = {"diagonal": "Diagonal", "triangular": "Triangular", "conv1x1": "1x1 convolution (LU)"}

fig, axes = plt.subplots(1, 4, figsize=(7, 1.95))
plot_samples(axes[0], sampler(4000), title="data")
rows = []
for ax, (kind, label) in zip(axes[1:], kinds.items()):
    print(f"  training {kind}")
    model = build_flow(kind)
    _, secs = train_flow(model, sampler, steps=args.steps, lr=1e-2)
    nll = test_nll(model, sampler)
    plot_density(ax, model.log_prob, title=f"{label}\nNLL {nll:.2f}")
    rows.append((label, nll, sum(p.numel() for p in model.parameters()), secs))
fig.tight_layout()
save_fig(fig, "fig_linear_flows")

tex = "\\begin{tabular}{lrr}\n\\hline\nFlow & Test NLL & Params \\\\\n\\hline\n"
tex += "".join(f"{l} & {n:.3f} & {k} \\\\\n" for l, n, k, _ in rows)
tex += "\\hline\n\\end{tabular}\n"
save_results("linear_flows", [dict(zip(["flow", "nll", "params", "seconds"], r)) for r in rows], tex)
