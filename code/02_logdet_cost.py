"""Section: training / computational shortcuts.
Why flows restrict themselves to diagonal or triangular Jacobians: computing
log|det J| of a dense d x d matrix costs O(d^3); diagonal and triangular cost O(d)."""
import timeit

import numpy as np
import matplotlib.pyplot as plt

from nf.common import save_fig, save_results

rng = np.random.default_rng(0)
dims = [16, 32, 64, 128, 256, 512, 1024, 2048]
results = {"dim": dims, "dense": [], "triangular": [], "diagonal": []}

for d in dims:
    A = rng.standard_normal((d, d)) + d * np.eye(d)
    T = np.tril(A)
    v = np.abs(rng.standard_normal(d)) + 0.1
    reps = 3 if d >= 1024 else 20
    t = lambda stmt: min(timeit.repeat(stmt, number=1, repeat=reps))
    results["dense"].append(t(lambda: np.linalg.slogdet(A)))
    results["triangular"].append(t(lambda: np.log(np.abs(np.diagonal(T))).sum()))
    results["diagonal"].append(t(lambda: np.log(v).sum()))
    # sanity check: the shortcut gives the same answer as the general method
    assert np.isclose(np.linalg.slogdet(T)[1], np.log(np.abs(np.diagonal(T))).sum())
    print(f"  d={d:>5}  dense {results['dense'][-1]*1e3:8.3f} ms   "
          f"triangular {results['triangular'][-1]*1e3:7.4f} ms")

fig, ax = plt.subplots(figsize=(3.3, 2.4))
for key, style in [("dense", "o-"), ("triangular", "s-"), ("diagonal", "^-")]:
    ax.loglog(dims, np.array(results[key]) * 1e3, style, ms=3, label=key)
ax.set_xlabel("dimension $d$")
ax.set_ylabel("time for log|det| (ms)")
ax.legend(frameon=False)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
save_fig(fig, "fig_logdet_cost")

speedup = results["dense"][-1] / results["triangular"][-1]
save_results("logdet_cost", results,
             f"\\newcommand{{\\logdetSpeedup}}{{{speedup:,.0f}}}\n"
             f"\\newcommand{{\\logdetMaxDim}}{{{dims[-1]}}}\n")
