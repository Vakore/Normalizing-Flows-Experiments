"""Section: building blocks.
Visualises the change-of-variables formula in 1-D:
    z ~ N(0, 1),  x = f(z),  p_X(x) = p_Z(f^{-1}(x)) * |d f^{-1}/dx|
and checks the formula against a histogram of transformed samples.
Pure NumPy - no training involved."""
import numpy as np
import matplotlib.pyplot as plt

from nf.common import save_fig

f = lambda z: z + 1.5 * np.tanh(2 * z)               # f'(z) = 1 + 3 sech^2(2z) > 0  -> bijective
df = lambda z: 1 + 3 / np.cosh(2 * z) ** 2
p_z = lambda z: np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)

z_grid = np.linspace(-6, 6, 20001)
f_inv = lambda x: np.interp(x, f(z_grid), z_grid)   # f is monotone, so interpolation inverts it

rng = np.random.default_rng(0)
x_samples = f(rng.standard_normal(200_000))
xs = np.linspace(-5, 5, 800)
p_x = p_z(f_inv(xs)) / df(f_inv(xs))                 # change of variables

fig, axes = plt.subplots(1, 3, figsize=(7, 2.1))
zs = np.linspace(-4, 4, 400)
axes[0].plot(zs, p_z(zs), color="#1f4e79")
axes[0].set_title(r"base $p_Z(z)=\mathcal{N}(0,1)$")
axes[1].plot(zs, f(zs), color="#b5452c")
axes[1].plot(zs, zs, "--", color="gray", lw=0.8)
axes[1].set_title(r"bijection $x=f(z)$")
axes[2].hist(x_samples, bins=150, range=(-5, 5), density=True, color="#cccccc", label="samples")
axes[2].plot(xs, p_x, color="#b5452c", label="formula")
axes[2].set_title(r"result $p_X(x)$")
axes[2].set_ylim(0, p_x.max() * 1.4)  # headroom for the legend
axes[2].legend(frameon=False, fontsize=6, loc="upper left", ncol=2)
for ax in axes:
    ax.set_yticks([])
fig.tight_layout()
save_fig(fig, "fig_change_of_variables")

hist, edges = np.histogram(x_samples, bins=150, range=(-5, 5), density=True)
centers = 0.5 * (edges[1:] + edges[:-1])
err = np.abs(hist - p_z(f_inv(centers)) / df(f_inv(centers))).max()
print(f"  max |histogram - formula| = {err:.4f}  (small => formula verified)")
