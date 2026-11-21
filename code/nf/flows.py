"""Flow layers used in the paper.

Convention: every layer is a bijection f that maps DATA x -> LATENT z and returns
    z, log|det df/dx|
so the change-of-variables formula reads

    log p_X(x) = log p_Z(f(x)) + log|det J_f(x)|.

`inverse` maps latent -> data and is what we use for sampling.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(n_in, n_out, hidden=64, layers=2):
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, n_out))
    return nn.Sequential(*mods)


class Flow(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    def inverse(self, z):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Linear flows                                                                #
# --------------------------------------------------------------------------- #
class DiagonalFlow(Flow):
    """z = x * exp(s) + b. Jacobian is diagonal, so log|det| = sum(s): O(d).
    Cheap, but it can only rescale/shift each axis independently.
    (Glow's ActNorm is exactly this layer with data-dependent initialisation.)"""

    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        z = x * torch.exp(self.log_scale) + self.shift
        return z, self.log_scale.sum().expand(x.shape[0])

    def inverse(self, z):
        return (z - self.shift) * torch.exp(-self.log_scale)


class TriangularFlow(Flow):
    """z = L x + b with L lower-triangular and positive diagonal.
    The determinant of a triangular matrix is the product of its diagonal, so
    log|det| = sum(log diag L): O(d) instead of the O(d^3) of a dense matrix.
    The inverse is a triangular solve (O(d^2))."""

    def __init__(self, dim):
        super().__init__()
        self.strict = nn.Parameter(torch.zeros(dim, dim))
        self.log_diag = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.register_buffer("mask", torch.tril(torch.ones(dim, dim), -1))

    def L(self):
        return self.strict * self.mask + torch.diag(torch.exp(self.log_diag))

    def forward(self, x):
        z = x @ self.L().T + self.shift
        return z, self.log_diag.sum().expand(x.shape[0])

    def inverse(self, z):
        return torch.linalg.solve_triangular(self.L(), (z - self.shift).T, upper=False).T


class Invertible1x1Conv(Flow):
    """Glow's invertible 1x1 convolution with LU-parameterised weight
        W = P L (U + diag(s)),
    P a fixed permutation, L unit lower-triangular, U strictly upper-triangular.
    log|det W| = sum(log|s|) is O(C); on an image with H*W pixels the same W is
    applied at every pixel, so log|det J| = H * W * sum(log|s|).

    Accepts images (N, C, H, W) or plain vectors (N, C), which are treated as
    1x1 images - so the 2-D experiments use exactly the same layer."""

    def __init__(self, channels):
        super().__init__()
        w = torch.linalg.qr(torch.randn(channels, channels))[0]  # random rotation
        P, L, U = torch.linalg.lu(w)
        s = torch.diagonal(U)
        lower = torch.tril(torch.ones(channels, channels), -1)
        self.register_buffer("P", P)
        self.register_buffer("sign_s", torch.sign(s))
        self.register_buffer("lower_mask", lower)
        self.register_buffer("eye", torch.eye(channels))
        self.L = nn.Parameter(L * lower)
        self.U = nn.Parameter(torch.triu(U, 1))
        self.log_s = nn.Parameter(torch.log(torch.abs(s)))

    def weight(self):
        L = self.L * self.lower_mask + self.eye
        U = self.U * self.lower_mask.T + torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.P @ L @ U

    def forward(self, x):
        W = self.weight()
        if x.dim() == 2:
            return x @ W.T, self.log_s.sum().expand(x.shape[0])
        n, c, h, w = x.shape
        z = F.conv2d(x, W.view(c, c, 1, 1))
        return z, (h * w * self.log_s.sum()).expand(n)

    def inverse(self, z):
        W_inv = torch.inverse(self.weight())
        if z.dim() == 2:
            return z @ W_inv.T
        c = z.shape[1]
        return F.conv2d(z, W_inv.view(c, c, 1, 1))


# --------------------------------------------------------------------------- #
# Coupling and autoregressive flows                                           #
# --------------------------------------------------------------------------- #
class AffineCoupling(Flow):
    """RealNVP affine coupling. Split x into (x_a, x_b) with a binary mask:
        z_a = x_a
        z_b = x_b * exp(s(x_a)) + t(x_a)
    The Jacobian is triangular, so log|det| = sum(s), and s, t can be ANY
    neural network - they never need to be inverted. That is the trick that
    makes coupling flows expressive yet cheap."""

    def __init__(self, dim, mask, hidden=64):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.net = mlp(dim, 2 * dim, hidden)
        nn.init.zeros_(self.net[-1].weight)  # start as the identity map
        nn.init.zeros_(self.net[-1].bias)

    def _scale_shift(self, x_masked):
        s, t = self.net(x_masked).chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)  # tanh keeps scales bounded -> stable training
        t = t * (1 - self.mask)
        return s, t

    def forward(self, x):
        xa = x * self.mask
        s, t = self._scale_shift(xa)
        z = xa + (1 - self.mask) * (x * torch.exp(s) + t)
        return z, s.sum(-1)

    def inverse(self, z):
        za = z * self.mask
        s, t = self._scale_shift(za)
        return za + (1 - self.mask) * ((z - t) * torch.exp(-s))


class MaskedLinear(nn.Linear):
    def __init__(self, n_in, n_out, mask):
        super().__init__(n_in, n_out)
        self.register_buffer("mask", mask.float())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Masked autoencoder: output i only sees inputs 1..i-1 (autoregressive)."""

    def __init__(self, dim, hidden=64, layers=2):
        super().__init__()
        deg_in = torch.arange(1, dim + 1)
        deg_h = torch.arange(hidden) % max(1, dim - 1) + 1
        mods, prev = [], deg_in
        for i in range(layers):
            n_in = dim if i == 0 else hidden
            mods += [MaskedLinear(n_in, hidden, deg_h[:, None] >= prev[None, :]), nn.SiLU()]
            prev = deg_h
        deg_out = torch.cat([deg_in, deg_in])  # outputs: (mu_1..mu_d, alpha_1..alpha_d)
        last = MaskedLinear(hidden, 2 * dim, deg_out[:, None] > prev[None, :])
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*mods, last)

    def forward(self, x):
        return self.net(x)


class MAF(Flow):
    """Masked autoregressive flow layer:
        z_i = (x_i - mu_i(x_<i)) * exp(-alpha_i(x_<i))
    Density evaluation (x -> z) is one parallel pass; sampling (z -> x) needs d
    sequential passes. Jacobian is triangular: log|det| = -sum(alpha)."""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.dim = dim
        self.made = MADE(dim, hidden)

    def _params(self, x):
        mu, alpha = self.made(x).chunk(2, dim=-1)
        return mu, torch.tanh(alpha) * 2.0

    def forward(self, x):
        mu, alpha = self._params(x)
        return (x - mu) * torch.exp(-alpha), -alpha.sum(-1)

    def inverse(self, z):
        x = torch.zeros_like(z)
        for i in range(self.dim):  # one pass per dimension: the cost of autoregression
            mu, alpha = self._params(x)
            x = x.clone()
            x[:, i] = z[:, i] * torch.exp(alpha[:, i]) + mu[:, i]
        return x


class Reverse(Flow):
    """Permutation (volume preserving, log|det| = 0) so later layers transform
    the dimensions earlier layers left untouched."""

    def forward(self, x):
        return x.flip(-1), x.new_zeros(x.shape[0])

    def inverse(self, z):
        return z.flip(-1)


# --------------------------------------------------------------------------- #
# The full model                                                              #
# --------------------------------------------------------------------------- #
class NormalizingFlow(nn.Module):
    def __init__(self, flows, dim):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.dim = dim

    def forward(self, x):
        """Data -> latent. Log-determinants of a composition simply add up."""
        logdet = torch.zeros(x.shape[0], device=x.device)
        for f in self.flows:
            x, ld = f(x)
            logdet = logdet + ld
        return x, logdet

    def log_prob(self, x):
        z, logdet = self(x)
        log_pz = -0.5 * (z ** 2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return log_pz + logdet

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.dim)
        for f in reversed(self.flows):
            z = f.inverse(z)
        return z


def alternating_mask(dim, i):
    return (torch.arange(dim) + i) % 2


def build_flow(kind, dim=2, n_layers=8, hidden=64):
    if kind == "diagonal":
        layers = [DiagonalFlow(dim)]
    elif kind == "triangular":
        layers = [TriangularFlow(dim)]
    elif kind == "conv1x1":
        layers = [Invertible1x1Conv(dim), DiagonalFlow(dim)]  # DiagonalFlow adds the bias
    elif kind == "realnvp":
        layers = [AffineCoupling(dim, alternating_mask(dim, i), hidden) for i in range(n_layers)]
    elif kind == "maf":
        layers = []
        for _ in range(n_layers // 2):
            layers += [MAF(dim, hidden), Reverse()]
    elif kind == "glow":  # Glow step = ActNorm -> 1x1 conv -> affine coupling
        layers = []
        for i in range(n_layers):
            layers += [DiagonalFlow(dim), Invertible1x1Conv(dim),
                       AffineCoupling(dim, alternating_mask(dim, i), hidden)]
    else:
        raise ValueError(f"unknown flow type: {kind}")
    return NormalizingFlow(layers, dim)
