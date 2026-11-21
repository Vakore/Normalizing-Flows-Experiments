"""2-D toy datasets. Two dimensions keep everything fast on a laptop CPU and,
more importantly, let us *plot* the learned densities."""
import math
import torch


def two_moons(n, noise=0.05):
    n1 = n // 2
    n2 = n - n1
    t1 = torch.rand(n1) * math.pi
    t2 = torch.rand(n2) * math.pi
    upper = torch.stack([torch.cos(t1), torch.sin(t1)], 1)
    lower = torch.stack([1 - torch.cos(t2), 0.5 - torch.sin(t2)], 1)
    x = torch.cat([upper, lower]) + noise * torch.randn(n, 2)
    x = (x - torch.tensor([0.5, 0.25])) * 2.0  # center and scale to roughly [-3, 3]
    return x[torch.randperm(n)]


def eight_gaussians(n, radius=2.0, std=0.15):
    angles = torch.arange(8) * (2 * math.pi / 8)
    centers = radius * torch.stack([torch.cos(angles), torch.sin(angles)], 1)
    idx = torch.randint(0, 8, (n,))
    return centers[idx] + std * torch.randn(n, 2)


def checkerboard(n):
    x1 = torch.rand(n) * 4 - 2
    x2 = torch.rand(n) - torch.randint(0, 2, (n,)).float() * 2
    x2 = x2 + torch.floor(x1) % 2
    return torch.stack([x1, x2], 1) * 1.25


DATASETS = {
    "two_moons": two_moons,
    "eight_gaussians": eight_gaussians,
    "checkerboard": checkerboard,
}
