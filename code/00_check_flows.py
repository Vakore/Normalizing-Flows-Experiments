"""Sanity checks for every layer in nf/flows.py:
  1. invertibility:  inverse(forward(x)) == x
  2. log-determinant: the cheap formula matches the brute-force
     log|det J| computed from the full autograd Jacobian.
Run this first - if it fails, every later result is meaningless."""
import torch

from nf.flows import (DiagonalFlow, TriangularFlow, Invertible1x1Conv, AffineCoupling,
                      MAF, Reverse, alternating_mask, build_flow)

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
dim = 4


def randomise(layer):
    with torch.no_grad():  # move away from the identity initialisation
        for p in layer.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return layer


layers = {
    "DiagonalFlow": DiagonalFlow(dim),
    "TriangularFlow": TriangularFlow(dim),
    "Invertible1x1Conv": Invertible1x1Conv(dim),
    "AffineCoupling": AffineCoupling(dim, alternating_mask(dim, 0)),
    "MAF": MAF(dim),
    "Reverse": Reverse(),
}
ok = True
for name, layer in layers.items():
    layer = randomise(layer)
    x = torch.randn(5, dim)
    z, logdet = layer(x)
    inv_err = (layer.inverse(z) - x).abs().max().item()
    brute = torch.stack([
        torch.linalg.slogdet(torch.autograd.functional.jacobian(lambda v: layer(v[None])[0][0], xi))[1]
        for xi in x])
    ld_err = (brute - logdet).abs().max().item()
    passed = inv_err < 1e-8 and ld_err < 1e-8
    ok &= passed
    print(f"  {name:<18} inverse err {inv_err:.1e}   logdet err {ld_err:.1e}   {'PASS' if passed else 'FAIL'}")

# 1x1 convolution on an actual image tensor: log|det| scales with H*W
conv = randomise(Invertible1x1Conv(3))
img = torch.randn(2, 3, 8, 8)
z, ld = conv(img)
inv_err = (conv.inverse(z) - img).abs().max().item()
expected = 64 * torch.linalg.slogdet(conv.weight())[1]
passed = inv_err < 1e-8 and abs(ld[0].item() - expected.item()) < 1e-8
ok &= passed
print(f"  {'1x1 conv on images':<18} inverse err {inv_err:.1e}   logdet = H*W*log|det W|   {'PASS' if passed else 'FAIL'}")

print("\nAll checks passed." if ok else "\nSOME CHECKS FAILED.")
raise SystemExit(0 if ok else 1)
