"""Radial basis building blocks. All functions take a 1-D tensor of distances ``x`` with shape ``[N]``
and return ``[N, nfunc]`` (basis) or ``[N]`` (cutoffs)."""
import math

import torch


def chebvander(x, deg):
    """Chebyshev polynomials T_0..T_{deg-1} of the first kind (at least T_0, T_1 for deg > 0), shape [N, ncols]."""
    v = [torch.ones_like(x)]
    if deg > 0:
        x2 = 2 * x
        v.append(x)
        for i in range(2, deg):
            v.append(v[i - 1] * x2 - v[i - 2])
    return torch.stack(v, dim=-1)


def legendre(x, order):
    """Legendre polynomials P_0..P_{order-1} (at least P_0, P_1 for order > 0), shape [N, ncols]."""
    p = [torch.ones_like(x)]
    if order > 0:
        p.append(x)
        for i in range(2, order):
            p.append(((2 * i - 1) * p[i - 1] * x - (i - 1) * p[i - 2]) / i)
    return torch.stack(p, dim=-1)


def sinc(x):
    nz = x != 0
    safe = torch.where(nz, x, torch.ones_like(x))
    return torch.where(nz, torch.sin(safe) / safe, torch.ones_like(x))


def _fn(x, rc, n):
    pi = math.pi
    pref = (
        ((-1.0) ** n)
        * math.sqrt(2.0)
        * pi
        / rc ** (3.0 / 2)
        * (n + 1)
        * (n + 2)
        / math.sqrt((n + 1) ** 2 + (n + 2) ** 2)
    )
    return pref * (sinc(x * (n + 1) * pi / rc) + sinc(x * (n + 2) * pi / rc))


def simplified_bessel(x, rc, deg):
    rc = float(rc)
    sbf = [_fn(x, rc, 0.0)]
    d = [1.0]
    if deg > 0:
        for i in range(1, deg):
            n = float(i)
            en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
            dn = 1 - en / d[i - 1]
            d.append(dn)
            sbf.append(1 / math.sqrt(d[i]) * (_fn(x, rc, n) + math.sqrt(en / d[i - 1]) * sbf[i - 1]))
    return torch.stack(sbf, dim=-1)


def cutoff_func_cos(x, rcut):
    cutoff = (1.0 + torch.cos(math.pi * x / rcut)) * 0.5
    return torch.where(x < rcut, cutoff, torch.zeros_like(x))


def cutoff_func_poly(r, rin, delta):
    """Polynomial cutoff descending from 1 at ``rin - delta`` to 0 at ``rin`` (same branch order as the C++ code).

    For ``delta == 0`` the transition zone is empty (step function)."""
    d = float(delta) if float(delta) > 0 else 1.0  # keeps the unused branch finite when delta == 0
    x = 1 - 2 * (1 + (r - rin) / d)
    f = 7.5 * (x / 4 - x**3 / 6 + x**5 / 20)
    return torch.where(
        r <= rin - delta,
        torch.ones_like(r),
        torch.where(r >= rin, torch.zeros_like(r), 0.5 * (1 + f)),
    )
