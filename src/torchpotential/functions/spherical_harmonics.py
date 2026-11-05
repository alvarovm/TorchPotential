"""Complex spherical harmonics, represented as (real, imag) tensor pairs."""
import math

import numpy as np
import torch


class SphericalHarmonics:
    """Real/imaginary parts of Y_lm for ``l = 0..lmax``, flattened l-major with ``m = -l..l``.

    ``compute_ylm(rhat)`` with ``rhat`` of shape ``[N, 3]`` returns two tensors of shape ``[N, (lmax+1)**2]``.
    ``l_tile`` (long tensor, ``(lmax+1)**2``) holds the value of ``l`` of every flattened column.
    """

    def __init__(self, lmax, device=None, dtype=torch.float64):
        self._lmax = int(lmax)
        self.device = device
        self.dtype = dtype
        self.alm, self.blm = self._pre_compute()
        self._build_index()

    # index helpers --------------------------------------------------------------------------------
    def lm1d(self, l, m):
        return m + l * (l + 1) // 2

    def lmsh(self, l, m):
        return l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2

    def _pre_compute(self):
        alm = [0.0]
        blm = [0.0]
        for i in range(1, self._lmax + 1):
            l = float(i)
            lsq = l * l
            ld = 2 * l
            l1 = 4 * lsq - 1
            l2 = lsq - ld + 1
            for j in range(0, i + 1):
                m = float(j)
                msq = m * m
                if i == j:
                    alm.append(-math.sqrt(1.0 + 0.5 / m))
                    blm.append(0.0)
                else:
                    alm.append(math.sqrt(l1 / (lsq - msq)))
                    blm.append(-math.sqrt((l2 - msq) / (4 * l2 - 1)))
        return alm, blm

    def _build_index(self):
        src, sign, conj, l_tile = [], [], [], []
        for l in range(0, self._lmax + 1):
            for m in range(-l, l + 1):
                src.append(self.lmsh(l, m))
                l_tile.append(l)
                if m < 0:
                    sign.append(1.0 if abs(m) % 2 == 0 else -1.0)
                    conj.append(-1.0)
                else:
                    sign.append(1.0)
                    conj.append(1.0)
        kw = dict(device=self.device)
        self._src = torch.tensor(src, dtype=torch.long, **kw)
        self._sign = torch.tensor(sign, dtype=self.dtype, **kw)
        self._conj = torch.tensor(conj, dtype=self.dtype, **kw)
        self.l_tile = torch.tensor(l_tile, dtype=torch.long, **kw)

    # Legendre part --------------------------------------------------------------------------------
    def legendre(self, x):
        """Normalised associated Legendre functions as a list indexed by ``lm1d(l, m)``; each of shape [N]."""
        pi = math.pi
        y00 = math.sqrt(1.0 / (4.0 * pi))
        plm = [x * 0 + y00]
        if self._lmax > 0:
            sq3o4pi = math.sqrt(3.0 / (4.0 * pi))
            sq3o8pi = math.sqrt(3.0 / (8.0 * pi))
            plm.append(sq3o4pi * x)  # (1,0)
            plm.append(x * 0 - sq3o8pi)  # (1,1)

            for l in range(2, self._lmax + 1):
                for m in range(0, l + 1):
                    if m == l - 1:
                        dl = math.sqrt(2.0 * m + 3.0)
                        plm.append(x * dl * plm[self.lm1d(l - 1, l - 1)])
                    elif m == l:
                        plm.append(self.alm[self.lm1d(l, l)] * plm[self.lm1d(l - 1, l - 1)])
                    else:
                        plm.append(
                            self.alm[self.lm1d(l, m)]
                            * (x * plm[self.lm1d(l - 1, m)] + self.blm[self.lm1d(l, m)] * plm[self.lm1d(l - 2, m)])
                        )
        return plm

    def compute_ylm(self, rhat):
        rx, ry, rz = rhat[:, 0], rhat[:, 1], rhat[:, 2]
        phase_r, phase_i = rx, ry
        plm = self.legendre(rz)

        ylm_r, ylm_i = [], []
        for l in range(0, self._lmax + 1):  # m = 0
            p = plm[self.lm1d(l, 0)]
            ylm_r.append(p)
            ylm_i.append(torch.zeros_like(p))
        for l in range(1, self._lmax + 1):  # m = 1
            p = plm[self.lm1d(l, 1)]
            ylm_r.append(phase_r * p)
            ylm_i.append(phase_i * p)

        phasem_r, phasem_i = phase_r, phase_i
        for m in range(2, self._lmax + 1):
            pr_tmp = phasem_r
            phasem_r = phasem_r * phase_r - phasem_i * phase_i
            phasem_i = pr_tmp * phase_i + phasem_i * phase_r
            for l in range(m, self._lmax + 1):
                p = plm[self.lm1d(l, m)]
                ylm_r.append(phasem_r * p)
                ylm_i.append(phasem_i * p)

        ylm_r = torch.stack(ylm_r, dim=1)
        ylm_i = torch.stack(ylm_i, dim=1)
        out_r = ylm_r[:, self._src] * self._sign
        out_i = ylm_i[:, self._src] * (self._sign * self._conj)
        return out_r, out_i
