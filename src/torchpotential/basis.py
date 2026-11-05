"""NumPy bookkeeping of B-basis functions (m-combinations, generalized Clebsch-Gordan coefficients, ...)."""
import numpy as np


def merge_lm(l, m):
    return l * (l + 1) + m


class BBasisFunc:
    def __init__(self, bbasisfunc, lmax=None):
        self.rank = bbasisfunc.rank
        self.ns = bbasisfunc.ns
        self.ls = bbasisfunc.ls
        self.mu0 = bbasisfunc.mu0
        self.mus = bbasisfunc.mus
        self.genCG = np.reshape(bbasisfunc.gen_cgs, [-1, 1])
        self.ms = bbasisfunc.ms_combs
        self.coefs = bbasisfunc.coeffs

        self.munlm = self.get_munlm()
        self.msum = np.zeros(len(self.ms)).astype(np.int32)

        if lmax is not None:
            self.adjust_m_index(lmax)

    def get_munlm(self):
        munlm = [np.zeros((len(self.ms), 4)).astype(np.int32) for _ in range(self.rank)]
        for c in range(len(self.ms)):
            for r in range(self.rank):
                munlm[r][c] = np.array([self.mus[r], self.ns[r] - 1, self.ls[r], self.ms[c][r]])

        return munlm

    def adjust_m_index(self, lmax):
        for r in range(self.rank):
            self.munlm[r][:, -1] += lmax


class BBasisFuncSet:
    """All B-functions of one rank of one central element.

    After construction ``munlm[r]`` has columns ``[mu, n-1, merged_lm]`` for each of the ``rank`` factors and
    each (function, m-combination) row; ``msum`` maps rows to the function index and ``genCG`` holds the
    generalized Clebsch-Gordan coefficients.
    """

    def __init__(self, list_of_bbasisfunc, rank):
        self.rank = rank
        self.nfunc = len(list_of_bbasisfunc)
        self.munlm = self.get_munlm(list_of_bbasisfunc)
        self.msum = self.get_msum(list_of_bbasisfunc)
        self.genCG = self.get_gen_cg(list_of_bbasisfunc)
        self.coefs = self.get_coefs(list_of_bbasisfunc)

        for r in range(self.rank):
            self.munlm[r][:, -2] = merge_lm(self.munlm[r][:, -2], self.munlm[r][:, -1])
            self.munlm[r] = self.munlm[r][:, :-1]

    def get_munlm(self, list_of_bbasisfunc):
        return [np.vstack([f.munlm[i] for f in list_of_bbasisfunc]) for i in range(self.rank)]

    def get_gen_cg(self, list_of_bbasisfunc):
        return np.vstack([f.genCG for f in list_of_bbasisfunc])

    def get_msum(self, list_of_bbasisfunc):
        return np.hstack([f.msum + i for i, f in enumerate(list_of_bbasisfunc)])

    def get_coefs(self, list_of_bbasisfunc):
        return np.vstack([f.coefs for f in list_of_bbasisfunc])


class ConfigBasis:
    """``central_atom[e][k]`` is the :class:`BBasisFuncSet` of rank ``k + 2`` of central element ``e``."""

    def __init__(self, bbasisfunc, ranksmax, nelem):
        self.central_atom = []
        for elem in range(nelem):
            total_rank = []
            for r in range(2, ranksmax[elem] + 1):  # rank 1 is handled separately
                funcs_of_rank = [f for f in bbasisfunc[elem] if f.rank == r]
                total_rank.append(BBasisFuncSet(funcs_of_rank, rank=r) if funcs_of_rank else None)
            self.central_atom.append(total_rank)
