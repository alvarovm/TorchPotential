import logging
import math
from itertools import combinations_with_replacement

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchpotential.basis import BBasisFunc, ConfigBasis
from torchpotential.functions import radial_functions
from torchpotential.functions.spherical_harmonics import SphericalHarmonics
from torchpotential.indexing import IndexLayout
from torchpotential.utils.device import get_device
from torchpotential.utils.symbols import symbol_to_atomic_number

log = logging.getLogger(__name__)

_SMOOTH_GRID_POINTS = 100


def _contract_radial(g, crad):
    """``einsum('jk,nlk->jnl', g, crad)`` as one 2-D matmul (einsum lowers this to a very slow bmm on CPU)."""
    nrad, lp1, nbase = crad.shape
    return torch.matmul(g, crad.reshape(nrad * lp1, nbase).t()).reshape(g.shape[0], nrad, lp1)


def phi_core_per_bond(d, seg, n_u, bi, distance_type):
    """Core repulsion summed over the neighbours of each central atom of one bond type, shape [n_u].

    Pair term (as in the C++ evaluator): ``pre * exp(-lambda r^2) / r * (1 + cos(pi r / rcut)) / 2``,
    zero for ``lambda r^2 >= 50``; for ``inner_cutoff_type == 'distance'`` it is additionally multiplied by the
    descending polynomial inner cutoff ``poly(r, r_in, delta_in)``.
    """
    lr2 = bi.core_lmbda * d**2
    phi = bi.core_pre * torch.exp(-lr2) / d
    phi = torch.where(lr2 < 50.0, phi, torch.zeros_like(phi))
    phi = phi * radial_functions.cutoff_func_cos(d, bi.rcut)
    if distance_type:
        phi = phi * radial_functions.cutoff_func_poly(d, bi.rcut_in, bi.dcut_in)
    return phi.new_zeros(n_u).index_add(0, seg, phi)


class _BondInfo:
    """Python-float copy of everything the energy needs from a bond/embedding specification."""

    def __init__(self, bond_spec, embed_spec):
        self.nradmax = int(bond_spec.nradmax)
        self.lmax = int(bond_spec.lmax)
        self.nradbase = int(bond_spec.nradbasemax)
        self.rcut = float(bond_spec.rcut)
        self.lmbda = float(bond_spec.radparameters[0])
        self.radbasename = bond_spec.radbasename
        # the C++ evaluator uses |prefactor| and |lambda| (repulsion is strictly positive and decaying)
        self.core_lmbda = abs(float(bond_spec.lambdahc))
        self.core_pre = abs(float(bond_spec.prehc))
        self.inner_cutoff_type = str(bond_spec.inner_cutoff_type)
        self.rcut_in = float(bond_spec.rcut_in)
        self.dcut_in = float(bond_spec.dcut_in)
        self.ndensity = int(embed_spec.ndensity)
        self.rho_core_cutoff = float(embed_spec.rho_core_cutoff)
        self.drho_core_cutoff = float(embed_spec.drho_core_cutoff)


class ACE(nn.Module):
    """Atomic Cluster Expansion potential with trainable coefficients ``fit_coefs`` (float64).

    Radial coefficients come first in ``fit_coefs`` (``total_num_crad`` entries), followed by the basis
    coefficients, in the order of ``ACEBBasisSet.all_coeffs``.
    """

    def __init__(self, potconfig, compute_smoothness=False, compute_orthogonality=False, device=None):
        super().__init__()
        self.device = torch.device(device) if device is not None else get_device()
        self.compute_smoothness = compute_smoothness
        self.compute_orthogonality = compute_orthogonality
        self.radbasetype = "ChebPow"
        self.BbasisFuncs = None

        if isinstance(potconfig, str):
            raise NotImplementedError(
                "Initialization from anything but BBasisConf is not implemented for multicomponent ACE"
            )
        self._init_basis_configuration_from_bbasisconf(potconfig)

        self.ncoef = 0 if self.tmp_coefs is None else len(self.tmp_coefs)
        self.reg_l1 = 0
        self.reg_l2 = 0
        self.fit_coefs = nn.Parameter(
            torch.as_tensor(np.asarray(self.tmp_coefs), dtype=torch.float64, device=self.device)
        )
        self.aux = [] if (self.compute_smoothness or self.compute_orthogonality) else None
        self.factor4pi = math.sqrt(4 * math.pi)
        self.sh = SphericalHarmonics(self.lmax, device=self.device)
        self._smooth_grid = None

    # ------------------------------------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------------------------------------
    def _init_basis_configuration_from_bbasisconf(self, bbasisfunconf):
        from pyace.basis import BBasisConfiguration, ACEBBasisSet

        assert isinstance(
            bbasisfunconf, BBasisConfiguration
        ), "provided configuration is not an instance of BBasisConfiguration"

        dev = self.device
        self.bbasisset = ACEBBasisSet(bbasisfunconf)
        self.nelements = self.bbasisset.nelements
        self.elem_names = self.bbasisset.elements_name
        self.elems_Z = [symbol_to_atomic_number[symbol] for symbol in self.elem_names]
        self.elem_idx = list(range(self.nelements))

        self.nradmax = self.bbasisset.nradmax
        self.lmax = self.bbasisset.lmax
        self.nradbase = self.bbasisset.nradbase
        self.bond_specs = self.bbasisset.map_bond_specifications
        self.embed_spec = self.bbasisset.map_embedding_specifications

        self.bond_combs = list(self.bond_specs.keys())
        self.bond_info = {c: _BondInfo(self.bond_specs[c], self.embed_spec[c[0]]) for c in self.bond_combs}
        self.index_layout = IndexLayout(self.elems_Z, self.bond_combs)

        cutoff_types = {bi.inner_cutoff_type for bi in self.bond_info.values()}
        if len(cutoff_types) != 1:
            raise ValueError("Mixed inner_cutoff_type values {} are not supported".format(sorted(cutoff_types)))
        self.inner_cutoff_type = cutoff_types.pop()
        if self.inner_cutoff_type == "zbl":
            raise NotImplementedError(
                "inner_cutoff_type 'zbl' (ZBL core repulsion) is not implemented in torchpotential; "
                "use 'density' or 'distance'"
            )
        if self.inner_cutoff_type not in ("density", "distance"):
            raise ValueError("Unknown inner_cutoff_type '{}'".format(self.inner_cutoff_type))

        self.ndensity = self.bbasisset.ndensitymax
        self.rcut = self.bbasisset.cutoffmax

        self.tmp_coefs = np.array(self.bbasisset.all_coeffs, dtype=np.float64)
        self.tmp_coefs[self.tmp_coefs == 0] += 1e-32

        self.total_num_crad = 0
        bond_mus_unique = list(combinations_with_replacement(range(self.nelements), 2))
        unique_bond_to_slice = {}
        for mus in bond_mus_unique:
            if mus not in self.bond_specs:
                continue
            bond = self.bond_specs[mus]
            n_crad = int(np.prod(np.shape(bond.radcoefficients)))
            unique_bond_to_slice[mus] = [self.total_num_crad, self.total_num_crad + n_crad]
            self.total_num_crad += n_crad
        self.bond_to_slice = {k: unique_bond_to_slice[tuple(sorted(k))] for k in self.bond_specs.keys()}

        coefs_part = {}
        coefs_r1_part = {c: [] for c in self.bond_combs}
        self.coefs_rk_part = {c: [] for c in self.bond_combs}
        count = 0
        self.ranksmax = []
        for ne in range(self.nelements):
            coefs_part[ne] = []
            basis = self.bbasisset.basis_rank1[ne] + self.bbasisset.basis[ne]
            rank = 0
            for f in basis:
                r = f.rank
                for _ in f.coeffs:
                    key = (f.mu0, f.mus[0])
                    if r == 1:
                        coefs_r1_part[key] += [count]
                    else:
                        coefs_part[ne] += [count]
                    self.coefs_rk_part[key] += [count + self.total_num_crad]
                    count += 1
                rank = max(rank, r)
            self.ranksmax.append(rank)
        as_idx = lambda lst: torch.as_tensor(np.asarray(lst, dtype=np.int64), device=dev)
        self.coefs_part = {ne: as_idx(v) for ne, v in coefs_part.items()}
        self.coefs_r1_part = {c: as_idx(coefs_r1_part[c]) for c in self.bond_combs}
        self.rankmax = max(self.ranksmax)

        bbasisfuncspecs = [[BBasisFunc(f) for f in self.bbasisset.basis[ne]] for ne in range(self.nelements)]
        self.config = ConfigBasis(bbasisfuncspecs, self.ranksmax, self.nelements)
        self._init_basis_tensors()

    def _init_basis_tensors(self):
        """Device tensors for every (central element, rank >= 2): flat gather indices, CG and function index."""
        dev = self.device
        row = self.nradmax * (self.lmax + 1) ** 2
        nlm = (self.lmax + 1) ** 2
        self._basis_t = []
        for e in range(self.nelements):
            per_rank = []
            for fs in self.config.central_atom[e]:
                if fs is None:
                    per_rank.append(None)
                    continue
                idx = []
                for r in range(fs.rank):
                    mnl = fs.munlm[r]  # [mu, n-1, merged_lm]
                    flat = mnl[:, 0].astype(np.int64) * row + mnl[:, 1].astype(np.int64) * nlm + mnl[:, 2]
                    idx.append(torch.as_tensor(flat, device=dev))
                per_rank.append(
                    dict(
                        idx=idx,
                        cg=torch.as_tensor(np.asarray(fs.genCG, dtype=np.float64).reshape(-1), device=dev),
                        msum=torch.as_tensor(np.asarray(fs.msum, dtype=np.int64), device=dev),
                        nfunc=fs.nfunc,
                    )
                )
            self._basis_t.append(per_rank)

    # ------------------------------------------------------------------------------------------------
    # coefficients
    # ------------------------------------------------------------------------------------------------
    def set_coefs(self, coefs):
        with torch.no_grad():
            self.fit_coefs.copy_(torch.as_tensor(np.asarray(coefs), dtype=torch.float64, device=self.device))

    def get_coefs(self):
        return self.fit_coefs

    def get_updated_config(self, updating_coefs=None, prefix=None):
        if updating_coefs is not None:
            self.set_coefs(updating_coefs)
        self.bbasisset.all_coeffs = self.fit_coefs.detach().cpu().numpy()
        return self.bbasisset.to_BBasisConfiguration()

    def compute_regularization(self):
        basis_coefs = self.fit_coefs[self.total_num_crad :]
        self.reg_l1 = torch.sum(torch.abs(basis_coefs))
        self.reg_l2 = torch.sum(basis_coefs**2)

    # ------------------------------------------------------------------------------------------------
    # radial functions
    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def cheb_exp_cos(d_ij, nfuncs, cutoff, lmbda):
        d_ch = 1.0 - 2.0 * ((torch.exp(-lmbda * (d_ij / cutoff - 1.0)) - 1.0) / (math.exp(lmbda) - 1.0))
        chebpol = radial_functions.chebvander(d_ch, nfuncs)
        cos_cut = radial_functions.cutoff_func_cos(d_ij, cutoff).unsqueeze(-1)
        gk = torch.where(chebpol == 1.0, chebpol * cos_cut, 0.5 * (1 - chebpol) * cos_cut)
        if nfuncs == 1:
            return gk[:, :-1]
        return gk

    @staticmethod
    def cheb_pow(d_ij, nfuncs, cutoff, lmbda):
        d_ch = 2.0 * (1.0 - torch.abs(1.0 - d_ij / cutoff) ** lmbda) - 1.0
        chebpol = radial_functions.chebvander(d_ch, nfuncs + 1)
        chebpol = 0.5 - 0.5 * chebpol[:, 1:]
        return torch.where((d_ij < cutoff).unsqueeze(-1), chebpol, torch.zeros_like(chebpol))

    @staticmethod
    def bessel_s(d_ij, nfuncs, cutoff):
        y = radial_functions.simplified_bessel(d_ij, cutoff, nfuncs)
        return torch.where((d_ij < cutoff).unsqueeze(-1), y, torch.zeros_like(y))

    @staticmethod
    def legendre(d_ij, nfuncs, cutoff, lmbda):
        d_l = 2.0 * (1.0 - torch.abs(1.0 - d_ij / cutoff) ** lmbda) - 1.0
        legpol = radial_functions.legendre(d_l, nfuncs + 1)
        legpol = (legpol[:, 1:] - 1) / 2
        return torch.where((d_ij < cutoff).unsqueeze(-1), legpol, torch.zeros_like(legpol))

    def radial_function(self, d_ij, nfunc, ftype, cutoff, lmbda):
        """Radial basis for a 1-D tensor of distances, shape [N] -> [N, nfunc]."""
        if ftype == "ChebExpCos":
            return self.cheb_exp_cos(d_ij, nfunc, cutoff, lmbda)
        elif ftype == "ChebPow":
            return self.cheb_pow(d_ij, nfunc, cutoff, lmbda)
        elif ftype == "TEST_LegendrePow":
            return self.legendre(d_ij, nfunc, cutoff, lmbda)
        elif ftype == "TEST_SBessel":
            import warnings

            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn('Name "TEST_SBessel" is deprecated, use "SBessel" instead.', DeprecationWarning)
            return self.bessel_s(d_ij, nfunc, cutoff)
        elif ftype == "SBessel":
            return self.bessel_s(d_ij, nfunc, cutoff)
        else:
            raise ValueError("Unknown radial function type {}".format(ftype))

    # ------------------------------------------------------------------------------------------------
    # embedding / core repulsion
    # ------------------------------------------------------------------------------------------------
    def embedding_function(self, rho, mexp, ftype="FinnisSinclairShiftedScaled"):
        if ftype == "FinnisSinclairShiftedScaled":
            return self.f_exp_shsc(rho, mexp)
        elif ftype == "FinnisSinclair":
            return self.f_exp_old(rho, mexp)
        raise ValueError("Unknown embedding function type {}".format(ftype))

    def f_exp_old(self, rho, mexp):
        return torch.where(torch.abs(rho) < 1e-10, mexp * rho, self.en_func_old(rho, mexp))

    @staticmethod
    def en_func_old(rho, mexp):
        y1 = 10.0 * rho**2
        g = torch.where(30.0 < y1, 0.0 * rho, torch.exp(-y1))
        omg = 1.0 - g
        a = torch.abs(rho)
        y3 = torch.pow(omg * a + 1e-20, mexp)
        y2 = mexp * g * a
        return torch.sign(rho) * (y3 + y2)

    @staticmethod
    def f_exp_shsc(rho, mexp):
        if abs(mexp - 1.0) < 1e-10:
            return rho
        arho = torch.abs(rho)
        exprho = torch.exp(-arho)
        nx = 1.0 / mexp
        xoff = nx ** (nx / (1.0 - nx)) * exprho
        yoff = nx ** (1 / (1.0 - nx)) * exprho
        return torch.sign(rho) * (torch.pow(xoff + arho, mexp) - yoff)

    @staticmethod
    def f_cut_core_rep(rho_core, rho_core_cut, drho_core_cut):
        return torch.where(
            rho_core_cut <= rho_core,
            torch.zeros_like(rho_core),
            torch.where(
                rho_core < rho_core_cut - drho_core_cut,
                torch.ones_like(rho_core),
                radial_functions.cutoff_func_poly(rho_core, rho_core_cut, drho_core_cut),
            ),
        )

    # ------------------------------------------------------------------------------------------------
    # B-basis
    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def complexmul(r1, im1, r2, im2):
        return r1 * r2 - im1 * im2, im2 * r1 + im1 * r2

    def compute_bbasis_funcs(self, a_r, a_i, cntrl_at, rank):
        """``a_r``, ``a_i``: [n_at, nelem * nradmax * nlm] A-functions of the atoms of element ``cntrl_at``.
        Returns the B-functions of the given rank as [n_at, nfunc]."""
        bt = self._basis_t[cntrl_at][rank - 2]
        idx = bt["idx"]
        prod_r, prod_i = self.complexmul(a_r[:, idx[0]], a_i[:, idx[0]], a_r[:, idx[1]], a_i[:, idx[1]])
        for k in range(2, rank):
            prod_r, prod_i = self.complexmul(prod_r, prod_i, a_r[:, idx[k]], a_i[:, idx[k]])
        b_base = prod_r * bt["cg"]
        out = b_base.new_zeros(b_base.shape[0], bt["nfunc"])
        return out.index_add(1, bt["msum"], b_base)

    # ------------------------------------------------------------------------------------------------
    # energy
    # ------------------------------------------------------------------------------------------------
    def eval_atomic_energy(self, r_ij, index):
        dt = r_ij.dtype
        nat = index.nat
        nel = self.nelements
        nlm = (self.lmax + 1) ** 2
        want_a = self.rankmax > 1

        d_ij = torch.linalg.norm(r_ij, dim=1)
        rhat = r_ij / d_ij.unsqueeze(-1)
        ylm_r, ylm_i = self.sh.compute_ylm(rhat)
        ylm_r = ylm_r * self.factor4pi
        ylm_i = ylm_i * self.factor4pi

        radial_coefs = self.fit_coefs[: self.total_num_crad]
        basis_coefs = self.fit_coefs[self.total_num_crad :]
        dsor = r_ij.new_zeros(())

        distance_type = self.inner_cutoff_type == "distance"
        rho_r1, phi_core, a_vals_r, a_vals_i = [], [], [], []
        for c_i, c in enumerate(self.bond_combs):
            bi = self.bond_info[c]
            pidx = index.bond_pairs[c_i]
            seg = index.bond_sum_ind[c_i]
            n_u = index.bond_atoms[c_i].shape[0]
            d = d_ij[pidx]

            crad = radial_coefs[slice(*self.bond_to_slice[c])].reshape(bi.nradmax, bi.lmax + 1, bi.nradbase)
            if self.rankmax > 1:
                cr = crad.permute(1, 0, 2)  # [l, nrad, nbase]
                wwt = torch.matmul(cr, cr.transpose(1, 2))
                wtw = torch.matmul(cr.transpose(1, 2), cr)
                dsor = (
                    dsor
                    + torch.mean((wwt - torch.eye(bi.nradmax, dtype=dt, device=r_ij.device)) ** 2)
                    + torch.mean((wtw - torch.eye(bi.nradbase, dtype=dt, device=r_ij.device)) ** 2)
                )

            phi_ij = phi_core_per_bond(d, seg, n_u, bi, distance_type)
            phi_core.append(phi_ij)

            comb_gjk = self.radial_function(d, bi.nradbase, bi.radbasename, bi.rcut, bi.lmbda)  # [n, nradbase]
            if distance_type:
                comb_gjk = comb_gjk * (1 - radial_functions.cutoff_func_poly(d, bi.rcut_in, bi.dcut_in)).unsqueeze(-1)
            gk_i = comb_gjk.new_zeros(n_u, comb_gjk.shape[1]).index_add(0, seg, comb_gjk)

            coefs_r1 = basis_coefs[self.coefs_r1_part[c]].reshape(-1, bi.ndensity)
            e = gk_i @ coefs_r1
            rho_r1.append(F.pad(e, (0, self.ndensity - bi.ndensity)))

            if want_a:
                rj_nl = _contract_radial(comb_gjk, crad)
                rj_nl = F.pad(rj_nl, (0, self.lmax - bi.lmax, 0, self.nradmax - bi.nradmax))
                rj_nlm = rj_nl[:, :, self.sh.l_tile]  # [n, nradmax, nlm]
                aj_r = rj_nlm * ylm_r[pidx].unsqueeze(1)
                aj_i = rj_nlm * ylm_i[pidx].unsqueeze(1)
                width = self.nradmax * nlm
                a_vals_r.append(aj_r.new_zeros(n_u, *aj_r.shape[1:]).index_add(0, seg, aj_r).reshape(n_u, width))
                a_vals_i.append(aj_i.new_zeros(n_u, *aj_i.shape[1:]).index_add(0, seg, aj_i).reshape(n_u, width))

        if self.compute_orthogonality:
            self.aux.append(dsor.reshape(1, 1))

        atoms_cat = index.atoms_cat
        rho_r1 = rho_r1[0].new_zeros(nat, self.ndensity).index_add(0, atoms_cat, torch.cat(rho_r1, 0))
        phi_core = d_ij.new_zeros(nat).index_add(0, atoms_cat, torch.cat(phi_core, 0))

        if want_a:
            a_r = d_ij.new_zeros(nat * nel, self.nradmax * nlm).index_add(0, index.a_row_cat, torch.cat(a_vals_r, 0))
            a_i = d_ij.new_zeros(nat * nel, self.nradmax * nlm).index_add(0, index.a_row_cat, torch.cat(a_vals_i, 0))
            a_r = a_r.reshape(nat, -1)
            a_i = a_i.reshape(nat, -1)

        at_nrgs = []
        self.BbasisFuncs = [] if want_a else None
        for e in range(nel):
            atoms = index.elem_atoms[e]
            ndensity = self.embed_spec[e].ndensity
            fs_parameters = self.embed_spec[e].FS_parameters
            rho = rho_r1[atoms][:, :ndensity]
            if want_a and self.ranksmax[e] > 1:
                a_r_e, a_i_e = a_r[atoms], a_i[atoms]
                basis = [
                    self.compute_bbasis_funcs(a_r_e, a_i_e, e, k)
                    for k in range(2, self.ranksmax[e] + 1)
                    if self._basis_t[e][k - 2] is not None
                ]
                if basis:
                    basis = torch.cat(basis, dim=1)
                    self.BbasisFuncs.append(basis)
                    coefs = basis_coefs[self.coefs_part[e]].reshape(-1, ndensity)
                    rho = basis @ coefs + rho
            safe_rho = torch.where(rho != 0, rho, rho + 1e-32)
            en_sum = 0
            for dens in range(ndensity):
                en_sum = en_sum + fs_parameters[2 * dens] * self.embedding_function(
                    safe_rho[:, dens], float(fs_parameters[2 * dens + 1]), ftype=self.embed_spec[e].npoti
                )
            # inner cutoff acts on the total core density of the atom (all neighbours of all species)
            rho_core_e = phi_core[atoms]
            inner_cut = self.f_cut_core_rep(
                rho_core_e, self.embed_spec[e].rho_core_cutoff, self.embed_spec[e].drho_core_cutoff
            )
            at_nrgs.append(en_sum * inner_cut + rho_core_e)

        e_atom = torch.cat(at_nrgs, 0)[index.inv_perm]
        return e_atom.unsqueeze(-1)

    def compute_atomic_energy(self, r_ij, index):
        """Atomic energies [nat, 1] from pair vectors ``r_ij`` [npair, 3] and the batch's :class:`IndexPartitioner`."""
        if self.aux is not None:
            self.aux = []
        e_atom = self.eval_atomic_energy(r_ij, index)
        if self.compute_smoothness:
            self.compute_smoothness_reg()
        return e_atom

    # ------------------------------------------------------------------------------------------------
    # smoothness regularization
    # ------------------------------------------------------------------------------------------------
    def _get_smooth_grid(self):
        """Radial basis and its first two distance derivatives on a fixed grid (the basis is not trainable)."""
        if self._smooth_grid is not None:
            return self._smooth_grid
        from torch.func import jvp

        d_cont = torch.linspace(1e-10, self.rcut - 1e-5, _SMOOTH_GRID_POINTS, dtype=torch.float64, device=self.device)
        ones = torch.ones_like(d_cont)
        grid = {"d": d_cont, "dx": float(d_cont[1] - d_cont[0]), "g": {}}
        for c in self.bond_combs:
            bi = self.bond_info[c]

            def g(x, bi=bi):
                return self.radial_function(x, bi.nradbase, bi.radbasename, bi.rcut, bi.lmbda)

            def g1(x, g=g):
                return jvp(g, (x,), (ones,))[1]

            g0 = g(d_cont)
            d1 = g1(d_cont)
            d2 = jvp(g1, (d_cont,), (ones,))[1]
            grid["g"][c] = (g0.detach(), d1.detach(), d2.detach())
        self._smooth_grid = grid
        return grid

    @staticmethod
    def integrate(func, x, dx, rcut):
        f = torch.sum(torch.abs(func), dim=(1, 2))
        return torch.sum(x**2 * f) * dx / rcut**2

    def compute_smoothness_reg(self):
        grid = self._get_smooth_grid()
        d_cont, dx = grid["d"], grid["dx"]
        radial_coefs = self.fit_coefs[: self.total_num_crad]
        ints = [[], [], []]
        for c in self.bond_combs:
            bi = self.bond_info[c]
            crad = radial_coefs[slice(*self.bond_to_slice[c])].reshape(bi.nradmax, bi.lmax + 1, bi.nradbase)
            for k, gk in enumerate(grid["g"][c]):
                ints[k].append(self.integrate(_contract_radial(gk, crad), d_cont, dx, bi.rcut))
        for k in range(3):
            self.aux.append(torch.stack(ints[k]).mean().reshape(1, 1))

    # ------------------------------------------------------------------------------------------------
    def selective_fitting(self, list_of_bonds, basis_factor=1.0, rad_coefs_factor=1.0):
        assert len(list_of_bonds) <= len(self.bond_combs), ValueError(
            "Number of requested bond types ({}) exceeds the number specified in the potential ({})".format(
                len(list_of_bonds), len(self.bond_combs)
            )
        )
        for bond in list_of_bonds:
            assert bond in self.bond_combs, ValueError(
                "Bond type {} is not in the potential's list of bond combinations".format(bond)
            )
        assert basis_factor in (0.0, 1.0), ValueError("Only 0 and 1 are allowed values for a factor")
        assert rad_coefs_factor in (0.0, 1.0), ValueError("Only 0 and 1 are allowed values for a factor")

        factor_list = np.zeros_like(self.tmp_coefs).astype(np.float64)
        for bond in list_of_bonds:
            factor_list[self.coefs_rk_part[bond]] = basis_factor
            factor_list[slice(*self.bond_to_slice[bond])] = rad_coefs_factor

        return factor_list
