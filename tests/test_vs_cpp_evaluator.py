"""Energies/forces of the PyTorch implementation vs. the C++ ACE evaluator (independent of TensorFlow).

The core repulsion, the embedding and the inner cutoff follow the C++ evaluator (``inner_cutoff_type``
``density`` and ``distance``); ``zbl`` is rejected with ``NotImplementedError``.
"""
import os

import numpy as np
import pytest

from pyace import PyACECalculator
from pyace.basis import BBasisConfiguration, ACEBBasisSet

from torchpotential.potentials.ace import ACE
from torchpotential.torchpot import TorchPotential

from helpers import make_structures, make_batch

HERE = os.path.dirname(os.path.abspath(__file__))
YAMLS = ["Al-r1234l12_crad_dif_2.yaml", "Al-Ni_opt_all.yaml", "ternary_AlCuZn.yaml", "Cu.pbe.in.yaml",
         "Ni_2dens_bug_case.yaml"]


def _config(yaml_name, **block_attrs):
    """BBasisConfiguration with attributes of every block overridden (e.g. ``rho_cut=0.5``)."""
    bconf = BBasisConfiguration(os.path.join(HERE, "data", "yaml", yaml_name))
    for blk in bconf.funcspecs_blocks:
        for k, v in block_attrs.items():
            setattr(blk, k, v)
    return bconf


def _cpp_calc(bconf):
    return PyACECalculator(basis_set=ACEBBasisSet(bconf))


def _compare_with_cpp(bconf, seed, min_dist, rtol=1e-8):
    elements = list(ACEBBasisSet(bconf).elements_name)
    cutoff = float(max(b.rcutij for b in bconf.funcspecs_blocks)) + 0.05
    structs = make_structures(elements, seed=seed, min_dist=min_dist)
    batch = make_batch(structs, cutoff)

    e, f, _ = TorchPotential(ACE(bconf)).evaluate(batch)
    e, f = e.cpu().numpy().reshape(-1), f.cpu().numpy()

    calc = _cpp_calc(bconf)
    off = np.cumsum([0] + [len(s) for s in structs])
    n_checked = 0
    for i, s in enumerate(structs):
        if not s.pbc.all():
            continue  # the torch pipeline wraps clusters into a box; compare periodic structures only
        a = s.copy()
        a.calc = calc
        e_ref, f_ref = a.get_potential_energy(), a.get_forces()
        assert abs(e[i] - e_ref) < rtol * max(1.0, abs(e_ref)), (i, e[i], e_ref)
        np.testing.assert_allclose(f[off[i]:off[i + 1]], f_ref, rtol=rtol, atol=rtol * max(1.0, np.abs(f_ref).max()))
        n_checked += 1
    assert n_checked >= 3


@pytest.mark.parametrize("yaml_name", YAMLS)
def test_energy_forces_vs_cpp(yaml_name):
    _compare_with_cpp(_config(yaml_name), seed=sum(map(ord, yaml_name)), min_dist=2.0)


def _core_effect(yaml_name, seed, min_dist, core=(300.0, 6.0), **block_attrs):
    """Effect of switching the core repulsion on, in both evaluators: ``(torch dE, cpp dE, torch dF, cpp dF)``.

    Taking the difference cancels the interpolation error of the C++ radial splines (large at short distances) in the
    parts of the potential that do not depend on the core repulsion."""
    out = []
    for core_rep in ([0.0, 1.0], list(core)):
        bconf = _config(yaml_name, core_rep_parameters=core_rep, **block_attrs)
        elements = list(ACEBBasisSet(bconf).elements_name)
        cutoff = float(max(b.rcutij for b in bconf.funcspecs_blocks)) + 0.05
        structs = [s for s in make_structures(elements, seed=seed, min_dist=min_dist, exact_min_dist=True)
                   if s.pbc.all()]
        e, f, _ = TorchPotential(ACE(bconf)).evaluate(make_batch(structs, cutoff))
        calc = _cpp_calc(bconf)
        e_cpp, f_cpp = [], []
        for s in structs:
            a = s.copy()
            a.calc = calc
            e_cpp.append(a.get_potential_energy())
            f_cpp.append(a.get_forces())
        out.append((e.cpu().numpy().reshape(-1), f.cpu().numpy(), np.array(e_cpp), np.concatenate(f_cpp)))
    (e0, f0, ec0, fc0), (e1, f1, ec1, fc1) = out
    return e1 - e0, ec1 - ec0, f1 - f0, fc1 - fc0


def _assert_core_effect_matches(yaml_name, seed, min_dist, rtol, **kw):
    de_t, de_c, df_t, df_c = _core_effect(yaml_name, seed, min_dist, **kw)
    assert np.abs(de_c).max() > 1e-3, "core repulsion has no effect, the test is meaningless"
    np.testing.assert_allclose(de_t, de_c, rtol=rtol, atol=rtol * np.abs(de_c).max())
    np.testing.assert_allclose(df_t, df_c, rtol=rtol, atol=rtol * np.abs(df_c).max())


@pytest.mark.parametrize("yaml_name", YAMLS)
def test_core_repulsion_vs_cpp(yaml_name):
    """Pair term ``pre exp(-lambda r^2)/r (1+cos(pi r/rcut))/2`` summed over all neighbours of all species."""
    # rtol: the C++ radial/core splines (bin 0.001) are accurate to ~1e-8 between nodes
    _assert_core_effect_matches(yaml_name, seed=sum(map(ord, yaml_name)) + 1, min_dist=0.9, rtol=1e-7)


@pytest.mark.parametrize("yaml_name, species", [("Al-Ni_opt_all.yaml", ("Al", "Ni")), ("Al-Ni_opt_all.yaml", ("Ni", "Ni")),
                                                 ("ternary_AlCuZn.yaml", ("Cu", "Zn")),
                                                 ("Ni_2dens_bug_case.yaml", ("Ni", "Ni"))])
@pytest.mark.parametrize("rho_cut, drho_cut", [(0.3, 0.2), (2.0, 1.5), (0.05, 0.04)])
def test_inner_cutoff_zone_vs_cpp(yaml_name, species, rho_cut, drho_cut):
    """Dimers at off-grid distances: the core density of the atoms sweeps through the polynomial cutoff zone
    (and the energy is small, so the interpolation error of the C++ splines does not hide the comparison)."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    bconf = _config(yaml_name, rho_cut=rho_cut, drho_cut=drho_cut, core_rep_parameters=[300.0, 6.0])
    tp = TorchPotential(ACE(bconf))
    calc = _cpp_calc(bconf)
    cutoff = float(max(b.rcutij for b in bconf.funcspecs_blocks)) + 0.05
    rng = np.random.default_rng(5)
    structs = []
    for d in np.sort(rng.uniform(0.75, 1.7, size=40)):
        s = Atoms(list(species), positions=[[0, 0, 0], [d, 0, 0]], cell=np.eye(3) * 20.0, pbc=True)
        s.calc = SinglePointCalculator(s, energy=0.0, forces=np.zeros((2, 3)))
        structs.append(s)
    e, f, _ = tp.evaluate(make_batch(structs, cutoff))
    e, f = e.cpu().numpy().reshape(-1), f.cpu().numpy()
    e_ref, f_ref = [], []
    for s in structs:
        a = s.copy()
        a.calc = calc
        e_ref.append(a.get_potential_energy())
        f_ref.append(a.get_forces())
    e_ref, f_ref = np.array(e_ref), np.concatenate(f_ref)
    np.testing.assert_allclose(e, e_ref, rtol=1e-7, atol=1e-7 * max(1.0, np.abs(e_ref).max()))
    np.testing.assert_allclose(f, f_ref, rtol=1e-7, atol=1e-7 * np.abs(f_ref).max())


@pytest.mark.parametrize("yaml_name", ["Al-Ni_opt_all.yaml", "ternary_AlCuZn.yaml"])
def test_distance_inner_cutoff_vs_cpp(yaml_name):
    """``inner_cutoff_type: distance``: pair repulsion and the radial basis are cut by the polynomial inner cutoff."""
    attrs = dict(inner_cutoff_type="distance", r_in=2.6, delta_in=1.0)
    _compare_with_cpp(_config(yaml_name, **attrs), seed=sum(map(ord, yaml_name)) + 3, min_dist=2.0, rtol=1e-6)
    _assert_core_effect_matches(yaml_name, seed=sum(map(ord, yaml_name)) + 4, min_dist=1.4, rtol=1e-7, **attrs)


def test_zbl_inner_cutoff_is_rejected():
    bconf = _config("Al-Ni_opt_all.yaml", inner_cutoff_type="zbl")
    with pytest.raises(NotImplementedError, match="zbl"):
        ACE(bconf)


def test_batch_missing_bond_types():
    """A ternary potential evaluated on single-element / binary structures (bond types absent in the batch)."""
    from ase.build import bulk

    bconf = _config("ternary_AlCuZn.yaml")
    tp = TorchPotential(ACE(bconf))
    calc = _cpp_calc(bconf)
    structs = []
    for comp in (["Al"] * 4, ["Cu"] * 4, ["Al", "Al", "Cu", "Cu"], ["Zn", "Cu", "Zn", "Zn"]):
        s = bulk("Al", "fcc", a=4.2, cubic=True)
        s.set_chemical_symbols(comp)
        s.rattle(0.05, seed=3)
        structs.append(s)
    from ase.calculators.singlepoint import SinglePointCalculator

    for s in structs:
        s.calc = SinglePointCalculator(s, energy=0.0, forces=np.zeros((len(s), 3)))
    # every structure alone (so that some bond types are absent) and all together
    for group in [[s] for s in structs] + [structs]:
        e, f, _ = tp.evaluate(make_batch(group, 4.05))
        for k, s in enumerate(group):
            a = s.copy()
            a.calc = calc
            assert abs(e.cpu().numpy().reshape(-1)[k] - a.get_potential_energy()) < 1e-9
