import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from pyace.atomicenvironment import generate_tp_atoms


def _stretch_to_min_dist(atoms, min_dist, exact=False):
    """Scale a periodic structure so that its shortest interatomic distance (incl. images) is >= min_dist
    (``exact=True``: == min_dist, i.e. structures are compressed as well)."""
    from ase.neighborlist import neighbor_list

    d = neighbor_list("d", atoms, 3.0)
    if len(d) and (d.min() < min_dist or exact):
        f = min_dist / d.min()
        atoms.set_cell(atoms.cell.array * f, scale_atoms=True)
    return atoms


def make_structures(elements, seed, min_dist=None, exact_min_dist=False):
    rng = np.random.default_rng(seed)
    els = list(elements)

    def assign(n):
        return [els[i % len(els)] for i in range(n)]

    structs = []
    a = bulk(els[0], "fcc", a=4.0, cubic=True)
    a.set_chemical_symbols(assign(len(a)))
    a.rattle(0.08, seed=int(rng.integers(1e6)))
    structs.append(a)
    b = bulk(els[0], "bcc", a=3.2, cubic=True).repeat((2, 1, 1))
    b.set_chemical_symbols(assign(len(b))[::-1])
    b.rattle(0.05, seed=int(rng.integers(1e6)))
    structs.append(b)
    cell = np.array([[3.4, 0.2, 0.1], [0.3, 3.7, -0.2], [0.1, 0.4, 3.9]])
    structs.append(Atoms(assign(3), scaled_positions=rng.random((3, 3)) * 0.8 + 0.05, cell=cell, pbc=True))
    structs.append(Atoms(assign(5), positions=rng.normal(size=(5, 3)) * 1.2, pbc=False))
    structs.append(Atoms(els[-1], positions=[[0, 0, 0]], cell=np.eye(3) * 2.6, pbc=True))
    if min_dist is not None:
        structs = [_stretch_to_min_dist(s, min_dist, exact_min_dist) if s.pbc.all() else s for s in structs]
    for s in structs:
        n = len(s)
        s.calc = SinglePointCalculator(s, energy=float(-3.0 * n + rng.normal() * 0.3), forces=rng.normal(size=(n, 3)) * 0.4)
    return structs


def make_batch(structs, cutoff, weights_seed=None):
    from torchpotential.utils.utilities import supply_batch

    rng = np.random.default_rng(weights_seed) if weights_seed is not None else None
    tp = []
    for s in structs:
        t = generate_tp_atoms(s, cutoff=cutoff)
        if t is None:
            continue
        t = dict(t)
        if rng is not None:
            t["_eweights"] = rng.uniform(0.5, 1.5, size=(1, 1))
            t["_fweights"] = rng.uniform(0.5, 1.5, size=(len(t["_forces"]), 1))
        tp.append(t)
    return supply_batch(tp)


def make_df(structs, cutoff, seed):
    """Fitting dataframe with ``tp_atoms`` / ``w_energy`` / ``w_forces`` columns (as produced by pyace)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for s in structs:
        t = generate_tp_atoms(s, cutoff=cutoff)
        if t is None:
            continue
        rows.append(dict(tp_atoms=dict(t), w_energy=float(rng.uniform(0.5, 1.5)),
                         w_forces=rng.uniform(0.5, 1.5, size=len(s)), NUMBER_OF_ATOMS=len(s),
                         energy_corrected=s.get_potential_energy()))
    return pd.DataFrame(rows)
