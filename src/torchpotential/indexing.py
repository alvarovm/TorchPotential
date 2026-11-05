"""Device-resident batch container and the per-batch index bookkeeping (``IndexPartitioner``).

In the TensorFlow implementation the partition/stitch indices were rebuilt inside the graph on every
evaluation. They only depend on the (fixed) batch and on the element/bond layout of the potential, so here they
are computed once per (batch, layout) and cached on the :class:`Batch`.
"""
import numpy as np
import torch

from torchpotential.utils.device import get_device

_FLOAT_KEYS = ("positions", "offsets", "cell", "energy", "forces", "eweights", "fweights")
_INT_KEYS = ("ind_i", "ind_j", "mu_i", "mu_j", "pair_map", "cell_map", "e_map")


class Batch:
    """A batch of structures (output of ``supply_batch``) as torch tensors on ``device``.

    Tensors: ``positions`` (scaled, [nat,3]), ``offsets`` ([npair,3]), ``cell`` ([ncell,3,3]), ``ind_i``, ``ind_j``,
    ``mu_i``, ``mu_j`` (atomic numbers), ``pair_map``, ``cell_map``, ``e_map`` and, when present, ``energy``
    ([ns,1]), ``forces`` ([nat,3]), ``eweights`` ([ns,1]), ``fweights`` ([nat,1]).
    """

    def __init__(self, data, device=None):
        self.device = torch.device(device) if device is not None else get_device()
        for k in _FLOAT_KEYS:
            if k in data and data[k] is not None:
                setattr(self, k, torch.as_tensor(np.asarray(data[k]), dtype=torch.float64, device=self.device))
            else:
                setattr(self, k, None)
        for k in _INT_KEYS:
            setattr(self, k, torch.as_tensor(np.asarray(data[k]), dtype=torch.long, device=self.device))

        e_map = np.asarray(data["e_map"])
        self.n_atoms = int(e_map.shape[0])
        self.n_struct = int(e_map.max()) + 1 if self.n_atoms else 0
        natoms = np.bincount(e_map, minlength=self.n_struct).astype(np.float64).reshape(-1, 1)
        self.natoms = torch.as_tensor(natoms, device=self.device)
        self._np = {k: np.asarray(data[k]) for k in ("ind_i", "mu_i", "mu_j")}
        self._index_cache = {}

    @classmethod
    def ensure(cls, data, device=None):
        return data if isinstance(data, Batch) else cls(data, device=device)

    def get_index(self, layout):
        """Cached :class:`IndexPartitioner` for ``layout`` (see ``ACE.index_layout``)."""
        key = layout.key
        idx = self._index_cache.get(key)
        if idx is None:
            idx = IndexPartitioner(self, layout)
            self._index_cache[key] = idx
        return idx


class IndexLayout:
    """Element / bond layout of a potential that determines the index structures."""

    def __init__(self, elems_Z, bond_combs):
        self.elems_Z = [int(z) for z in elems_Z]
        self.bond_combs = [tuple(c) for c in bond_combs]
        self.nelem = len(self.elems_Z)
        self.key = (tuple(self.elems_Z), tuple(self.bond_combs))
        zmax = max(self.elems_Z)
        self.z2idx = np.full(zmax + 1, -1, dtype=np.int64)
        for i, z in enumerate(self.elems_Z):
            self.z2idx[z] = i


class IndexPartitioner:
    """Index permutations for a batch.

    Attributes (``i`` runs over the bond combinations ``bond_combs``, ``e`` over the elements):

    * ``nat``: number of atoms
    * ``atomic_mu``: element index of every atom [nat]
    * ``elem_atoms[e]``: atoms of element ``e`` (ascending); ``inv_perm``: ``cat(per-element values)[inv_perm]``
      gives the values in atom order
    * ``bond_pairs[i]``: indices of the pairs belonging to bond ``i``
    * ``bond_atoms[i]``: atoms having at least one pair of bond ``i`` (ascending)
    * ``bond_sum_ind[i]``: for every pair of bond ``i`` the position of its central atom inside ``bond_atoms[i]``
    * ``atoms_cat``: concatenation of ``bond_atoms``; ``a_row_cat``: ``atom * nelem + mu_j`` rows of the
      ``[nat * nelem, ...]`` neighbour-density tensor written by each bond
    """

    def __init__(self, batch, layout):
        dev = batch.device
        nelem = layout.nelem
        ind_i = batch._np["ind_i"].astype(np.int64)
        nat = int(batch.n_atoms)
        if ind_i.size == 0 or int(ind_i.max()) + 1 != nat or np.bincount(ind_i, minlength=nat).min() == 0:
            raise ValueError(
                "Every atom of a structure must have at least one neighbour within the cutoff "
                "(found {} atoms, neighbour list covers {})".format(nat, 0 if ind_i.size == 0 else int(ind_i.max()) + 1)
            )
        self.nat = nat

        mu_i = self._to_index(batch._np["mu_i"], layout)
        mu_j = self._to_index(batch._np["mu_j"], layout)

        atomic_mu = np.zeros(nat, dtype=np.int64)
        atomic_mu[ind_i] = mu_i
        self.atomic_mu = torch.as_tensor(atomic_mu, device=dev)
        elem_atoms = [np.nonzero(atomic_mu == e)[0] for e in range(nelem)]
        self.elem_atoms = [torch.as_tensor(a, device=dev) for a in elem_atoms]
        # concatenating per-element results and indexing with ``inv_perm`` restores the atom order
        self.inv_perm = torch.as_tensor(np.argsort(np.concatenate(elem_atoms), kind="stable"), device=dev)

        comb_lookup = np.full(nelem * nelem, -1, dtype=np.int64)
        for i, (a, b) in enumerate(layout.bond_combs):
            comb_lookup[a * nelem + b] = i
        comb_id = comb_lookup[mu_i * nelem + mu_j]
        if (comb_id < 0).any():
            raise ValueError("Found pairs of elements that are not described by the potential's bond specifications")

        self.bond_pairs, self.bond_atoms, self.bond_sum_ind = [], [], []
        atoms_cat, a_row_cat = [], []
        for i, (a, b) in enumerate(layout.bond_combs):
            pairs = np.nonzero(comb_id == i)[0]
            atoms, inverse = np.unique(ind_i[pairs], return_inverse=True)
            self.bond_pairs.append(torch.as_tensor(pairs, device=dev))
            self.bond_atoms.append(torch.as_tensor(atoms, device=dev))
            self.bond_sum_ind.append(torch.as_tensor(inverse.reshape(-1), device=dev))
            atoms_cat.append(atoms)
            a_row_cat.append(atoms * nelem + b)
        self.atoms_cat = torch.as_tensor(np.concatenate(atoms_cat), device=dev)
        self.a_row_cat = torch.as_tensor(np.concatenate(a_row_cat), device=dev)
        self.n_struct = batch.n_struct

    @staticmethod
    def _to_index(z, layout):
        z = np.asarray(z, dtype=np.int64)
        if z.size and (z.max() >= len(layout.z2idx) or z.min() < 0):
            raise ValueError("Structure contains atomic numbers that are not part of the potential")
        idx = layout.z2idx[z]
        if (idx < 0).any():
            bad = sorted(set(int(v) for v in z[idx < 0]))
            raise ValueError("Structure contains atomic numbers {} that are not part of the potential".format(bad))
        return idx
