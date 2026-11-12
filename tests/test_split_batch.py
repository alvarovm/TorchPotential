"""``split_batch``: pieces must reassemble to the original batch, and evaluating them piecewise must reproduce
evaluating the whole merged batch at once (used by ``FitTorchPotential.predict`` to bound the transient device
memory of batches that must not stay resident, see its docstring)."""
import os

import numpy as np
import pytest

from pyace.basis import BBasisConfiguration, ACEBBasisSet

from torchpotential.potentials.ace import ACE
from torchpotential.torchpot import TorchPotential
from torchpotential.utils.utilities import split_batch

from helpers import make_batch, make_structures

HERE = os.path.dirname(os.path.abspath(__file__))


def _batch():
    structs = make_structures(["Al"], seed=11, min_dist=2.0)
    basis = BBasisConfiguration(os.path.join(HERE, "data", "yaml", "Al-r1234l12_crad_dif_2.yaml"))
    for block in basis.funcspecs_blocks:
        block.core_rep_parameters = [0.0, 1.0]
    cutoff = max(b.rcutij for b in basis.funcspecs_blocks) + 0.05
    return make_batch(structs, cutoff, weights_seed=11), basis


@pytest.mark.parametrize("max_structures", [1, 2, 3, 100])
def test_pieces_reassemble_to_the_original_batch(max_structures):
    b, _ = _batch()
    n_structures = int(b["e_map"].max()) + 1
    pieces = split_batch(b, max_structures)
    assert sum(int(p["e_map"].max()) + 1 if len(p["e_map"]) else 0 for p in pieces) == n_structures
    assert sum(len(p["ind_i"]) for p in pieces) == len(b["ind_i"])
    assert sum(len(p["positions"]) for p in pieces) == len(b["positions"])

    for p in pieces:
        n_struct_p = int(p["e_map"].max()) + 1
        assert p["e_map"].min() == 0
        assert p["cell_map"].min() == 0 and p["cell_map"].max() == n_struct_p - 1
        assert p["pair_map"].min() == 0 and p["pair_map"].max() == n_struct_p - 1
        assert p["ind_i"].min() == 0 and p["ind_i"].max() == len(p["positions"]) - 1
        assert p["ind_j"].min() >= 0 and p["ind_j"].max() <= p["ind_i"].max()
        assert len(p["energy"]) == n_struct_p == len(p["eweights"]) == len(p["cell"])

    # concatenating every field in piece order reconstructs the original per-structure data exactly
    np.testing.assert_array_equal(np.concatenate([p["positions"] for p in pieces]), b["positions"])
    np.testing.assert_array_equal(np.concatenate([p["forces"] for p in pieces]), b["forces"])
    np.testing.assert_array_equal(np.concatenate([p["energy"] for p in pieces]), b["energy"])
    np.testing.assert_array_equal(np.concatenate([p["cell"] for p in pieces]), b["cell"])


def test_no_split_below_the_threshold_returns_the_same_object():
    b, _ = _batch()
    n_structures = int(b["e_map"].max()) + 1
    assert split_batch(b, n_structures) == [b]
    assert split_batch(b, n_structures + 10) == [b]


@pytest.mark.parametrize("max_structures", [1, 2, 3])
def test_split_evaluation_matches_the_whole_batch(max_structures):
    b, basis = _batch()
    ace = ACE(basis, device="cpu")
    tp = TorchPotential(ace, loss_specs=None)

    e_whole, f_whole, s_whole = tp.evaluate(b)
    pieces = split_batch(b, max_structures)
    e_parts = [tp.evaluate(p)[0].detach().numpy() for p in pieces]
    f_parts = [tp.evaluate(p)[1].detach().numpy() for p in pieces]

    np.testing.assert_allclose(np.concatenate(e_parts), e_whole.detach().numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.concatenate(f_parts), f_whole.detach().numpy(), rtol=1e-10, atol=1e-12)
