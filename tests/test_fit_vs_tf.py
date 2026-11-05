"""End-to-end fit trajectories vs. references recorded with the TensorFlow implementation."""
import os

import numpy as np
import pytest

from pyace.basis import BBasisConfiguration, ACEBBasisSet

from torchpotential.constants import *
from torchpotential.fit import FitTorchPotential
from torchpotential.potentials.ace import ACE
from torchpotential.torchpot import TorchPotential

from fit_cases import CASES, TORCH_ONLY_CASES
from helpers import make_structures, make_df

HERE = os.path.dirname(os.path.abspath(__file__))


def run_case(case, start=None):
    bconf = BBasisConfiguration(os.path.join(HERE, "data", "yaml", case["yaml"]))
    for blk in bconf.funcspecs_blocks:  # the TF references contain no core repulsion (see test_tf_parity.load)
        blk.core_rep_parameters = [0.0, 1.0]
    elements = list(ACEBBasisSet(bconf).elements_name)
    cutoff = float(max(b.rcutij for b in bconf.funcspecs_blocks)) + 0.05
    structs = make_structures(elements, seed=case["seed"], min_dist=2.0)
    df = make_df(structs, cutoff, seed=case["seed"])
    n_batches = int(np.ceil(len(df) / case["batch_size"]))
    aux = case["aux"]
    specs = {LOSS_TYPE: "per-atom", LOSS_FORCE_FACTOR: case["kappa"], LOSS_ENERGY_FACTOR: 1 - case["kappa"],
             L1_REG: case["L1"] / n_batches, L2_REG: case["L2"] / n_batches}
    if aux:
        specs[AUX_LOSS_FACTOR] = [w / n_batches for w in case["aux_w"]]
    ace = ACE(bconf, compute_orthogonality=aux, compute_smoothness=aux)
    if start is None:
        rng = np.random.default_rng(case["seed"] + 11)
        c0 = ace.get_coefs().detach().cpu().numpy()
        start = c0 + case["perturb"] * rng.normal(size=c0.shape)
    ace.set_coefs(start)
    fitter = FitTorchPotential(TorchPotential(ace, loss_specs=specs), display_step=1000)
    fitter.nfuncs = bconf.total_number_of_functions
    jf = None
    if case["mask_radial"]:
        jf = np.ones(ace.ncoef)
        jf[: ace.total_num_crad] = 0.0
    np.random.seed(123)
    fitter.fit(df.copy(), niter=case["niter"], optimizer=case["optimizer"], batch_size=case["batch_size"],
               jacobian_factor=jf, fit_metric_callback=lambda d: None)
    final = fitter.fit_coefs if fitter.res_opt is None else fitter.res_opt.x
    return fitter, np.asarray(final), df


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fit_trajectory_matches_tf(case):
    ref = np.load(os.path.join(HERE, "data", "tf_fit_ref", case["name"] + ".npz"))
    fitter, final, df = run_case(case, start=ref["start"])
    tol = 1e-5 if case["optimizer"] == "Adam" else 1e-7
    hist = np.asarray(fitter.loss_history)
    assert len(hist) == len(ref["loss_history"])
    np.testing.assert_allclose(hist, ref["loss_history"], rtol=tol)
    np.testing.assert_allclose(final, ref["final"], rtol=tol, atol=tol * np.abs(ref["final"]).max())

    pred = fitter.predict(df.copy())
    np.testing.assert_allclose(pred["energy_pred"], ref["e_pred"], rtol=tol, atol=1e-9)
    f_pred = np.concatenate([np.asarray(x).reshape(-1, 3) for x in pred["forces_pred"]])
    np.testing.assert_allclose(f_pred, ref["f_pred"], rtol=tol, atol=tol * np.abs(ref["f_pred"]).max())
    np.testing.assert_allclose(np.asarray(pred["stress_pred"]), ref["s_pred"], rtol=tol,
                               atol=tol * np.abs(ref["s_pred"]).max())


@pytest.mark.parametrize("case", TORCH_ONLY_CASES, ids=[c["name"] for c in TORCH_ONLY_CASES])
def test_least_squares_path_runs_and_decreases_loss(case):
    fitter, final, df = run_case(case)
    assert len(fitter.loss_history) >= 2
    assert fitter.loss_history[-1] < fitter.loss_history[0]
    assert np.all(np.isfinite(final))
