"""Generate TensorFlow fit-trajectory references (run once in the legacy TF env `ace`):

    CUDA_VISIBLE_DEVICES=-1 mamba run -n ace python tests/tools/gen_tf_fit_reference.py
"""
import os
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import numpy as np

from pyace.basis import BBasisConfiguration, ACEBBasisSet
from tensorpotential.compat.pace.potentials.ace import ACE
from tensorpotential.compat.pace.tensorpot import TensorPotential
from tensorpotential.compat.pace.fit import FitTensorPotential
from tensorpotential.compat.pace.constants import (LOSS_TYPE, LOSS_FORCE_FACTOR, LOSS_ENERGY_FACTOR, L1_REG, L2_REG,
                                                   AUX_LOSS_FACTOR)
from helpers import make_structures, make_df
import fit_cases as C

OUT = os.path.join(HERE, "..", "data", "tf_fit_ref")


def run(case):
    bconf = BBasisConfiguration(os.path.join(HERE, "..", "data", "yaml", case["yaml"]))
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
    rng = np.random.default_rng(case["seed"] + 11)
    c0 = ace.get_coefs().numpy()
    ace.set_coefs(c0 + case["perturb"] * rng.normal(size=c0.shape))
    start = ace.get_coefs().numpy().copy()
    tp = TensorPotential(ace, loss_specs=specs)
    fitter = FitTensorPotential(tp, display_step=1000)
    fitter.nfuncs = bconf.total_number_of_functions
    jf = None
    if case["mask_radial"]:
        jf = np.ones(ace.ncoef)
        jf[: ace.total_num_crad] = 0.0
    np.random.seed(123)
    fitter.fit(df.copy(), niter=case["niter"], optimizer=case["optimizer"], batch_size=case["batch_size"],
               jacobian_factor=jf, fit_metric_callback=lambda d: None)
    final = fitter.fit_coefs if fitter.res_opt is None else fitter.res_opt.x
    pred = fitter.predict(df.copy())
    np.savez_compressed(os.path.join(OUT, case["name"] + ".npz"), start=start, final=np.asarray(final),
                        loss_history=np.asarray(fitter.loss_history), e_pred=np.asarray(pred["energy_pred"]),
                        f_pred=np.concatenate([np.asarray(x).reshape(-1, 3) for x in pred["forces_pred"]]),
                        s_pred=np.asarray(pred["stress_pred"]),
                        nit=np.asarray(-1 if fitter.res_opt is None else fitter.res_opt.get("nit", -1)))
    print(case["name"], "n_loss_evals", len(fitter.loss_history), "loss", fitter.loss_history[0], "->",
          fitter.loss_history[-1])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for c in C.CASES:
        run(c)
