"""Generate TensorFlow reference fixtures for TorchPotential parity tests.

Run ONCE, in the legacy TF env (`ace`), while TF is still available:

    CUDA_VISIBLE_DEVICES=-1 mamba run -n ace python tests/tools/gen_tf_reference.py

Each case writes tests/data/tf_ref/<case>.npz holding the batched inputs, the
initial/perturbed coefficients and the reference outputs (energy, forces,
stress, loss, d(loss)/d(coefs), regularization components, hessian).
The BBasis yaml files are copied into tests/data/yaml.
"""
import os
import shutil
import sys
import traceback
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from pyace.basis import BBasisConfiguration, ACEBBasisSet
from pyace.atomicenvironment import generate_tp_atoms

from tensorpotential.compat.pace.potentials.ace import ACE
from tensorpotential.compat.pace.tensorpot import TensorPotential
from tensorpotential.compat.pace.utils.utilities import supply_batch
from tensorpotential.compat.pace.constants import (LOSS_TYPE, LOSS_FORCE_FACTOR, LOSS_ENERGY_FACTOR,
                                                   L1_REG, L2_REG, AUX_LOSS_FACTOR)

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.normpath(os.path.join(HERE, ".."))
PYACE_TESTS = "/home/vama/soft/ace/python-ace/tests"
OUT = os.path.join(TESTS, "data", "tf_ref")
YAML_OUT = os.path.join(TESTS, "data", "yaml")

CASES = [
    # name, yaml
    ("unary_r1234", "Al-r1234l12_crad_dif_2.yaml"),
    ("binary_AlNi", "Al-Ni_opt_all.yaml"),
    ("ternary_AlCuZn", "ternary_AlCuZn.yaml"),
    ("unary_lmax6_cu", "Cu.pbe.in.yaml"),
    ("unary_sbessel_rank1", "SBessel_rank1.yaml"),
    ("unary_2dens_ni", "Ni_2dens_bug_case.yaml"),
]

LOSS_VARIANTS = {
    "peratom_reg": {LOSS_TYPE: "per-atom", "kappa": 0.4, "L1": 1e-3, "L2": 2e-3, "aux": True},
    "perstruct": {LOSS_TYPE: "per-structure", "kappa": 0.7, "L1": 0.0, "L2": 0.0, "aux": False},
}


def make_structures(elements, rcut, seed):
    rng = np.random.default_rng(seed)
    els = list(elements)

    def assign(n):
        return [els[i % len(els)] for i in range(n)]

    structs = []
    # periodic, cubic fcc
    a = bulk(els[0], "fcc", a=4.0, cubic=True)
    a.set_chemical_symbols(assign(len(a)))
    a.rattle(0.08, seed=int(rng.integers(1e6)))
    structs.append(a)
    # periodic bcc, 2x1x1
    b = bulk(els[0], "bcc", a=3.2, cubic=True).repeat((2, 1, 1))
    b.set_chemical_symbols(assign(len(b))[::-1])
    b.rattle(0.05, seed=int(rng.integers(1e6)))
    structs.append(b)
    # triclinic, 3 atoms
    cell = np.array([[3.4, 0.2, 0.1], [0.3, 3.7, -0.2], [0.1, 0.4, 3.9]])
    c = Atoms(assign(3), scaled_positions=rng.random((3, 3)) * 0.8 + 0.05, cell=cell, pbc=True)
    structs.append(c)
    # non-periodic cluster (exercises enforce_pbc)
    pos = rng.normal(size=(5, 3)) * 1.2
    d = Atoms(assign(5), positions=pos, pbc=False)
    structs.append(d)
    # single-atom periodic cell
    e = Atoms(els[-1], positions=[[0, 0, 0]], cell=np.eye(3) * 2.6, pbc=True)
    structs.append(e)

    for s in structs:
        n = len(s)
        s.calc = SinglePointCalculator(s, energy=float(-3.0 * n + rng.normal() * 0.3),
                                       forces=rng.normal(size=(n, 3)) * 0.4)
    return structs


def build_batch(structs, cutoff, seed):
    rng = np.random.default_rng(seed + 1)
    tp = []
    for s in structs:
        t = generate_tp_atoms(s, cutoff=cutoff)
        if t is None:
            continue
        t = dict(t)
        t["_eweights"] = rng.uniform(0.5, 1.5, size=(1, 1))
        t["_fweights"] = rng.uniform(0.5, 1.5, size=(len(t["_forces"]), 1))
        tp.append(t)
    return supply_batch(tp)


def run_case(name, yaml_name):
    local = os.path.join(TESTS, "data", "yaml_src", yaml_name)
    src = local if os.path.exists(local) else os.path.join(PYACE_TESTS, yaml_name)
    shutil.copy(src, os.path.join(YAML_OUT, yaml_name))
    bconf = BBasisConfiguration(src)
    bset = ACEBBasisSet(bconf)
    elements = list(bset.elements_name)
    rcut = max(b.rcutij for b in bconf.funcspecs_blocks)
    cutoff = float(rcut) + 0.05
    seed = sum(map(ord, name))
    structs = make_structures(elements, cutoff, seed)
    batch = build_batch(structs, cutoff, seed)

    for vname, v in LOSS_VARIANTS.items():
        has_aux = v["aux"]
        ace = ACE(bconf, compute_orthogonality=has_aux, compute_smoothness=has_aux)
        rng = np.random.default_rng(seed + 7)
        coefs0 = ace.get_coefs().numpy().copy()
        coefs = coefs0 + 0.03 * rng.normal(size=coefs0.shape)
        ace.set_coefs(coefs)
        naux = (1 + 3) if has_aux else 0
        loss_specs = {LOSS_TYPE: v[LOSS_TYPE], LOSS_FORCE_FACTOR: v["kappa"],
                      LOSS_ENERGY_FACTOR: 1 - v["kappa"], L1_REG: v["L1"], L2_REG: v["L2"]}
        if has_aux:
            loss_specs[AUX_LOSS_FACTOR] = [0.3, 0.1, 0.2, 0.05][:naux]
        tp = TensorPotential(ace, loss_specs=loss_specs)

        loss, grad, e, f = tp.external_fit(coefs, batch, eager=False)
        e_ev, f_ev, s_ev = tp.evaluate(batch)
        reg = np.array(tp.reg_components.numpy())

        out = {f"in_{k}": np.asarray(vv) for k, vv in batch.items()}
        out.update(
            coefs0=coefs0, coefs=coefs, loss=loss.numpy(), grad=grad.numpy(), e=e.numpy(), f=f.numpy(),
            e_eval=e_ev.numpy(), f_eval=f_ev.numpy(), s_eval=s_ev.numpy(), reg_components=reg,
            loss_type=np.array(v[LOSS_TYPE]), kappa=np.array(v["kappa"]), L1=np.array(v["L1"]),
            L2=np.array(v["L2"]), aux_factors=np.array(loss_specs.get(AUX_LOSS_FACTOR, [0.0])),
            has_aux=np.array(has_aux), elements=np.array(elements), ncoef=np.array(ace.ncoef),
        )
        # updated config round trip: save yaml with perturbed coefs
        upd = ace.get_updated_config(updating_coefs=coefs)
        upd_yaml = os.path.join(YAML_OUT, f"{name}__{vname}__updated.yaml")
        upd.save(upd_yaml)
        out["updated_yaml"] = np.array(os.path.basename(upd_yaml))

        # hessian only for the tiny cases (single batch of small structures)
        if vname == "perstruct" and batch["positions"].shape[0] <= 30:
            small = build_batch(structs[2:4], cutoff, seed)
            try:
                eh, hess = tp.evaluate_hessian({**small})
                out["hess_in_keys"] = np.array(sorted(small.keys()))
                for k, vv in small.items():
                    out[f"hin_{k}"] = np.asarray(vv)
                out["hess_e"] = eh.numpy()
                out["hess"] = hess.numpy()
            except Exception as ex:  # noqa
                print("  hessian skipped:", ex)

        np.savez_compressed(os.path.join(OUT, f"{name}__{vname}.npz"), **out)
        print(f"{name}/{vname}: ncoef={ace.ncoef} nat={batch['positions'].shape[0]} "
              f"npairs={batch['ind_i'].shape[0]} loss={float(loss):.6f} |grad|={np.linalg.norm(grad.numpy()):.4e} "
              f"reg={reg.ravel()[:4]}")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(YAML_OUT, exist_ok=True)
    failed = []
    selected = sys.argv[1:]  # optionally regenerate only the named cases
    for name, yml in CASES:
        if selected and name not in selected:
            continue
        try:
            run_case(name, yml)
        except Exception:
            traceback.print_exc()
            failed.append(name)
    print("FAILED:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
