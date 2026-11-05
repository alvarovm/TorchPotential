"""Definitions of the end-to-end fit cases shared by the TF reference generator and the torch tests."""
CASES = [
    dict(name="bfgs_unary_aux", yaml="Al-r1234l12_crad_dif_2.yaml", seed=21, kappa=0.4, L1=1e-3, L2=2e-3,
         aux=True, aux_w=[0.3, 0.1, 0.2, 0.05], optimizer="BFGS", niter=12, batch_size=2, mask_radial=False,
         perturb=0.02),
    dict(name="lbfgsb_binary_mask", yaml="Al-Ni_opt_all.yaml", seed=22, kappa=0.7, L1=0.0, L2=1e-3, aux=False,
         aux_w=[], optimizer="L-BFGS-B", niter=12, batch_size=3, mask_radial=True, perturb=0.02),
    dict(name="adam_binary", yaml="Al-Ni_opt_all.yaml", seed=24, kappa=0.5, L1=0.0, L2=0.0, aux=False, aux_w=[],
         optimizer="Adam", niter=5, batch_size=2, mask_radial=False, perturb=0.02),
]

# the legacy TF least-squares path (trf/dogbox/lm) is broken, so there is no TF reference for it
TORCH_ONLY_CASES = [
    dict(name="trf_unary_2dens", yaml="Ni_2dens_bug_case.yaml", seed=23, kappa=0.5, L1=1e-3, L2=1e-3, aux=False,
         aux_w=[], optimizer="trf", niter=8, batch_size=2, mask_radial=False, perturb=0.02),
]
