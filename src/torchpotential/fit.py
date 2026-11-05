import logging
import time

import numpy as np
import torch
from scipy.optimize import minimize, dual_annealing

from torchpotential.constants import *
from torchpotential.fitmetrics import FitMetrics
from torchpotential.indexing import Batch
from torchpotential.utils.utilities import batching_data

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

NUMPY_OPTIMIZERS = ["BFGS", "L-BFGS-B", "CG", "trf", "dogbox", "lm", "dual_annealing"]
TORCH_OPTIMIZERS = ["Adam", "SGD"]


class _KerasAdam:
    """Adam with the update rule of ``tf.keras`` (epsilon applied to ``sqrt(v)`` *before* bias correction),
    which differs slightly from ``torch.optim.Adam``. Keeps the optimizer behaviour of the TF implementation."""

    def __init__(self, params, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-7):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, beta_1, beta_2, eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        alpha = self.lr * (1 - self.b2**self.t) ** 0.5 / (1 - self.b1**self.t)
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m.add_((g - m) * (1 - self.b1))
            v.add_((g * g - v) * (1 - self.b2))
            p.sub_(alpha * m / (v.sqrt() + self.eps))


class FitTorchPotential:
    """Fits the coefficients of a :class:`TorchPotential` (API-compatible with ``FitTensorPotential``)."""

    def __init__(self, tensorpot, display_step=20, batch_size=10, eager=False):
        self.tensorpot = tensorpot
        self.ncoefs = self.tensorpot.potential.ncoef
        self.fit_coefs = None
        self.jacobian_factor = None
        self.loss_history = []
        self.iter_num = 0
        self.eval_count = 0
        self.last_loss = 0
        self.global_callback = None
        self.display_step = display_step
        self.data_df = None
        self.test_df = None
        self.res_opt = None
        self.batch_size = batch_size
        self.n_batches = None
        self.metrics = None
        self.test_metrics = None
        self.optmzr = None
        self.eager = eager  # kept for API compatibility: execution is always eager
        self.fit_metrics_data_dict = {}
        self.last_fit_metric_data = None
        self.last_test_metric_data = None
        self.fit_metric_callback = None
        self.test_metric_callback = None
        self.nfuncs = None
        self._tbatches = {}  # id(numpy batch dict) -> (numpy batch dict, device Batch)
        self._test_batches = None
        self._train_batches = None

    # ------------------------------------------------------------------------------------------------
    # device batches
    # ------------------------------------------------------------------------------------------------
    def _register(self, np_batches):
        """Move numpy batches to the device once; keeps a reference so that ``id`` stays valid."""
        for b in np_batches:
            self._tbatches[id(b)] = (b, Batch(b, device=self.tensorpot.device))
        return np_batches

    def _tbatch(self, b):
        entry = self._tbatches.get(id(b))
        if entry is None or entry[0] is not b:
            entry = (b, Batch(b, device=self.tensorpot.device))
            self._tbatches[id(b)] = entry
        return entry[1]

    def _forget(self, np_batches):
        for b in np_batches:
            self._tbatches.pop(id(b), None)

    def fit(
        self,
        df,
        test_df=None,
        batch_size=100,
        optimizer="BFGS",
        coefs=None,
        niter=100,
        jacobian_factor=None,
        callback=None,
        save_coefs=False,
        options=None,
        fit_metric_callback=None,
        test_metric_callback=None,
    ):
        if fit_metric_callback is not None:
            self.fit_metric_callback = fit_metric_callback
        if test_metric_callback is not None:
            self.test_metric_callback = test_metric_callback

        self._tbatches = {}
        self.data_df = df
        self.batch_size = batch_size
        batches = self._register(batching_data(self.data_df, batch_size=batch_size))
        self._train_batches = batches
        self.n_batches = len(batches)

        w_e = np.vstack([b["eweights"] for b in batches])
        w_f = np.vstack([b["fweights"] for b in batches])
        force_scale = float(self.tensorpot.loss_specs[LOSS_FORCE_FACTOR])
        energy_scale = float(self.tensorpot.loss_specs[LOSS_ENERGY_FACTOR])
        self.metrics = FitMetrics(w_e, w_f, energy_scale, force_scale, self.ncoefs)
        self.metrics.nfuncs = self.nfuncs

        self._test_batches = None
        if test_df is not None:
            self.test_df = test_df
            test_batches = self._register(batching_data(self.test_df, batch_size=batch_size))
            self._test_batches = test_batches
            test_w_e = np.vstack([b["eweights"] for b in test_batches])
            test_w_f = np.vstack([b["fweights"] for b in test_batches])
            self.test_metrics = FitMetrics(test_w_e, test_w_f, energy_scale, force_scale, self.ncoefs)
            self.test_metrics.nfuncs = self.nfuncs

        self.global_callback = callback
        if jacobian_factor is not None:
            self.jacobian_factor = np.array(jacobian_factor)

        if coefs is None:
            coefs = self.tensorpot.potential.get_coefs().detach().cpu().numpy().copy()
        else:
            try:
                assert len(coefs) == self.ncoefs
            except Exception:
                raise ValueError(
                    "Number of provided coefficients {0} is different from the numbers used in model {1}".format(
                        len(coefs), self.ncoefs
                    )
                )

        t0 = time.time()
        if optimizer in NUMPY_OPTIMIZERS:
            if optimizer in ["trf", "dogbox", "lm"]:
                from scipy.optimize import least_squares

                ls_opt_func = LMfunc(self, w_e)

                res_opt = least_squares(
                    ls_opt_func.func,
                    coefs,
                    jac=ls_opt_func.jac,
                    method=optimizer,
                    args=(batches,),
                    ftol=1e-8,
                    verbose=1,
                    max_nfev=niter,
                )
            elif optimizer == "dual_annealing":
                da_opt_func = DAfunc(self)

                bounds = [(-10000, 10000) for _ in coefs]
                options = {"disp": True, "maxiter": niter, "jac": False}
                minimizer_kwargs = {
                    "method": "BFGS",
                    "jac": da_opt_func.jac,
                    "options": {"gtol": 1e-8, "disp": True, "maxiter": niter},
                }
                res_opt = dual_annealing(
                    da_opt_func.func,
                    bounds,
                    args=(batches,),
                    maxiter=niter,
                    local_search_options=minimizer_kwargs,
                    callback=self.callback,
                    x0=coefs,
                )
            else:
                if options is None:
                    options = {"gtol": 1e-8, "disp": True, "maxiter": niter}
                options["maxiter"] = niter
                log.info("Minimizer options: {}".format(options))
                # reset fit_metrics_data_dict here and every time in self.callback. No need to store complete history
                self.fit_metrics_data_dict = {}
                res_opt = minimize(
                    self.fit_func,
                    coefs,
                    method=optimizer,
                    jac=True,
                    args=(batches,),
                    options=options,
                    callback=self.callback,
                )
                self.process_test_metric()

            self.res_opt = res_opt
            self.fit_coefs = res_opt.x
            self.tensorpot.potential.set_coefs(self.fit_coefs)

        elif optimizer in TORCH_OPTIMIZERS:
            self.opt = self.get_optimzer(optimizer)
            for epoch in range(niter):
                self.torch_fit_func(batches)

                self.fit_coefs = self.tensorpot.potential.get_coefs().detach().cpu().numpy().copy()
                self.callback(self.fit_coefs)
        else:
            raise ValueError(
                "Unknown optimizer `{}`. Should be one of {}".format(optimizer, NUMPY_OPTIMIZERS + TORCH_OPTIMIZERS)
            )

        if save_coefs:
            self.save_coefs()
        x = self.fit_coefs
        self.update_last_metrics(x)
        print("Fitting took {0:6.2f} seconds".format(time.time() - t0))

    def restart_fit(self, df, batch_size=100, optimizer="BFGS", niter=100, jacobian_factor=None):
        if self.eval_count == 0:
            raise AttributeError("Cannot restart, no fit has been done so far")
        else:
            self.fit(
                df,
                batch_size=batch_size,
                optimizer=optimizer,
                coefs=self.fit_coefs,
                niter=niter,
                jacobian_factor=jacobian_factor,
            )

    def get_optimzer(self, opt_name):
        params = [self.tensorpot.potential.fit_coefs]
        if opt_name == "Adam":
            return _KerasAdam(params, lr=1e-3)
        elif opt_name == "SGD":
            return torch.optim.SGD(params, lr=1e-3)
        raise ValueError("Unknown optimizer {}".format(opt_name))

    def _collect_metrics_snapshot(self, de, de_pa, df, total_na):
        self.metrics.regs = self.get_reg_components()
        self.metrics.reg_weights = self.get_reg_weights()
        self.metrics.compute_metrics(
            np.vstack(de), np.vstack(de_pa), np.vstack(df), np.vstack(total_na), dataframe=self.data_df
        )

    def _record_evaluation(self, x, tot_loss, errors, t0):
        """Common bookkeeping after one evaluation of the full training set at coefficients ``x``.

        ``errors`` is a list of the per-batch ``compute_batch_errors`` results.
        Returns the metrics snapshot (also stored under ``hash(x)``).
        """
        de, de_pa, df, total_na = ([e[k] for e in errors] for k in range(4))
        self.fit_coefs = x
        self.eval_count += 1
        self.eval_time = time.time() - t0
        self._collect_metrics_snapshot(de, de_pa, df, total_na)
        self.metrics.loss = tot_loss
        self.metrics.eval_time = self.eval_time
        # snapshot of FitMetrics: loss, e_loss, f_loss, reg_loss, RMSE, MAE, MAX_E (E,F), timing
        curr_fit_metrics_data = self.metrics.to_FitMetricsDict()
        curr_fit_metrics_data["eval_count"] = self.eval_count
        curr_fit_metrics_data["iter_num"] = self.iter_num
        self.fit_metrics_data_dict[hash(np.asarray(x).tobytes())] = curr_fit_metrics_data
        self.last_fit_metric_data = curr_fit_metrics_data

        if self.iter_num == 0:
            if self.fit_metric_callback is not None:
                self.fit_metric_callback(curr_fit_metrics_data)
            else:
                print_detailed_metrics(curr_fit_metrics_data, title="Initial state:")
                print_extended_metrics(curr_fit_metrics_data, title="INIT STATS")
            self.process_test_metric(title="TEST INIT STATS")
            self.iter_num += 1

        self.loss_history.append(tot_loss)
        self.last_loss = self.loss_history[-1]
        return curr_fit_metrics_data

    def torch_fit_func(self, batches):
        tot_loss = 0
        errors = []
        t0 = time.time()
        idx = np.arange(len(batches))
        np.random.shuffle(idx)
        potential = self.tensorpot.potential
        for ind in idx:
            b = batches[ind]
            loss, grad_loss, e, f = self.tensorpot.native_fit(self._tbatch(b))
            self.opt.zero_grad(set_to_none=True)
            potential.fit_coefs.grad = grad_loss
            self.opt.step()
            tot_loss += float(loss)
            errors.append(self.compute_batch_errors(b, e.cpu().numpy(), f.cpu().numpy()))

        x = potential.get_coefs().detach().cpu().numpy().copy()
        self._record_evaluation(x, tot_loss, errors, t0)
        return tot_loss

    def _apply_jacobian_factor(self, tot_jac):
        if self.jacobian_factor is not None:
            try:
                assert len(self.jacobian_factor) == tot_jac.shape[0]
                tot_jac = tot_jac * self.jacobian_factor
            except Exception:
                raise ValueError(
                    "Size of the provided Jacobian factor ({0}) is not compatible with Jacobian size ({1}).".format(
                        len(self.jacobian_factor), tot_jac.shape[0]
                    )
                )
        return tot_jac

    def fit_func(self, x, batches):
        tot_loss = 0
        tot_jac = 0
        errors = []
        t0 = time.time()
        for b in batches:
            loss, jac, e, f = self.tensorpot.external_fit(x, self._tbatch(b), eager=self.eager)
            tot_jac += jac.cpu().numpy()
            tot_loss += float(loss)
            errors.append(self.compute_batch_errors(b, e.cpu().numpy(), f.cpu().numpy()))

        self._record_evaluation(x, tot_loss, errors, t0)
        return tot_loss, self._apply_jacobian_factor(tot_jac)

    def compute_batch_errors(self, b, e, f):
        e_true = b["energy"]
        f_true = b["forces"]
        _, nat = np.unique(b["e_map"], return_counts=True)
        de = e_true - e
        de_pa = de / nat.reshape(-1, 1)
        df = f_true - f

        return de, de_pa, df, nat.reshape(-1, 1)

    def callback(self, x, *args, **kwargs):
        # call global callback
        self.metrics.record_time(self.eval_time)

        # get true metrics data that correspond to actual successfull x
        true_fit_metric_data = self.update_last_metrics(x)

        if self.fit_metric_callback is not None:
            # example of fit_metric_callback is:  print_extended_metrics
            self.fit_metric_callback(true_fit_metric_data)
        else:
            if self.iter_num % self.display_step == 0:
                print_extended_metrics(true_fit_metric_data)
            else:
                print_detailed_metrics(true_fit_metric_data)
        # test metric
        if self.iter_num % self.display_step == 0:
            self.process_test_metric(updating_coefs=x)
        self.iter_num += 1

        if self.iter_num % self.display_step == 0:
            self.tensorpot.potential.get_updated_config(prefix=self.iter_num)

        if self.global_callback is not None:
            self.global_callback(x, *args, **kwargs)

    def process_test_metric(self, title="TEST_STATS", updating_coefs=None):
        if self.test_df is not None:
            curr_test_metrics_data = self.compute_test_metric(updating_coefs=updating_coefs)
            curr_test_metrics_data["eval_count"] = self.eval_count
            curr_test_metrics_data["iter_num"] = self.iter_num
            self.last_test_metric_data = curr_test_metrics_data
            if self.test_metric_callback is not None:
                self.test_metric_callback(curr_test_metrics_data)
            else:
                print_extended_metrics(curr_test_metrics_data, title=title)

    def update_last_metrics(self, x):
        x_hash = hash(x.tobytes())
        true_fit_metric_data = self.fit_metrics_data_dict[x_hash]
        true_fit_metric_data["iter_num"] = self.iter_num
        true_fit_metric_data["eval_count"] = len(self.loss_history)
        # update self.metrics with this true metrics data
        self.metrics.from_FitMetricsDict(true_fit_metric_data)
        self.last_fit_metric_data = true_fit_metric_data
        return true_fit_metric_data

    def _batches_for(self, datadf):
        """(numpy batches, whether they are temporary) for ``datadf``; training/test batches are reused."""
        if datadf is None:
            datadf = self.data_df
        if datadf is self.test_df and self._test_batches is not None:
            return self._test_batches, False
        if datadf is self.data_df and self._train_batches is not None:
            return self._train_batches, False
        return batching_data(datadf, batch_size=self.batch_size), True

    def predict(self, datadf=None):
        batches, temporary = self._batches_for(datadf)

        pred_dict = {
            "energy_pred": [],
            "forces_pred": [],
            "stress_pred": [],
            "de": [],
            "de_pa": [],
            "df": [],
            "total_na": [],
        }
        for b in batches:
            _, partition = np.unique(b["e_map"], return_counts=True)
            for i in range(1, len(partition)):
                partition[i] += partition[i - 1]
            e, f, s = self.tensorpot.evaluate(self._tbatch(b))
            e, f, s = e.cpu().numpy(), f.cpu().numpy(), s.cpu().numpy()
            errors = self.compute_batch_errors(b, e, f)
            e_pred = e.reshape(-1)
            f_pred = np.split(f.reshape(-1, 3), partition)[:-1]
            s_pred = s.reshape(-1, 3, 3)
            pred_dict["energy_pred"] += list(e_pred)
            pred_dict["forces_pred"] += list(f_pred)
            pred_dict["stress_pred"] += list(s_pred)
            pred_dict["de"] += list(errors[0].reshape(-1))
            pred_dict["de_pa"] += list(errors[1].reshape(-1))
            pred_dict["df"] += list(np.split(errors[2].reshape(-1, 3), partition)[:-1])
            pred_dict["total_na"].append(errors[3])
        if temporary:
            self._forget(batches)

        return pred_dict

    def get_fitting_data(self):
        return self.data_df

    def get_reg_components(self):
        rc = self.tensorpot.reg_components
        if isinstance(rc, torch.Tensor):
            return rc.detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
        return np.zeros((0, 1))

    def get_reg_weights(self):
        loss_specs = self.tensorpot.loss_specs
        l1 = np.array(loss_specs[L1_REG]).reshape(-1, 1)
        l2 = np.array(loss_specs[L2_REG]).reshape(-1, 1)
        aux = np.array(loss_specs[AUX_LOSS_FACTOR]).reshape(-1, 1)

        return np.vstack([l1, l2, aux]) * self.n_batches

    def save_coefs(self):
        np.savetxt("Fit_coefficients_step_{0}".format(self.eval_count), self.fit_coefs)

    def get_fitted_coefficients(self):
        if self.eval_count > 0:
            return self.fit_coefs
        else:
            raise AttributeError("Coefficients have not been fitted yet")

    def print_detailed_metrics(self, title="Iteration:"):
        print_detailed_metrics(self.last_fit_metric_data, title=title)

    def print_extended_metrics(self, title="FIT_STATS"):
        print_extended_metrics(self.last_fit_metric_data, title=title)

    def compute_test_metric(self, updating_coefs=None):
        if self.test_df is not None:
            self.tensorpot.potential.get_updated_config(updating_coefs=updating_coefs, prefix=self.iter_num)
            # 'energy_pred', 'forces_pred', 'stress_pred', 'de', 'de_pa', 'df', 'total_na'
            test_pred = self.predict(datadf=self.test_df)
            self.test_metrics.regs = self.get_reg_components()
            self.test_metrics.reg_weights = self.get_reg_weights()
            self.test_metrics.compute_metrics(
                np.vstack(test_pred["de"]),
                np.vstack(test_pred["de_pa"]),
                np.vstack(test_pred["df"]),
                np.vstack(test_pred["total_na"]),
                dataframe=self.test_df,
            )
            curr_test_metrics_data = self.test_metrics.to_FitMetricsDict()
            curr_test_metrics_data["loss"] = (
                curr_test_metrics_data["e_loss_contrib"]
                + curr_test_metrics_data["f_loss_contrib"]
                + curr_test_metrics_data["l1_reg_contrib"]
                + curr_test_metrics_data["l2_reg_contrib"]
                + sum(curr_test_metrics_data["extra_regularization_contrib"])
            )
            curr_test_metrics_data["iter_num"] = self.iter_num
            return curr_test_metrics_data


class LMfunc:
    """Residual/Jacobian provider for ``scipy.optimize.least_squares`` (trf, dogbox, lm)."""

    def __init__(self, tensorpot_fit, w_e):
        self.tensorpot_fit = tensorpot_fit
        self.resid_value = None
        self.jacob_value = None
        self.x_current = None
        self.w_e = w_e

    def run_fit(self, x, batches):
        fit = self.tensorpot_fit
        tot_res = []
        tot_jac = []
        errors = []
        t0 = time.time()
        nregs = len(fit.get_reg_components())
        for index, b in enumerate(batches):
            loss, jac, e, f = fit.tensorpot.external_vector_fit(x, fit._tbatch(b))
            nregs = len(fit.get_reg_components())
            if (index + 1) < len(batches):
                tot_jac.append(jac.cpu().numpy().reshape(-1, len(x))[:-nregs])
                tot_res.append(loss.cpu().numpy().reshape(-1, 1)[:-nregs])
            else:
                tot_jac.append(jac.cpu().numpy().reshape(-1, len(x)))
                tot_res.append(loss.cpu().numpy().reshape(-1, 1))
            errors.append(fit.compute_batch_errors(b, e.cpu().numpy(), f.cpu().numpy()))
        tot_res = np.vstack(tot_res).reshape(-1)
        tot_jac = np.vstack(tot_jac)
        tot_res[-nregs:] *= len(batches)

        scalar_loss = np.sum(
            self.w_e.reshape(-1, 1) * (tot_res[:-nregs].reshape(-1, 1) / self.w_e.reshape(-1, 1)) ** 2
        ) + np.sum(tot_res[-nregs:])
        fit._record_evaluation(x.copy(), float(scalar_loss), errors, t0)

        return tot_res, tot_jac

    def update(self, x, batches):
        resid_value, jacob_value = self.run_fit(x, batches)
        self.resid_value = resid_value.copy()
        self.jacob_value = jacob_value.copy()
        self.x_current = x.copy()

    def func(self, x, batches):
        self.update(x, batches)
        self.tensorpot_fit.callback(x)
        return self.resid_value

    def jac(self, x, batches):
        if (x != self.x_current).any():
            self.update(x, batches)
        return self.jacob_value


class DAfunc:
    """Objective/gradient provider for ``scipy.optimize.dual_annealing``."""

    def __init__(self, tensorpot_fit):
        self.tensorpot_fit = tensorpot_fit
        self.loss_value = None
        self.jacob_value = None
        self.x_current = None
        self.batches = None

    def run_fit(self, x, batches):
        fit = self.tensorpot_fit
        tot_loss = 0
        tot_jac = 0
        errors = []
        t0 = time.time()
        for b in batches:
            loss, jac, e, f = fit.tensorpot.external_fit(x, fit._tbatch(b), eager=fit.eager)
            tot_jac += jac.cpu().numpy()
            tot_loss += float(loss)
            errors.append(fit.compute_batch_errors(b, e.cpu().numpy(), f.cpu().numpy()))

        fit._record_evaluation(x.copy(), tot_loss, errors, t0)
        self.last_loss = fit.loss_history[-1]
        return tot_loss, fit._apply_jacobian_factor(tot_jac)

    def update(self, x, batches):
        loss_value, jacob_value = self.run_fit(x, batches)
        self.loss_value = loss_value.copy()
        self.jacob_value = jacob_value.copy()
        self.x_current = x.copy()

    def func(self, x, batches):
        self.batches = batches
        self.update(x, batches)
        self.tensorpot_fit.callback(x)
        return self.loss_value

    def jac(self, x):
        if (x != self.x_current).any():
            self.update(x, self.batches)
        return self.jacob_value


def print_extended_metrics(fit_metrics_dict, title="FIT_STATS"):
    # (self, iter_num, total_loss, reg_comps, reg_weights, title='FIT STATS', nfuncs=None):

    iter_num = fit_metrics_dict["iter_num"]
    total_loss = fit_metrics_dict["loss"]

    str0 = "\n" + "-" * 44 + title + "-" * 44 + "\n"
    str1 = "{prefix:<11} #{iter_num:<4}".format(prefix="Iteration:", iter_num=iter_num)
    str1 += "{prefix:<8}".format(prefix="Loss:")
    str1 += "{prefix:>8} {tot_loss:>1.4e} ({fr:3.0f}%) ".format(
        prefix="Total: ", tot_loss=total_loss, fr=100
    )
    str1 += "\n"

    fr = fit_metrics_dict["e_loss_contrib"] / total_loss * 100 if total_loss > 0 else 0
    str2 = "{prefix:>33} {e_loss:>1.4e} ({fr:3.0f}%) ".format(
        prefix="Energy: ", e_loss=fit_metrics_dict["e_loss_contrib"], fr=fr
    )
    str2 += "\n"

    fr = fit_metrics_dict["f_loss_contrib"] / total_loss * 100 if total_loss > 0 else 0
    str3 = "{prefix:>33} {f_loss:>1.4e} ({fr:3.0f}%) ".format(
        prefix="Force: ", f_loss=fit_metrics_dict["f_loss_contrib"], fr=fr
    )
    str3 += "\n"

    l1 = fit_metrics_dict["l1_reg_contrib"]
    l2 = fit_metrics_dict["l2_reg_contrib"]

    fr = l1 / total_loss * 100 if total_loss != 0 else 0
    str4 = "{prefix:>33} {l1:>1.4e} ({fr:3.0f}%) ".format(prefix="L1: ", l1=l1, fr=fr)
    str4 += "\n"
    fr = l2 / total_loss * 100 if total_loss != 0 else 0
    str4 += "{prefix:>33} {l2:>1.4e} ({fr:3.0f}%) ".format(prefix="L2: ", l2=l2, fr=fr)
    str4 += "\n"

    reg_comps = fit_metrics_dict["extra_regularization_contrib"]
    str5 = ""
    for i, comp in enumerate(reg_comps):
        str5 += "{prefix:>33} ".format(prefix="Smooth_w{}: ".format(i + 1))
        str5 += "{s1:>1.4e} ".format(s1=comp)
        str5 += "({fr:3.0f}%) ".format(fr=comp / total_loss * 100)
        str5 += "\n"

    nfuncs = fit_metrics_dict.get("nfuncs")
    ncoefs = fit_metrics_dict.get("ncoefs")

    if nfuncs is None:
        line = "Number of params.: "
    else:
        line = "Number of params./funcs: "
    str6 = "{prefix:>20}".format(prefix=line) + "{ncoefs:>6d}".format(ncoefs=ncoefs)
    if nfuncs is not None:
        str6 += "/{nfuncs:<6d}".format(nfuncs=nfuncs)

    avg_t = fit_metrics_dict["eval_time"] / fit_metrics_dict["nat"]  # in sec/atom
    str6 += "{prefix:>42}".format(
        prefix="Avg. time: "
    ) + "{avg_t:>10.2f} {un:<6}".format(avg_t=avg_t * 1e6, un="mcs/at")

    str6 += "\n" + "-" * 97 + "\n"
    str_loss = str0 + str1 + str2 + str3 + str4 + str5 + str6
    ##############################
    er_str_h = (
        "{:>9}".format("")
        + "{:^22}".format("Energy/at, meV/at")
        + "{:^22}".format("Energy_low/at, meV/at")
        + "{:^22}".format("Force, meV/A")
        + "{:^22}\n".format("Force_low, meV/A")
    )

    er_rmse = "{prefix:>9} ".format(prefix="RMSE: ")
    er_rmse += (
        "{:>14.2f}".format(fit_metrics_dict["rmse_epa"] * 1e3)
        + "{:>21.2f}".format(fit_metrics_dict["low_rmse_epa"] * 1e3)
        + "{:>21.2f}".format(fit_metrics_dict["rmse_f"] * 1e3)
        + "{:>24.2f}\n".format(fit_metrics_dict["low_rmse_f"] * 1e3)
    )
    er_mae = "{prefix:>9} ".format(prefix="MAE: ")
    er_mae += (
        "{:>14.2f}".format(fit_metrics_dict["mae_epa"] * 1e3)
        + "{:>21.2f}".format(fit_metrics_dict["low_mae_epa"] * 1e3)
        + "{:>21.2f}".format(fit_metrics_dict["mae_f"] * 1e3)
        + "{:>24.2f}\n".format(fit_metrics_dict["low_mae_f"] * 1e3)
    )
    er_max = "{prefix:>9} ".format(prefix="MAX_AE: ")
    er_max += (
        "{:>14.2f}".format(fit_metrics_dict["max_abs_epa"] * 1e3)
        + "{:>21.2f}".format(fit_metrics_dict["low_max_abs_epa"] * 1e3)
        + "{:>21.2f}".format(fit_metrics_dict["max_abs_f"] * 1e3)
        + "{:>24.2f}\n".format(fit_metrics_dict["low_max_abs_f"] * 1e3)
    )
    er_str = er_str_h + er_rmse + er_mae + er_max + "-" * 97  # + '\n'
    log.info(str_loss + er_str)


def print_detailed_metrics(fit_metrics_dict, title="Iteration:"):
    # fit_metrics_dict
    iter_num = fit_metrics_dict["iter_num"]
    total_loss = fit_metrics_dict["loss"]
    avg_t = fit_metrics_dict["eval_time"] / fit_metrics_dict["nat"]  # in sec/atom
    log.info(
        "{:<12}".format(title)
        + "#{iter_num:<5}".format(iter_num=iter_num)
        + "{:<14}".format(
            "({numeval} evals):".format(numeval=fit_metrics_dict["eval_count"])
        )
        + "{:>10}".format("Loss: ")
        + "{loss: >3.6f}".format(loss=total_loss)
        + "{str1:>21}{rmse_epa:>.2f} ({low_rmse_e:>.2f}) meV/at".format(
            str1=" | RMSE Energy(low): ",
            rmse_epa=1e3 * fit_metrics_dict["rmse_epa"],
            low_rmse_e=1e3 * fit_metrics_dict["low_rmse_epa"],
        )
        + "{str3:>16}{rmse_f:>.2f} ({low_rmse_f:>.2f}) meV/A".format(
            str3=" | Forces(low): ",
            rmse_f=1e3 * fit_metrics_dict["rmse_f"],
            low_rmse_f=1e3 * fit_metrics_dict["low_rmse_f"],
        )
        + " | Time/eval: {:>6.2f} mcs/at".format(avg_t * 1e6)
    )
