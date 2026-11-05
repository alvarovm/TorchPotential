import torch

from torchpotential.constants import *
from torchpotential.indexing import Batch

_LOSS_TYPES = ["per-atom", "per-structure"]


class TorchPotential:
    """Energy/force evaluation and loss/gradient computation for a potential (currently :class:`ACE`).

    API-compatible with ``tensorpotential.compat.pace.tensorpot.TensorPotential``. ``data`` arguments are either
    the numpy dicts produced by ``batching_data`` or :class:`torchpotential.indexing.Batch` objects (preferred in
    loops: a ``Batch`` lives on the device and caches its index structures). All returned tensors are detached.
    ``compile_vector_loss`` / ``jit_op_compile`` are accepted for API compatibility and ignored (eager execution).
    """

    def __init__(self, potential, loss_specs=None, compile_vector_loss=False, jit_op_compile=False):
        self.potential = potential
        self.device = potential.device
        self.loss_specs = self._set_default_loss_specs() if loss_specs is None else self._set_loss_specs(loss_specs)
        self.compile_vector_loss = compile_vector_loss
        self.reg_components = []

    # ------------------------------------------------------------------------------------------------
    # loss specification
    # ------------------------------------------------------------------------------------------------
    def _set_default_loss_specs(self):
        return {
            LOSS_TYPE: "per-atom",
            LOSS_FORCE_FACTOR: 0.0,
            LOSS_ENERGY_FACTOR: 1.0,
            L1_REG: 0.0,
            L2_REG: 0.0,
            AUX_LOSS_FACTOR: [0.0],
        }

    def _set_loss_specs(self, specs):
        valid_loss_types = ["per-atom", "per-structure", "sqrt"]
        assert specs[LOSS_TYPE] in valid_loss_types, "{0} is not a supported loss type".format(specs[LOSS_TYPE])
        return {
            LOSS_TYPE: specs[LOSS_TYPE],
            LOSS_FORCE_FACTOR: float(specs.get(LOSS_FORCE_FACTOR, 0.0)),
            LOSS_ENERGY_FACTOR: float(specs.get(LOSS_ENERGY_FACTOR, 1.0)),
            L1_REG: float(specs.get(L1_REG, 0.0)),
            L2_REG: float(specs.get(L2_REG, 0.0)),
            AUX_LOSS_FACTOR: [float(v) for v in specs.get(AUX_LOSS_FACTOR, [0.0])],
        }

    # ------------------------------------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------------------------------------
    def _batch(self, data):
        return Batch.ensure(data, device=self.device)

    @staticmethod
    def compute_positions(batch):
        """Cartesian positions from scaled positions and the per-atom cell."""
        return torch.einsum("ni,nij->nj", batch.positions, batch.cell[batch.cell_map])

    @staticmethod
    def compute_pair_vectors(batch, pos, cell):
        j_ofst = torch.einsum("pi,pij->pj", batch.offsets, cell[batch.pair_map])
        r_i = pos[batch.ind_i]
        r_j = pos[batch.ind_j] + j_ofst
        return r_j - r_i

    def forward_pass(self, batch, pos, cell=None):
        """Total energy per structure, shape [n_struct, 1]."""
        if cell is None:
            cell = batch.cell
        r_ij = self.compute_pair_vectors(batch, pos, cell)
        index = batch.get_index(self.potential.index_layout)
        e_atomic = self.potential.compute_atomic_energy(r_ij, index)
        return e_atomic.new_zeros(batch.n_struct, 1).index_add(0, batch.e_map, e_atomic)

    # ------------------------------------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------------------------------------
    def _regularization(self):
        """Adds the L1/L2 terms and auxiliary losses; returns (list of weighted terms, reg_components)."""
        ls = self.loss_specs
        self.potential.compute_regularization()
        terms = [self.potential.reg_l1 * ls[L1_REG], self.potential.reg_l2 * ls[L2_REG]]
        reg_components = [self.potential.reg_l1.reshape(1, 1), self.potential.reg_l2.reshape(1, 1)]
        aux = self.potential.aux
        if aux is not None:
            factors = ls[AUX_LOSS_FACTOR]
            if len(factors) > len(aux):
                raise ValueError(
                    "{} auxiliary loss factors given, but the potential computes only {} auxiliary terms".format(
                        len(factors), len(aux)
                    )
                )
            for i, fac in enumerate(factors):
                terms.append(torch.squeeze(aux[i] * fac))
                reg_components.append(aux[i])
        return terms, reg_components

    def compute_loss(self, batch, e, f):
        ls = self.loss_specs
        loss_type = ls[LOSS_TYPE]
        if loss_type == "per-structure":
            total_loss = ls[LOSS_ENERGY_FACTOR] * torch.sum(batch.eweights * (e - batch.energy) ** 2)
        elif loss_type == "per-atom":
            total_loss = ls[LOSS_ENERGY_FACTOR] * torch.sum(batch.eweights * ((e - batch.energy) / batch.natoms) ** 2)
        else:
            raise NotImplementedError("Loss type '{}' is not implemented".format(loss_type))

        total_loss = total_loss + ls[LOSS_FORCE_FACTOR] * torch.sum(batch.fweights * (f - batch.forces) ** 2)

        terms, reg_components = self._regularization()
        for t in terms:
            total_loss = total_loss + t
        self.reg_components = torch.stack([r.detach().reshape(1, 1) for r in reg_components])
        return total_loss

    def compute_vector_loss(self, batch, e):
        """Residual vector (energy residuals followed by regularization terms) for least-squares solvers."""
        ls = self.loss_specs
        assert ls[LOSS_TYPE] == "per-atom", "Only per-atom loss type is compatible with vector loss"

        e_res = ls[LOSS_ENERGY_FACTOR] * (batch.eweights * ((e - batch.energy) / batch.natoms))
        residuals = [e_res.reshape(-1)]
        terms, reg_components = self._regularization()
        residuals += [t.reshape(-1) for t in terms]
        self.reg_components = torch.stack([r.detach().reshape(1, 1) for r in reg_components])
        return torch.cat(residuals, dim=0)

    # ------------------------------------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------------------------------------
    def compute_grad(self, batch):
        pos = self.compute_positions(batch).detach().requires_grad_(True)
        coefs = self.potential.fit_coefs
        with torch.enable_grad():
            e = self.forward_pass(batch, pos)
            need_f_graph = self.loss_specs[LOSS_FORCE_FACTOR] != 0.0
            (dedr,) = torch.autograd.grad(e.sum(), pos, create_graph=need_f_graph)
            f = -dedr
            loss = self.compute_loss(batch, e, f)
            (grad_loss,) = torch.autograd.grad(loss, coefs, allow_unused=True)
        if grad_loss is None:
            grad_loss = torch.zeros_like(coefs)
        return loss.detach(), grad_loss.detach(), e.detach(), f.detach()

    def compute_vector_grad(self, batch):
        pos = self.compute_positions(batch).detach().requires_grad_(True)
        coefs = self.potential.fit_coefs
        with torch.enable_grad():
            e = self.forward_pass(batch, pos)
            loss = self.compute_vector_loss(batch, e)
            (dedr,) = torch.autograd.grad(e.sum(), pos, retain_graph=True)
            f = -dedr
            rows = []
            for i in range(loss.shape[0]):
                (g,) = torch.autograd.grad(loss[i], coefs, retain_graph=i + 1 < loss.shape[0], allow_unused=True)
                rows.append(torch.zeros_like(coefs) if g is None else g)
            jac = torch.stack(rows)
        return loss.detach(), jac.detach(), e.detach(), f.detach()

    def compute_energy_forces_stress(self, batch):
        pos = self.compute_positions(batch).detach().requires_grad_(True)
        cell = batch.cell.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            e = self.forward_pass(batch, pos, cell)
            dpos, dcell = torch.autograd.grad(e.sum(), [pos, cell], allow_unused=True)
        if dcell is None:
            dcell = torch.zeros_like(cell)
        return e.detach(), -dpos.detach(), dcell.detach()

    def compute_hessian(self, batch):
        pos = self.compute_positions(batch).detach().requires_grad_(True)
        with torch.enable_grad():
            e = self.forward_pass(batch, pos)
            (f,) = torch.autograd.grad(e.sum(), pos, create_graph=True)
            f_flat = f.reshape(-1)
            rows = []
            for i in range(f_flat.shape[0]):
                (g,) = torch.autograd.grad(f_flat[i], pos, retain_graph=True, allow_unused=True)
                rows.append(torch.zeros_like(pos).reshape(-1) if g is None else g.reshape(-1))
            hess = torch.stack(rows)
        return e.detach(), hess.detach()

    # ------------------------------------------------------------------------------------------------
    # public API (mirrors TensorPotential)
    # ------------------------------------------------------------------------------------------------
    def external_fit(self, coefs, data, eager=False):
        self.potential.set_coefs(coefs)
        return self.native_fit(data, eager=eager)

    def native_fit(self, data, eager=False):
        return self.compute_grad(self._batch(data))

    def external_vector_fit(self, coefs, data):
        self.potential.set_coefs(coefs)
        return self.compute_vector_grad(self._batch(data))

    def evaluate(self, data):
        return self.compute_energy_forces_stress(self._batch(data))

    eager_evaluate = evaluate

    def evaluate_hessian(self, data):
        return self.compute_hessian(self._batch(data))

    def evaluate_loss(self, data):
        return self.native_fit(data)

    eager_evaluate_loss = evaluate_loss
