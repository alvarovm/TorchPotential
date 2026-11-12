"""Data-parallel evaluation of the fit objective on several devices.

The fit loss is a sum over batches, so the batches of a fit can be spread over several devices. The main process
(rank 0) keeps the optimizer, callbacks and metrics and evaluates its own share of the batches; every other rank is a
worker process (``python -m torchpotential.distributed``) that owns one device, a copy of the potential and its share
of the batches. For every evaluation of the objective rank 0 sends the coefficients to the workers, evaluates its own
shard, and then sums loss and gradient and merges the per-batch errors (in the original batch order).

Transport is a ``multiprocessing`` connection over a socket pair inherited by the worker (POSIX only), so no ports or
process groups are needed, and a dead worker is detected instead of hanging the fit.
"""
import atexit
import logging
import os
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Connection, Pipe

import numpy as np

from torchpotential.utils.utilities import batch_errors

log = logging.getLogger(__name__)

ENV_WORKER_TIMEOUT = "TORCHPOTENTIAL_WORKER_TIMEOUT"
_POLL_INTERVAL = 0.5


class WorkerError(RuntimeError):
    """A worker process failed or died. Out-of-memory failures keep the 'out of memory' text of the original error,
    so that ``utils.device.is_oom_error`` recognises them."""


def evaluate_batches(tensorpot, x, items):
    """Loss, gradient and per-batch errors of ``items`` at coefficients ``x``.

    ``items`` is a list of ``(global_index, numpy_batch, device_batch)``. Returns ``(loss, grad, errors)`` with
    ``errors`` a list of ``(global_index, (de, de_pa, df, nat))``.
    """
    tot_loss = 0.0
    tot_grad = 0
    errors = []
    for gi, nb, tb in items:
        loss, grad, e, f = tensorpot.external_fit(x, tb)
        tot_grad += grad.cpu().numpy()
        tot_loss += float(loss)
        errors.append((gi, batch_errors(nb, e.cpu().numpy(), f.cpu().numpy())))
    return tot_loss, tot_grad, errors


def merge_results(results):
    """Sum loss and gradient of the per-rank results and order the errors by global batch index."""
    tot_loss = 0.0
    tot_grad = 0
    errors = []
    for loss, grad, errs in results:
        tot_loss += loss
        tot_grad = tot_grad + grad
        errors += errs
    errors.sort(key=lambda t: t[0])
    return tot_loss, tot_grad, [e for _, e in errors]


def batch_cost(b):
    """Relative cost of evaluating a numpy batch (number of pairs plus atoms)."""
    return int(len(b["ind_i"]) + len(b["e_map"]))


def shard_batches(costs, n_ranks):
    """Assign batches to ranks (longest-processing-time first). Returns one ascending list of batch indices per rank.

    Rank 0 receives the most expensive batch; every rank gets at least one batch when there are at least
    ``n_ranks`` batches.
    """
    order = sorted(range(len(costs)), key=lambda i: (-costs[i], i))
    loads = [0] * n_ranks
    shards = [[] for _ in range(n_ranks)]
    for i in order:
        r = min(range(n_ranks), key=lambda k: (loads[k], k))
        shards[r].append(i)
        loads[r] += max(int(costs[i]), 1)
    return [sorted(s) for s in shards]


class DevicePool:
    """Worker processes for ranks ``1..len(devices)-1``; rank 0 (``devices[0]``) is the calling process."""

    def __init__(self, devices, num_threads=None, mem_limit=0, timeout=None):
        self.devices = [str(d) for d in devices]
        if len(self.devices) < 2:
            raise ValueError("A DevicePool needs at least two devices (rank 0 is the calling process)")
        self.num_threads = num_threads
        self.mem_limit = int(mem_limit or 0)
        env_timeout = os.environ.get(ENV_WORKER_TIMEOUT)
        self.timeout = float(env_timeout) if env_timeout else timeout
        self.world_size = len(self.devices)
        self.active = []  # ranks (>= 1) holding at least one batch for the current fit
        self._procs = []
        self._conns = []
        self._closed = False
        self._broken = False
        atexit.register(self.close)
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------------------------------------
    # life cycle
    # ------------------------------------------------------------------------------------------------
    def matches(self, devices, num_threads, mem_limit):
        return (
            not self._closed
            and not self._broken
            and self.devices == [str(d) for d in devices]
            and self.num_threads == num_threads
            and self.mem_limit == int(mem_limit or 0)
        )

    def _start(self):
        for rank in range(1, self.world_size):
            parent, child = Pipe(duplex=True)
            env = dict(os.environ)
            if self.num_threads:
                for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCHPOTENTIAL_NUM_THREADS"):
                    env[var] = str(int(self.num_threads))
            cmd = [
                sys.executable,
                "-m",
                "torchpotential.distributed",
                "--fd",
                str(child.fileno()),
                "--rank",
                str(rank),
                "--device",
                self.devices[rank],
                "--mem-limit",
                str(self.mem_limit),
            ]
            proc = subprocess.Popen(cmd, env=env, pass_fds=[child.fileno()])
            child.close()
            self._procs.append(proc)
            self._conns.append(parent)
        for rank in range(1, self.world_size):
            self._expect_ok(rank, "start")
        log.info("Started %d worker process(es) on devices %s", self.world_size - 1, self.devices[1:])

    def close(self):
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.send(("stop",))
            except Exception:
                pass
        deadline = time.time() + 10.0
        for proc in self._procs:
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except Exception:
                proc.kill()
                proc.wait()
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass
        self._procs, self._conns = [], []
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------
    # messaging
    # ------------------------------------------------------------------------------------------------
    def _send(self, rank, msg):
        try:
            self._conns[rank - 1].send(msg)
        except (OSError, EOFError) as exc:
            self._broken = True
            raise WorkerError("worker rank {} is not reachable: {}".format(rank, exc)) from exc

    def _recv(self, rank):
        conn, proc = self._conns[rank - 1], self._procs[rank - 1]
        start = time.time()
        while not conn.poll(_POLL_INTERVAL):
            if proc.poll() is not None and not conn.poll(0):
                self._broken = True
                raise WorkerError("worker rank {} died (exit code {})".format(rank, proc.returncode))
            if self.timeout is not None and time.time() - start > self.timeout:
                self._broken = True
                raise WorkerError("worker rank {} did not answer within {} s".format(rank, self.timeout))
        try:
            return conn.recv()
        except (EOFError, OSError) as exc:
            self._broken = True
            raise WorkerError("worker rank {} died while answering: {}".format(rank, exc)) from exc

    def _expect_ok(self, rank, what):
        reply = self._recv(rank)
        if reply[0] != "ok":
            raise WorkerError("worker rank {} failed during {}: {}".format(rank, what, reply[1]))
        return reply

    def _gather(self, ranks, what):
        """Receive one reply from each rank (always all of them, to keep the protocol in step); return the replies
        and the first error, if any."""
        replies, error = {}, None
        for rank in ranks:
            reply = self._recv(rank)
            if reply[0] == "ok":
                replies[rank] = reply
            elif error is None:
                error = WorkerError("worker rank {} failed during {}: {}".format(rank, what, reply[1]))
        return replies, error

    # ------------------------------------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------------------------------------
    def setup(self, potential, loss_specs, worker_items):
        """Give the workers the potential and their batches.

        ``worker_items[k]`` is the list of ``(global_index, numpy_batch)`` of rank ``k + 1``; ranks with an empty list
        stay idle. Any previous fit state of the workers is released first.
        """
        if self._broken or self._closed:
            raise WorkerError("the worker pool is not usable (a worker failed earlier)")
        assert len(worker_items) == self.world_size - 1
        config = potential.get_updated_config()
        self.active = [r + 1 for r, items in enumerate(worker_items) if items]
        for rank in range(1, self.world_size):
            payload = dict(
                config=config,
                loss_specs=loss_specs,
                compute_smoothness=bool(potential.compute_smoothness),
                compute_orthogonality=bool(potential.compute_orthogonality),
                items=worker_items[rank - 1],
            )
            self._send(rank, ("setup", payload))
        _, error = self._gather(range(1, self.world_size), "setup")
        if error is not None:
            self.active = []
            raise error

    def evaluate(self, x, local_fn):
        """Evaluate all shards at ``x``. ``local_fn()`` evaluates the shard of rank 0 and returns
        ``(loss, grad, errors)``. Returns the list of the results of all active ranks (rank 0 first)."""
        if self._broken or self._closed:
            raise WorkerError("the worker pool is not usable (a worker failed earlier)")
        x = np.ascontiguousarray(x, dtype=np.float64)
        for rank in self.active:
            self._send(rank, ("eval", x))
        local, local_exc = None, None
        try:
            local = local_fn()
        except Exception as exc:  # keep the protocol in step: collect the workers' answers first
            local_exc = exc
        except BaseException:
            self.close()
            raise
        replies, error = self._gather(self.active, "evaluation")
        if local_exc is not None:
            raise local_exc
        if error is not None:
            raise error
        return [local] + [replies[r][1] for r in self.active]

    def release(self):
        """Free the batches and the cached memory of the workers."""
        if self._broken or self._closed:
            return
        for rank in range(1, self.world_size):
            self._send(rank, ("release",))
        _, error = self._gather(range(1, self.world_size), "release")
        self.active = []
        if error is not None:
            raise error


# ----------------------------------------------------------------------------------------------------
# worker process
# ----------------------------------------------------------------------------------------------------
def _worker_main(fd, rank, device, mem_limit):
    import gc

    import torch

    from torchpotential.indexing import Batch
    from torchpotential.potentials.ace import ACE
    from torchpotential.torchpot import TorchPotential
    from torchpotential.utils import device as device_utils
    from torchpotential.utils.utilities import apply_memory_limit

    conn = Connection(fd)
    dev = torch.device(device)
    device_utils.set_device(dev)
    device_utils.set_num_threads(None)
    if dev.type in ("cuda", "xpu"):
        getattr(torch, dev.type).set_device(dev)
        apply_memory_limit(dev, mem_limit)

    state = {}

    def free():
        state.clear()
        gc.collect()
        if dev.type in ("cuda", "xpu"):
            getattr(torch, dev.type).empty_cache()

    conn.send(("ok", rank))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break  # the main process is gone
        cmd = msg[0]
        if cmd == "stop":
            break
        reply, failure = None, None
        try:
            if cmd == "setup":
                free()
                p = msg[1]
                ace = ACE(
                    p["config"],
                    compute_smoothness=p["compute_smoothness"],
                    compute_orthogonality=p["compute_orthogonality"],
                    device=dev,
                )
                state["tp"] = TorchPotential(ace, loss_specs=p["loss_specs"])
                state["items"] = [(gi, nb, Batch(nb, device=dev)) for gi, nb in p["items"]]
                reply = ("ok",)
            elif cmd == "eval":
                reply = ("ok", evaluate_batches(state["tp"], msg[1], state["items"]))
            elif cmd == "release":
                free()
                reply = ("ok",)
            else:
                failure = "unknown command {!r}".format(cmd)
        except Exception as exc:
            from torchpotential.utils.device import is_oom_error

            failure = "{}: {}".format(type(exc).__name__, exc)
            if not is_oom_error(exc):
                failure += "\n" + traceback.format_exc()
        # outside of the except block, so that the traceback no longer keeps the failed tensors alive
        if failure is not None:
            if cmd in ("setup", "eval"):
                free()
            reply = ("error", failure)
        conn.send(reply)
    conn.close()


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="torchpotential worker process (started by DevicePool)")
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--mem-limit", type=int, default=0)
    args = parser.parse_args(argv)
    _worker_main(args.fd, args.rank, args.device, args.mem_limit)


if __name__ == "__main__":
    _main()
