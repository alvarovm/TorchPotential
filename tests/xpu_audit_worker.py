"""Test-only entry point: audit real device tensors while running the production worker loop."""
import json
import os
from pathlib import Path
import sys

import torch

from torchpotential.torchpot import TorchPotential


def audited_external_fit(expected, report_path):
    expected = torch.device(expected)
    report_path = Path(report_path)
    original = TorchPotential.external_fit

    def checked(self, coefs, data, eager=False):
        assert self.device == expected
        assert self.potential.get_coefs().device == expected
        assert data.device == expected
        assert all(t.device == expected for t in vars(data).values() if isinstance(t, torch.Tensor))
        assert torch.xpu.current_device() == expected.index
        result = original(self, coefs, data, eager=eager)
        assert all(t.device == expected and t.dtype == torch.float64 for t in result)
        torch.xpu.synchronize(expected)
        allocated = torch.xpu.memory_allocated(expected)
        assert allocated > 0
        if not report_path.exists():
            report_path.write_text(json.dumps(dict(
                pid=os.getpid(), device=str(expected), model_device=str(self.potential.get_coefs().device),
                batch_device=str(data.device), result_devices=[str(t.device) for t in result],
                allocated_bytes=allocated,
            )))
        return result

    return checked


if __name__ == "__main__":
    from torchpotential.distributed import _main

    argv = sys.argv[1:]
    device = argv[argv.index("--device") + 1]
    rank = int(argv[argv.index("--rank") + 1])
    report = Path(os.environ["TORCHPOTENTIAL_XPU_AUDIT_DIR"]) / "rank_{}.json".format(rank)
    TorchPotential.external_fit = audited_external_fit(device, report)
    _main(argv)
