"""Parity with the released model after only the missing/broken dispatch is repaired."""
from pathlib import Path
import types

import pytest
import torch

from famenet import FAMENet
from famenet.checkpoint import migrate_weights


def load_original():
    root = Path(__file__).resolve().parents[2]
    if not (root / "model" / "moe.py").exists():
        pytest.skip("Original source unavailable")
    # Execute exact released classes, removing only unused plotting/vision imports.
    # This keeps the actual upstream convolutions, gate and LU math independent.
    import sys
    package = types.ModuleType("_fame_legacy")
    package.__path__ = [str(root / "model")]
    sys.modules[package.__name__] = package
    for name in ("thops", "modules", "refine", "moe"):
        module = types.ModuleType(package.__name__ + "." + name)
        module.__package__ = package.__name__
        sys.modules[module.__name__] = module
        source = (root / "model" / (name + ".py")).read_text(encoding="utf-8")
        source = "\n".join(line for line in source.splitlines()
                           if not line.startswith(("import matplotlib", "import torchvision")))
        exec(compile(source, str(root / "model" / (name + ".py")), "exec"), module.__dict__)
    return sys.modules[package.__name__ + ".moe"]


@pytest.mark.parametrize("width", [4,32])
def test_upstream_repaired_dispatch_parity(width):
    legacy = load_original()
    def corrected(self, x):
        x = self.pre_fuse(x)
        g = self.gate(x)
        # Dense reference avoids the release's missing num_experts, incorrect
        # skip condition and last-expert-only return value.
        result = sum(expert(x)*g[:,i,None,None,None] for i,expert in enumerate(self.expert_networks_d))
        return result, g
    for name in ("LfExpert","HfExpert","Decoder"):
        getattr(legacy, name).forward = corrected
    torch.manual_seed(8)
    old = legacy.Net(4,width,{}).eval()
    new = FAMENet(width=width, routing="legacy", dispatch="sparse").eval()
    new.load_state_dict(migrate_weights(new,old.state_dict()))
    ms = torch.rand(2,4,3,4)
    pan = torch.rand(2,1,12,16)
    torch.manual_seed(99)
    a, mask, gates = old(ms,None,pan)
    torch.manual_seed(99)
    b = new(ms,pan,sample_mask=True)
    torch.testing.assert_close(a,b.hrms,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(mask,b.mask)
    for g,h in zip(gates,b.gates):
        torch.testing.assert_close(g,h,atol=2e-5,rtol=2e-5)
