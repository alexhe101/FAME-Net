import pytest
import torch

from famenet import FAMENet, load_weights
from famenet.checkpoint import migrate_weights


def test_strict_migration(tmp_path):
    model = FAMENet(width=4)
    state = {"module." + k: v.clone() for k,v in model.state_dict().items()}
    state["module.msencoder.conv1.identity.weight"] = torch.randn(4,4,1,1)
    state["module.msencoder.conv1.conv_3.bias"] = torch.randn(4)
    path = tmp_path / "legacy.pth"
    torch.save({"net": state}, path)
    clone = FAMENet(width=4)
    load_weights(clone, path)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, clone.state_dict()[k])
    state["module.typo"] = torch.zeros(1)
    with pytest.raises(ValueError, match="Unexpected"):
        migrate_weights(clone, state)


def test_missing_tensor():
    model = FAMENet(width=4)
    state = dict(model.state_dict())
    state.pop("msconv.weight")
    with pytest.raises(ValueError, match="Missing"):
        migrate_weights(model, state)
