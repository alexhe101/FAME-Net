import pytest
import torch

from famenet import FAMENet, fame_loss
from famenet.routing import GateNetwork, ExpertMixture
from famenet.losses import load_loss


def sample(batch=3, bands=4):
    return torch.rand(batch, bands, 4, 5), torch.rand(batch, 1, 16, 20)


@pytest.mark.parametrize("routing", ["legacy", "paper"])
def test_model(routing):
    torch.manual_seed(4)
    model = FAMENet(width=4, routing=routing).eval()
    ms, pan = sample()
    a, b = model(ms, pan), model(ms, pan)
    assert a.hrms.shape == (3, 4, 16, 20)
    torch.testing.assert_close(a.hrms, b.hrms)
    torch.testing.assert_close(a.mask.sum(dim=1), torch.ones_like(a.mask[:, 0]))
    assert all(g.shape == (3, 4) for g in a.gates)
    for g in a.gates:
        assert (g.count_nonzero(dim=1) == 2).all()
        torch.testing.assert_close(g.sum(dim=1), torch.ones(3))


@pytest.mark.parametrize("kind", ["low", "high", "fusion"])
def test_mixed_batch_sparse_matches_dense(kind):
    torch.manual_seed(6)
    sparse = ExpertMixture(4, 4, 2, kind, "paper", "sparse")
    dense = ExpertMixture(4, 4, 2, kind, "paper", "dense")
    dense.load_state_dict(sparse.state_dict())
    # Force different routes in the same batch; reproduces the legacy dispatch bug.
    def route(x):
        return x.new_tensor([[.3,.7,0,0],[0,.4,.6,0],[.5,0,0,.5]])
    sparse.gate.forward = route
    dense.gate.forward = route
    a = torch.rand(3, 16 if kind == "fusion" else 8, 6, 7, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    y, g = sparse(a)
    z, h = dense(b)
    torch.testing.assert_close(y, z, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(g, h)
    y.sum().backward()
    z.sum().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("experts,k", [(1,1),(4,2),(4,4)])
def test_zero_variance_router(experts,k):
    gate = GateNetwork(4, experts, k, "legacy")
    with torch.no_grad():
        gate.fc0.weight.zero_()
        gate.fc0.bias.zero_()
    out = gate(torch.zeros(2,4,4,4))
    assert torch.isfinite(out).all()
    assert (out.count_nonzero(1) == k).all()


def test_paper_router_aux_gradient():
    router = GateNetwork(4, 4, 2, "paper")
    weights = router(torch.randn(3, 4, 5, 5))
    (weights * torch.arange(4)).sum().backward()
    assert router.fc0.weight.grad is not None
    assert router.fc0.weight.grad.abs().sum() > 0


def test_training_and_amp():
    model = FAMENet(width=4, routing="paper")
    ms, pan = sample(2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(ms, pan)
        loss = fame_loss(output, torch.rand_like(output.hrms), output.mask.detach())
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert model.maskp.conv.weight.grad is not None
    assert model.msconv.weight.grad is not None
    assert torch.isfinite(model.msconv.weight.grad).all()


def test_shape_and_load_loss():
    net = FAMENet(width=4, bands=8, scale=2)
    ms = torch.rand(1,8,4,5)
    pan = torch.rand(1,1,8,10)
    assert net(ms, pan).hrms.shape == (1,8,8,10)
    with pytest.raises(ValueError):
        net(ms, pan[..., :7, :])
    torch.testing.assert_close(load_loss(torch.tensor([[.5,.5,0,0],[0,0,.5,.5]])), torch.tensor(0.))

def test_default_paper_profile_and_equations():
    import torch.nn.functional as F
    model = FAMENet(width=4)
    assert model.config["routing"] == "paper"
    assert model.refine.kernel_size == (1, 1)
    gate = model.decoder.gate
    x = torch.randn(2,4,8,8)
    pooled = x.mean(dim=(-1,-2)) + x.amax(dim=(-1,-2))
    v = gate.fc1(pooled) + F.softplus(gate.fc0(pooled))
    vals, ids = v.topk(2, dim=1)
    expected = torch.zeros_like(v).scatter(1, ids, vals.softmax(1))
    torch.testing.assert_close(gate(x), expected)
    assert model.maskp.relu.negative_slope == 0
    assert FAMENet(width=4,routing="legacy").decoder.pre_fuse[0].F.relu_1.negative_slope == 1
