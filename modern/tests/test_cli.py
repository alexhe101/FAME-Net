import numpy as np
import torch

from famenet import FAMENet
from famenet.cli import main


def test_infer_and_train(tmp_path):
    inp = tmp_path / "sample.npz"
    target = np.ones((4,8,8), dtype=np.float32) * .5
    np.savez(inp, ms=np.ones((4,2,2), dtype=np.float32)*.5,
             pan=np.ones((1,8,8), dtype=np.float32)*.5,
             target=target, mask=np.stack([np.zeros((8,8)),np.ones((8,8))]).astype(np.float32))
    ckpt = tmp_path / "weights.pth"
    torch.save(FAMENet(width=4).state_dict(), ckpt)
    output = tmp_path / "result.npz"
    main(["infer","--width","4","--checkpoint",str(ckpt),"--input",str(inp),"--output",str(output)])
    with np.load(output) as data:
        assert data["hrms"].shape == (4,8,8)
        assert np.isfinite(data["hrms"]).all()
    output.unlink()  # Keep only training samples in the input directory.
    main(["train","--width","4","--data",str(tmp_path),"--output",str(tmp_path/"run"),
          "--epochs","1","--batch","1"])
    assert (tmp_path/"run"/"latest.pth").exists()
