import numpy as np
from famenet.mask import dct_mask

def test_constant_and_texture():
    x = np.ones((4,8,8), dtype=np.float32)
    a = dct_mask(x, cutoff=.25, threshold=.01)
    assert a.shape == (2,8,8)
    assert not a[0].any()
    y = (np.indices((8,8)).sum(axis=0) % 2).astype(np.float32)
    b = dct_mask(np.repeat(y[None],4,axis=0), cutoff=.25, threshold=.01)
    assert b[0].any()
    np.testing.assert_allclose(b.sum(axis=0), 1)
