"""Configurable reconstruction of DCT-derived mask targets (paper Fig. 1).
The private sensor-specific cutoff and binarization rules were not released.
"""
import numpy as np

def dct_mask(hrms: np.ndarray, *, cutoff: float, threshold: float) -> np.ndarray:
    from scipy.fft import dctn, idctn
    if hrms.ndim != 3 or not np.isfinite(hrms).all():
        raise ValueError("Expected finite CHW HRMS")
    if not 0 < cutoff < 1 or threshold < 0:
        raise ValueError("cutoff must be in (0,1); threshold >= 0")
    _, h, w = hrms.shape
    radius = np.hypot(np.arange(h)[:, None] / h, np.arange(w)[None, :] / w)
    spec = dctn(hrms.astype(np.float32), axes=(-2,-1), norm="ortho")
    spec *= radius >= cutoff
    detail = np.abs(idctn(spec, axes=(-2,-1), norm="ortho")).mean(axis=0)
    high = (detail > threshold).astype(np.float32)
    return np.stack((high, 1-high))

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    a = parser.parse_args()
    with np.load(a.input, allow_pickle=False) as src:
        values = {k: src[k] for k in src.files if k != "mask"}
    values["mask"] = dct_mask(values["target"], cutoff=a.cutoff, threshold=a.threshold)
    out = Path(a.output)
    if out.suffix != ".npz" or out.resolve() == Path(a.input).resolve():
        raise ValueError("Choose a different .npz output")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **values)

if __name__ == "__main__":
    main()
