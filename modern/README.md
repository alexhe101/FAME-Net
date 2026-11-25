# FAME-Net Modern — paper-first reconstruction

The default model implements the written method in [He et al., AAAI 2024](https://arxiv.org/abs/2401.02151). This is not an exact claim about the original unpublished training implementation. The released code contains runtime errors and mismatches with the paper. The original files are retained as a historical record.

## Implementation decisions

| Component | Paper-first profile (default) | Legacy profile (`routing="legacy"`) |
|:--|:--|:--|
| Mask predictor (Eq. 2–5) | Conv–ReLU–Conv, hard Gumbel-Softmax at training | Original LeakyReLU 0.1, optional original sampled evaluation |
| Frequency split (Eq. 6–9) | HF channel 0, LF channel 1; complementary masks | Same |
| HF / LF experts | HIN blocks / residual 3×3 conv blocks | Same |
| Gate (Eq. 10–13) | GAP+GMP, FC + Softplus(FC), top-k, softmax of selected scores | Original LeakyReLU logits + normalized auxiliary term for selection; softmax of selected logits |
| Fusion (Fig. 2, Eq. 14) | Concatenation MS, PAN, LF, HF; ReLU fusion blocks; 1×1 output projection | Original MS, HF, PAN, LF; LeakyReLU fusion; attention refinement + bicubic residual |
| Coupling internals | HIN slope 0.2 | Original positional-argument slope 1.0 inside coupling blocks |
| Dispatch | Correct per-sample sparse expert selection; dense reference available | Same bug fix |
| Loss (Eq. 15–19) | L1 reconstruction + L1 mask + sum of three SCVs | Corrected loss, not original broken solver |

Both profiles retain the released LU pre-fusion blocks, channel widths, and residual encoders as implementation details not fully specified in the paper. They are not implied by the displayed equations alone. Paper profile deliberately uses the 1×1 reconstruction projection written in Eq. 14, instead of claiming that the larger released attention/residual head is identical. Use `head="legacy"` (or `--head legacy`) with paper routing to investigate that variant. Legacy compatibility requires matching the original profile and dimensions, and `sample_mask=True` to reproduce stochastic mask sampling with a matched RNG seed. Neither profile reproduces the broken unmodified dispatch.

### Routing and optimization

For each of the three expert banks, weights have shape [B,E]. The default N=4, k=2 selects two experts per image, not per pixel. The frequency mask itself is spatial. Sparse dispatch computes only the selected batch items, using index_select and index_add; dense dispatch computes all experts as a numerical reference. Dynamic nonzero indices can incur CUDA synchronization, so benchmark both rather than assuming sparse execution is faster. The normalized gate and LU matrix reconstruction run in FP32 during BF16 autocast (FP64 for double precision).

The paper's Softplus term is implemented literally as a deterministic learned score contribution, not an additional random draw. Both FC layers receive task gradients through selected scores in paper mode. Legacy fc0 is frozen because its scores only affect discrete selection.

The loss is L = Lrec + alpha * Lmask + beta * Lload, with Lrec and Lmask as L1. Lload uses SCV of expert importance summed over the batch for each of HF, LF and fusion. The paper does not specify batch reduction or variance convention; batch aggregation and population variance are explicit choices to balance utilization across images. Hard top-k is not a guarantee against expert collapse. Monitor routing statistics. Alpha starts at 0.001, linearly reaches zero at 70% of training, and beta=0.1; the old solver's incompatible coefficients are not reused.

### Install

```bash
python -m pip install -e './modern[dev]'
python -m pytest modern/tests -q -rs
```

Inference requires PyTorch and NumPy; the dev extra adds SciPy for DCT labels and legacy regression. Python 3.10+, PyTorch 2.4+. Keep model weights in FP32 for BF16 autocast.

## NPZ data contract

Each file is an explicitly paired scene; no pairing by directory listing, hidden normalization or uint8 conversion is performed. Use co-registered floating CHW arrays, identical sensor band order and radiometry:

| Key | Shape, 4 bands and 4× scale | Required |
|:--|:--|:--|
| ms | [4,h,w] | Always |
| pan | [1,4h,4w] | Always |
| target | [4,4h,4w] | Training |
| mask | [2,4h,4w] | Training while alpha > 0 |

Mask channel order is HF, LF; values must be in [0,1] and sum to one per pixel. Use same-sized patches within a batch, and a consistent set of keys in all samples. The trainer expects preprocessed paired samples rather than implementing all sensor-specific Wald/MTF degradation steps.

### DCT-derived supervision

The optional label tool performs a DCT, removes a low-frequency disk, inverse transforms, then thresholds the mean absolute detail across spectral bands:

```bash
python -m famenet.mask --input paired_sample.npz --output train_npz/sample.npz --cutoff 0.25 --threshold 0.02
```

These example thresholds are illustrative for normalized images, not recovered settings from the paper. The exact cutoff, aggregation, binarization and preprocessing used to make the private labels are not documented. Tune per sensor, or supply your own original mask labels. Use `--alpha 0` only when intentionally training without mask supervision.

## Training, inference and benchmarking

```bash
famenet train --data train_npz --output runs/fame --device cuda --amp
famenet infer --checkpoint runs/fame/latest.pth --input sample.npz --output result.npz
famenet infer --routing legacy --checkpoint original.pth --input sample.npz --output result.npz --sample-mask
famenet benchmark --device cuda --dispatch sparse --batch 4
famenet benchmark --device cuda --dispatch dense --batch 4
famenet benchmark --device cuda --dispatch dense --compile
```

The single-device trainer uses Adam (5e-4 by default), cosine scheduling, gradient clipping, and atomic latest checkpoint writes. It saves configuration and optimizer/scheduler states; `--checkpoint` initializes model weights only, not full training resume. For historical checkpoints, choose legacy mode. The loader uses weights_only=True and strict active keys and shapes, strips one module. prefix, discards only known inactive legacy HIN layers, and checks recorded configuration for modern checkpoints. Output NPZ retains all spectral bands, mask and the three complete weight vectors. No pretrained weights are provided. Benchmarks are synthetic synchronized model latency, not measured speedups against the paper.

## Validation scope

Tests cover model shapes, deterministic evaluation, the paper gate formula, sparse-vs-dense forward/gradient equivalence on a mixed-route batch, finite routing at zero variance, BF16 CPU autocast, DCT label generation, strict checkpoints and one-step synthetic training. A separate legacy test executes the original classes with only their invalid dispatch replaced by a dense reference; it does not treat the broken source as ground truth for paper mode. [GitHub Actions run 45544297751](https://github.com/alexhe101/FAME-Net/actions/runs/45544297751) passed all 18 tests in both Python 3.10 and 3.12. Real-data training, original checkpoint reproduction, full-reference/no-reference metrics and CUDA performance remain separate validation requirements.
