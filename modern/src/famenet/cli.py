"""Single-device inference, profiling and supervised training."""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_weights
from .data import NPZDataset, read_sample
from .losses import fame_loss, mask_weight
from .model import FAMENet


def make_model(a):
    return FAMENet(bands=a.bands, width=a.width, experts=a.experts, top_k=a.top_k,
                   scale=a.scale, routing=a.routing, dispatch=a.dispatch, head=a.head).to(a.device)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def amp(a):
    return torch.autocast(device_type=torch.device(a.device).type, dtype=torch.bfloat16, enabled=a.amp)


def infer(a):
    model = make_model(a).eval()
    load_weights(model, a.checkpoint)
    sample = read_sample(a.input)
    with torch.inference_mode(), amp(a):
        result = model(sample["ms"][None].to(a.device), sample["pan"][None].to(a.device),
                       sample_mask=a.sample_mask)
    output = Path(a.output)
    if output.suffix != ".npz":
        raise ValueError("Use an .npz output to retain all multispectral bands")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, hrms=result.hrms[0].float().cpu().numpy(),
                        mask=result.mask[0].float().cpu().numpy(),
                        gates=torch.stack(result.gates).float().cpu().numpy())


def benchmark(a):
    if a.repeats < 1 or a.warmup < 0 or a.height % a.scale or a.image_width % a.scale:
        raise ValueError("Invalid benchmark shape or repetitions")
    model = make_model(a).eval()
    ms = torch.rand(a.batch, a.bands, a.height // a.scale, a.image_width // a.scale, device=a.device)
    pan = torch.rand(a.batch, 1, a.height, a.image_width, device=a.device)
    call = torch.compile(model) if a.compile else model
    if torch.device(a.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(a.device)
    times = []
    with torch.inference_mode(), amp(a):
        for i in range(a.warmup + a.repeats):
            sync(a.device)
            start = time.perf_counter()
            call(ms, pan)
            sync(a.device)
            if i >= a.warmup:
                times.append((time.perf_counter() - start) * 1000)
    print(json.dumps(dict(median_ms=statistics.median(times), batch=a.batch,
                          parameters=sum(p.numel() for p in model.parameters()),
                          peak_allocated_bytes=(torch.cuda.max_memory_allocated(a.device)
                              if torch.device(a.device).type == "cuda" else None)), indent=2))


def train(a):
    torch.manual_seed(a.seed)
    model = make_model(a)
    if a.checkpoint:
        load_weights(model, a.checkpoint)
    loader = DataLoader(NPZDataset(a.data), batch_size=a.batch, shuffle=True,
                        num_workers=a.workers, pin_memory=torch.device(a.device).type == "cuda")
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=5e-8)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(a.epochs):
        model.train()
        sums = torch.zeros(4, device=a.device)
        alpha = mask_weight(epoch, a.epochs, a.alpha)
        for batch in loader:
            batch = {k: v.to(a.device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with amp(a):
                result = model(batch["ms"], batch["pan"])
                terms = fame_loss(result, batch["target"], batch.get("mask"), alpha=alpha, beta=a.beta)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
            sums += torch.stack([t.detach() for t in terms])
        scheduler.step()
        print(json.dumps(dict(epoch=epoch + 1, alpha=alpha,
                              losses=(sums / len(loader)).cpu().tolist())))
        tmp = output / "latest.tmp"
        torch.save(dict(net=model.state_dict(), config=model.config, epoch=epoch+1,
                        optimizer=opt.state_dict(), scheduler=scheduler.state_dict()), tmp)
        tmp.replace(output / "latest.pth")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("infer", infer), ("benchmark", benchmark), ("train", train)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--device", default="cpu")
        p.add_argument("--amp", action="store_true", help="BF16 autocast")
        p.add_argument("--bands", type=int, default=4)
        p.add_argument("--width", type=int, default=32)
        p.add_argument("--experts", type=int, default=4)
        p.add_argument("--top-k", type=int, default=2)
        p.add_argument("--scale", type=int, default=4)
        p.add_argument("--routing", choices=["legacy", "paper"], default="paper")
        p.add_argument("--dispatch", choices=["sparse", "dense"], default="sparse")
        p.add_argument("--head", choices=["paper", "legacy"], default=None,
                       help="Defaults to reconstruction head of selected routing profile")
        if name == "infer":
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--input", required=True)
            p.add_argument("--output", required=True)
            p.add_argument("--sample-mask", action="store_true")
        elif name == "benchmark":
            p.add_argument("--batch", type=int, default=1)
            p.add_argument("--height", type=int, default=128)
            p.add_argument("--image-width", type=int, default=128)
            p.add_argument("--warmup", type=int, default=3)
            p.add_argument("--repeats", type=int, default=10)
            p.add_argument("--compile", action="store_true", help="Experimental; use dense dispatch")
        else:
            p.add_argument("--data", required=True)
            p.add_argument("--output", required=True)
            p.add_argument("--checkpoint")
            p.add_argument("--epochs", type=int, default=1000)
            p.add_argument("--batch", type=int, default=4)
            p.add_argument("--workers", type=int, default=0)
            p.add_argument("--lr", type=float, default=5e-4)
            p.add_argument("--alpha", type=float, default=0.001)
            p.add_argument("--beta", type=float, default=0.1)
            p.add_argument("--seed", type=int, default=123)
    a = parser.parse_args(argv)
    if a.command == "train" and (a.epochs < 1 or a.batch < 1):
        parser.error("epochs and batch must be positive")
    a.fn(a)


if __name__ == "__main__":
    main()
