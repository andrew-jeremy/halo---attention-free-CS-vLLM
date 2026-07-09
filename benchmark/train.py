"""
Unified trainer for the three-way comparison (HALO / GPT / Mamba), byte-level LM.

Example (DGX):
  python train.py --model halo --preset 10m --data data/enwik8 \
      --steps 50000 --batch 32 --seq 256 --lr 3e-3 --seed 0 --out runs/halo_s0

Fairness rules (see PROTOCOL.md): identical data, seq, batch, token budget,
optimizer family, and schedule for all models; only lr is tuned per model.
Logs CSV (step, train nats, val bpc, lr, sec) and checkpoints best/last.

Andrew Kiruluta, UC Berkeley, CA. 2026.
"""

import argparse, csv, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from models.halo_gpu import HALOBench
from models.transformer import GPT
from models.mamba_minimal import Mamba

PRESETS = {
    # ~10M learned params each (verify with the printed count; adjust to match ±5%)
    "halo":  {"10m":  dict(n_dict=1280, k=32, n_layers=4, n_bands=6,
                           band_dims=[48, 80, 112, 128, 176, 224], d_rank=160),
              "tiny": dict(n_dict=512, k=16, n_layers=2, n_bands=3,
                           band_dims=[96, 128, 160], d_rank=64)},
    "gpt":   {"10m":  dict(d=384, n_layers=6, n_heads=6),
              "tiny": dict(d=128, n_layers=2, n_heads=4)},
    "mamba": {"10m":  dict(d_model=512, n_layers=6),
              "tiny": dict(d_model=128, n_layers=2)},
}


def build(model, preset, seq, vocab=256):
    cfg = PRESETS[model][preset]
    if model == "halo":
        return HALOBench(vocab=vocab, **cfg)
    if model == "gpt":
        return GPT(vocab=vocab, block_size=seq, **cfg)
    return Mamba(vocab=vocab, **cfg)


def get_batch(arr, seq, bsz, device, rng):
    ix = rng.integers(0, len(arr) - seq - 1, size=bsz)
    x = torch.stack([torch.from_numpy(arr[i:i + seq].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(arr[i + 1:i + seq + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, arr, seq, bsz, device, n_batches=16, seed=1234):
    model.eval()
    rng = np.random.default_rng(seed)          # fixed windows -> low-variance metric
    tot = 0.0
    for _ in range(n_batches):
        x, y = get_batch(arr, seq, bsz, device, rng)
        tot += F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).item()
    model.train()
    return tot / n_batches                     # nats/byte


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["halo", "gpt", "mamba"], required=True)
    p.add_argument("--preset", default="10m", choices=["tiny", "10m"])
    p.add_argument("--data", required=True, help="dir with train/val.bin (see data.py)")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--out", default=None)
    p.add_argument("--amp", default="bf16", choices=["bf16", "off"])
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or f"runs/{args.model}_{args.preset}_s{args.seed}"
    os.makedirs(out, exist_ok=True)
    lr = args.lr or {"halo": 3e-3, "gpt": 6e-4, "mamba": 1e-3}[args.model]

    train_arr = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint8, mode="r")
    val_arr = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint8, mode="r")

    model = build(args.model, args.preset, args.seq).to(device)
    n_par = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    n_buf = sum(b.numel() for b in model.buffers())
    print(f"{args.model}/{args.preset}: {n_par/1e6:.2f}M learned params, "
          f"{n_buf/1e6:.2f}M frozen buffer elements, device={device}, lr={lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    sched = lambda s: min(s / max(args.warmup, 1), 1.0) * \
        (0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0))) * 0.9 + 0.1)
    start = 0
    ck = os.path.join(out, "last.pt")
    if args.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); start = st["step"]
        print(f"resumed at step {start}")

    logf = open(os.path.join(out, "log.csv"), "a", newline="")
    log = csv.writer(logf)
    if start == 0:
        log.writerow(["step", "train_nats", "val_bpc", "lr", "sec"])
        json.dump(vars(args) | {"params": n_par}, open(os.path.join(out, "config.json"), "w"))

    rng = np.random.default_rng(args.seed)
    amp = torch.autocast(device_type=device, dtype=torch.bfloat16,
                         enabled=(args.amp == "bf16" and device == "cuda"))
    best = float("inf"); t0 = time.time()
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr * sched(step)
        x, y = get_batch(train_arr, args.seq, args.batch, device, rng)
        with amp:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            total = loss + model.aux_loss().to(loss.device)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

        if step % args.eval_every == 0 or step == args.steps - 1:
            vb = evaluate(model, val_arr, args.seq, args.batch, device) / math.log(2)
            log.writerow([step, f"{loss.item():.4f}", f"{vb:.4f}",
                          f"{opt.param_groups[0]['lr']:.2e}", int(time.time() - t0)])
            logf.flush()
            print(f"step {step:>6}  train {loss.item():.3f} nats  "
                  f"val {vb:.4f} bpc  [{time.time()-t0:.0f}s]")
            torch.save({"m": model.state_dict(), "o": opt.state_dict(),
                        "step": step + 1, "model": args.model, "preset": args.preset,
                        "seq": args.seq}, ck)
            if vb < best:
                best = vb
                torch.save({"m": model.state_dict(), "model": args.model,
                            "preset": args.preset, "seq": args.seq, "val_bpc": vb},
                           os.path.join(out, "best.pt"))
    print(f"done. best val {best:.4f} bpc -> {out}/best.pt")


if __name__ == "__main__":
    main()
