"""
Final test-set evaluation: sliding-window bits/char, the convention used by
published enwik8 numbers (each byte is predicted with substantial left context;
only the last `stride` positions of each window are scored, so no byte is ever
scored with a short context except at the very start of the test set).

  python eval.py --ckpt runs/halo_10m_s0/best.pt --data data/enwik8

Andrew Kiruluta, UC Berkeley, CA. 2026.
"""

import argparse, math, os
import numpy as np
import torch
import torch.nn.functional as F

from train import build


@torch.no_grad()
def sliding_bpc(model, arr, seq, stride, device, batch=32, limit=None):
    n = len(arr) if limit is None else min(limit, len(arr))
    starts = list(range(0, n - seq - 1, stride))
    tot_nats, tot_count = 0.0, 0
    for i in range(0, len(starts), batch):
        ix = starts[i:i + batch]
        x = torch.stack([torch.from_numpy(arr[j:j + seq].astype(np.int64)) for j in ix]).to(device)
        y = torch.stack([torch.from_numpy(arr[j + 1:j + seq + 1].astype(np.int64)) for j in ix]).to(device)
        logits = model(x)
        for row, j in enumerate(ix):
            lo = 0 if j == 0 else seq - stride    # score full first window, tail after
            nats = F.cross_entropy(logits[row, lo:], y[row, lo:], reduction="sum")
            tot_nats += nats.item()
            tot_count += seq - lo
    return tot_nats / tot_count / math.log(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--stride", type=int, default=None, help="default seq/2")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="max bytes to score (debug)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build(st["model"], st["preset"], st["seq"]).to(device)
    model.load_state_dict(st["m"]); model.eval()

    arr = np.memmap(os.path.join(args.data, f"{args.split}.bin"), dtype=np.uint8, mode="r")
    stride = args.stride or st["seq"] // 2
    bpc = sliding_bpc(model, arr, st["seq"], stride, device, args.batch, args.limit)
    print(f"{st['model']}/{st['preset']}  {args.split} bits/char: {bpc:.4f} "
          f"(seq {st['seq']}, stride {stride}, {len(arr):,} bytes)")


if __name__ == "__main__":
    main()
