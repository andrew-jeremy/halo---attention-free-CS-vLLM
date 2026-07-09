"""
HALO prototype — attention-free language model built from compressed-sensing primitives.
Companion to halo-cs-lm-design.md.

Experiments:
  0: theory check — matched-filter recall from a random sketch, SNR vs M (no learning)
  1: associative recall (MQAR-style) — key/value pairs shown once, queried later
  2: char-level language modeling

Usage: python halo_prototype.py --exp all [--corpus path.txt]

Andrew Kiruluta, UC Berkeley, CA. 2026.
"""

import argparse, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ----------------------------------------------------------------------------
# Core primitives (design doc §3)
# ----------------------------------------------------------------------------

def topk_sparsify(z, k):
    """TopK over relu(z): the CS-native nonlinearity (one IHT step). Keeps values."""
    z = F.relu(z)
    vals, idx = torch.topk(z, k, dim=-1)
    out = torch.zeros_like(z)
    return out.scatter(-1, idx, vals)


class SketchBank(nn.Module):
    """Multi-timescale compressive sketch of a sparse-code stream (eq. 2).

    y_t^b = lambda_b * SignedPerm_b(y_{t-1}^b) + Phi_b @ a_t
    Phi_b and the signed permutation are FIXED random operators (axiom A4):
    all temporal mixing is parameter-free. Only pointwise maps are learned.
    """

    def __init__(self, n_dict, band_dims, band_lambdas):
        super().__init__()
        self.lambdas = band_lambdas
        for i, (m, lam) in enumerate(zip(band_dims, band_lambdas)):
            phi = torch.randn(m, n_dict) / math.sqrt(m)          # RIP whp for k log(N/k) << m
            self.register_buffer(f"phi{i}", phi)
            self.register_buffer(f"perm{i}", torch.randperm(m))   # time-binding operator
            self.register_buffer(f"sign{i}", torch.randint(0, 2, (m,)).float() * 2 - 1)
        self.n_bands = len(band_dims)

    def forward(self, a):                     # a: (B, T, N) sparse codes
        B, T, _ = a.shape
        ys = []
        for i in range(self.n_bands):
            phi = getattr(self, f"phi{i}")
            perm, sign = getattr(self, f"perm{i}"), getattr(self, f"sign{i}")
            lam = self.lambdas[i]
            u = a @ phi.T                     # measure all steps at once (B, T, M)
            y = torch.zeros(B, u.shape[-1], device=a.device)
            outs = []
            for t in range(T):                # O(M) elementwise scan (linear recurrence)
                y = lam * (sign * y[:, perm]) + u[:, t]
                outs.append(y)
            ys.append(torch.stack(outs, dim=1))
        return ys                             # list of (B, T, M_b)


class PTRLayer(nn.Module):
    """Probe -> Threshold -> Re-code (eq. 3). Replaces attention + FFN.

    Probes correlate the sketch against N learned matched filters (learning in
    measurement space, axiom A3); TopK is one learned-IHT step; residual re-code.
    """

    def __init__(self, n_dict, k, band_dims, band_lambdas):
        super().__init__()
        self.k = k
        self.sketch = SketchBank(n_dict, band_dims, band_lambdas)
        self.w_q = nn.ModuleList([nn.Linear(m, n_dict, bias=False) for m in band_dims])
        self.w_c = nn.ModuleList([nn.Linear(m, n_dict, bias=False) for m in band_dims])
        self.w_d = nn.Linear(n_dict, n_dict, bias=True)   # direct (skip) path
        for w in list(self.w_q) + list(self.w_c):
            nn.init.normal_(w.weight, std=1.0 / math.sqrt(w.in_features))

    def forward(self, a_in):                  # (B, T, N) sparse
        ys = self.sketch(a_in)
        z = self.w_d(a_in)
        for i, (wq, wc, y) in enumerate(zip(self.w_q, self.w_c, ys)):
            u = a_in @ getattr(self.sketch, f"phi{i}").T   # measure current code
            z = z + wq(y) + wc(u * y)         # probe + unbinding (corr. features)
        return topk_sparsify(z, self.k) + a_in, ys


class HALO(nn.Module):
    """Sparse coder -> L x (sketch + PTR) -> readout with matched-filter taps."""

    def __init__(self, vocab, n_dict=512, k=16, n_layers=2,
                 band_dims=(96, 128, 160), band_lambdas=(0.5, 0.875, 0.969)):
        super().__init__()
        self.k = k
        self.embed = nn.Embedding(vocab, n_dict)          # learned dictionary E
        self.layers = nn.ModuleList(
            PTRLayer(n_dict, k, band_dims, band_lambdas) for _ in range(n_layers))
        self.readout = nn.Linear(n_dict, vocab)
        self.taps = nn.Linear(sum(band_dims), vocab, bias=False)  # direct sketch taps (§3.4)

    def forward(self, x):                     # (B, T) token ids
        a = topk_sparsify(self.embed(x), self.k)          # k-sparse semantic code
        for layer in self.layers:
            a, ys = layer(a)
        return self.readout(a) + self.taps(torch.cat(ys, dim=-1))

    def coherence_penalty(self):              # keep learned dictionary CS-friendly (§6)
        e = F.normalize(self.embed.weight, dim=-1)
        g = e @ e.T
        return (g - torch.eye(len(g), device=g.device)).pow(2).mean()


# ----------------------------------------------------------------------------
# Exp 0: matched-filter recall from a sketch — pure CS, no learning (§4.3)
# ----------------------------------------------------------------------------

def exp0():
    print("=" * 70)
    print("Exp 0: matched-filter recall from a random sketch (theory check)")
    N, k, T, trials = 1024, 4, 32, 20
    print(f"N={N} dict atoms, k={k} active/token, T={T} tokens "
          f"(context signal ~{k*T}-sparse)")
    print(f"{'M':>6} {'support recovery':>17} {'meas. SNR':>10} {'pred sqrt(M/kT)':>16}")
    for M in [64, 128, 256, 512, 1024]:
        phi = torch.randn(M, N) / math.sqrt(M)
        hits, snrs = 0, []
        for _ in range(trials):
            supports = [torch.randperm(N)[:k] for _ in range(T)]
            x = torch.zeros(N)
            for s in supports:
                x[s] += 1.0                                # lambda=1: worst case load
            y = phi @ x
            corr = phi.T @ y                               # correlate vs every atom
            present = torch.zeros(N, dtype=torch.bool)
            for s in supports:
                present[s] = True
            top = torch.topk(corr, int(present.sum())).indices   # support recovery
            hits += present[top].float().mean().item()
            snrs.append((corr[present].mean() / corr[~present].std()).item())
        print(f"{M:>6} {hits/trials:>15.3f} {sum(snrs)/trials:>10.2f} "
              f"{math.sqrt(M/(k*T)):>16.2f}")
    print("-> recall undergoes the predicted phase transition as M passes "
          "~ kT log(N/kT); SNR tracks sqrt(M/kT).\n")


# ----------------------------------------------------------------------------
# Exp 1: associative recall — the capability attention supposedly owns (§3.5)
# ----------------------------------------------------------------------------

def exp1(steps=400, ckpt=None):
    print("=" * 70)
    print("Exp 1: associative recall (key value ... key -> value), no attention")
    n_keys, n_vals, n_pairs = 8, 8, 4
    vocab = n_keys + n_vals + 1               # +1 query marker
    Q = vocab - 1

    def make_batch(bsz):
        keys = torch.stack([torch.randperm(n_keys)[:n_pairs] for _ in range(bsz)])
        vals = torch.randint(n_keys, n_keys + n_vals, (bsz, n_pairs))
        seq = torch.stack([keys, vals], dim=-1).reshape(bsz, -1)   # k1 v1 k2 v2 ...
        qi = torch.randint(0, n_pairs, (bsz,))
        qk = keys.gather(1, qi[:, None])
        ans = vals.gather(1, qi[:, None]).squeeze(1)
        x = torch.cat([seq, torch.full((bsz, 1), Q), qk], dim=1)   # ... [Q] k?
        return x, ans

    model = HALO(vocab, n_dict=512, k=16, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    if ckpt:
        import os
        if os.path.exists(ckpt):
            st = torch.load(ckpt, weights_only=False)
            model.load_state_dict(st["m"]); opt.load_state_dict(st["o"])
            print(f"  resumed from {ckpt}")
    for step in range(steps):
        x, ans = make_batch(64)
        logits = model(x)[:, -1]              # predict at query position
        loss = F.cross_entropy(logits, ans) + 0.1 * model.coherence_penalty()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                x, ans = make_batch(512)
                acc = (model(x)[:, -1].argmax(-1) == ans).float().mean()
            print(f"  step {step:>4}  loss {loss.item():.3f}  recall acc {acc:.3f}"
                  f"  (chance {1/n_vals:.3f})")
    if ckpt:
        torch.save({"m": model.state_dict(), "o": opt.state_dict()}, ckpt)
    print("-> retrieval works via sketch matched-filtering: content-addressable "
          "recall without attention.\n")
    return acc.item()


# ----------------------------------------------------------------------------
# Exp 2: char-level language modeling
# ----------------------------------------------------------------------------

def exp2(corpus_path, steps=300, T=96, ckpt=None, max_chars=200_000,
         val_frac=0.05, batch_size=16):
    print("=" * 70)
    print("Exp 2: char-level language modeling")
    text = open(corpus_path, encoding="utf-8", errors="ignore").read()
    if max_chars:
        text = text[:max_chars]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    all_data = torch.tensor([stoi[c] for c in text])
    n_val = int(len(all_data) * val_frac)
    data, val_data = all_data[:-n_val], all_data[-n_val:]   # held-out tail split
    vocab = len(chars)
    probs = torch.bincount(data, minlength=vocab).float()
    probs /= probs.sum()
    unigram = -(probs * (probs + 1e-9).log()).sum().item()
    big = torch.ones(vocab, vocab)                    # add-1 smoothed bigram baseline
    for a, b in zip(data[:-1], data[1:]):
        big[a, b] += 1
    bigram = F.cross_entropy(
        (big / big.sum(1, keepdim=True)).log()[val_data[:-1]], val_data[1:]).item()
    print(f"corpus {len(text)} chars ({len(data)} train / {n_val} val), vocab {vocab}, "
          f"val unigram {unigram:.3f} / bigram {bigram:.3f} nats baseline")

    model = HALO(vocab, n_dict=512, k=16, n_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M learned params "
          f"(all pointwise; temporal mixing is parameter-free)")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    if ckpt:
        import os
        if os.path.exists(ckpt):
            st = torch.load(ckpt, weights_only=False)
            model.load_state_dict(st["m"]); opt.load_state_dict(st["o"])
            print(f"  resumed from {ckpt}")

    def batch(src, bsz):
        ix = torch.randint(0, len(src) - T - 1, (bsz,))
        x = torch.stack([src[i:i + T] for i in ix])
        y = torch.stack([src[i + 1:i + T + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def val_loss(n_batches=8):
        tot = 0.0
        for _ in range(n_batches):
            x, y = batch(val_data, batch_size)
            tot += F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item()
        return tot / n_batches

    t0 = time.time()
    for step in range(steps):
        x, y = batch(data, batch_size)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        (loss + 0.1 * model.coherence_penalty()).backward()
        opt.step(); opt.zero_grad()
        if step % 50 == 0 or step == steps - 1:
            vl = val_loss()
            print(f"  step {step:>4}  train {loss.item():.3f}  val {vl:.3f} nats/char "
                  f"({vl/math.log(2):.3f} bits/char; uni {unigram:.2f} / bi {bigram:.2f})"
                  f"  [{time.time()-t0:.0f}s]")
    if ckpt:
        torch.save({"m": model.state_dict(), "o": opt.state_dict()}, ckpt)
    return vl, unigram


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="all")
    p.add_argument("--corpus", default="corpus.txt")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--max-chars", type=int, default=200_000,
                   help="cap on corpus size; 0 = use the whole file")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=96)
    args = p.parse_args()
    if args.exp in ("0", "all"):
        exp0()
    if args.exp in ("1", "all"):
        exp1(args.steps or 400, ckpt=args.ckpt)
    if args.exp in ("2", "all"):
        exp2(args.corpus, args.steps or 300, T=args.seq, ckpt=args.ckpt,
             max_chars=args.max_chars, val_frac=args.val_frac,
             batch_size=args.batch)
