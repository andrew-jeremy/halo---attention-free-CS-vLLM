"""
GPU-scale HALO for benchmarking (byte-level LM).

Differences from the toy prototype (halo_prototype.py), all engineering:
  * Time-binding by per-channel complex rotation (a diagonal unitary) instead of a
    signed permutation. Same role -- pushes different lags into incoherent subspaces --
    but diagonal, so the sketch recurrence admits a *chunked parallel scan* (big
    triangular matmuls instead of T sequential steps). This is the S5/HRR-in-Fourier
    trick; the CS story is unchanged.
  * W_d (direct path) is low-rank to keep the parameter count matched to baselines.
  * Per-band LayerNorm on sketches before probing (calibrates measurement scales
    across horizons; RIP holds up to the per-band scale).
  * The scan runs in fp32 even under bf16 autocast (long products need the mantissa).

Self-test:  python halo_gpu.py   (checks chunked scan == sequential reference)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_sparsify(z, k):
    z = F.relu(z)
    vals, idx = torch.topk(z, k, dim=-1)
    return torch.zeros_like(z).scatter(-1, idx, vals)


class RotScan(nn.Module):
    """y_t = lambda * R(theta) y_{t-1} + u_t, R = per-pair 2D rotation (fixed random).

    Chunked parallel form: within a chunk of length C, with complex alpha = lambda*e^{i theta},
        z_c = alpha^{c+1} z_carry + sum_{i<=c} alpha^{c-i} v_i ,
    computed as one triangular (C x C x M/2) contraction. Powers alpha^j, j <= C,
    never explode (|alpha| <= 1), so this is numerically stable at any chunk size.
    """

    def __init__(self, m, lam):
        super().__init__()
        assert m % 2 == 0
        theta = torch.rand(m // 2) * 2 * math.pi
        self.register_buffer("ar", lam * torch.cos(theta))   # Re(alpha)
        self.register_buffer("ai", lam * torch.sin(theta))   # Im(alpha)

    def _powers(self, c):                       # alpha^0 .. alpha^c : (c+1, M/2)
        pr = [torch.ones_like(self.ar)]
        pi = [torch.zeros_like(self.ai)]
        for _ in range(c):
            r = pr[-1] * self.ar - pi[-1] * self.ai
            i = pr[-1] * self.ai + pi[-1] * self.ar
            pr.append(r); pi.append(i)
        return torch.stack(pr), torch.stack(pi)

    def forward(self, u, chunk=64):             # u: (B, T, M) real
        B, T, M = u.shape
        with torch.autocast(device_type=u.device.type, enabled=False):
            u = u.float()
            v = u.view(B, T, M // 2, 2)
            vr, vi = v[..., 0], v[..., 1]
            pr, pi = self._powers(min(chunk, T))
            zr = u.new_zeros(B, M // 2)
            zi = u.new_zeros(B, M // 2)
            outs = []
            for s in range(0, T, chunk):
                c = min(chunk, T - s)
                d = torch.arange(c, device=u.device)
                D = d[:, None] - d[None, :]                  # lag matrix (c, c)
                mask = (D >= 0).float()
                Tr = pr[D.clamp(min=0)] * mask[..., None]    # (c, c, M/2)
                Ti = pi[D.clamp(min=0)] * mask[..., None]
                br, bi = vr[:, s:s + c], vi[:, s:s + c]
                outr = torch.einsum("cim,bim->bcm", Tr, br) - torch.einsum("cim,bim->bcm", Ti, bi)
                outi = torch.einsum("cim,bim->bcm", Tr, bi) + torch.einsum("cim,bim->bcm", Ti, br)
                # carry contribution: alpha^{c+1} * z_carry
                cr, ci = pr[1:c + 1], pi[1:c + 1]            # (c, M/2)
                outr = outr + cr[None] * zr[:, None] - ci[None] * zi[:, None]
                outi = outi + cr[None] * zi[:, None] + ci[None] * zr[:, None]
                zr, zi = outr[:, -1], outi[:, -1]
                outs.append(torch.stack([outr, outi], dim=-1).reshape(B, c, M))
            return torch.cat(outs, dim=1)

    def sequential(self, u):                    # reference implementation (for tests)
        B, T, M = u.shape
        v = u.float().view(B, T, M // 2, 2)
        zr = u.new_zeros(B, M // 2); zi = u.new_zeros(B, M // 2)
        outs = []
        for t in range(T):
            r = self.ar * zr - self.ai * zi + v[:, t, :, 0]
            i = self.ar * zi + self.ai * zr + v[:, t, :, 1]
            zr, zi = r, i
            outs.append(torch.stack([zr, zi], dim=-1).reshape(B, M))
        return torch.stack(outs, dim=1)


class PTRLayerGPU(nn.Module):
    def __init__(self, n_dict, k, band_dims, band_lambdas, d_rank):
        super().__init__()
        self.k = k
        self.scans = nn.ModuleList(RotScan(m, l) for m, l in zip(band_dims, band_lambdas))
        for i, m in enumerate(band_dims):
            self.register_buffer(f"phi{i}", torch.randn(m, n_dict) / math.sqrt(m))
        self.norms = nn.ModuleList(nn.LayerNorm(m) for m in band_dims)
        self.w_q = nn.ModuleList(nn.Linear(m, n_dict, bias=False) for m in band_dims)
        self.w_c = nn.ModuleList(nn.Linear(m, n_dict, bias=False) for m in band_dims)
        self.w_d1 = nn.Linear(n_dict, d_rank, bias=False)     # low-rank direct path
        self.w_d2 = nn.Linear(d_rank, n_dict, bias=True)
        for w in list(self.w_q) + list(self.w_c):
            nn.init.normal_(w.weight, std=1.0 / math.sqrt(w.in_features))

    def forward(self, a_in):
        z = self.w_d2(self.w_d1(a_in))
        ys = []
        for i, (scan, norm, wq, wc) in enumerate(zip(self.scans, self.norms, self.w_q, self.w_c)):
            u = a_in @ getattr(self, f"phi{i}").T
            y = norm(scan(u))
            z = z + wq(y) + wc(u * y)          # probe + holographic unbinding
            ys.append(y)
        return topk_sparsify(z, self.k) + a_in, ys


class HALOBench(nn.Module):
    def __init__(self, vocab=256, n_dict=1536, k=32, n_layers=4, n_bands=6,
                 band_dims=None, d_rank=192):
        super().__init__()
        self.cfg = dict(vocab=vocab, n_dict=n_dict, k=k, n_layers=n_layers,
                        n_bands=n_bands, band_dims=band_dims, d_rank=d_rank)
        if band_dims is None:
            band_dims = [64 * min(b + 1, 6) for b in range(n_bands)]
        lambdas = [1 - 2.0 ** -(b + 1) for b in range(n_bands)]   # horizons 2,4,...,2^B
        self.k = k
        self.embed = nn.Embedding(vocab, n_dict)
        self.layers = nn.ModuleList(
            PTRLayerGPU(n_dict, k, band_dims, lambdas, d_rank) for _ in range(n_layers))
        self.readout = nn.Linear(n_dict, vocab)
        self.taps = nn.Linear(sum(band_dims), vocab, bias=False)

    def forward(self, x):
        a = topk_sparsify(self.embed(x), self.k)
        for layer in self.layers:
            a, ys = layer(a)
        return self.readout(a) + self.taps(torch.cat(ys, dim=-1))

    def coherence_penalty(self):
        e = F.normalize(self.embed.weight, dim=-1)
        g = e @ e.T
        return (g - torch.eye(len(g), device=g.device)).pow(2).mean()

    def aux_loss(self):
        return 0.1 * self.coherence_penalty()


if __name__ == "__main__":
    torch.manual_seed(0)
    scan = RotScan(64, lam=0.9)
    u = torch.randn(3, 200, 64)
    err = (scan(u, chunk=64) - scan.sequential(u)).abs().max().item()
    print(f"chunked-vs-sequential scan max abs err: {err:.2e}")
    assert err < 1e-4, "scan mismatch"
    m = HALOBench(vocab=256, n_dict=512, k=16, n_layers=2, n_bands=4, d_rank=64)
    x = torch.randint(0, 256, (2, 128))
    print("forward:", m(x).shape, "params:", sum(p.numel() for p in m.parameters()))
    print("OK")
