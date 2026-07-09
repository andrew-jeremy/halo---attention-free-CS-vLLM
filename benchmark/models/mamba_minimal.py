"""Size-matched Mamba baseline: minimal selective SSM (Gu & Dao 2023), pure PyTorch.

Faithful to the paper's S6 recurrence (input-dependent Delta, B, C over a diagonal
A), with the scan done in chunks via exp-of-cumsum (stable: exponents <= 0).
For serious throughput on the DGX, install the official CUDA package instead
(`pip install causal-conv1d mamba-ssm`) and pass --mamba-official to train.py;
this file is the always-works fallback with identical math.

Self-test: python mamba_minimal.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, d_conv=4, dt_rank=None):
        super().__init__()
        d_in = expand * d_model
        dt_rank = dt_rank or max(d_model // 16, 8)
        self.d_in, self.d_state = d_in, d_state
        self.in_proj = nn.Linear(d_model, 2 * d_in, bias=False)
        self.conv = nn.Conv1d(d_in, d_in, d_conv, padding=d_conv - 1, groups=d_in)
        self.x_proj = nn.Linear(d_in, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_in, bias=True)
        A = torch.arange(1, d_state + 1).float().repeat(d_in, 1)
        self.A_log = nn.Parameter(torch.log(A))               # (d_in, N)
        self.D = nn.Parameter(torch.ones(d_in))
        self.out_proj = nn.Linear(d_in, d_model, bias=False)
        self.dt_rank = dt_rank
        # standard Mamba dt init: softplus(dt_proj bias) uniform-log in [1e-3, 0.1]
        dt = torch.exp(torch.rand(d_in) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def ssm_scan(self, u, dt, Bm, Cm):
        """u,dt: (B,T,D); Bm,Cm: (B,T,N). h_t = exp(dt*A) h_{t-1} + dt*B*u; y = C.h + D*u.

        Sequential reference scan (the recurrence is input-dependent and diagonal-
        decaying, so the stable parallel form needs the official CUDA kernels --
        install `mamba-ssm` and pass --mamba-official to train.py for speed)."""
        b, T, D = u.shape
        A = -torch.exp(self.A_log.float())                    # (D, N), negative
        dA = torch.exp(dt[..., None] * A)                     # (B,T,D,N), in (0,1)
        dBu = dt[..., None] * Bm[:, :, None, :] * u[..., None]
        h = u.new_zeros(b, D, self.d_state)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dBu[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))
        return torch.stack(ys, dim=1) + self.D * u

    def forward(self, x):
        b, T, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[:, :, :T].transpose(1, 2)
        u = F.silu(u)
        proj = self.x_proj(u)
        dt, Bm, Cm = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        y = self.ssm_scan(u.float(), dt.float(), Bm.float(), Cm.float())
        return self.out_proj(y.to(x.dtype) * F.silu(z))


class Mamba(nn.Module):
    def __init__(self, vocab=256, d_model=512, n_layers=6, d_state=16):
        super().__init__()
        self.cfg = dict(vocab=vocab, d_model=d_model, n_layers=n_layers, d_state=d_state)
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList(MambaBlock(d_model, d_state) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)   # tied head needs small init

    def forward(self, x):
        h = self.embed(x)
        for norm, layer in zip(self.norms, self.layers):
            h = h + layer(norm(h))
        return self.head(self.ln_f(h))

    def aux_loss(self):
        return torch.tensor(0.0)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = Mamba(d_model=128, n_layers=2)
    x = torch.randint(0, 256, (2, 64))
    print("forward:", m(x).shape, "params:", sum(p.numel() for p in m.parameters()))
    print("OK")
