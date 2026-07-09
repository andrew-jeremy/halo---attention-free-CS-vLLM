"""Size-matched transformer baseline: a standard pre-norm GPT (nanoGPT-style).

Self-test: python transformer.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab=256, d=384, n_layers=6, n_heads=6, block_size=256):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, n_layers=n_layers, n_heads=n_heads,
                        block_size=block_size)
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight            # weight tying (standard)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.block_size
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_f(h))

    def aux_loss(self):
        return torch.tensor(0.0)


if __name__ == "__main__":
    m = GPT(d=128, n_layers=2, n_heads=4, block_size=128)
    x = torch.randint(0, 256, (2, 128))
    print("forward:", m(x).shape, "params:", sum(p.numel() for p in m.parameters()))
    print("OK")
