# HALO — Holographic Autoregressive Language Operator

An attention-free language model built entirely from compressed-sensing primitives.
No attention, no softmax over positions, no KV cache.

- **Context = compressive sketch.** The model never stores the sequence; each layer keeps a
  fixed-size multi-timescale RIP measurement `y_t = λ·Π(y_{t−1}) + Φ·s_t` of the k-sparse
  token trajectory. Φ and Π are *fixed random isometries* — zero learned temporal-mixing
  parameters.
- **Computation in measurement space.** Probes = matched filters; the unbinding path
  `(Φa ⊙ y)` does content-dependent retrieval holographically; TopK (the only nonlinearity)
  is one step of iterative hard thresholding, so depth = unrolled sparse inference.
- **Guarantees.** State size follows a formula (`M_b ≈ C·k·2^b·log N` per dyadic band);
  retrieval SNR is `√(M/kT_eff)` — verified empirically to two decimals.
- **Serving.** O(1) memory per sequence (kilobytes), per-token compute independent of
  context length, prefill = parallel associative scan, state fork = memcpy.

See `paper/halo_manuscript.tex` (and `.pdf`) for the full manuscript, and
`docs/halo-cs-lm-design.md` for the working design document.

## Quickstart

```bash
pip install -r requirements.txt

# Exp 0: sketch obeys CS theory (no learning) — SNR vs sqrt(M/kT)
python halo_prototype.py --exp 0

# Exp 1: associative recall without attention (94.9% vs 12.5% chance)
python halo_prototype.py --exp 1 --steps 1300 --ckpt e1.pt

# Exp 2: char-level LM (0.94 nats/char vs 2.47 bigram baseline)
python halo_prototype.py --exp 2 --corpus corpus.txt --steps 500 --ckpt e2.pt
```

`--ckpt` saves/resumes model+optimizer state, so long runs can be chained.
`corpus.txt` is a 300 KB smoke-test corpus (public-domain license texts); any UTF-8 text works.

To train on standard benchmarks (tiny Shakespeare, enwik8, text8), e.g.:

```bash
curl -LO http://mattmahoney.net/dc/enwik8.zip && unzip enwik8.zip
python halo_prototype.py --exp 2 --corpus enwik8 --steps 100000 \
    --max-chars 0 --seq 256 --batch 32 --ckpt enwik8.pt
```

Exp 2 keeps a held-out tail split (`--val-frac`, default 5%) and reports val loss in both
nats/char and bits/char. **See [TRAINING.md](TRAINING.md)** for the full training walkthrough
of both experiments: data pipelines, hyperparameters, expected learning curves, benchmark
commands, and the scaling recipe.

## Results (CPU, torch 2.4.1, 1.43M params)

| Experiment | Result |
|---|---|
| Sketch SNR vs predicted √(M/kT) | 0.71/1.00/1.46/2.02/2.83 vs 0.71/1.00/1.41/2.00/2.83 |
| Associative recall (chance 12.5%) | **94.9%** (additive-probe ablation: 46%) |
| Char-LM (uni 3.19 / bi 2.47 nats) | **0.94 nats/char** @ 500 steps |

## Repo layout

## GPU benchmark (DGX Spark / any CUDA machine)

`benchmark/` contains a size-matched three-way comparison — HALO vs Transformer vs
Mamba, ~10M params each, byte-level enwik8, test bits/char — with a frozen protocol:

```bash
cd benchmark
python data.py --dataset enwik8
python train.py --model halo --preset 10m --data data/enwik8 --steps 50000
python eval.py  --ckpt runs/halo_10m_s0/best.pt --data data/enwik8
```

See `benchmark/README.md` (DGX Spark runbook) and `benchmark/PROTOCOL.md`. The GPU
HALO uses rotation binding + a chunked parallel scan (verified against the sequential
reference to 2e-6) so training speed is matmul-bound like the baselines.

## Repo layout

```
halo_prototype.py   # toy model + all three paper experiments (~280 lines)
TRAINING.md         # detailed training guide (Exp 1 + Exp 2 on benchmarks)
corpus.txt          # sample training text for Exp 2
benchmark/          # GPU harness: data.py, train.py, eval.py, models/, PROTOCOL.md
paper/              # LaTeX manuscript + compiled PDF
docs/               # design documents
requirements.txt
```
