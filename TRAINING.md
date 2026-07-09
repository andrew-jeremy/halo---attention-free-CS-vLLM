# TRAINING.md — How HALO Trains

This document walks through the full training process for the two learned experiments in
`halo_prototype.py`: **Exp 1** (associative recall) and **Exp 2** (character-level language
modeling on standard benchmark datasets). Exp 0 involves no training — it is a closed-form
check that the sketch obeys compressed-sensing theory — so it is not covered here.

---

## 1. What "training" means in HALO

HALO deliberately trains only a subset of the network. Understanding this split is the key
to everything below.

| Component | Symbol | Status |
|---|---|---|
| Measurement operators (one per band per layer) | Φ | **Frozen** random Gaussian, registered as buffers |
| Time-binding operators | Π (signed permutations) | **Frozen** random |
| Band decays | λ_b ∈ {0.5, 0.875, 0.969} | **Fixed** (dyadic ladder) |
| Sparse dictionary / embedding | E (V×N) | Learned |
| Probe filters (per band per layer) | W_q | Learned |
| Unbinding filters (per band per layer) | W_c | Learned |
| Direct path | W_d | Learned |
| Readout + sketch taps | W_o, W_r | Learned |

All temporal mixing — everything that moves information *across positions* — is
parameter-free. Gradient descent only shapes the pointwise maps: what enters the sketch
(E), and what is read out of it (W_q, W_c, W_d, W_o, W_r). The optimizer therefore never
touches the recurrence, which is why training is stable: the scan has spectral radius
λ_b < 1 by construction, so there are no exploding gradients through time, and the frozen
Φ/Π provide the same near-isometric embedding at step 0 as at step 100k.

Both experiments use the same loop skeleton:

```
batch → forward (sparse-code → sketch scan → PTR layers → logits)
      → cross-entropy + 0.1 · coherence_penalty
      → backward → Adam(lr=3e-3) step
```

The coherence penalty `‖ĒᵀĒ − I‖²_offdiag` (on the row-normalized embedding) keeps the
learned dictionary incoherent — the assumption under which the RIP/recall guarantees hold.
Gradients pass through TopK exactly on the surviving coordinates; the selection mask is
piecewise constant, so no straight-through estimator is needed.

---

## 2. Exp 1 — Associative recall (MQAR-style)

### 2.1 The task

The model sees 4 key–value pairs once, then a query marker and one of the keys, and must
emit the bound value. This is the canonical "you need attention for this" task
(Arora et al., *Zoology*, 2023). Chance is 1/8 = 12.5%.

```
sequence:  k₃ v₃ k₁ v₁ k₇ v₇ k₂ v₂ [Q] k₇
target:                                 v₇     (loss on final position only)
```

Vocabulary: 8 keys (ids 0–7), 8 values (ids 8–15), 1 query marker (id 16). Keys within a
sequence are sampled *without* replacement (`randperm`), values independently at random.

### 2.2 The data

There is no dataset. Every batch is synthesized on the fly inside `make_batch()`:

```python
keys = randperm(8)[:4]  per row      # 4 distinct keys
vals = randint(8, 16, (4,))          # 4 random values
seq  = interleave(keys, vals)        # k v k v k v k v
x    = [seq, Q, keys[random_index]]  # append query
y    = vals[same_index]              # supervision: the bound value
```

Fresh sampling every step means the model cannot memorize pairs — it must learn the
*retrieval circuit*: layer 1 forms bigram (key, value) atoms by coincidence detection over
the fast band (λ=0.5, horizon ≈ 2, i.e., "previous token"); layer 2's sketch superposes
those atoms; at the query, the unbinding path `(Φa ⊙ y)` correlates the query key's
measurement against that sketch and TopK selects the matching bigram atom.

### 2.3 Hyperparameters

| Parameter | Value |
|---|---|
| Dictionary size N / sparsity k | 512 / 16 |
| Layers / bands | 2 / 3 (M = 96, 128, 160) |
| Batch size | 64 |
| Optimizer / lr | Adam / 3e-3 |
| Loss | CE at answer position + 0.1·coherence |
| Eval | 512 fresh sequences every 100 steps |

### 2.4 Run it

```bash
python halo_prototype.py --exp 1 --steps 1300 --ckpt e1.pt
```

`--ckpt` saves `{model, optimizer}` state at the end of the run and resumes if the file
exists, so `4 × --steps 330` is exactly equivalent to one 1300-step run (same optimizer
moments throughout; only the random batches differ).

### 2.5 What to expect (actual run, CPU, ~0.08 s/step)

| Step | Loss | Recall accuracy |
|---|---|---|
| 0 | 18.4 | 12.5% (chance) |
| 100 | 1.70 | 40.8% |
| 300 | 1.16 | 51.2% |
| 660 | 0.42 | 85.2% |
| 990 | 0.09 | 91.8% |
| ~1300 | 0.01–0.2 | **94.9%** |

The learning curve has a characteristic shape: fast rise to ~45% (the additive-probe
circuit: unigram/recency statistics), a plateau, then a second rise as the bilinear
unbinding circuit comes online. **Ablation:** delete the `W_c` term in `PTRLayer.forward`
and the model stalls permanently at the plateau (~46%) — additive functionals of the sketch
provably cannot condition retrieval on the query; the unbinding path is what buys recall.

Harder task variants: raise `n_pairs` (memory load → tests the √(M/kT_eff) SNR law),
widen the key/value vocabularies, or insert distractor tokens between the pairs and the
query (tests band horizons).

---

## 3. Exp 2 — Char-level LM on benchmark datasets

### 3.1 The data pipeline

`exp2()` reads any UTF-8 file, builds a char vocabulary, and makes a **held-out tail
split**: the last `--val-frac` (default 5%) of the file is validation and is never sampled
for training. Training batches are random `--seq`-length windows (default 96) of the train
region with next-char targets. Reported numbers:

- `train` — CE loss of the current batch (nats/char)
- `val` — CE averaged over 8 random held-out batches (nats/char **and bits/char**;
  bits = nats / ln 2, the unit used in published enwik8/text8 results)
- `uni` / `bi` — unigram and add-1-smoothed bigram baselines *computed on the val split*

The shipped `corpus.txt` (300 KB of public-domain license text) is a smoke-test corpus,
not a benchmark. For real experiments, use the standard char-LM datasets:

### 3.2 Benchmark datasets

**Tiny Shakespeare** (1.1 MB — the classic small-scale sanity benchmark):

```bash
curl -LO https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python halo_prototype.py --exp 2 --corpus input.txt \
    --steps 5000 --max-chars 0 --ckpt shakespeare.pt
```

**enwik8** (100 MB Wikipedia XML; the standard char-level benchmark, reported in
bits/char; conventional split is 90/5/5M):

```bash
curl -LO http://mattmahoney.net/dc/enwik8.zip && unzip enwik8.zip
python halo_prototype.py --exp 2 --corpus enwik8 \
    --steps 100000 --max-chars 0 --val-frac 0.05 \
    --seq 256 --batch 32 --ckpt enwik8.pt
```

**text8** (100 MB, enwik8 stripped to lowercase a–z + space — easier, vocab 27):

```bash
curl -LO http://mattmahoney.net/dc/text8.zip && unzip text8.zip
python halo_prototype.py --exp 2 --corpus text8 \
    --steps 100000 --max-chars 0 --seq 256 --batch 32 --ckpt text8.pt
```

Word/BPE-level benchmarks (WikiText-2/103, PTB) require swapping the char vocabulary for
a tokenizer (e.g., `tiktoken`); the model code is unchanged — only `V` grows, and N/k
should grow with it (see §3.5).

### 3.3 Relevant flags

| Flag | Default | Meaning |
|---|---|---|
| `--corpus` | `corpus.txt` | any UTF-8 text file |
| `--max-chars` | 200000 | corpus cap; **use `0` for benchmarks** (whole file) |
| `--val-frac` | 0.05 | held-out tail fraction |
| `--seq` | 96 | training window length |
| `--batch` | 16 | batch size |
| `--steps` | 300 | optimizer steps this run |
| `--ckpt` | none | save/resume model+optimizer (chain long runs) |

### 3.4 What to expect (reference run: corpus.txt, toy config, CPU)

| Step | Train (nats) | Context |
|---|---|---|
| 0 | ~38 | random logits (matched-filter taps start uncalibrated — large but harmless) |
| 50 | 2.06 | already below the 2.47 bigram baseline |
| 200 | 1.28 | exploiting structure beyond bigrams |
| 500 | **0.94** | ≈ 1.36 bits/char on this (easy, repetitive) corpus |

Healthy-run signatures: the val curve tracks train with a small gap (fresh random windows
each step ≈ single-epoch regime on large corpora); loss should cross the **bigram**
baseline within a few hundred steps — if it stalls at the unigram level, the sketch path
is not being used (check that Φ buffers loaded and λ values are the dyadic ladder).

### 3.5 Scaling to honest benchmark numbers

The toy config (N=512, k=16, 2 layers, bands with horizons 2/8/32) will train on enwik8
but will not approach published bits/char — its longest memory is ~32 characters. The
knobs, and the theory that sets them (design doc §4.2: M_b ≈ C·k·2^b·log N per band):

| Knob | Toy | Benchmark-serious | Why |
|---|---|---|---|
| Bands B | 3 (horizon 32) | 8–11 (horizon 256–2048) | horizons double per band |
| Band widths M_b | 96–160 | grow ∝ k·2^b·log N | capacity formula, eq. (5) |
| Dictionary N | 512 | 4096–16384 | more atoms = more concepts before coherence degrades |
| Sparsity k | 16 | 32–64 | richer codes; keep k·T_eff ≪ M_b for slow bands |
| Layers L | 2 | 4–8 | more unrolled IHT steps (deeper sparse inference) |
| seq/batch | 96/16 | 512+/64+ (GPU) | expose slow bands to long-range structure |
| lr schedule | constant 3e-3 | warmup + cosine to ~3e-4 | standard practice at scale |

Two code notes for large runs: (1) the training scan is a Python loop over T — correct but
slow; for GPU-scale work replace it with an associative scan (the recurrence is linear, so
it parallelizes exactly like S5/Mamba); (2) the corpus is loaded into one tensor — for
100 MB+ files use a memory-mapped array.

### 3.6 Monitoring specific to HALO

- **Coherence penalty term**: should stay small and stable. If it climbs while loss falls,
  the dictionary is buying accuracy by violating incoherence — raise its weight (0.1 →
  0.3) or the recall guarantees quietly erode.
- **Atom usage**: track how often each of the N atoms survives TopK. A few dead atoms are
  normal; mass die-off means k is too small for the current loss surface — anneal k from
  ~2× its target down over early training, or add the MoE-style load-balancing loss
  (design doc §6).
- **Band ablation at eval**: zeroing one band's sketch at eval time and measuring the loss
  jump tells you which timescales the model actually uses — a cheap interpretability probe
  unique to this architecture.

---

## 4. Reproducing the paper numbers

```bash
python halo_prototype.py --exp 0                                  # Table 2: SNR vs √(M/kT)
python halo_prototype.py --exp 1 --steps 1300 --ckpt e1.pt        # §6.2: 94.9% recall
python halo_prototype.py --exp 2 --corpus corpus.txt --steps 500 --ckpt e2.pt   # §6.3
```

Seeds: the script sets `torch.manual_seed(0)` at import; Exp 1/2 numbers reproduce to
within normal minibatch noise (±1–2% recall, ±0.05 nats).
