# PROTOCOL.md — Frozen Comparison Protocol

Fixed **before** any full run. Deviations must be reported.

## Models

Three byte-level (vocab 256) language models, learned parameters matched within ±5%:

| Model | Preset `10m` | Learned params |
|---|---|---|
| HALO | N=1280, k=32, L=4, 6 bands (M=48…224, horizons 2–64), low-rank direct path r=160 | 10.37M |
| Transformer | pre-norm GPT, d=384, L=6, h=6, learned pos-emb, tied head | 10.84M |
| Mamba | selective SSM (S6), d_model=512, L=6, d_state=16, expand=2 | 10.31M |

HALO additionally carries 5.9M *frozen random* buffer elements (Φ, rotation phases);
these are not trained and are reported separately, along with per-sequence state size.

## Data

enwik8, standard split: train = first 90 MB, val = next 5 MB, test = last 5 MB.
(Dry runs: tiny Shakespeare, 90/5/5%.) Prepared by `data.py`; raw bytes, no tokenizer.

## Training (identical for all models)

| Item | Value |
|---|---|
| Sequence length / batch | 256 / 32 (8,192 tokens/step) |
| Steps | 50,000 (≈ 410M tokens ≈ 4.5 epochs) |
| Optimizer | AdamW, β=(0.9, 0.95), wd=0.1, grad-clip 1.0 |
| Schedule | 1k warmup, cosine to 10% of peak |
| Precision | bf16 autocast (scans in fp32) |
| Seeds | 3 per model (0, 1, 2) |
| Peak lr | the ONLY per-model knob: grid {1e-3, 3e-3, 6e-3} HALO; {3e-4, 6e-4, 1e-3} GPT; {6e-4, 1e-3, 3e-3} Mamba, chosen on val with seed 0, 5k-step pilots |

## Evaluation

Model selection: best val bits/char (fixed val windows, logged every 1k steps).
Final metric: **test bits/char, sliding window** (window 256, stride 128; only the
final 128 positions of each window scored) via `eval.py`. Report mean ± std over seeds.

Also reported: tokens/sec (same hardware), learned params, per-sequence inference
state (HALO: Σ M_b × L floats = 3,072 floats ≈ 12 KB; GPT: KV cache 256×384×2×6 ≈ 4.5 MB
at seq 256 and growing linearly; Mamba: d_inner×d_state×L = 98K floats ≈ 393 KB).

## Ablations (HALO only, 1 seed)

1. No unbinding path (delete `W_c` term).
2. Band knockout at eval: zero each band's sketch, measure Δbpc per horizon.
