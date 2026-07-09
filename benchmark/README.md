# HALO Benchmark — DGX Spark Runbook

Three-way, size-matched comparison (HALO vs Transformer vs Mamba) on enwik8,
reported in test bits/char. Targets an NVIDIA DGX Spark (GB10 Grace Blackwell,
128 GB unified memory, aarch64) but runs on any CUDA machine — or CPU for dry runs.

Read `PROTOCOL.md` first: it freezes the comparison rules (data, budget, matching,
metric) so the result is defensible.

---

## 0. Environment setup (DGX Spark)

The Spark is aarch64 + CUDA, so use NVIDIA's PyTorch container (recommended) or
NVIDIA's aarch64 wheels. From DGX OS:

```bash
# Option A (recommended): NGC container — PyTorch preinstalled for GB10
docker run --gpus all --ipc=host -it --rm \
    -v $HOME/halo:/work -w /work/benchmark \
    nvcr.io/nvidia/pytorch:25.03-py3        # or newer tag

# Option B: native venv
python3 -m venv ~/halo-env && source ~/halo-env/bin/activate
pip install torch numpy   # ensure the wheel reports +cuda for aarch64/sbsa

# sanity
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Optional, for a faster Mamba baseline (official CUDA kernels; the bundled
`mamba_minimal.py` is the always-works pure-PyTorch fallback with identical math):

```bash
pip install causal-conv1d mamba-ssm --no-build-isolation   # builds from source on aarch64
```

Unzip `halo-repo.zip` to `$HOME/halo` so this folder is `~/halo/benchmark`.

## 1. Download and prepare the benchmark data

```bash
cd ~/halo/benchmark
python data.py --dataset shakespeare      # 1.1 MB  — for the dry run
python data.py --dataset enwik8           # 100 MB  — the real benchmark
```

This writes `data/<name>/{train,val,test}.bin` (raw bytes) + `meta.json`.
enwik8 uses the standard 90/5/5 MB split; everything is byte-level (vocab 256),
matching published bits/char conventions. No tokenizer, no preprocessing.

## 2. Dry run — tiny Shakespeare (~10 min total)

Verifies the harness end-to-end before spending real compute:

```bash
for M in halo gpt mamba; do
  python train.py --model $M --preset tiny --data data/shakespeare \
      --steps 2000 --batch 32 --seq 256 --eval-every 500 --out runs/dry_$M
done
```

Expect all three val bits/char curves to fall well below ~4.0 within 2k steps, and
the printed learned-param counts to be sane. If HALO stalls above the ~3.2 unigram
level, something is broken — stop and debug.

## 3. Learning-rate pilots (protocol §Training)

The only per-model tuning allowed. 5k-step pilots on enwik8, seed 0:

```bash
for LR in 1e-3 3e-3 6e-3; do
  python train.py --model halo --preset 10m --data data/enwik8 \
      --steps 5000 --lr $LR --seed 0 --out runs/pilot_halo_$LR
done
# same for gpt over {3e-4,6e-4,1e-3} and mamba over {6e-4,1e-3,3e-3}
```

Pick each model's peak lr by best 5k-step val bpc (in `runs/*/log.csv`), write the
choices into PROTOCOL.md, and don't touch them again.

## 4. Full training runs — 3 models × 3 seeds

```bash
for M in halo gpt mamba; do
  for S in 0 1 2; do
    python train.py --model $M --preset 10m --data data/enwik8 \
        --steps 50000 --batch 32 --seq 256 --lr <chosen> --seed $S \
        --out runs/${M}_10m_s${S}
  done
done
```

Notes for the Spark:
- Each run logs val bits/char every 1k steps to `runs/<name>/log.csv` and keeps
  `best.pt` (lowest val) + `last.pt` (resumable with `--resume`).
- 128 GB unified memory is far more than these 10M-param models need; the batch/seq
  are fixed by protocol for fairness, not by memory. Runs can execute concurrently
  (e.g., 3 seeds of one model at once) — GB10 has the headroom at this scale.
- Rough wall-clock at this scale: hours per GPT run; HALO similar order (chunked
  scan); Mamba with `mamba_minimal` is a few× slower (sequential scan) — use
  `mamba-ssm` if it built, it is the same model class either way.
- Everything is checkpointed: `--resume` continues an interrupted run exactly.

## 5. Validation and test

Validation happens automatically during training (fixed windows, every 1k steps) and
selects `best.pt`. Final numbers — run once, at the end, on the untouched test split:

```bash
for M in halo gpt mamba; do
  for S in 0 1 2; do
    python eval.py --ckpt runs/${M}_10m_s${S}/best.pt --data data/enwik8 --split test
  done
done
```

`eval.py` uses sliding-window scoring (window 256, stride 128) — the convention
behind published enwik8 numbers. Report mean ± std over the 3 seeds per model,
alongside learned params, tokens/sec, and per-sequence state size (see PROTOCOL.md).

## 6. Ablations (the two reviewers will ask for)

```bash
# (a) no-unbinding: comment out the `wc(u * y)` term in models/halo_gpu.py
#     (PTRLayerGPU.forward), retrain 1 seed, re-eval.
# (b) band knockout: at eval, zero one band's sketch and measure the bpc jump —
#     shows which temporal horizons carry the bits.
```

## 7. Reporting

The deliverable is one table:

| Model | Learned params | State @ seq 256 | Test bpc (mean±std) | Tokens/s |
|---|---|---|---|---|
| Transformer (GPT) | 10.8M | 4.5 MB KV, grows O(T) | … | … |
| Mamba (S6) | 10.3M | 393 KB, O(1) | … | … |
| HALO | 10.4M (+5.9M frozen random) | **12 KB, O(1)** | … | … |

plus the val-bpc training curves (`log.csv` files plot directly). Add the table as
§6.4 of the manuscript. Framing that matches the evidence: HALO's claim is the
memory/fidelity trade-off with provable capacity — competitive bpc at 30× less
state than Mamba and ~400× less than the KV cache is a strong result even if the
transformer wins raw bpc at this scale.

## Context-length caveat

The `10m` HALO preset's slowest band has horizon 2^6 = 64 < seq 256: by the capacity
formula (design doc §4.2) the model keeps high-fidelity recall to ~64 bytes and gist
beyond. If long-range recall matters more than parameter matching, add bands
(`n_bands=8`, band_dims extended) and note the param delta in the report.
