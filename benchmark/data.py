"""
Dataset preparation for the HALO benchmark. Everything is byte-level (vocab 256),
which is the convention for enwik8 and works for any text file.

  python data.py --dataset shakespeare              # 1.1 MB, minutes-scale runs
  python data.py --dataset enwik8                   # 100 MB, the standard benchmark
  python data.py --dataset file --path my.txt       # any local text/binary file

Writes <out>/<name>/{train,val,test}.bin (raw uint8) + meta.json.
Splits: enwik8 uses the standard 90/5/5 MB; others use 90/5/5 percent.
"""

import argparse, json, os, urllib.request, zipfile

URLS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "enwik8": "http://mattmahoney.net/dc/enwik8.zip",
}


def fetch(dataset, path=None):
    if dataset == "file":
        assert path, "--path required for --dataset file"
        return open(path, "rb").read()
    url = URLS[dataset]
    print(f"downloading {url} ...")
    fn, _ = urllib.request.urlretrieve(url)
    if url.endswith(".zip"):
        with zipfile.ZipFile(fn) as z:
            return z.read(z.namelist()[0])
    return open(fn, "rb").read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["shakespeare", "enwik8", "file"], required=True)
    p.add_argument("--path", default=None, help="local file for --dataset file")
    p.add_argument("--out", default="data")
    args = p.parse_args()

    raw = fetch(args.dataset, args.path)
    name = args.dataset if args.dataset != "file" else os.path.basename(args.path)
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)

    if args.dataset == "enwik8":                      # standard 90/5/5 MB split
        n1, n2 = 90_000_000, 95_000_000
    else:                                             # 90/5/5 %
        n1, n2 = int(len(raw) * 0.90), int(len(raw) * 0.95)
    splits = {"train": raw[:n1], "val": raw[n1:n2], "test": raw[n2:]}
    for k, v in splits.items():
        with open(os.path.join(d, f"{k}.bin"), "wb") as f:
            f.write(v)
    json.dump({"vocab_size": 256, "bytes": {k: len(v) for k, v in splits.items()}},
              open(os.path.join(d, "meta.json"), "w"), indent=2)
    print(f"wrote {d}: " + ", ".join(f"{k}={len(v):,}B" for k, v in splits.items()))


if __name__ == "__main__":
    main()
