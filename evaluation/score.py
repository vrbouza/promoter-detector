"""Score/do the inference of every model on the sets its comparisons need: results/scores/<model>/<set>.tsv (cached).

Models: the best and every seed of each training run (outputs/<run>), and the two published models.
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force", action="store_true", help="recompute existing score files")
parser.add_argument("--gpu", type=int, default=None,
                    help="GPU index in CUDA order (0 = fastest); default: the first GPU, or the CPU if there is none")
args = parser.parse_args()
# One GPU, chosen before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = (str(args.gpu) if args.gpu is not None
                                      else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

import torch

import common

# Training run -> (test sets its comparisons use, its own training set for the gap plot).
RUNS = {"task1": (["multispecies", "sigma70", "pneumo_test"], "train_task1"),
        "task2": (["pneumo_test"], "train_task2"),
        "task3": (["task3_test", "sigma70"], "train_task3"),
        "task3_authors": (["task3_test", "sigma70"], "train_task3")}
PUBLISHED = {"published-2025": ["multispecies", "sigma70"], "published-2024": ["multispecies", "sigma70"]}


def models(): # An iterator... I think? yield was for iterators, right?
    """(name, path or published id, sets). Best models also score their own training set."""
    for run, (sets, train) in RUNS.items():
        d = common.ROOT / f"outputs/{run}"
        if (d / "best").exists():
            yield run, d / "best", sets + [train]
        for m in sorted(d.glob("seed_*/model")):
            yield f"{run}_{m.parent.name}", m, sets
    for name, sets in PUBLISHED.items():
        yield name, name, sets


def main():
    if args.gpu is not None and not torch.cuda.is_available():
        sys.exit(f"GPU {args.gpu} not found")
    print(f"Scoring on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    pool = common.load_pool() ## This gets the "pool" of GPUs ready
    s1, s2, s3 = (common.load_split(t, pool) for t in (1, 2, 3))
    data = {"multispecies": pool[pool.origin == "hf_multispecies"], "sigma70": common.load_sigma70(),
            "pneumo_test": s2["test"], "task3_test": s3["test"],
            "train_task1": s1["train"], "train_task2": s2["train"], "train_task3": s3["train"]}
    tokenizer = common.load_tokenizer()
    for name, path, sets in models():
        out_dir = common.ROOT / f"results/scores/{name}"
        # A best model is re-scored when its seed changed (e.g. after adding seeds): a copy of best/SEED tracks it.
        seed = (path / "SEED").read_text() if name in RUNS else None
        stale = seed is not None and (not (out_dir / "SEED").exists() or (out_dir / "SEED").read_text() != seed)
        todo = [s for s in sets if args.force or stale or not (out_dir / f"{s}.tsv").exists()]
        if not todo:
            continue
        model = common.load_model(path)
        masked = common.MASKED_IDS.get(name, (2, 3))
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in todo:
            d = data[s].copy()
            d["score"] = common.score(model, tokenizer, d.segment, masked)
            cols = [c for c in ("segment_id", "origin", "y", "kind", "sigma", "score") if c in d]
            d[cols].to_csv(out_dir / f"{s}.tsv", sep="\t", index=False)
            print(f"{name:22s} {s:14s} n={len(d)}")
        if seed is not None:
            (out_dir / "SEED").write_text(seed)


if __name__ == "__main__":
    main()
