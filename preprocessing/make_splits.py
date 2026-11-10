"""Write data/splits/task{1,2,3}.tsv (segment_id, split) and data/splits/summary.tsv.

Task 1: 10% of HF train as validation (random, like the paper); test = multispecies + our test.
Task 2: task 1's HF split + our data. 
Task 3/new comparison/actual development: HF train + multispecies re-split by species + our data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import ROOT, SEED, load_pool, write_tsv

OUT = ROOT / "data/splits"


def strata(df):
    """Stratum per row: species (HF promoters; rare ones pooled), prom_class (HF negatives), sigma/src (ours)."""
    sp = df.species.fillna("")
    counts = sp[df.y == 1].value_counts()
    sp = sp.where(~sp.isin(counts[counts < 10].index), "other")
    hf = np.where(df.y == 1, "hf|" + sp, "hf|neg|" + df.kind)
    pn = np.where(df.y == 1, "pn|" + df.sigma, "pn|neg|" + df.kind)
    return pd.Series(np.where(df.origin == "pneumo", pn, hf), index=df.index)


def three_way(df, first, second):
    """Stratified train/val/test: `first` (fraction or rows) goes to test, then `second` of the rest to val."""
    rest, test = train_test_split(df, test_size=first, stratify=df.stratum, random_state=SEED)
    train, val = train_test_split(rest, test_size=second, stratify=rest.stratum, random_state=SEED)
    return train, val, test


def tagged(**parts):
    return pd.concat([pd.DataFrame({"segment_id": d.segment_id, "split": s}) for s, d in parts.items()])


def main():
    pool = load_pool()
    pool["stratum"] = strata(pool)
    hf_train = pool[pool.origin == "hf_train"]
    ms = pool[pool.origin == "hf_multispecies"]
    pn = pool[pool.origin == "pneumo"]

    pn_tr, pn_va, pn_te = three_way(pn, 0.1, 420)
    hf_tr, hf_va = train_test_split(hf_train, test_size=0.1, random_state=SEED)
    t3_tr, t3_va, t3_te = three_way(pd.concat([hf_train, ms]), 24568, 24567)

    splits = {
        1: tagged(train=hf_tr, val=hf_va, test=pd.concat([ms, pn_te])),
        2: tagged(train=pd.concat([hf_tr, pn_tr]), val=pd.concat([hf_va, pn_va]), test=pn_te),
        3: tagged(train=pd.concat([t3_tr, pn_tr]), val=pd.concat([t3_va, pn_va]), test=pd.concat([t3_te, pn_te])),
    }
    expected = {1: (200782, 22310, 22582 + 420), 2: (204139, 22730, 420), 3: (199896, 24987, 24988)}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, df in splits.items():
        n = tuple(int((df.split == s).sum()) for s in ("train", "val", "test"))
        assert n == expected[task], f"task {task}: {n} != {expected[task]}"
        assert df.segment_id.is_unique, f"task {task}: overlapping splits"
        df.to_csv(OUT / f"task{task}.tsv", sep="\t", index=False)
        m = df.merge(pool, on="segment_id")
        rows.append(m.groupby(["origin", "stratum", "split"]).size().unstack(fill_value=0).reset_index()
                    .assign(task=task))
        print(f"task {task}: train/val/test = {n}")
    summary = pd.concat(rows)[["task", "origin", "stratum", "train", "val", "test"]].fillna(0)
    write_tsv(summary.astype({c: int for c in ("train", "val", "test")}), OUT / "summary.tsv")


if __name__ == "__main__":
    main()
