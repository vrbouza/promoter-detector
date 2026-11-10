"""Comparison B: task 1 vs task 2 (task 1 + 90% of our data) on the held-out 10% of our D39V/TIGR4 data.

Also: AUC per sigma factor (promoters of one class vs all negatives) and task 2's training-vs-test view.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score

import common

out = common.ROOT / "results/B_pneumo"
entries = [(f"task{t}", common.read_scores(f"task{t}", "pneumo_test")) for t in (1, 2)]
seeds = [(f"task{t}", m, common.read_scores(m, "pneumo_test")) for t in (1, 2) for m in common.seed_models(f"task{t}")]
common.compare(out, "B: our test (10% of D39V/TIGR4)", entries, seeds)

rows = []
for label, d in entries:
    neg = d[d.y == 0]
    for sigma in ("SigA", "SigX", "unassigned"):
        pos = d[(d.y == 1) & (d.sigma == sigma)]
        both = pd.concat([pos, neg])
        rows.append({"model": label, "sigma": sigma, "n_pos": len(pos), "n_neg": len(neg),
                     "auc_vs_all_negatives": roc_auc_score(both.y, both.score)})
common.write_tsv(rows, out / "by_sigma.tsv")
print(pd.DataFrame(rows).to_string(index=False))

train = common.read_scores("task2", "train_task2")
common.generalisation(out, "task2", [("train (all)", train), ("train (our data)", train[train.origin == "pneumo"]),
                                     ("our test", entries[1][1])],
                      gof_pairs=[("train (our data)", "our test")])  # like for like: our sequences only
