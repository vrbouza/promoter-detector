"""Summarise the seeds of finished training runs: seeds.tsv (validation metrics per seed, plus mean and SD) and best/. This is run automatically by default, but if you added more seeds, or something broke and you need to re-run it, you can run it manually here.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runs", nargs="+", help="training output folders, e.g. outputs/task3")
args = parser.parse_args()

for run in map(Path, args.runs):
    files = sorted(run.glob("seed_*/val.tsv"))  # one row per seed, written by the training script
    assert files, f"no finished seeds in {run}"
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files]).sort_values("seed", ignore_index=True)
    stats = df.drop(columns="seed").select_dtypes("number").agg(["mean", "std"]).reset_index(names="seed")
    pd.concat([df, stats]).to_csv(run / "seeds.tsv", sep="\t", index=False, float_format="%.4f")

    best = int(df.loc[df.val_auc.idxmax(), "seed"])
    shutil.rmtree(run / "best", ignore_errors=True)
    shutil.copytree(run / f"seed_{best}/model", run / "best")
    (run / "best/SEED").write_text(f"{best}\n")
    print(f"{run}: best seed {best}\n{df.to_string(index=False)}\n")
