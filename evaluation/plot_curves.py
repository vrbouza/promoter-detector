"""Training curves per run, one colour per seed: training and validation loss, and validation AUC, against epoch.

Reads the Trainer log of each seed (last checkpoint's trainer_state.json). Writes results/training_curves/<name>_trainvalloss.png
and <name>_valAUC.png, where <name> is the run. Dotted vertical lines mark the step of each seed's chosen model (task 3: lowest validation loss; authors' recipe: the last step). Runs with the authors' code log the notebook's AUC (on a raw logit); task 3 logs the AUC on the class probability.
"""
import argparse
import json
from pathlib import Path

from matplotlib.lines import Line2D

from common import ROOT, plt  # common sets the non-interactive matplotlib backend

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runs", nargs="*", default=["outputs/task1", "outputs/task2", "outputs/task3", "outputs/task3_authors"])
args = parser.parse_args()


def trainer_log(seed_dir):
    """Trainer state of one seed, from its last checkpoint (None if it has none)."""
    states = sorted(seed_dir.glob("checkpoints/checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[1]))
    return json.loads(states[-1].read_text()) if states else None


for run in map(Path, args.runs):
    best = (run / "best/SEED").read_text().strip() if (run / "best/SEED").exists() else None
    seeds = [(d.name, s) for d in sorted(run.glob("seed_*")) if (s := trainer_log(d))]
    if not seeds:
        print(f"No seeds found in {run}")
        continue

    state = seeds[0][1]
    steps_per_epoch = state["global_step"] / state["epoch"]
    note = f"batch size {state['train_batch_size']}, {round(steps_per_epoch):,} steps (batches) per epoch"
    # Create figures
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4.2))
    fig_auc, ax_auc = plt.subplots(figsize=(6, 4.2))
    
    # Create legend
    for i, (seed, state) in enumerate(seeds):
        log, color = state["log_history"], f"C{i}"
        label = seed + (" (best)" if seed == f"seed_{best}" else "")
        train = [(e["epoch"], e["loss"]) for e in log if "loss" in e]
        val = [(e["epoch"], e["eval_loss"], e["eval_roc_auc"]) for e in log if "eval_loss" in e]
        ax_loss.plot(*zip(*train), color=color, lw=0.8, alpha=0.7, ls="--")
        ax_loss.plot([v[0] for v in val], [v[1] for v in val], color=color, lw=1.5, label=label)
        ax_auc.plot([v[0] for v in val], [v[2] for v in val], color=color, lw=1.5, label=label)
        chosen = (state["best_global_step"] or state["global_step"]) / steps_per_epoch  # no best step: the last one
        for ax in (ax_loss, ax_auc):
            ax.axvline(chosen, color=color, ls=":", lw=1.2)

    # Plot the loss: one legend for the line styles, one for the seeds (colours).
    chosen_line = Line2D([], [], color="grey", ls=":", lw=1.2, label="chosen model")
    styles = [Line2D([], [], color="grey", lw=0.8, ls="--", label="training"),
              Line2D([], [], color="grey", lw=1.5, label="validation"), chosen_line]
    ax_loss.add_artist(ax_loss.legend(handles=styles, fontsize=8, loc="upper center"))
    ax_loss.legend(fontsize=8, loc="upper right")
    ax_loss.set(xlabel="epoch", ylabel="loss")
    
    # Plot the auc
    ax_auc.legend(handles=ax_auc.get_legend_handles_labels()[0] + [chosen_line], fontsize=8, loc="lower right")
    ax_auc.set(xlabel="epoch", ylabel="validation AUC")

    name = run.name.replace("task3", "newcomparison")
    for fig, ax, suffix in ((fig_loss, ax_loss, "trainvalloss"), (fig_auc, ax_auc, "valAUC")):
        ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=8, ha="left", va="bottom")  # where a title would be
        out = ROOT / f"results/training_curves/{name}_{suffix}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"{run} -> {out}")
