"""How the training scripts use GPUs: one GPU (or the CPU) per process; several GPUs = a queue of one process per seed. Small helper script
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def add_args(parser):
    parser.add_argument("--gpu", type=int, nargs="+", default=None,
                        help="GPU index(es) in CUDA order (0 = fastest); several share the seeds as a queue; "
                             "default: the first GPU, or the CPU if there is none")
    parser.add_argument("--all-gpus", action="store_true", help="share the seeds between all visible GPUs")
    parser.add_argument("--no-summary", action="store_true", help=argparse.SUPPRESS)  # set for the per-seed processes


def setup(args, out_dir):
    """Call before importing torch. Several GPUs: run this script once per seed (queue_seeds), summarise and exit.
    Otherwise show this process a single GPU: the one asked for, or the first visible one (none: the Trainer uses the CPU).
    """
    gpus = args.gpu or (visible_gpus() if args.all_gpus else [])
    if len(gpus) > 1:
        queue_seeds(gpus, args.seeds, Path(out_dir))
        finish(args, out_dir)
        sys.exit()
    first = str(gpus[0]) if gpus else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = first


def visible_gpus():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [int(g) for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g.strip()]
    import torch  # only counts the GPUs: this process won't train
    return list(range(torch.cuda.device_count()))


def queue_seeds(gpus, seeds, out_dir):
    """A queue: each GPU trains one seed at a time (its own process of this script) and takes the next seed when free.

    Faster GPUs thus train more seeds. A GPU whose seed fails gets no more seeds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [a for a in sys.argv[1:] if a != "--all-gpus"]
    todo, usable, running, failed = list(seeds), list(gpus), {}, []
    while running or (todo and usable):
        for gpu in usable:
            if gpu not in running and todo:
                seed = todo.pop(0)
                log = out_dir / f"seed_{seed}.log"
                # argparse keeps the last value of a repeated option, so these replace the user's --gpu and --seeds
                cmd = [sys.executable, sys.argv[0], *argv, "--gpu", str(gpu), "--seeds", str(seed), "--no-summary"]
                print(f"seed {seed} -> GPU {gpu} (log: {log})", flush=True)
                running[gpu] = (seed, log, subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                                            env={**os.environ, "PYTHONUNBUFFERED": "1"}))
        time.sleep(10)
        for gpu, (seed, log, job) in list(running.items()):
            if job.poll() is None:
                continue
            del running[gpu]
            if job.returncode == 0:
                print(f"seed {seed} on GPU {gpu}: done", flush=True)
            else:
                print(f"seed {seed} on GPU {gpu}: FAILED, see {log}", flush=True)
                failed.append(seed)
                usable.remove(gpu)
    if failed or todo:
        sys.exit(f"not summarising: failed seeds {failed}, seeds never started {todo}")


def finish(args, out_dir):
    """Summarise the run (training/summarise_seeds.py), unless this is one of the per-seed processes of a queue."""
    if not args.no_summary:
        subprocess.run([sys.executable, str(Path(__file__).with_name("summarise_seeds.py")), str(out_dir)], check=True)


def device(args):
    """Call after importing torch: name of the device this process trains on. A GPU asked for with --gpu must exist."""
    import torch
    assert torch.cuda.device_count() <= 1, "several GPUs visible: the Trainer would use DataParallel"
    if not torch.cuda.is_available():
        if args.gpu:
            sys.exit(f"GPU {args.gpu[0]} not found")
        return "CPU"
    return torch.cuda.get_device_name(0)
