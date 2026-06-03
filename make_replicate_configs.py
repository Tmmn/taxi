"""
Make replicate copies of configs to create replications

The simulator does not seed its RNG, so K copies of one config produce K independent runs.
That is what lets us test whether the algorithms really differ at a fixed (d, R, behaviour, ic) point instead of seeing one noisy number.

Copies are written next to the originals in the same experiment folder, named <base>_rep_NN.conf, and run with the usual command:

python batch_run.py configs/exp1_imbalanced/

A config is selected when its filename matches the d, one of the R values, the
behaviour, and (if given) the initial condition. Without --ic both ic variants (base, home) match.

Examples:
    python make_replicate_configs.py --d 258 --R 0.5 --behaviour go_back --ic home --k 10
    python make_replicate_configs.py --d 258,447 --R 0.5,1.0 --behaviour go_back --ic home --k 10 --dry-run
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIGS_ROOT = ROOT / "configs"


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _r_token(r: str) -> str:
    return f"_R_{float(r):.2f}".replace(".", "_") + "_"

def matches(name: str, ds, rs, behaviours, ics) -> bool:
    if not any(f"_d_{d}_" in name for d in ds):
        return False
    if not any(_r_token(r) in name for r in rs):
        return False
    if not any(f"_behav_{b}_" in name for b in behaviours):
        return False
    if ics and not any(f"_ic_{ic}_" in name for ic in ics):
        return False
    return True

def discover_experiments(include_max3: bool) -> list[str]:
    return sorted(p.name for p in CONFIGS_ROOT.iterdir()
                  if p.is_dir() and p.name.startswith("exp") and (include_max3 or "max3" not in p.name))

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", default="", help="comma-separated experiment folder names (default: every exp* folder)")
    ap.add_argument("--d", default="258", help="comma-separated d values")
    ap.add_argument("--R", default="1.0,0.5", help="comma-separated R values")
    ap.add_argument("--behaviour", default="stay", help="comma-separated behaviour values")
    ap.add_argument("--ic", default="", help="comma-separated ic values (default: all)")
    ap.add_argument("--k", type=int, default=10, help="replicate copies per matched config")
    ap.add_argument("--include-max3", action="store_true", help="also use the *_max3 experiment folders (default: base experiments only)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    experiments = _csv(args.experiments) or discover_experiments(args.include_max3)
    ds, rs = _csv(args.d), _csv(args.R)
    behaviours, ics = _csv(args.behaviour), _csv(args.ic)

    print(f"configs root : {CONFIGS_ROOT}")
    print(f"experiments  : {experiments}")
    print(f"d / R        : {ds} / {rs}")
    print(f"behaviour/ic : {behaviours} / {ics or ['(all)']}")
    print(f"K            : {args.k}")
    print(f"dry-run      : {args.dry_run}\n")

    grand_total = 0
    for exp in experiments:
        exp_dir = CONFIGS_ROOT / exp
        if not exp_dir.is_dir():
            print(f"[skip] {exp}: no such folder")
            continue

        sources = [f for f in sorted(exp_dir.glob("*.conf"))
                   if "_rep_" not in f.name
                   and matches(f.name, ds, rs, behaviours, ics)]
        if not sources:
            print(f"[skip] {exp}: no configs match")
            continue

        n_runs = len(sources) * args.k
        grand_total += n_runs
        print(f"{exp}: {len(sources)} cell(s) x {args.k} -> {n_runs} replicate runs")
        for src in sources:
            print(f"    {src.stem}")
        if args.dry_run:
            continue
        for src in sources:
            for i in range(1, args.k + 1):
                shutil.copyfile(src, exp_dir / f"{src.stem}_rep_{i:02d}.conf")

    print(f"\nTotal replicate runs {'planned' if args.dry_run else 'written'}: {grand_total}")
    if not args.dry_run:
        print("Now run each experiment folder as usual")

if __name__ == "__main__":
    main()
