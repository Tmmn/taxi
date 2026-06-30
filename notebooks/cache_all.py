"""
cache_all.py  --  Pre-warm all notebook caches for every experiment group.

Run this once (or after each new batch of results) so that exp_analysis.ipynb
and other notebooks load instantly via load_*_cached() without re-parsing .gz files.

Pickle files written per group into notebooks/cache/:
  <group>_summary.pkl                       -- one summary row per run (~seconds)
  <group>_fleet_ts.pkl                      -- per-taxi timeseries, representative runs (~minutes)
  <group>_regional.pkl                      -- full per-request data with region tags (~minutes)
  <group>_region_req_origin_avg.pkl         -- per-region request-based safety scalars (every run)
  <group>_region_req_metrics_origin_avg.pkl -- per-region safety + wait + service scalars (every run)

Usage
-----
  # from anywhere in the project (script resolves paths itself):
  python notebooks/cache_all.py

  # force-regenerate everything even if caches are fresh:
  python notebooks/cache_all.py --force

  # specific groups only:
  python notebooks/cache_all.py --groups exp1_balanced,exp2_balanced

  # skip the heavy per-taxi and per-request caches (just summaries):
  python notebooks/cache_all.py --summary-only

  # run N groups in parallel (default: min(4, n_groups)):
  python notebooks/cache_all.py --workers 6

  # sequential (equivalent to --workers 1):
  python notebooks/cache_all.py --workers 1

Notes on data correctness
--------------------------
- Wait times (wait_to_assign, wait_to_pickup) are computed only for completed
  trips (mode == "done").  Dropped requests are excluded because they often
  lack pickup/dropoff timestamps.  The drop_rate column captures their share.
- service_rate = n_done / n_total where n_total includes in-flight requests
  that did not finish by simulation end -- so it is a true completion rate.
- Regional analysis tags each request by its *origin* region.  drop_rate per
  region is fraction of requests from that region that were dropped.
- Representative runs (one per R × behaviour × initial_conditions combo,
  closest to median d) are used for fleet_ts and regional caches to keep file
  sizes manageable.  For cross-experiment comparison load the same
  (R, d, behaviour, initial_conditions) row from each group's summary instead.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Run with CWD = notebooks/ so that relative paths in exp_utils (../results,
# ../configs) resolve correctly, regardless of where the script is invoked from.
_NOTEBOOKS_DIR = Path(__file__).resolve().parent
os.chdir(_NOTEBOOKS_DIR)
sys.path.insert(0, str(_NOTEBOOKS_DIR))

from exp_utils import (  # noqa: E402  (import after chdir is intentional)
    GROUP_CONFIGS,
    get_representative_runs,
    load_fleet_ts_cached,
    load_region_req_cached,
    load_region_req_metrics_cached,
    load_regional_data_cached,
    load_regions,
    load_run_summaries_cached,
)

CACHE_DIR = _NOTEBOOKS_DIR / "cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _print_header(group: str, label: str) -> None:
    bar = "-" * 62
    print(f"\n{bar}")
    print(f"  {group}  │  {label}")
    print(bar)


# ---------------------------------------------------------------------------
# Per-group caching
# ---------------------------------------------------------------------------

def cache_group(
    group: str,
    cfg: dict,
    *,
    force: bool,
    summary_only: bool,
) -> bool:
    """Cache all artifacts for one group.  Returns True if the group was processed."""
    results_dir = Path(cfg["results_dir"])
    if not results_dir.exists():
        print(f"  [skip] {group}: {results_dir} not found")
        return False

    _print_header(group, cfg.get("label", ""))
    t_group = time.time()

    # -- 1. Run summaries ------------------------------------------------------
    t = time.time()
    df = load_run_summaries_cached(results_dir, CACHE_DIR, group, force=force)
    if df.empty:
        print("  [warn] empty summary – skipping fleet_ts and regional")
        return True
    print(f"  summaries : {len(df)} runs  ({_fmt(time.time() - t)})")

    if summary_only:
        return True

    # -- 2. Fleet time-series (representative runs) ----------------------------
    rep = get_representative_runs(df)
    t = time.time()
    fleet_ts = load_fleet_ts_cached(results_dir, rep, CACHE_DIR, group, force=force)
    print(f"  fleet_ts  : {len(fleet_ts)} representative runs  ({_fmt(time.time() - t)})")

    # -- 3. Regional per-request data ------------------------------------------
    regions_file = cfg.get("regions_file")
    if regions_file is None or not Path(regions_file).exists():
        print(f"  [skip] no regions file for {group} – skipping regional caches")
    else:
        regions_cfg = load_regions(regions_file)
        regions_list = regions_cfg.get("regions", regions_cfg)
        t = time.time()
        regional = load_regional_data_cached(
            results_dir, rep, regions_list, CACHE_DIR, group, force=force
        )
        print(f"  regional  : {len(regional)} representative runs  ({_fmt(time.time() - t)})")

        # -- 4. Request-based regional safety scalars (every run, all algorithms) --
        t = time.time()
        region_req = load_region_req_cached(
            results_dir, CACHE_DIR, group, regions_list, force=force
        )
        print(f"  region_req: {len(region_req)} runs  ({_fmt(time.time() - t)})")

        # -- 5. Request-based regional safety + wait + service scalars -------------
        t = time.time()
        region_req_metrics = load_region_req_metrics_cached(
            results_dir, CACHE_DIR, group, regions_list, force=force
        )
        print(f"  region_req_metrics: {len(region_req_metrics)} runs  ({_fmt(time.time() - t)})")

    print(f"  -- group total: {_fmt(time.time() - t_group)}")
    return True


# ---------------------------------------------------------------------------
# Parallel worker (top-level so it is picklable on Windows/spawn)
# ---------------------------------------------------------------------------

def _cache_group_worker(args: tuple) -> tuple:
    """
    Worker entry point for ProcessPoolExecutor.

    Captures all stdout from cache_group so that output from concurrent
    workers doesn't interleave. Returns (group, processed, captured_output).
    """
    group, cfg, force, summary_only = args
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        processed = cache_group(group, cfg, force=force, summary_only=summary_only)
    except Exception as exc:
        sys.stdout = old_stdout
        return group, False, buf.getvalue(), f"ERROR: {exc}"
    finally:
        sys.stdout = old_stdout
    return group, processed, buf.getvalue(), None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate caches even if they are still fresh",
    )
    parser.add_argument(
        "--groups", default="",
        metavar="GROUP[,GROUP...]",
        help="Comma-separated subset of groups to cache (default: all with results)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Only cache run summaries; skip heavy fleet_ts and regional caches",
    )
    parser.add_argument(
        "--workers", type=int, default=0, metavar="N",
        help=(
            "Number of groups to process in parallel.  "
            "0 (default) = min(4, n_groups).  1 = sequential."
        ),
    )
    args = parser.parse_args()

    # Resolve which groups to process
    if args.groups:
        requested = [g.strip() for g in args.groups.split(",") if g.strip()]
        unknown = [g for g in requested if g not in GROUP_CONFIGS]
        if unknown:
            print(f"Unknown group(s): {unknown}")
            print(f"Available: {sorted(GROUP_CONFIGS)}")
            sys.exit(1)
        configs = {g: GROUP_CONFIGS[g] for g in requested}
    else:
        configs = GROUP_CONFIGS

    n_workers = args.workers if args.workers > 0 else min(4, len(configs))

    print(f"Cache directory : {CACHE_DIR.resolve()}")
    print(f"Groups          : {list(configs)}")
    print(f"Force           : {args.force}")
    print(f"Summary-only    : {args.summary_only}")
    print(f"Workers         : {n_workers}")

    t_total = time.time()
    n_ok, n_skip = 0, 0

    if n_workers == 1 or len(configs) == 1:
        # Sequential path — no subprocess overhead for single-group runs
        for group, cfg in configs.items():
            processed = cache_group(
                group, cfg, force=args.force, summary_only=args.summary_only
            )
            if processed:
                n_ok += 1
            else:
                n_skip += 1
    else:
        tasks = [
            (g, c, args.force, args.summary_only)
            for g, c in configs.items()
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_cache_group_worker, t): t[0]
                for t in tasks
            }
            for fut in as_completed(futures):
                group, processed, output, error = fut.result()
                # Print captured output as a single block (no interleaving)
                print(output, end="")
                if error:
                    print(f"  [ERROR] {group}: {error}")
                if processed:
                    n_ok += 1
                else:
                    n_skip += 1

    bar = "═" * 62
    print(f"\n{bar}")
    print(f"  Finished.  {n_ok} group(s) cached, {n_skip} skipped.")
    print(f"  Total time: {_fmt(time.time() - t_total)}")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()