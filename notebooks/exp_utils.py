"""
exp_utils.py - shared utilities for experiment analysis notebooks.

Usage:
    import sys; sys.path.insert(0, '.')
    from exp_utils import load_run_summaries, load_request_df, ...

Provides
--------
  parse_run_id              Extract parameter dict from run-id string
  load_run_summaries        Fast: one row per run (aggregates + last-line snapshots)
  load_final_taxi_snapshot  Per-taxi summary from last batch of per_taxi_metrics
  load_service_stats        Service rate, wait times, drop reasons from per_request_metrics
  load_request_df           Full per-request DataFrame with deduplication
  build_wait_metrics        Add wait_to_assign / wait_to_pickup / trip_time columns
  load_per_taxi_batches     All batch snapshots as list[dict]
  build_fleet_timeseries    Wide per-batch DataFrame with fleet-level means
  gini / atkinson / ratio_2020 / calc_inequality   Inequality metrics
  load_regions              Parse a regions JSON config file
  assign_region             Tag a single (x, y) with a region id
  tag_requests_with_regions Add origin_region / destination_region columns
  aggregate_region_stats    Per-region mean, count, p90 for any metric list
  compute_regional_gini     Gini of per-region means (scalar fairness metric)
  plot_region_heatmap       Draw a region heatmap on an existing matplotlib Axes
  get_representative_runs   Select representative run subset from a summary DataFrame
  fmt_display               Format a DataFrame for display without Jinja2/pandas Styler
  GROUP_CONFIGS             Dict mapping group name -> (results_dir, regions_file, ...)

  -- Pickle-cache helpers (avoid re-parsing .gz files on every notebook run) --
  load_run_summaries_cached   Cached load_run_summaries (invalidates when any run_* file changes)
  load_fleet_ts_cached        Cached per-taxi timeseries for representative runs
  load_regional_data_cached   Cached full per-request DataFrames for regional analysis
"""
from __future__ import annotations

import gzip
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from scipy import stats

# ---------------------------------------------------------------------------
# Path defaults - override in each notebook if your layout differs
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path("../results")
CONFIGS_ROOT = Path("../configs")

# Short, consistent legend labels keyed by experiment id. These match the Exp1-Exp7
# naming used in the thesis prose so a reader does not have to map a raw algorithm
# string (e.g. nearest_two_sided_region_pass_pref) back to an experiment number.
EXP_SHORT: Dict[str, str] = {
    "exp1": "Exp1 nearest",
    "exp2": "Exp2 region",
    "exp3": "Exp3 distance",
    "exp4": "Exp4 passenger",
    "exp5": "Exp5 two-sided",
    "exp6": "Exp6 safety-2s",
    "exp7": "Exp7 safety-obj",
}


def exp_label(exp: str, algo: Optional[str] = None) -> str:
    """Short legend label for an experiment id, falling back to 'exp: algo'."""
    if exp in EXP_SHORT:
        return EXP_SHORT[exp]
    return f"{exp}: {algo}" if algo else exp


# Natural full ranges for the regional heatmap metrics, so colour encodes the same
# absolute value in every figure rather than each figure's own data min/max. The
# safety score is bounded 0-100 (taxis start at 100 and decay); service/completion
# rates are fractions 0-1; wait-to-pickup is capped at a fixed ceiling that covers
# every observed regional wait with headroom.
METRIC_RANGE: Dict[str, Tuple[float, float]] = {
    "service_rate": (0.0, 1.0),
    "non_completion_rate": (0.0, 1.0),
    "mean_driver_average_safety_score": (0.0, 100.0),
    "mean_driver_safety_score_pickup": (0.0, 100.0),
    "mean_wait_to_pickup": (0.0, 45.0),
}


def apply_pub_style() -> None:
    """Set matplotlib rcParams for legible, tight figures in the thesis.

    Figures are authored large and downscaled to the text width in LaTeX, which
    shrinks any default-size text below readability. Bumping the base sizes and
    cropping whitespace on save keeps labels, ticks, and legends readable after
    the downscale.
    """
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "figure.titlesize": 17,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.04,
    })


def legend_outside(fig, ax=None, handles=None, labels=None, ncol=1,
                   fontsize=12, **kw):
    """Place a shared legend just outside the axes on the right edge.

    apply_pub_style sets savefig.bbox='tight', so the saved canvas expands to
    include the legend rather than letting it sit on top of the lines (which made
    the trajectory panels unreadable) or clipping it off.
    """
    if handles is None:
        src = ax if ax is not None else fig.axes[0]
        handles, labels = src.get_legend_handles_labels()
    return fig.legend(handles, labels, loc='center left',
                      bbox_to_anchor=(1.0, 0.5), frameon=True,
                      ncol=ncol, fontsize=fontsize, **kw)


GROUP_CONFIGS: Dict[str, dict] = {
    # -- Exp 1: nearest (baseline) --------------------------------------------
    "exp1_balanced": {
        "results_dir": RESULTS_ROOT / "exp1_balanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_balanced.json",
        "label": "Exp 1 - Balanced",
        "algorithm": "nearest",
        "balance": "balanced",
        "experiment": "exp1",
    },
    "exp1_imbalanced": {
        "results_dir": RESULTS_ROOT / "exp1_imbalanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_extreme_imbalanced.json",
        "label": "Exp 1 - Imbalanced",
        "algorithm": "nearest",
        "balance": "imbalanced",
        "experiment": "exp1",
    },
    # -- Exp 2: nearest_region_pref -------------------------------------------
    "exp2_balanced": {
        "results_dir": RESULTS_ROOT / "exp2_balanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_balanced.json",
        "label": "Exp 2 - Balanced",
        "algorithm": "nearest_region_pref",
        "balance": "balanced",
        "experiment": "exp2",
    },
    "exp2_imbalanced": {
        "results_dir": RESULTS_ROOT / "exp2_imbalanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_extreme_imbalanced.json",
        "label": "Exp 2 - Imbalanced",
        "algorithm": "nearest_region_pref",
        "balance": "imbalanced",
        "experiment": "exp2",
    },
    "exp2_balanced_max3": {
        "results_dir": RESULTS_ROOT / "exp2_balanced_max3",
        "regions_file": CONFIGS_ROOT / "regions_big_city_balanced.json",
        "label": "Exp 2 - Balanced (max3)",
        "algorithm": "nearest_region_pref",
        "balance": "balanced",
        "experiment": "exp2",
    },
    "exp2_imbalanced_max3": {
        "results_dir": RESULTS_ROOT / "exp2_imbalanced_max3",
        "regions_file": CONFIGS_ROOT / "regions_big_city_extreme_imbalanced.json",
        "label": "Exp 2 - Imbalanced (max3)",
        "algorithm": "nearest_region_pref",
        "balance": "imbalanced",
        "experiment": "exp2",
    },
    # -- Exp 3: nearest_distance_pref -----------------------------------------
    "exp3_balanced": {
        "results_dir": RESULTS_ROOT / "exp3_balanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_balanced.json",
        "label": "Exp 3 - Balanced",
        "algorithm": "nearest_distance_pref",
        "balance": "balanced",
        "experiment": "exp3",
    },
    "exp3_imbalanced": {
        "results_dir": RESULTS_ROOT / "exp3_imbalanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_extreme_imbalanced.json",
        "label": "Exp 3 - Imbalanced",
        "algorithm": "nearest_distance_pref",
        "balance": "imbalanced",
        "experiment": "exp3",
    },
    "exp3_balanced_max3": {
        "results_dir": RESULTS_ROOT / "exp3_balanced_max3",
        "regions_file": CONFIGS_ROOT / "regions_big_city_balanced.json",
        "label": "Exp 3 - Balanced (max3)",
        "algorithm": "nearest_distance_pref",
        "balance": "balanced",
        "experiment": "exp3",
    },
    "exp3_imbalanced_max3": {
        "results_dir": RESULTS_ROOT / "exp3_imbalanced_max3",
        "regions_file": CONFIGS_ROOT / "regions_big_city_extreme_imbalanced.json",
        "label": "Exp 3 - Imbalanced (max3)",
        "algorithm": "nearest_distance_pref",
        "balance": "imbalanced",
        "experiment": "exp3",
    },
    # -- Exp 4: nearest_passenger_pref ----------------------------------------
    "exp4_balanced": {
        "results_dir": RESULTS_ROOT / "exp4_balanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_balanced.json",
        "label": "Exp 4 - Balanced",
        "algorithm": "nearest_passenger_pref",
        "balance": "balanced",
        "experiment": "exp4",
    },
    "exp4_imbalanced": {
        "results_dir": RESULTS_ROOT / "exp4_imbalanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_extreme_imbalanced.json",
        "label": "Exp 4 - Imbalanced",
        "algorithm": "nearest_passenger_pref",
        "balance": "imbalanced",
        "experiment": "exp4",
    },
    # -- Exp 5: nearest_two_sided_region_pass_pref -----------------------------
    "exp5_balanced": {
        "results_dir": RESULTS_ROOT / "exp5_balanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_balanced.json",
        "label": "Exp 5 - Balanced",
        "algorithm": "nearest_two_sided_region_pass_pref",
        "balance": "balanced",
        "experiment": "exp5",
    },
    "exp5_imbalanced": {
        "results_dir": RESULTS_ROOT / "exp5_imbalanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_extreme_imbalanced.json",
        "label": "Exp 5 - Imbalanced",
        "algorithm": "nearest_two_sided_region_pass_pref",
        "balance": "imbalanced",
        "experiment": "exp5",
    },
    "exp5_balanced_max3": {
        "results_dir": RESULTS_ROOT / "exp5_balanced_max3",
        "regions_file": CONFIGS_ROOT / "regions_big_city_balanced.json",
        "label": "Exp 5 - Balanced (max3)",
        "algorithm": "nearest_two_sided_region_pass_pref",
        "balance": "balanced",
        "experiment": "exp5",
    },
    "exp5_imbalanced_max3": {
        "results_dir": RESULTS_ROOT / "exp5_imbalanced_max3",
        "regions_file": CONFIGS_ROOT / "regions_big_city_extreme_imbalanced.json",
        "label": "Exp 5 - Imbalanced (max3)",
        "algorithm": "nearest_two_sided_region_pass_pref",
        "balance": "imbalanced",
        "experiment": "exp5",
    },
    # -- Exp 6: safety_objective_two_sided ------------------------------------
    "exp6_balanced": {
        "results_dir": RESULTS_ROOT / "exp6_balanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_balanced.json",
        "label": "Exp 6 - Balanced",
        "algorithm": "safety_objective_two_sided",
        "balance": "balanced",
        "experiment": "exp6",
    },
    "exp6_imbalanced": {
        "results_dir": RESULTS_ROOT / "exp6_imbalanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_extreme_imbalanced.json",
        "label": "Exp 6 - Imbalanced",
        "algorithm": "safety_objective_two_sided",
        "balance": "imbalanced",
        "experiment": "exp6",
    },
    "exp6_balanced_max3": {
        "results_dir": RESULTS_ROOT / "exp6_balanced_max3",
        "regions_file": CONFIGS_ROOT / "regions_big_city_balanced.json",
        "label": "Exp 6 - Balanced (max3)",
        "algorithm": "safety_objective_two_sided",
        "balance": "balanced",
        "experiment": "exp6",
    },
    "exp6_imbalanced_max3": {
        "results_dir": RESULTS_ROOT / "exp6_imbalanced_max3",
        "regions_file": CONFIGS_ROOT / "regions_big_city_extreme_imbalanced.json",
        "label": "Exp 6 - Imbalanced (max3)",
        "algorithm": "safety_objective_two_sided",
        "balance": "imbalanced",
        "experiment": "exp6",
    },
    # -- Exp 7: safety_objective -----------------------------------------------
    "exp7_balanced": {
        "results_dir": RESULTS_ROOT / "exp7_balanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_balanced.json",
        "label": "Exp 7 - Balanced",
        "algorithm": "safety_objective",
        "balance": "balanced",
        "experiment": "exp7",
    },
    "exp7_imbalanced": {
        "results_dir": RESULTS_ROOT / "exp7_imbalanced",
        "regions_file": CONFIGS_ROOT / "regions_big_city_extreme_imbalanced.json",
        "label": "Exp 7 - Imbalanced",
        "algorithm": "safety_objective",
        "balance": "imbalanced",
        "experiment": "exp7",
    },
}


# ---------------------------------------------------------------------------
# Run-ID parsing
# ---------------------------------------------------------------------------

def parse_run_id(run_id: str) -> dict:
    """Extract parameter dict from a run-id string."""
    params: dict = {"run_id": run_id}

    m = re.search(r"_d_(\d+(?:\.\d+)?)(?:_|$)", run_id)
    if m:
        params["d"] = float(m.group(1))

    # R stored with underscores instead of decimal point: R_0_50 -> 0.50
    m = re.search(r"_R_(\d+)_(\d+)(?:_|$)", run_id)
    if m:
        params["R"] = float(f"{m.group(1)}.{m.group(2)}")

    m = re.search(r"_alg_(.+?)_geom_", run_id)
    if m:
        params["matching"] = m.group(1)

    m = re.search(r"_geom_(\d+)(?:_|$)", run_id)
    if m:
        params["geom"] = int(m.group(1))

    m = re.search(r"_behav_(\w+?)(?:_ic_|$)", run_id)
    if m:
        params["behaviour"] = m.group(1)

    m = re.search(r"_ic_(\w+?)(?:_reset|$)", run_id)
    if m:
        params["initial_conditions"] = m.group(1)

    # Replicate copies share an identical point and differ only by a _rep_NN
    # suffix (see make_replicate_configs.py).  is_rep lets the sweep keep
    # one run per point while the per-point tests pool all replicates.
    m = re.search(r"_rep_(\d+)(?:_|$)", run_id)
    params["replicate"] = int(m.group(1)) if m else None
    params["is_rep"] = m is not None

    return params


# ---------------------------------------------------------------------------
# Low-level file helpers
# ---------------------------------------------------------------------------

def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _last_nonempty_line(path: Path) -> Optional[str]:
    last = None
    with _open_text(path) as f:
        for line in f:
            if line.strip():
                last = line.strip()
    return last


# ---------------------------------------------------------------------------
# Inequality metrics (no external dependencies)
# ---------------------------------------------------------------------------

def gini(x) -> float:
    """Gini coefficient of array x (0 = perfectly equal, 1 = maximally unequal)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if len(x) == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    total = cumx[-1]
    if total == 0:
        return 0.0
    return float((n + 1 - 2 * np.sum(cumx) / total) / n)


def atkinson(x, epsilon: float = 1.0) -> float:
    """Atkinson index of array x (higher = more inequality, sensitive to lower tail)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.nan
    mu = float(np.mean(x))
    if mu == 0:
        return np.nan
    if abs(epsilon - 1.0) < 1e-9:
        geo_mean = float(np.exp(np.mean(np.log(x))))
        return 1.0 - geo_mean / mu
    power_mean = float(np.mean(x ** (1 - epsilon)) ** (1 / (1 - epsilon)))
    return 1.0 - power_mean / mu


def ratio_2020(x, pct: float = 20.0) -> float:
    """Ratio of mean in top-pct% to mean in bottom-pct%."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    upper_thres = np.percentile(x, 100 - pct)
    lower_thres = np.percentile(x, pct)
    upper_vals = x[x >= upper_thres]
    lower_vals = x[x <= lower_thres]
    if len(lower_vals) == 0 or float(np.mean(lower_vals)) == 0:
        return np.nan
    return float(np.mean(upper_vals)) / float(np.mean(lower_vals))


def calc_inequality(x) -> dict:
    """Return dict with gini, atkinson, ratio_2020 for array x."""
    return {"gini": gini(x), "atkinson": atkinson(x), "ratio_2020": ratio_2020(x)}


# ---------------------------------------------------------------------------
# Fast summary loading (one row per run)
# ---------------------------------------------------------------------------

def load_final_taxi_snapshot(ptm_path: Path) -> dict:
    """
    Read the last batch snapshot from per_taxi_metrics and return a flat dict
    of fleet-level summary stats (means, stds, inequality indices).
    """
    line = _last_nonempty_line(ptm_path)
    if line is None:
        return {}
    snap = json.loads(line)
    out: dict = {}

    def _extract(key, out_prefix):
        vals = [float(v) for v in snap.get(key, [])
                if v is not None and np.isfinite(float(v))]
        if vals:
            arr = np.array(vals)
            out[f"{out_prefix}_mean"] = float(np.mean(arr))
            out[f"{out_prefix}_std"] = float(np.std(arr))
            out[f"{out_prefix}_median"] = float(np.median(arr))
            out[f"{out_prefix}_p10"] = float(np.percentile(arr, 10))
            out[f"{out_prefix}_p90"] = float(np.percentile(arr, 90))
        return vals

    safety_vals = _extract("safety_score", "final_safety")
    sat_vals = _extract("satisfaction_score", "final_satisfaction")
    income_vals = _extract("trip_income", "final_income")

    if income_vals:
        arr = np.array(income_vals)
        out["income_gini"] = gini(arr)
        out["income_atkinson"] = atkinson(arr)
        out["income_ratio_2020"] = ratio_2020(arr)

    if safety_vals:
        out["safety_gini"] = gini(np.array(safety_vals))

    for key in ["time_serving", "time_waiting", "time_to_request",
                "time_cruising", "time_on_break"]:
        vals = [float(v) for v in snap.get(key, [])
                if v is not None and np.isfinite(float(v))]
        if vals:
            out[f"final_{key}_mean"] = float(np.mean(vals))

    return out


def load_service_stats(prm_path: Path) -> dict:
    """
    Compute service rate, wait times, and drop-reason counts from
    per_request_metrics.

    The file is an append-only log:
      - Intermediate lines: done/dropped requests flushed at the end of each
        batch (removed from memory after writing).
      - Final line: in-flight requests (pending/serving/waiting) that never
        finished; these are appended at simulation end.
    Each request appears in exactly one line, so we read all lines and
    concatenate - the last line must NOT be used alone.
    """
    requests = []
    with _open_text(prm_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            requests.extend(json.loads(line).get("requests", []))
    if not requests:
        return {}

    n_done = sum(1 for r in requests if r.get("mode") == "done")
    n_dropped = sum(1 for r in requests if r.get("mode") == "dropped")
    n_total = len(requests)

    reasons: dict = {}
    for r in requests:
        if r.get("mode") == "dropped":
            reason = r.get("cancellation_reason") or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1

    wait_assign, wait_pickup, trip_times, pickup_dists = [], [], [], []
    for r in requests:
        if r.get("mode") != "done":
            continue
        ts = r.get("timestamp")
        asgn = r.get("assignment")
        pkup = r.get("pickup")
        drop = r.get("dropoff")
        dist = r.get("assigned_taxi_distance")
        if ts is not None and asgn is not None:
            wait_assign.append(float(asgn) - float(ts))
        if ts is not None and pkup is not None:
            wait_pickup.append(float(pkup) - float(ts))
        if pkup is not None and drop is not None:
            trip_times.append(float(drop) - float(pkup))
        if dist is not None:
            pickup_dists.append(float(dist))

    out: dict = {
        "n_requests": n_total,
        "n_done": n_done,
        "n_dropped": n_dropped,
        "service_rate": n_done / n_total if n_total else None,
        **{f"drop_{k}": v for k, v in reasons.items()},
    }

    for col_name, values in [
        ("wait_to_assign", wait_assign),
        ("wait_to_pickup", wait_pickup),
        ("trip_time", trip_times),
        ("pickup_dist", pickup_dists),
    ]:
        if values:
            arr = np.array(values)
            out[f"{col_name}_mean"] = float(np.mean(arr))
            out[f"{col_name}_p25"]  = float(np.percentile(arr, 25))
            out[f"{col_name}_p50"]  = float(np.percentile(arr, 50))
            out[f"{col_name}_p75"]  = float(np.percentile(arr, 75))
            out[f"{col_name}_p90"]  = float(np.percentile(arr, 90))

    return out


def load_run_summaries(
    results_dir: Path,
    pattern: str = "run_*_aggregates.csv.gz",
    load_taxi_snapshot: bool = True,
    load_request_stats: bool = True,
    include_reps: bool = False,
) -> pd.DataFrame:
    """
    Load one summary row per run from a results directory.
    Reads last row of aggregates + optionally last-line snapshots of per_taxi
    and per_request files.  Adds fleet utilization and parsed parameter columns.

    Replicate runs (filename `..._rep_NN`, see make_replicate_configs.py) share a
    point with the x1 run and are EXCLUDED by default so that callers which assume
    one row per point do not double-count.  Pass include_reps=True to get them (the
    per-point significance tests do this); the original x1 run counts as a replicate.
    """
    agg_files = sorted(Path(results_dir).glob(pattern))
    if not agg_files:
        print(f"  No files matching '{pattern}' in {results_dir}")
        return pd.DataFrame()

    rows = []
    for agg_path in agg_files:
        run_id = re.sub(r"^run_", "", re.sub(r"_aggregates\.csv\.gz$", "", agg_path.name))
        try:
            agg = pd.read_csv(agg_path, index_col=0)
            if agg.empty:
                continue
            row = agg.iloc[-1].to_dict()
        except Exception as e:
            print(f"  {agg_path.name}: {e}")
            continue

        row.update(parse_run_id(run_id))

        if load_taxi_snapshot:
            ptm = agg_path.with_name(agg_path.name.replace("_aggregates.csv.gz",
                                                             "_per_taxi_metrics.json.gz"))
            if ptm.exists():
                row.update(load_final_taxi_snapshot(ptm))

        if load_request_stats:
            prm = agg_path.with_name(agg_path.name.replace("_aggregates.csv.gz",
                                                             "_per_request_metrics.json.gz"))
            if prm.exists():
                row.update(load_service_stats(prm))

        # Fleet utilization: fraction of total tracked time spent en-route+serving
        time_cols = ["avg_time_serving", "avg_time_waiting", "avg_time_to_request",
                     "avg_time_cruising", "avg_time_on_break"]
        present = [c for c in time_cols if c in row and row[c] is not None
                   and np.isfinite(float(row[c]))]
        if present:
            total = sum(float(row[c]) for c in present)
            if total > 0 and "avg_time_serving" in row:
                row["utilization"] = float(row["avg_time_serving"]) / total

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ["d", "R", "geom"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if not include_reps and "is_rep" in df.columns:
        df = df[~df["is_rep"].astype(bool)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Full data loading (for deep-dive sections)
# ---------------------------------------------------------------------------

def load_request_df(prm_path: Path) -> pd.DataFrame:
    """
    Load all requests from per_request_metrics into a tidy DataFrame.
    Reads every snapshot line and deduplicates by request_id (keeps last state).
    """
    latest: dict = {}
    with _open_text(prm_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for req in json.loads(line).get("requests", []):
                rid = req.get("request_id")
                if rid is not None:
                    latest[rid] = req

    def _coord(point):
        if isinstance(point, (list, tuple)) and len(point) == 2:
            return point[0], point[1]
        return None, None

    rows = []
    for req in latest.values():
        ox, oy = _coord(req.get("origin"))
        dx, dy = _coord(req.get("destination"))
        rows.append({
            "request_id":                   req.get("request_id"),
            "mode":                          req.get("mode"),
            "cancellation_reason":           req.get("cancellation_reason"),
            "timestamp":                     req.get("timestamp"),
            "assignment":                    req.get("assignment"),
            "pickup":                        req.get("pickup"),
            "dropoff":                       req.get("dropoff"),
            "taxi_id":                       req.get("taxi_id"),
            "assigned_taxi_distance":        req.get("assigned_taxi_distance"),
            "ox": ox, "oy": oy, "dx": dx, "dy": dy,
            "driver_safety_score_start":     req.get("driver_safety_score_start"),
            "driver_safety_score_end":       req.get("driver_safety_score_end"),
            "driver_average_safety_score":   req.get("driver_average_safety_score"),
            "driver_safety_score_pickup":    req.get("driver_safety_score_pickup"),
        })

    return pd.DataFrame(rows)


def build_wait_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add wait_to_assign, wait_to_pickup, assigned_to_pickup, trip_time columns."""
    df = df.copy()
    for col in ["timestamp", "assignment", "pickup", "dropoff"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["wait_to_assign"]     = df["assignment"] - df["timestamp"]
    df["wait_to_pickup"]     = df["pickup"]     - df["timestamp"]
    df["assigned_to_pickup"] = df["pickup"]     - df["assignment"]
    df["trip_time"]          = df["dropoff"]    - df["pickup"]
    return df


def load_per_taxi_batches(ptm_path: Path) -> List[dict]:
    """Load all batch snapshots from per_taxi_metrics as a list of dicts."""
    rows = []
    with _open_text(ptm_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_fleet_timeseries(batches: List[dict]) -> pd.DataFrame:
    """
    Build a wide per-batch DataFrame with fleet-level mean/std/median for each
    scalar metric.  Much faster than building a full per-taxi timeseries.
    """
    SCALAR_METRICS = [
        "safety_score", "satisfaction_score", "satisfaction_delta",
        "trip_income", "trip_num_completed",
        "time_serving", "time_to_request", "time_cruising",
        "time_waiting", "time_on_break",
    ]
    records = []
    for batch in batches:
        ts = batch.get("timestamp")
        if ts is None:
            continue
        rec: dict = {"timestamp": int(ts)}
        for m in SCALAR_METRICS:
            vals = [float(v) for v in batch.get(m, [])
                    if v is not None and np.isfinite(float(v))]
            if vals:
                rec[f"{m}_mean"]   = float(np.mean(vals))
                rec[f"{m}_std"]    = float(np.std(vals))
                rec[f"{m}_median"] = float(np.median(vals))
        on_break = batch.get("on_break", [])
        if on_break:
            rec["break_share"] = float(sum(bool(b) for b in on_break) / len(on_break))
        records.append(rec)
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Region utilities
# ---------------------------------------------------------------------------

def load_regions(path: Path) -> dict:
    """Parse a regions JSON config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def assign_region(x, y, regions: list) -> Optional[str]:
    """Return the region id for coordinate (x, y), or None if unassigned."""
    if x is None or y is None:
        return None
    try:
        xi, yi = int(x), int(y)
    except (TypeError, ValueError):
        return None
    for r in regions:
        if r["x_min"] <= xi <= r["x_max"] and r["y_min"] <= yi <= r["y_max"]:
            return r["id"]
    return None


def tag_requests_with_regions(df: pd.DataFrame, regions: list) -> pd.DataFrame:
    """Add origin_region and destination_region columns to a request DataFrame."""
    df = df.copy()
    df["origin_region"] = [
        assign_region(x, y, regions)
        for x, y in zip(df.get("ox", pd.Series(dtype=object)),
                        df.get("oy", pd.Series(dtype=object)))
    ]
    df["destination_region"] = [
        assign_region(x, y, regions)
        for x, y in zip(df.get("dx", pd.Series(dtype=object)),
                        df.get("dy", pd.Series(dtype=object)))
    ]
    return df


def aggregate_region_stats(
    df: pd.DataFrame,
    regions: list,
    region_col: str,
    metric_cols: List[str],
) -> pd.DataFrame:
    """
    Compute per-region mean, std, count, and p90 for each metric in metric_cols.
    Returns one row per region (including regions with no data: count=0, mean=NaN).
    """
    valid = df.dropna(subset=[region_col])
    grouped = valid.groupby(region_col)

    rows = []
    for region in regions:
        rid = region["id"]
        row: dict = {
            "region_id":   rid,
            "region_name": region["name"],
            "x_min": region["x_min"], "x_max": region["x_max"],
            "y_min": region["y_min"], "y_max": region["y_max"],
        }
        if rid in grouped.groups:
            grp = grouped.get_group(rid)
            row["count"] = len(grp)
            # drop_rate counts only mode == "dropped"; non_completion_rate counts
            # any mode != "done".  The patience limit (max_request_waiting_time) is
            # now enforced for every matching algorithm (fixed in commit e683500),
            # so both rates are comparable across algorithms on post-fix data.
            row["drop_rate"] = (
                (grp["mode"] == "dropped").mean()
                if "mode" in grp.columns else np.nan
            )
            row["non_completion_rate"] = (
                (grp["mode"] != "done").mean()
                if "mode" in grp.columns else np.nan
            )
            for col in metric_cols:
                if col in grp.columns:
                    vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                    row[f"mean_{col}"] = float(vals.mean())  if len(vals) else np.nan
                    row[f"std_{col}"]  = float(vals.std())   if len(vals) > 1 else 0.0
                    row[f"p90_{col}"]  = float(vals.quantile(0.9)) if len(vals) else np.nan
        else:
            row["count"] = 0
            row["drop_rate"] = np.nan
            row["non_completion_rate"] = np.nan
            for col in metric_cols:
                row[f"mean_{col}"] = np.nan
                row[f"std_{col}"]  = np.nan
                row[f"p90_{col}"]  = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def compute_regional_gini(region_stats_df: pd.DataFrame, metric_col: str) -> float:
    """Gini of per-region means - a scalar measure of spatial inequality."""
    means = region_stats_df[metric_col].dropna().values
    return gini(means)


# ---------------------------------------------------------------------------
# Request-based regional safety
# ---------------------------------------------------------------------------
#
# The aggregates column region_safety_avg_<id> is a TAXI-POSITION metric: the
# simulation averages the safety score of taxis physically standing in each
# region, sampled over time.  It is only written by the region-aware algorithms
# (Exp2/5/6/7), so it cannot enter an all-experiment comparison.
#
# The request-based metric below instead tags each request by the region of its
# origin (or destination) and averages a per-request driver-safety field over the
# requests served from that region.  Every algorithm logs per_request_metrics, so
# this is computable for ALL seven experiments on ONE comparable scale.  These
# two numbers are NOT interchangeable (different basis, different scale).

REGION_REQ_SCORE_FIELD = "driver_average_safety_score"
REGION_REQ_POINT = "origin"  # "origin" (pickup area) or "destination" (drop-off area)
_SCORE_FIELD_SHORT = {
    "driver_average_safety_score": "avg",
    "driver_safety_score_pickup": "pickup",
    "driver_safety_score_start": "start",
    "driver_safety_score_end": "end",
}


def region_request_safety_means(
    prm_path: Path,
    regions: list,
    score_field: str = REGION_REQ_SCORE_FIELD,
    point: str = REGION_REQ_POINT,
) -> Dict[str, float]:
    """
    Per-region mean of *score_field* over the requests whose *point* (``origin``
    or ``destination``) falls in that region.  Requests are deduplicated by
    request_id (last state wins); regions with no scored request map to NaN.
    """
    if point not in ("origin", "destination"):
        raise ValueError("point must be 'origin' or 'destination'")

    latest: dict = {}
    with _open_text(prm_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for req in json.loads(line).get("requests", []):
                rid = req.get("request_id")
                if rid is not None:
                    latest[rid] = req

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for req in latest.values():
        pt = req.get(point)
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            continue
        region_id = assign_region(pt[0], pt[1], regions)
        if region_id is None:
            continue
        sc = req.get(score_field)
        if sc is None:
            continue
        sc = float(sc)
        if not np.isfinite(sc):
            continue
        sums[region_id] = sums.get(region_id, 0.0) + sc
        counts[region_id] = counts.get(region_id, 0) + 1

    return {
        r["id"]: (sums[r["id"]] / counts[r["id"]]) if counts.get(r["id"]) else np.nan
        for r in regions
    }


# Per-region service quality alongside safety, on the same request-based basis.
# region_request_safety_means above covers only driver safety; the regional
# fairness picture also needs wait-to-pickup and service rate per region, so they
# are computed here in one pass over the same per_request file.  Wait and service
# follow the run-level definitions in compute_request_stats: wait_to_pickup is
# (pickup - request timestamp) over served requests only, service rate is
# served / total requests tagged to the region.
REGION_REQ_FAMILIES = ("safety", "wait", "service")
# Which tail is the "worst" region for each family: higher safety / service is
# better (worst = min), lower wait is better (worst = max).
REGION_REQ_WORST = {"safety": "min", "wait": "max", "service": "min"}


def region_request_metrics(
    prm_path: Path,
    regions: list,
    score_field: str = REGION_REQ_SCORE_FIELD,
    point: str = REGION_REQ_POINT,
) -> Dict[str, Dict[str, float]]:
    """
    Per-region request-based driver safety, wait-to-pickup, and service rate in a
    single pass over *prm_path*.  Requests are deduplicated by request_id (last
    state wins) and tagged to a region by their *point* (``origin`` or
    ``destination``).  Returns ``{"safety": {rid: mean}, "wait": {rid: mean},
    "service": {rid: rate}}``; regions with no qualifying request map to NaN.

    safety  - mean of *score_field* over requests carrying a finite score.
    wait    - mean of (pickup - timestamp) over served (mode == "done") requests.
    service - served / total requests tagged to the region.
    """
    if point not in ("origin", "destination"):
        raise ValueError("point must be 'origin' or 'destination'")

    latest: dict = {}
    with _open_text(prm_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for req in json.loads(line).get("requests", []):
                rid = req.get("request_id")
                if rid is not None:
                    latest[rid] = req

    safety_sum: Dict[str, float] = {}; safety_cnt: Dict[str, int] = {}
    wait_sum: Dict[str, float] = {};   wait_cnt: Dict[str, int] = {}
    done: Dict[str, int] = {};         total: Dict[str, int] = {}
    for req in latest.values():
        pt = req.get(point)
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            continue
        region_id = assign_region(pt[0], pt[1], regions)
        if region_id is None:
            continue
        total[region_id] = total.get(region_id, 0) + 1
        served = req.get("mode") == "done"
        if served:
            done[region_id] = done.get(region_id, 0) + 1

        sc = req.get(score_field)
        if sc is not None:
            sc = float(sc)
            if np.isfinite(sc):
                safety_sum[region_id] = safety_sum.get(region_id, 0.0) + sc
                safety_cnt[region_id] = safety_cnt.get(region_id, 0) + 1

        if served:
            ts = req.get("timestamp"); pk = req.get("pickup")
            if ts is not None and pk is not None:
                w = float(pk) - float(ts)
                if np.isfinite(w):
                    wait_sum[region_id] = wait_sum.get(region_id, 0.0) + w
                    wait_cnt[region_id] = wait_cnt.get(region_id, 0) + 1

    safety = {r["id"]: (safety_sum[r["id"]] / safety_cnt[r["id"]])
              if safety_cnt.get(r["id"]) else np.nan for r in regions}
    wait = {r["id"]: (wait_sum[r["id"]] / wait_cnt[r["id"]])
            if wait_cnt.get(r["id"]) else np.nan for r in regions}
    service = {r["id"]: (done.get(r["id"], 0) / total[r["id"]])
               if total.get(r["id"]) else np.nan for r in regions}
    return {"safety": safety, "wait": wait, "service": service}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_region_heatmap(
    ax,
    region_stats_df: pd.DataFrame,
    metric_col: str,
    regions_cfg: dict,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt: str = ".2f",
    show_count: bool = True,
    colorbar: bool = True,
    show_labels: bool = True,
    label_fontsize: float = 12,
    square: bool = True,
) -> None:
    """
    Draw a region heatmap on an existing Axes.  region_stats_df must contain
    region_id, region_name, x_min/x_max/y_min/y_max and the metric_col column;
    metric_col is the column to colour by; regions_cfg is a dict with a 'grid'
    key (n, m).
    """
    n = regions_cfg["grid"]["n"]
    m = regions_cfg["grid"]["m"]
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, m - 0.5)
    if square:
        ax.set_aspect("equal", "box")
    else:
        # fill the whole axes cell (no square-aspect margins); the grid is square
        # so the geometric distortion is negligible and the panels pack with no gaps
        ax.set_aspect("auto")
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

    vals = region_stats_df[metric_col].dropna()
    if vmin is None:
        vmin = float(vals.min()) if not vals.empty else 0.0
    if vmax is None:
        vmax = float(vals.max()) if not vals.empty else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for _, row in region_stats_df.iterrows():
        width  = row["x_max"] - row["x_min"] + 1
        height = row["y_max"] - row["y_min"] + 1
        val = row[metric_col]
        if pd.isna(val):
            facecolor  = "lightgray"
            label_val  = "NA"
        else:
            facecolor = cmap_obj(norm(float(val)))
            label_val = f"{float(val):{fmt}}"

        ax.add_patch(patches.Rectangle(
            (row["x_min"] - 0.5, row["y_min"] - 0.5), width, height,
            linewidth=1.0, edgecolor="black", facecolor=facecolor, alpha=0.85,
        ))
        if show_labels:
            import textwrap
            cx = (row["x_min"] + row["x_max"]) / 2.0
            cy = (row["y_min"] + row["y_max"]) / 2.0
            count_str = f"\n({int(row['count'])})" if show_count and "count" in row else ""
            region_name = textwrap.fill(row.get("region_name", ""), width=15)
            ax.text(cx, cy, f"{region_name}\n{label_val}{count_str}",
                    ha="center", va="center", fontsize=label_fontsize)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)


# ---------------------------------------------------------------------------
# Convenience selectors
# ---------------------------------------------------------------------------

def get_representative_runs(
    df: pd.DataFrame,
    n_per_combo: int = 1,
    fix_cols: Optional[List[str]] = None,
    vary_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Select representative runs for deep-dive analysis.

    By default selects one run per (R, behaviour, initial_conditions)
    combination, preferring the median fleet-size value.  Returns a
    filtered DataFrame.
    """
    if fix_cols is None:
        fix_cols = []
    if vary_cols is None:
        vary_cols = ["R", "behaviour", "initial_conditions"]

    present_vary = [c for c in vary_cols if c in df.columns]
    if not present_vary:
        return df.head(n_per_combo)

    groups = df.groupby(present_vary)
    selected = []
    for _, grp in groups:
        if "d" in grp.columns:
            # prefer the run closest to the median d
            med_d = grp["d"].median()
            idx = (grp["d"] - med_d).abs().nsmallest(n_per_combo).index
            selected.append(grp.loc[idx])
        else:
            selected.append(grp.head(n_per_combo))
    return pd.concat(selected).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Display helpers (no Jinja2 / pandas Styler required)
# ---------------------------------------------------------------------------

def fmt_display(
    df: pd.DataFrame,
    fmt: Optional[Dict[str, str]] = None,
    na_rep: str = "-",
) -> pd.DataFrame:
    """Return a display copy of *df* with numeric columns formatted as strings.

    Drop-in replacement for ``df.style.format(fmt, na_rep=na_rep)`` that does
    not require Jinja2.  Background-gradient styling is silently omitted.  *fmt*
    maps a column name to a Python format spec (e.g. '{:.3f}') and *na_rep* is
    the string used for missing values.
    """
    out = df.copy().astype(object)
    for col, spec in (fmt or {}).items():
        if col not in out.columns:
            continue
        out[col] = out[col].apply(
            lambda v, s=spec: na_rep if (v is None or (isinstance(v, float) and pd.isna(v))) else s.format(v)
        )
    # Replace any remaining NaN / None with na_rep
    for col in out.columns:
        out[col] = out[col].apply(
            lambda v: na_rep if (v is None or (isinstance(v, float) and pd.isna(v))) else v
        )
    return out


def match_run_pairs(
    df_exp1: pd.DataFrame,
    df_exp2: pd.DataFrame,
    keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Inner-join exp1 and exp2 summary DataFrames on matching parameter keys.
    Returns one row per matched pair with _exp1 / _exp2 suffixes on metric cols.
    """
    if keys is None:
        keys = ["d", "R", "behaviour", "initial_conditions"]
    keys = [k for k in keys if k in df_exp1.columns and k in df_exp2.columns]

    merged = pd.merge(
        df_exp1, df_exp2,
        on=keys,
        how="inner",
        suffixes=("_exp1", "_exp2"),
        validate="1:1",
    )
    return merged


# ---------------------------------------------------------------------------
# Cross-experiment loading (one DataFrame spanning multiple GROUP_CONFIGS)
# ---------------------------------------------------------------------------

def load_experiment_set(
    geometry: str,
    cache_dir: Path = Path("cache"),
    experiments: Optional[List[str]] = None,
    include_max3: bool = False,
    include_reps: bool = False,
    force: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Load every GROUP_CONFIGS entry for one *geometry* ("balanced"/"imbalanced")
    into a single long DataFrame, one row per run.  Adds the descriptive
    columns ``group``, ``algorithm``, ``experiment``, ``geometry``, ``label``
    from GROUP_CONFIGS so plots can label/legend by algorithm directly.

    *geometry* matches the ``balance`` field in GROUP_CONFIGS and *cache_dir* is
    the pickle cache directory (relative to CWD, i.e. notebooks/).  *experiments*
    optionally whitelists experiment ids (e.g. ['exp1','exp2']); None loads every
    experiment for the geometry.  *include_max3* adds the *_max3 variant groups
    (default False).  *include_reps* keeps the replicate runs (`_rep_NN`); the
    default False keeps one row per point for the sweep/paired analysis.  *force*
    is forwarded to load_run_summaries_cached to ignore the cache, and any other
    keyword args go to load_run_summaries (e.g. load_request_stats).

    Groups whose results_dir does not exist (not yet run) are skipped silently;
    the set of loaded experiments is reported via the returned frame's
    ``experiment`` column.
    """
    frames = []
    for group, cfg in GROUP_CONFIGS.items():
        if cfg.get("balance") != geometry:
            continue
        if not include_max3 and group.endswith("_max3"):
            continue
        if experiments is not None and cfg.get("experiment") not in experiments:
            continue
        results_dir = Path(cfg["results_dir"])
        if not results_dir.exists():
            continue
        sub = load_run_summaries_cached(
            results_dir, cache_dir, group, force=force,
            include_reps=include_reps, **kwargs
        )
        if sub.empty:
            print(f"  [skip] {group}: empty summary")
            continue
        sub = sub.copy()
        sub["group"] = group
        sub["algorithm"] = cfg.get("algorithm")
        sub["experiment"] = cfg.get("experiment")
        sub["geometry"] = cfg.get("balance")
        sub["label"] = cfg.get("label")
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_ceiling_fraction(
    df: pd.DataFrame,
    floor_exp: str,
    ceiling_exp: str,
    metric: str,
    keys: Optional[List[str]] = None,
    min_gap: float = 0.0,
) -> pd.DataFrame:
    """
    For each run, express *metric* as a fraction of the way from a floor
    experiment to a ceiling experiment, matched on *keys*:

        ceiling_fraction = (value - floor) / (ceiling - floor)

    where ``floor`` is the floor_exp value and ``ceiling`` is the ceiling_exp
    value at the same (d, R, behaviour, initial_conditions) operating point.

    Returns *df* with three columns added: ``floor``, ``ceiling``,
    ``ceiling_fraction``.  Rows whose |ceiling - floor| <= *min_gap* get NaN
    ceiling_fraction (the normalisation is too noisy to be meaningful there).

    *df* must contain an ``experiment`` column (as produced by
    load_experiment_set).
    """
    if keys is None:
        keys = ["d", "R", "behaviour", "initial_conditions"]
    keys = [k for k in keys if k in df.columns]

    floor = (
        df[df["experiment"] == floor_exp][keys + [metric]]
        .rename(columns={metric: "floor"})
    )
    ceil = (
        df[df["experiment"] == ceiling_exp][keys + [metric]]
        .rename(columns={metric: "ceiling"})
    )
    bounds = floor.merge(ceil, on=keys, how="inner")

    out = df.merge(bounds, on=keys, how="left")
    gap = out["ceiling"] - out["floor"]
    out["ceiling_fraction"] = (out[metric] - out["floor"]) / gap
    out.loc[gap.abs() <= min_gap, "ceiling_fraction"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Pickle-cache helpers
# ---------------------------------------------------------------------------

def _cache_valid(cache_file: Path, results_dir: Path) -> bool:
    # Valid only if the cache is at least as new as every run_* file it derives
    # from; otherwise a regenerated result dir would silently serve stale rows.
    cache_file = Path(cache_file)
    if not cache_file.exists():
        return False
    cache_mtime = cache_file.stat().st_mtime
    # resolve() normalises the leading ../ so long run_* filenames stay under the
    # Windows MAX_PATH limit; without it stat() on the relative path can raise.
    results_dir = Path(results_dir).resolve()

    def _mtimes():
        for p in results_dir.glob("run_*"):
            try:
                yield p.stat().st_mtime
            except OSError:
                continue

    newest = max(_mtimes(), default=0.0)
    return cache_mtime >= newest


def _save_pickle(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_run_summaries_cached(
    results_dir: Path,
    cache_dir: Path,
    group: str,
    force: bool = False,
    include_reps: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Cached wrapper for load_run_summaries.

    On the first call the full ~6-minute parse runs and the result is written
    to ``<cache_dir>/<group>_summary.pkl``.  On subsequent calls the pickle is
    returned instantly, unless *force=True* or any ``run_*`` file in
    *results_dir* is newer than the cache.

    The pickle always stores every run, replicates included; the include_reps
    filter is applied on return, so both views are served from one cache.
    *group* is the cache-file stem (e.g. 'exp1_balanced'); any other keyword
    args are forwarded to load_run_summaries.
    """
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        if not include_reps and "is_rep" in df.columns:
            return df[~df["is_rep"].astype(bool)].reset_index(drop=True)
        return df

    cache_file = Path(cache_dir) / f"{group}_summary.pkl"
    if not force and _cache_valid(cache_file, results_dir):
        print(f"[cache HIT]  {cache_file.name}  ->  loading summary")
        return _filter(_load_pickle(cache_file))

    reason = "forced" if force else "cache miss / stale"
    print(f"[cache MISS] {reason} - computing run summaries (this takes a few minutes)...")
    df = load_run_summaries(results_dir, include_reps=True, **kwargs)
    _save_pickle(df, cache_file)
    print(f"[cache SAVE] {cache_file}")
    return _filter(df)


def load_fleet_ts_cached(
    results_dir: Path,
    rep_runs: pd.DataFrame,
    cache_dir: Path,
    group: str,
    force: bool = False,
) -> dict:
    """
    Cached wrapper that builds the ``fleet_ts`` dict (run_id -> (label, timeseries
    DataFrame)) used in the Worktime & Break section.

    The cache key is ``<cache_dir>/<group>_fleet_ts.pkl``.
    """
    cache_file = Path(cache_dir) / f"{group}_fleet_ts.pkl"
    if not force and _cache_valid(cache_file, results_dir):
        print(f"[cache HIT]  {cache_file.name}  ->  loading fleet timeseries")
        return _load_pickle(cache_file)

    reason = "forced" if force else "cache miss / stale"
    print(f"[cache MISS] {reason} - loading per-taxi batches for {len(rep_runs)} runs...")
    fleet_ts: dict = {}
    for _, row in rep_runs.iterrows():
        run_id = row["run_id"]
        ptm_path = Path(results_dir) / f"run_{run_id}_per_taxi_metrics.json.gz"
        if not ptm_path.exists():
            print(f"  Missing: {ptm_path.name}")
            continue
        batches = load_per_taxi_batches(ptm_path)
        ts = build_fleet_timeseries(batches)
        label = f"R={row.get('R', '?')} {row.get('behaviour', '')}"
        fleet_ts[run_id] = (label, ts)
        print(f"  {label}: {len(batches)} batches")

    _save_pickle(fleet_ts, cache_file)
    print(f"[cache SAVE] {cache_file}")
    return fleet_ts


def load_regional_data_cached(
    results_dir: Path,
    rep_regional: pd.DataFrame,
    regions: list,
    cache_dir: Path,
    group: str,
    force: bool = False,
) -> dict:
    """
    Cached wrapper that builds the ``regional_data`` dict
    (run_id -> (row_info, req_df)) used in the Regional Analysis section.

    The cache key is ``<cache_dir>/<group>_regional.pkl``.
    """
    cache_file = Path(cache_dir) / f"{group}_regional.pkl"
    if not force and _cache_valid(cache_file, results_dir):
        print(f"[cache HIT]  {cache_file.name}  ->  loading regional data")
        return _load_pickle(cache_file)

    reason = "forced" if force else "cache miss / stale"
    print(f"[cache MISS] {reason} - loading full per-request data for {len(rep_regional)} runs...")
    regional_data: dict = {}
    for _, row_info in rep_regional.iterrows():
        run_id = row_info["run_id"]
        prm_path = Path(results_dir) / f"run_{run_id}_per_request_metrics.json.gz"
        if not prm_path.exists():
            print(f"  Missing: {prm_path.name}")
            continue
        req_df = load_request_df(prm_path)
        req_df = build_wait_metrics(req_df)
        req_df = tag_requests_with_regions(req_df, regions)
        regional_data[run_id] = (row_info, req_df)
        label = f"R={row_info.get('R', '?')} {row_info.get('behaviour', '')}"
        n_done    = int((req_df["mode"] == "done").sum())    if "mode" in req_df.columns else len(req_df)
        n_dropped = int((req_df["mode"] == "dropped").sum()) if "mode" in req_df.columns else 0
        print(f"  {label}: {len(req_df)} requests, {n_done} done, {n_dropped} dropped")

    _save_pickle(regional_data, cache_file)
    print(f"[cache SAVE] {cache_file}")
    return regional_data


def load_region_req_cached(
    results_dir: Path,
    cache_dir: Path,
    group: str,
    regions: list,
    score_field: str = REGION_REQ_SCORE_FIELD,
    point: str = REGION_REQ_POINT,
    force: bool = False,
) -> pd.DataFrame:
    """
    One row per run (replicates included) of request-based regional safety:
    parses every ``run_*_per_request_metrics`` file in *results_dir*, tags each
    request to a region by its *point*, and reduces the per-region means to the
    three scalars used in the significance report:

      region_safety_req_mean - unweighted mean across regions
      region_safety_req_min  - worst-served region
      region_safety_req_gini - spatial inequality (lower = fairer)

    Per-region columns (``region_safety_req_<id>``) are also kept for transparency.
    The cache key includes *point* and *score_field* so changing either rebuilds
    rather than serving a stale metric.
    """
    # resolve() normalises the leading ../ so the longest run ids (safety_objective
    # reps) stay under the Windows MAX_PATH limit for glob()/open().
    results_dir = Path(results_dir).resolve()
    short = _SCORE_FIELD_SHORT.get(score_field, score_field)
    cache_file = Path(cache_dir) / f"{group}_region_req_{point}_{short}.pkl"
    if not force and _cache_valid(cache_file, results_dir):
        print(f"[cache HIT]  {cache_file.name}  ->  loading request-based regional safety")
        return _load_pickle(cache_file)

    reason = "forced" if force else "cache miss / stale"
    print(f"[cache MISS] {reason} - computing request-based regional safety "
          f"({point}/{short}) for {group}...")
    rows = []
    for prm in sorted(results_dir.glob("run_*_per_request_metrics.json.gz")):
        run_id = re.sub(r"^run_", "",
                        re.sub(r"_per_request_metrics\.json\.gz$", "", prm.name))
        means = region_request_safety_means(prm, regions, score_field, point)
        row: dict = {"run_id": run_id}
        row.update(parse_run_id(run_id))
        vals = np.array(list(means.values()), dtype=float)
        finite = vals[np.isfinite(vals)]
        row["region_safety_req_mean"] = float(np.mean(finite)) if finite.size else np.nan
        row["region_safety_req_min"] = float(np.min(finite)) if finite.size else np.nan
        row["region_safety_req_gini"] = gini(finite)
        for rid, v in means.items():
            row[f"region_safety_req_{rid}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["d", "R", "geom"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    _save_pickle(df, cache_file)
    print(f"[cache SAVE] {cache_file}  ({len(df)} runs)")
    return df


def load_region_req_set(
    geometry: str,
    cache_dir: Path = Path("cache"),
    experiments: Optional[List[str]] = None,
    score_field: str = REGION_REQ_SCORE_FIELD,
    point: str = REGION_REQ_POINT,
    include_reps: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """
    Request-based regional safety for every experiment of one *geometry*, stacked
    into one long DataFrame (mirrors load_experiment_set).  Adds the
    descriptive ``group``/``algorithm``/``experiment``/``geometry``/``label``
    columns so the significance helpers can group by ``experiment``.

    Each group's per-run scalars come from load_region_req_cached (its own
    regions file).  ``include_reps`` keeps/drops the ``_rep_NN`` runs exactly like
    load_experiment_set (the sweep wants one run per point; the focus
    replicate test wants them all).
    """
    frames = []
    for group, cfg in GROUP_CONFIGS.items():
        if cfg.get("balance") != geometry:
            continue
        if group.endswith("_max3"):
            continue
        if experiments is not None and cfg.get("experiment") not in experiments:
            continue
        results_dir = Path(cfg["results_dir"])
        if not results_dir.exists():
            continue
        regions_file = cfg.get("regions_file")
        if regions_file is None or not Path(regions_file).exists():
            print(f"  [skip] {group}: no regions file")
            continue
        regions = load_regions(regions_file).get("regions", [])
        sub = load_region_req_cached(
            results_dir, cache_dir, group, regions,
            score_field=score_field, point=point, force=force,
        )
        if sub.empty:
            continue
        sub = sub.copy()
        sub["group"] = group
        sub["algorithm"] = cfg.get("algorithm")
        sub["experiment"] = cfg.get("experiment")
        sub["geometry"] = cfg.get("balance")
        sub["label"] = cfg.get("label")
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if not include_reps and "is_rep" in out.columns:
        out = out[~out["is_rep"].astype(bool)].reset_index(drop=True)
    return out


def load_region_req_metrics_cached(
    results_dir: Path,
    cache_dir: Path,
    group: str,
    regions: list,
    score_field: str = REGION_REQ_SCORE_FIELD,
    point: str = REGION_REQ_POINT,
    force: bool = False,
) -> pd.DataFrame:
    """
    One row per run (replicates included) of request-based regional safety, wait,
    and service rate (see region_request_metrics).  Each family is reduced
    to three scalars -- ``region_<fam>_req_mean`` (unweighted regional mean),
    ``_worst`` (worst region, direction-aware via REGION_REQ_WORST), and
    ``_gini`` (spatial inequality, lower is fairer) -- and the per-region values
    are kept as ``region_<fam>_req_<id>`` for transparency.  Cached separately
    from load_region_req_cached so the safety-only cache is untouched.
    """
    results_dir = Path(results_dir).resolve()
    short = _SCORE_FIELD_SHORT.get(score_field, score_field)
    cache_file = Path(cache_dir) / f"{group}_region_req_metrics_{point}_{short}.pkl"
    if not force and _cache_valid(cache_file, results_dir):
        print(f"[cache HIT]  {cache_file.name}  ->  loading request-based regional metrics")
        return _load_pickle(cache_file)

    reason = "forced" if force else "cache miss / stale"
    print(f"[cache MISS] {reason} - computing request-based regional metrics "
          f"({point}/{short}) for {group}...")
    rows = []
    for prm in sorted(results_dir.glob("run_*_per_request_metrics.json.gz")):
        run_id = re.sub(r"^run_", "",
                        re.sub(r"_per_request_metrics\.json\.gz$", "", prm.name))
        m = region_request_metrics(prm, regions, score_field, point)
        row: dict = {"run_id": run_id}
        row.update(parse_run_id(run_id))
        for fam in REGION_REQ_FAMILIES:
            prefix = f"region_{fam}_req"
            vals = np.array(list(m[fam].values()), dtype=float)
            finite = vals[np.isfinite(vals)]
            row[f"{prefix}_mean"] = float(np.mean(finite)) if finite.size else np.nan
            if finite.size:
                row[f"{prefix}_worst"] = float(
                    np.max(finite) if REGION_REQ_WORST[fam] == "max" else np.min(finite))
            else:
                row[f"{prefix}_worst"] = np.nan
            row[f"{prefix}_gini"] = gini(finite)
            for rid, v in m[fam].items():
                row[f"{prefix}_{rid}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["d", "R", "geom"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    _save_pickle(df, cache_file)
    print(f"[cache SAVE] {cache_file}  ({len(df)} runs)")
    return df


def load_region_req_metrics_set(
    geometry: str,
    cache_dir: Path = Path("cache"),
    experiments: Optional[List[str]] = None,
    score_field: str = REGION_REQ_SCORE_FIELD,
    point: str = REGION_REQ_POINT,
    include_reps: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """
    Request-based regional safety, wait, and service rate for every experiment of
    one *geometry*, stacked into one long DataFrame (mirrors
    load_region_req_set).  Drives the regional significance report.
    """
    frames = []
    for group, cfg in GROUP_CONFIGS.items():
        if cfg.get("balance") != geometry:
            continue
        if group.endswith("_max3"):
            continue
        if experiments is not None and cfg.get("experiment") not in experiments:
            continue
        results_dir = Path(cfg["results_dir"])
        if not results_dir.exists():
            continue
        regions_file = cfg.get("regions_file")
        if regions_file is None or not Path(regions_file).exists():
            print(f"  [skip] {group}: no regions file")
            continue
        regions = load_regions(regions_file).get("regions", [])
        sub = load_region_req_metrics_cached(
            results_dir, cache_dir, group, regions,
            score_field=score_field, point=point, force=force,
        )
        if sub.empty:
            continue
        sub = sub.copy()
        sub["group"] = group
        sub["algorithm"] = cfg.get("algorithm")
        sub["experiment"] = cfg.get("experiment")
        sub["geometry"] = cfg.get("balance")
        sub["label"] = cfg.get("label")
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if not include_reps and "is_rep" in out.columns:
        out = out[~out["is_rep"].astype(bool)].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------
#
# Design constraints (see guides/STATISTICAL_ANALYSIS_PLAN.md):
#  - The x1 sweep (5 d x 3 R x 4 behaviour) holds DESIGN factors, not replicates;
#    algorithm comparisons across it must be PAIRED by (d,R,behaviour,ic), never
#    treated as i.i.d. samples -> paired_test (Wilcoxon signed-rank).
#  - Per-point run-level significance needs true replication; it exists only at the
#    focus setup below, replicated K times per algorithm and geometry -> replicate_test.
#  - Inequality indices (Gini) have no closed-form SE; never t-test them. Use a
#    bootstrap CI of the index (and of its difference) instead.
#  - Every p-value carries an effect size and a multiple-comparison correction
#    (Holm or BH) within its metric family.

FOCUS_SETUP = {"d": 258.0, "R": 0.5, "behaviour": "go_back", "initial_conditions": "home"}


def cliffs_delta(a, b) -> float:
    """Cliff's delta: P(a>b) - P(a<b), in [-1, 1]. Nonparametric dominance size."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan
    bs = np.sort(b)
    # for each a: count of b strictly below (a>b) and strictly above (a<b)
    n_lt = np.searchsorted(bs, a, side="left")              # b < a
    n_gt = nb - np.searchsorted(bs, a, side="right")        # b > a
    return float((n_lt.sum() - n_gt.sum()) / (na * nb))


def cliffs_magnitude(delta: float) -> str:
    """Romano et al. thresholds for |Cliff's delta|."""
    if delta is None or not np.isfinite(delta):
        return "na"
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def paired_test(
    df: pd.DataFrame,
    exp_a: str,
    exp_b: str,
    metric: str,
    keys: Optional[List[str]] = None,
    group_col: str = "experiment",
) -> dict:
    """
    Wilcoxon signed-rank on the x1 sweep, pairing exp_a vs exp_b runs by
    (d, R, behaviour, initial_conditions).  Positive deltas mean exp_a > exp_b.

    *df* is a long frame from load_experiment_set (must carry the
    ``group_col`` column).  ``group_col`` selects what exp_a / exp_b name: the
    ``experiment`` (algorithm-vs-algorithm, default), the ``group`` (max3 vs
    unlimited), or the ``geometry`` (balanced vs imbalanced within an algorithm).
    Replicate rows are dropped so each point contributes one pair.  Returns stat,
    p, Cliff's delta, n, sign agreement, mean/median delta.
    """
    if keys is None:
        keys = ["d", "R", "behaviour", "initial_conditions"]

    sub_a = df[df[group_col] == exp_a]
    sub_b = df[df[group_col] == exp_b]
    if "is_rep" in df.columns:
        sub_a = sub_a[~sub_a["is_rep"].astype(bool)]
        sub_b = sub_b[~sub_b["is_rep"].astype(bool)]

    merged = match_run_pairs(sub_a, sub_b, keys=keys)
    col_a, col_b = f"{metric}_exp1", f"{metric}_exp2"
    if col_a not in merged.columns or col_b not in merged.columns:
        raise KeyError(f"metric '{metric}' not found in matched pairs for {exp_a} vs {exp_b}")

    va = pd.to_numeric(merged[col_a], errors="coerce")
    vb = pd.to_numeric(merged[col_b], errors="coerce")
    mask = va.notna() & vb.notna()
    va, vb = va[mask].to_numpy(), vb[mask].to_numpy()
    delta = va - vb
    n = int(len(delta))

    n_pos = int(np.sum(delta > 0))
    n_neg = int(np.sum(delta < 0))
    n_zero = int(np.sum(delta == 0))
    direction = "a>b" if (np.median(delta) > 0) else ("a<b" if np.median(delta) < 0 else "tie")
    n_agree = max(n_pos, n_neg)

    try:
        stat, p = stats.wilcoxon(va, vb)
        stat, p = float(stat), float(p)
    except ValueError:
        # all-zero differences (identical) -> no signed-rank statistic
        stat, p = np.nan, np.nan

    d_cliff = cliffs_delta(va, vb)
    return {
        "exp_a": exp_a, "exp_b": exp_b, "metric": metric,
        "test": "wilcoxon_signed_rank",
        "n": n, "stat": stat, "p": p,
        "cliffs_delta": d_cliff, "magnitude": cliffs_magnitude(d_cliff),
        "mean_delta": float(np.mean(delta)) if n else np.nan,
        "median_delta": float(np.median(delta)) if n else np.nan,
        "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero,
        "sign_agreement": (n_agree / n) if n else np.nan,
        "sign_summary": f"{n_agree}/{n} {direction}",
    }


def replicate_test(
    df: pd.DataFrame,
    exp_a: str,
    exp_b: str,
    metric: str,
    setup: Optional[dict] = None,
    ci: float = 0.95,
    group_col: str = "experiment",
) -> dict:
    """
    Run-level test across the K replicates at one operating point (default
    FOCUS_SETUP).  *df* must be loaded with ``include_reps=True``; the
    original x1 run at the point counts as an extra replicate.  ``group_col``
    selects what exp_a / exp_b name (``experiment``, ``group`` or ``geometry``;
    see paired_test).

    Welch's t-test is primary (unequal variance, no pairing across algorithms);
    Mann-Whitney U is reported as a distribution-free fallback.  Positive
    mean_diff means exp_a > exp_b.  Returns both p-values, a Welch CI on the
    difference, Cohen's d and Cliff's delta.
    """
    if setup is None:
        setup = FOCUS_SETUP

    def _vals(exp):
        sub = df[df[group_col] == exp]
        for k, v in setup.items():
            sub = sub[sub[k] == v]
        return pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy()

    a, b = _vals(exp_a), _vals(exp_b)
    na, nb = len(a), len(b)
    out = {
        "exp_a": exp_a, "exp_b": exp_b, "metric": metric,
        "n_a": na, "n_b": nb,
        "mean_a": float(np.mean(a)) if na else np.nan,
        "mean_b": float(np.mean(b)) if nb else np.nan,
        "sd_a": float(np.std(a, ddof=1)) if na > 1 else np.nan,
        "sd_b": float(np.std(b, ddof=1)) if nb > 1 else np.nan,
    }
    if na < 2 or nb < 2:
        out.update({"mean_diff": np.nan, "t_stat": np.nan, "welch_p": np.nan,
                    "ci_lo": np.nan, "ci_hi": np.nan, "u_stat": np.nan,
                    "mw_p": np.nan, "cohens_d": np.nan, "cliffs_delta": np.nan,
                    "magnitude": "na"})
        return out

    mean_diff = float(np.mean(a) - np.mean(b))
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = float(np.sqrt(va / na + vb / nb))

    t_stat, welch_p = stats.ttest_ind(a, b, equal_var=False)
    if se > 0:
        welch_df = (va / na + vb / nb) ** 2 / (
            (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        )
        tcrit = float(stats.t.ppf(0.5 + ci / 2, welch_df))
        ci_lo, ci_hi = mean_diff - tcrit * se, mean_diff + tcrit * se
    else:
        ci_lo = ci_hi = mean_diff

    try:
        u_stat, mw_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        u_stat, mw_p = float(u_stat), float(mw_p)
    except ValueError:
        u_stat, mw_p = np.nan, np.nan

    pooled_sd = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    cohen_d = float(mean_diff / pooled_sd) if pooled_sd > 0 else np.nan
    d_cliff = cliffs_delta(a, b)

    out.update({
        "mean_diff": mean_diff, "diff_se": se,
        "t_stat": float(t_stat), "welch_p": float(welch_p),
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "u_stat": u_stat, "mw_p": mw_p,
        "cohens_d": cohen_d,
        "cliffs_delta": d_cliff, "magnitude": cliffs_magnitude(d_cliff),
    })
    return out


def bootstrap_gini_ci(values, n_boot: int = 1000, ci: float = 0.95,
                      seed: Optional[int] = None) -> dict:
    """Percentile bootstrap CI for the Gini of *values* (no closed-form SE exists)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if len(x) == 0:
        return {"gini": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.array([gini(x[rng.integers(0, n, n)]) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return {
        "gini": gini(x),
        "lo": float(np.percentile(boots, 100 * alpha)),
        "hi": float(np.percentile(boots, 100 * (1 - alpha))),
        "se": float(np.std(boots, ddof=1)),
        "n_boot": n_boot,
    }


def bootstrap_gini_diff_ci(values_a, values_b, n_boot: int = 1000, ci: float = 0.95,
                           seed: Optional[int] = None) -> dict:
    """
    Percentile bootstrap CI for gini(a) - gini(b), resampling each group
    independently.  ``significant`` is True when the CI excludes 0.
    """
    a = np.asarray(values_a, dtype=float); a = a[np.isfinite(a) & (a >= 0)]
    b = np.asarray(values_b, dtype=float); b = b[np.isfinite(b) & (b >= 0)]
    if len(a) == 0 or len(b) == 0:
        return {"gini_a": np.nan, "gini_b": np.nan, "diff": np.nan,
                "lo": np.nan, "hi": np.nan, "significant": False, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    boots = np.array([
        gini(a[rng.integers(0, na, na)]) - gini(b[rng.integers(0, nb, nb)])
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1 - alpha)))
    return {
        "gini_a": gini(a), "gini_b": gini(b), "diff": gini(a) - gini(b),
        "lo": lo, "hi": hi, "significant": (lo > 0 or hi < 0), "n_boot": n_boot,
    }


def holm_correct(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, returned in input order.

    NaN inputs (e.g. a degenerate Wilcoxon) are passed through as NaN and excluded
    from the family size."""
    p = np.asarray(pvalues, dtype=float)
    adj = np.full(p.shape, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if len(finite) == 0:
        return adj
    pf = p[finite]
    m = len(pf)
    order = np.argsort(pf)
    running = 0.0
    out = np.empty(m)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pf[idx])
        out[idx] = min(running, 1.0)
    adj[finite] = out
    return adj


def bh_correct(pvalues) -> np.ndarray:
    """Benjamini-Hochberg FDR adjusted p-values, returned in input order."""
    p = np.asarray(pvalues, dtype=float)
    adj = np.full(p.shape, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if len(finite) == 0:
        return adj
    pf = p[finite]
    m = len(pf)
    order = np.argsort(pf)
    prev = 1.0
    out = np.empty(m)
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, pf[idx] * m / (rank + 1))
        out[idx] = min(prev, 1.0)
    adj[finite] = out
    return adj


def _is_inequality_metric(metric: str) -> bool:
    return any(k in metric for k in ("gini", "atkinson", "ratio_2020"))


def _group_for(geometry: str, experiment: str, max3: bool = False) -> Optional[str]:
    for group, cfg in GROUP_CONFIGS.items():
        if cfg.get("balance") != geometry or cfg.get("experiment") != experiment:
            continue
        if group.endswith("_max3") == max3:
            return group
    return None


def pooled_focus_taxi_values(
    geometry: str,
    experiment: str,
    key: str,
    setup: Optional[dict] = None,
    cache_dir: Path = Path("cache"),
    max3: bool = False,
) -> np.ndarray:
    """
    Per-taxi values (last-batch snapshot) pooled across every replicate at the
    focus setup, for one experiment and geometry.  *key* is a per_taxi field such
    as ``"trip_income"`` or ``"safety_score"``.  Feeds the inequality bootstrap
    (bootstrap_gini_ci / bootstrap_gini_diff_ci), the one
    inequality test that needs the taxi-level vector rather than the run summary.
    """
    if setup is None:
        setup = FOCUS_SETUP
    group = _group_for(geometry, experiment, max3=max3)
    if group is None:
        return np.array([])
    cfg = GROUP_CONFIGS[group]
    results_dir = Path(cfg["results_dir"]).resolve()
    summ = load_run_summaries_cached(
        results_dir, cache_dir, group, include_reps=True,
        load_taxi_snapshot=False, load_request_stats=False,
    )
    runs = summ
    for k, v in setup.items():
        runs = runs[runs[k] == v]

    vals: List[float] = []
    for run_id in runs["run_id"]:
        ptm = results_dir / f"run_{run_id}_per_taxi_metrics.json.gz"
        if not ptm.exists():
            continue
        line = _last_nonempty_line(ptm)
        if line is None:
            continue
        snap = json.loads(line)
        vals.extend(float(v) for v in snap.get(key, [])
                    if v is not None and np.isfinite(float(v)))
    return np.array(vals)


def significance_summary(
    df_sweep: pd.DataFrame,
    df_reps: pd.DataFrame,
    comparisons: List[Tuple[str, str]],
    metrics: List[str],
    setup: Optional[dict] = None,
    group_col: str = "experiment",
    correction: str = "holm",
    keys: Optional[List[str]] = None,
    geometry: Optional[str] = None,
    boot_keys: Optional[dict] = None,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """
    One tidy row per (comparison, metric) combining the two significance views:

      - SWEEP:    paired Wilcoxon across the x1 sweep (df_sweep, one run per point),
        paired by (d,R,behaviour,ic).  This is the headline test.
      - FOCUS:    replicate test at FOCUS_SETUP (df_reps, include_reps=True).

    Effect sizes accompany every p-value (Cliff's delta on the sweep, Cohen's d
    and Cliff's delta on the focus).  P-values are corrected WITHIN each metric
    across the comparisons (the metric family) by Holm (default) or BH.

    Inequality metrics (gini / atkinson / ratio_2020) skip the parametric Welch
    t-test - an inequality index has no closed-form SE - so their focus p-value
    comes from Mann-Whitney; pair them with a bootstrap CI (see
    bootstrap_gini_diff_ci) for the headline inequality claim.

    *comparisons* is a list of (a, b) names interpreted via *group_col*; positive
    deltas / diffs mean a > b.  When *geometry* is given together with *boot_keys*,
    the inequality metrics named in *boot_keys* additionally get a pooled per-taxi
    bootstrap CI on the Gini difference at the focus point (columns ``foc_gini_a``,
    ``foc_gini_b``, ``foc_gini_diff``, ``foc_gini_ci_lo``, ``foc_gini_ci_hi``);
    this is only valid when *group_col* is ``experiment``.  *boot_keys* maps a
    metric to its per-taxi field name (e.g. {"income_gini": "trip_income",
    "safety_gini": "safety_score"}).
    """
    corr = bh_correct if correction == "bh" else holm_correct

    # Pool each experiment's per-taxi vector once up front; the bootstrap below
    # then reuses these instead of re-reading the summary pickle and per-taxi
    # files on every (comparison, metric) iteration.
    boot_vals: dict = {}
    do_boot = geometry is not None and bool(boot_keys) and group_col == "experiment"
    if do_boot:
        exps = sorted({e for pair in comparisons for e in pair})
        for metric, key in boot_keys.items():
            if metric not in metrics:
                continue
            for e in exps:
                boot_vals[(metric, e)] = pooled_focus_taxi_values(
                    geometry, e, key, setup=setup)

    rows = []
    for a, b in comparisons:
        for metric in metrics:
            if metric not in df_sweep.columns:
                continue
            pt = paired_test(df_sweep, a, b, metric, keys=keys, group_col=group_col)
            rt = replicate_test(df_reps, a, b, metric, setup=setup, group_col=group_col)
            is_ineq = _is_inequality_metric(metric)
            rep_p = rt["mw_p"] if is_ineq else rt["welch_p"]
            boot = {"gini_a": np.nan, "gini_b": np.nan, "diff": np.nan,
                    "lo": np.nan, "hi": np.nan}
            if do_boot and is_ineq and (metric, a) in boot_vals:
                boot = bootstrap_gini_diff_ci(
                    boot_vals[(metric, a)], boot_vals[(metric, b)],
                    n_boot=n_boot, seed=0)
            rows.append({
                "comparison": f"{a} vs {b}",
                "metric": metric,
                "sweep_n": pt["n"],
                "sweep_p": pt["p"],
                "sweep_cliff": pt["cliffs_delta"],
                "sweep_mag": pt["magnitude"],
                "sweep_mean_delta": pt["mean_delta"],
                "sweep_signs": pt["sign_summary"],
                "foc_diff": rt["mean_diff"],
                "foc_ci_lo": rt["ci_lo"],
                "foc_ci_hi": rt["ci_hi"],
                "foc_welch_p": np.nan if is_ineq else rt["welch_p"],
                "foc_mw_p": rt["mw_p"],
                "foc_cohens_d": np.nan if is_ineq else rt["cohens_d"],
                "foc_cliff": rt["cliffs_delta"],
                "foc_gini_a": boot["gini_a"],
                "foc_gini_b": boot["gini_b"],
                "foc_gini_diff": boot["diff"],
                "foc_gini_ci_lo": boot["lo"],
                "foc_gini_ci_hi": boot["hi"],
                "_rep_p": rep_p,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["sweep_p_adj"] = np.nan
    out["foc_p_adj"] = np.nan
    for _, grp in out.groupby("metric"):
        out.loc[grp.index, "sweep_p_adj"] = corr(grp["sweep_p"].to_numpy())
        out.loc[grp.index, "foc_p_adj"] = corr(grp["_rep_p"].to_numpy())
    out["sweep_sig"] = out["sweep_p_adj"] < 0.05
    out["foc_sig"] = out["foc_p_adj"] < 0.05
    return out.drop(columns="_rep_p")


# ---------------------------------------------------------------------------
# Safety trajectories (safety as evolution over time)
# ---------------------------------------------------------------------------

def _fleet_safety_traj(ptm_path: Path) -> pd.DataFrame:
    """Per-batch fleet safety distribution (mean/std/p10/p50/p90) plus break share.

    Computed straight from the per_taxi batches because the aggregates CSV carries
    no fleet safety column, only the per-region averages."""
    recs = []
    for batch in load_per_taxi_batches(ptm_path):
        ts = batch.get("timestamp")
        if ts is None:
            continue
        rec: dict = {"timestamp": int(ts)}
        vals = [float(v) for v in batch.get("safety_score", [])
                if v is not None and np.isfinite(float(v))]
        if vals:
            arr = np.array(vals)
            rec["safety_mean"] = float(arr.mean())
            rec["safety_std"] = float(arr.std())
            rec["safety_p10"] = float(np.percentile(arr, 10))
            rec["safety_p50"] = float(np.percentile(arr, 50))
            rec["safety_p90"] = float(np.percentile(arr, 90))
        on_break = batch.get("on_break", [])
        if on_break:
            rec["break_share"] = float(sum(bool(x) for x in on_break) / len(on_break))
        recs.append(rec)
    return pd.DataFrame(recs).sort_values("timestamp").reset_index(drop=True)


def _region_safety_traj(csv_path: Path) -> pd.DataFrame:
    """Long per-batch, per-region safety from run_*_region_safety_averages.csv.gz."""
    raw = pd.read_csv(csv_path, index_col=0)
    region_ids = sorted({c[: -len("_avg_safety_score")]
                         for c in raw.columns if c.endswith("_avg_safety_score")})
    recs = []
    for _, row in raw.iterrows():
        ts = int(row["timestamp"])
        for rid in region_ids:
            sc = row.get(f"{rid}_avg_safety_score")
            cnt = row.get(f"{rid}_taxi_count")
            recs.append({
                "timestamp": ts, "region_id": rid,
                "safety_mean": float(sc) if pd.notna(sc) else np.nan,
                "taxi_count": float(cnt) if pd.notna(cnt) else np.nan,
            })
    return pd.DataFrame(recs)


def load_safety_trajectories(
    geometry: str,
    experiments: Optional[List[str]] = None,
    agg: str = "fleet",
    d_focus: float = 258.0,
    cache_dir: Path = Path("cache"),
    include_reps: bool = True,
    include_max3: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """
    Long DataFrame of per-algorithm safety-over-time at the focus fleet size
    *d_focus*, mirroring load_experiment_set.

    agg='fleet'  -> per-batch fleet safety mean/std/p10/p50/p90 + break_share
                    (from the per_taxi batches).
    agg='region' -> per-batch per-region safety mean + taxi_count
                    (from run_*_region_safety_averages.csv.gz).

    The focus point (d_focus, R=0.5, behaviour=go_back, ic=home) carries all K
    replicates when ``include_reps`` is True, so the plots can draw replicate
    bands there; every other point at d_focus is a single run.  Cached to
    ``<cache_dir>/safety_traj_<geometry>_<agg>_d<d_focus>.pkl`` and invalidated
    when any contributing run_* file is newer.
    """
    if agg not in ("fleet", "region"):
        raise ValueError("agg must be 'fleet' or 'region'")
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"safety_traj_{geometry}_{agg}_d{int(round(d_focus))}.pkl"

    groups = []
    for group, cfg in GROUP_CONFIGS.items():
        if cfg.get("balance") != geometry:
            continue
        if not include_max3 and group.endswith("_max3"):
            continue
        if experiments is not None and cfg.get("experiment") not in experiments:
            continue
        if not Path(cfg["results_dir"]).exists():
            continue
        groups.append((group, cfg))

    if (not force and cache_file.exists()
            and all(_cache_valid(cache_file, Path(cfg["results_dir"])) for _, cfg in groups)):
        print(f"[cache HIT]  {cache_file.name}  ->  loading safety trajectories")
        return _load_pickle(cache_file)

    print(f"[cache MISS] building {agg} safety trajectories for {geometry} "
          f"(d={int(round(d_focus))}, {len(groups)} groups)...")
    frames = []
    for group, cfg in groups:
        # resolve() normalises the leading ../ so the long region_safety_averages
        # filenames stay under the Windows MAX_PATH limit; otherwise exists()/open
        # silently fail on the longest run ids (e.g. safety_objective_two_sided reps).
        results_dir = Path(cfg["results_dir"]).resolve()
        summ = load_run_summaries_cached(
            results_dir, cache_dir, group, include_reps=True,
            load_taxi_snapshot=False, load_request_stats=False,
        )
        runs = summ[summ["d"] == d_focus]
        if not include_reps and "is_rep" in runs.columns:
            runs = runs[~runs["is_rep"].astype(bool)]
        n_loaded = 0
        for _, r in runs.iterrows():
            run_id = r["run_id"]
            if agg == "fleet":
                path = results_dir / f"run_{run_id}_per_taxi_metrics.json.gz"
                if not path.exists():
                    continue
                traj = _fleet_safety_traj(path)
            else:
                path = results_dir / f"run_{run_id}_region_safety_averages.csv.gz"
                if not path.exists():
                    continue
                traj = _region_safety_traj(path)
            if traj.empty:
                continue
            traj["group"] = group
            traj["experiment"] = cfg.get("experiment")
            traj["algorithm"] = cfg.get("algorithm")
            traj["geometry"] = cfg.get("balance")
            traj["run_id"] = run_id
            for c in ["d", "R", "behaviour", "initial_conditions", "is_rep", "replicate"]:
                traj[c] = r.get(c)
            frames.append(traj)
            n_loaded += 1
        print(f"  {group}: {n_loaded} runs")

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _save_pickle(out, cache_file)
    print(f"[cache SAVE] {cache_file}  ({len(out)} rows)")
    return out
