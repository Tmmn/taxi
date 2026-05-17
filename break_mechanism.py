from __future__ import annotations

import numpy as np
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from city_model import Simulation, Taxi

RNG = np.random.default_rng()

def _sample_spec(spec: object) -> float:
    """Sample from a break parameter spec. Allows zero (for offsets and optional params)."""
    if isinstance(spec, (int, float)):
        val = float(spec)
        if val < 0:
            raise ValueError("Break parameter value must be >= 0.")
        return val
    if not isinstance(spec, dict):
        raise ValueError("Break parameter spec must be numeric or a dict.")
    dist = spec.get("dist", "uniform")
    if dist == "uniform":
        low = float(spec["low"])
        high = float(spec["high"])
        if high <= low or low < 0:
            raise ValueError("Uniform break spec requires 0 <= low < high.")
        return float(RNG.uniform(low, high))
    if dist == "fixed":
        val = float(spec["value"])
        if val < 0:
            raise ValueError("Fixed break spec value must be >= 0.")
        return val
    raise ValueError(f"Unsupported break spec dist: {dist}")


def _validate_spec(spec: object, name: str) -> None:
    try:
        _sample_spec(spec)
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{name}: {e}")


def _sample_break_target_from_spec(spec: object) -> float:
    """Sample a positive shift duration target. Rejects zero values."""
    if isinstance(spec, (int, float)):
        val = float(spec)
        if val <= 0:
            raise ValueError("Numeric shift_duration_tu must be > 0.")
        return val

    if not isinstance(spec, dict):
        raise ValueError("shift_duration_tu entries must be numeric or dict specs.")

    dist = spec.get("dist", "uniform")
    if dist == "uniform":
        low = float(spec["low"])
        high = float(spec["high"])
        if high <= low or low <= 0:
            raise ValueError("Uniform shift_duration_tu requires 0 < low < high.")
        return float(RNG.uniform(low, high))

    if dist == "fixed":
        val = float(spec["value"])
        if val <= 0:
            raise ValueError("Fixed shift_duration_tu value must be > 0.")
        return val

    raise ValueError(f"Unsupported shift_duration_tu dist: {dist}")


def _normalize_break_cohort_settings(raw_settings: object) -> dict[str, dict[str, object]]:
    if raw_settings is None:
        return {}
    if not isinstance(raw_settings, dict):
        raise ValueError("break_cohort_settings must be a dict with cohort ids as keys")

    normalized: dict[str, dict[str, object]] = {}
    for cohort_id, spec in raw_settings.items():
        cid = str(cohort_id)
        if spec is None:
            raise ValueError(f"break_cohort_settings['{cid}']: entry cannot be None.")
        if not isinstance(spec, dict):
            raise ValueError(f"break_cohort_settings['{cid}']: entry must be a dict.")

        rest_spec = spec.get("inter_shift_rest_tu")
        if rest_spec is None:
            raise ValueError(f"break_cohort_settings['{cid}']: inter_shift_rest_tu is required.")
        _validate_spec(rest_spec, f"inter_shift_rest_tu in cohort '{cid}'")

        intra_work_spec = spec.get("intra_shift_break_after_work_tu")
        intra_dur_spec = spec.get("intra_shift_break_duration_tu")
        if intra_work_spec is not None:
            _validate_spec(intra_work_spec, f"intra_shift_break_after_work_tu in cohort '{cid}'")
            if intra_dur_spec is None:
                raise ValueError(
                    f"break_cohort_settings['{cid}']: intra_shift_break_duration_tu must be set "
                    f"when intra_shift_break_after_work_tu is configured."
                )
        if intra_dur_spec is not None:
            _validate_spec(intra_dur_spec, f"intra_shift_break_duration_tu in cohort '{cid}'")

        demotivation_spec = spec.get("demotivation_threshold_tu")
        if demotivation_spec is not None:
            _validate_spec(demotivation_spec, f"demotivation_threshold_tu in cohort '{cid}'")
            if intra_dur_spec is None:
                raise ValueError(
                    f"break_cohort_settings['{cid}']: intra_shift_break_duration_tu must be set "
                    f"when demotivation_threshold_tu is configured."
                )

        offset_spec = spec.get("shift_start_offset_tu", 0)
        _validate_spec(offset_spec, f"shift_start_offset_tu in cohort '{cid}'")

        normalized[cid] = {
            "inter_shift_rest_tu": rest_spec,
            "intra_shift_break_after_work_tu": intra_work_spec,
            "intra_shift_break_duration_tu": intra_dur_spec,
            "demotivation_threshold_tu": demotivation_spec,
            "shift_start_offset_tu": offset_spec,
        }

    return normalized


def configure_breaks(sim: Simulation, config: dict) -> None:
    sim.driver_break_cohort_mix = config.get("driver_break_cohort_mix")
    sim.shift_duration_tu_config = config.get("shift_duration_tu")
    sim.use_break_cohorts = False
    sim.break_cohort_ids = []
    sim.break_cohort_probs = []

    if sim.driver_break_cohort_mix is not None or sim.shift_duration_tu_config is not None:
        if not isinstance(sim.driver_break_cohort_mix, dict) or not isinstance(sim.shift_duration_tu_config, dict):
            raise ValueError(
                "driver_break_cohort_mix and shift_duration_tu must both be dicts when cohort breaks are configured"
            )
        _initialize_break_cohort_config(sim)

    sim.day_length_tu = int(config.get("day_length_tu", 8640))
    if sim.day_length_tu <= 0:
        raise ValueError("day_length_tu must be a positive integer")

    sim.rush_windows_tu = config.get("rush_windows_tu", [])
    if not isinstance(sim.rush_windows_tu, list):
        raise ValueError("rush_windows_tu must be a list of {start, end} windows")

    for win in sim.rush_windows_tu:
        if not isinstance(win, dict) or "start" not in win or "end" not in win:
            raise ValueError("Each rush_windows_tu entry must contain numeric start and end")
        start = int(win["start"])
        end = int(win["end"])
        if start < 0 or end < 0 or start >= sim.day_length_tu or end >= sim.day_length_tu:
            raise ValueError("rush window bounds must be within [0, day_length_tu)")

    sim.p_defer_end_of_shift_in_rush = float(config.get("p_defer_end_of_shift_in_rush", 0.0))
    if not (0.0 <= sim.p_defer_end_of_shift_in_rush <= 1.0):
        raise ValueError("p_defer_end_of_shift_in_rush must be in [0, 1].")

    sim.max_break_deferral_tu = int(config.get("max_break_deferral_tu", 0))
    if sim.max_break_deferral_tu < 0:
        raise ValueError("max_break_deferral_tu must be >= 0.")

    sim.break_cohort_settings = _normalize_break_cohort_settings(config.get("break_cohort_settings"))


def _initialize_break_cohort_config(sim: Simulation) -> None:
    if sim.driver_break_cohort_mix is None or sim.shift_duration_tu_config is None:
        raise ValueError(
            "driver_break_cohort_mix and shift_duration_tu must both be set for cohort breaks"
        )

    cohort_mix = sim.driver_break_cohort_mix
    target_cfg = sim.shift_duration_tu_config

    if len(cohort_mix) == 0:
        raise ValueError("driver_break_cohort_mix must not be empty.")

    total = float(sum(cohort_mix.values()))
    if total <= 0:
        raise ValueError("driver_break_cohort_mix must sum to a positive value.")

    cohort_ids = []
    cohort_probs = []
    for cohort_id, weight in cohort_mix.items():
        if cohort_id not in target_cfg:
            raise ValueError(
                f"Missing shift_duration_tu config for cohort '{cohort_id}'."
            )
        w = float(weight)
        if w < 0:
            raise ValueError("driver_break_cohort_mix weights must be non-negative.")

        _sample_break_target_from_spec(target_cfg[cohort_id])  # validate spec at config time
        cohort_ids.append(cohort_id)
        cohort_probs.append(w)

    sim.break_cohort_ids = cohort_ids
    sim.break_cohort_probs = list(np.array(cohort_probs) / total)
    sim.use_break_cohorts = True


def assign_break_cohort_to_taxi(sim: Simulation, taxi: Taxi) -> Taxi:
    taxi.break_deferral_elapsed_tu = 0
    taxi.breaks_started_today = 0
    taxi.breaks_day_index = sim.time // sim.day_length_tu
    taxi.scheduled_return_time_tu = None
    taxi.break_is_end_of_shift = False

    if not sim.use_break_cohorts:
        taxi.break_profile_id = None
        taxi.shift_duration_tu = None
        taxi.shift_ended = False
        taxi.shift_start_work_time_tu = 0.0
        taxi.intra_shift_break_after_work_tu = None
        taxi.intra_shift_break_duration_tu = None
        taxi.demotivation_threshold_tu = None
        return taxi

    if sim.shift_duration_tu_config is None:
        raise ValueError("shift_duration_tu configuration is missing while cohorts are enabled.")

    cohort_idx = int(RNG.choice(len(sim.break_cohort_ids), p=sim.break_cohort_probs))
    cohort_id = sim.break_cohort_ids[cohort_idx]
    target_spec = sim.shift_duration_tu_config[cohort_id]
    taxi.break_profile_id = cohort_id
    taxi.shift_duration_tu = _sample_break_target_from_spec(target_spec)
    taxi.shift_ended = False

    settings = sim.break_cohort_settings.get(cohort_id, {})

    intra_work_spec = settings.get("intra_shift_break_after_work_tu")
    taxi.intra_shift_break_after_work_tu = _sample_spec(intra_work_spec) if intra_work_spec is not None else None

    intra_dur_spec = settings.get("intra_shift_break_duration_tu")
    taxi.intra_shift_break_duration_tu = _sample_spec(intra_dur_spec) if intra_dur_spec is not None else None

    demotivation_spec = settings.get("demotivation_threshold_tu")
    taxi.demotivation_threshold_tu = _sample_spec(demotivation_spec) if demotivation_spec is not None else None

    # Staggered shift start: negative offset desynchronizes fleet's first shift end
    offset_spec = settings.get("shift_start_offset_tu", 0)
    offset = _sample_spec(offset_spec)
    if taxi.shift_duration_tu is not None and offset >= taxi.shift_duration_tu:
        offset = taxi.shift_duration_tu - 1.0
    taxi.shift_start_work_time_tu = -offset

    return taxi


def put_taxi_on_break(sim: Simulation, t: Taxi) -> None:
    if not t.available or t.on_break:
        return

    sim.city.A[sim.city.coordinate_dict_ij_to_c[t.x][t.y]].remove(t.taxi_id)
    del sim.taxis_available[t.taxi_id]

    t.on_break = True
    t.available = False
    t.break_start_safety_score = float(t.safety_score)
    t.next_destination = deque()
    t.time_on_break_current = 0
    work_time = t.time_serving + t.time_to_request + t.time_cruising + t.time_waiting
    t.work_time_at_last_break = work_time
    t.break_deferral_elapsed_tu = 0
    current_day = sim.time // sim.day_length_tu
    if t.breaks_day_index != current_day:
        t.breaks_day_index = current_day
        t.breaks_started_today = 0
    t.breaks_started_today += 1

    if t.shift_ended:
        cohort_settings = sim.break_cohort_settings.get(t.break_profile_id, {})
        rest_spec = cohort_settings.get("inter_shift_rest_tu")
        # validated with configure_breaks
        rest = _sample_spec(rest_spec)
        t.scheduled_return_time_tu = sim.time + int(rest)
        t.break_is_end_of_shift = True
    else:
        dur = t.intra_shift_break_duration_tu
        if dur is None:
            raise ValueError(
                f"intra_shift_break_duration_tu not set for taxi {t.taxi_id} (cohort '{t.break_profile_id}'). "
                f"cannot take intra-shift break."
            )
        t.scheduled_return_time_tu = sim.time + int(dur)
        t.break_is_end_of_shift = False

    sim.taxis_on_break.add(t.taxi_id)
    sim.taxis[t.taxi_id] = t

    if sim.log:
        print(f"\ttaxi {t.taxi_id} going on break (end_of_shift={t.break_is_end_of_shift})")


def return_taxi_from_break(sim: Simulation, t: Taxi) -> None:
    if not t.on_break:
        return

    t.on_break = False
    t.available = True
    t.time_waiting_since_last_trip = 0
    t.scheduled_return_time_tu = None
    t.break_is_end_of_shift = False

    if t.break_is_end_of_shift and sim.use_break_cohorts and t.break_profile_id is not None:
        assert sim.shift_duration_tu_config is not None
        work_time = float(t.time_serving + t.time_to_request + t.time_cruising)
        t.shift_start_work_time_tu = work_time
        target_spec = sim.shift_duration_tu_config[t.break_profile_id]
        t.shift_duration_tu = _sample_break_target_from_spec(target_spec)
        t.shift_ended = False
        t.break_deferral_elapsed_tu = 0

    sim.taxis_on_break.remove(t.taxi_id)
    sim.taxis_available[t.taxi_id] = t
    sim.city.A[sim.city.coordinate_dict_ij_to_c[t.x][t.y]].add(t.taxi_id)

    sim.taxis[t.taxi_id] = t

    if sim.log:
        print(f"\ttaxi {t.taxi_id} returning from break")


def _is_in_window(time_of_day_tu: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= time_of_day_tu < end
    return time_of_day_tu >= start or time_of_day_tu < end


def _is_rush_time(sim: Simulation) -> bool:
    if len(sim.rush_windows_tu) == 0:
        return False
    time_of_day_tu = sim.time % sim.day_length_tu
    for win in sim.rush_windows_tu:
        if _is_in_window(time_of_day_tu, int(win["start"]), int(win["end"])):
            return True
    return False


def _should_defer_end_of_shift(sim: Simulation, taxi: Taxi) -> bool:
    if sim.p_defer_end_of_shift_in_rush <= 0.0 or sim.max_break_deferral_tu <= 0:
        return False
    if taxi.break_deferral_elapsed_tu >= sim.max_break_deferral_tu:
        return False
    if not _is_rush_time(sim):
        return False
    return RNG.random() < sim.p_defer_end_of_shift_in_rush


def check_and_manage_breaks(sim: Simulation) -> None:
    if not sim.use_break_cohorts:
        return

    current_day = sim.time // sim.day_length_tu
    for taxi_id in sim.taxis:
        t: Taxi = sim.taxis[taxi_id]
        # next day
        if t.breaks_day_index != current_day:
            t.breaks_day_index = current_day
            t.breaks_started_today = 0
            t.break_deferral_elapsed_tu = 0
            # shift_ended is NOT reset here - shifts are independent of calendar days
            sim.taxis[taxi_id] = t

    # loop taxis on break
    for taxi_id in sim.taxis_on_break.copy():
        t: Taxi = sim.taxis[taxi_id]
        if t.scheduled_return_time_tu is not None and sim.time >= t.scheduled_return_time_tu:
            return_taxi_from_break(sim, t)

    # loop available taxis
    for taxi_id in sim.taxis_available.keys().copy():
        t: Taxi = sim.taxis[taxi_id]

        total_work_time = t.time_serving + t.time_to_request + t.time_cruising + t.time_waiting

        # 1. check End-of-shift
        if (not t.shift_ended) and (t.shift_duration_tu is not None):
            work_time_since_shift_start = total_work_time - t.shift_start_work_time_tu
            if work_time_since_shift_start >= t.shift_duration_tu:
                if _should_defer_end_of_shift(sim, t):
                    t.break_deferral_elapsed_tu += 1
                    sim.taxis[taxi_id] = t
                    continue
                t.shift_ended = True
                sim.taxis[taxi_id] = t
                put_taxi_on_break(sim, t)
                continue

        # 2. check Intra-shift mandatory break (work-time threshold)
        if t.intra_shift_break_after_work_tu is not None:
            work_time_since_last_break = total_work_time - t.work_time_at_last_break
            if work_time_since_last_break >= t.intra_shift_break_after_work_tu:
                put_taxi_on_break(sim, t)
                continue

        # 3. check Intra-shift demotivation break (waiting-time threshold)
        if t.demotivation_threshold_tu is not None:
            if t.time_waiting_since_last_trip >= t.demotivation_threshold_tu:
                put_taxi_on_break(sim, t)
