#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:00:15 2017

@author: bokanyie
"""

import numpy as np
import pandas as pd
import json
from random import choice, shuffle
import matplotlib.pyplot as plt
from time import time

import gzip
import shutil
import os

# special data types
from collections import deque
from queue import Queue
from randomdict import RandomDict

from geometry import City
from break_mechanism import (
    configure_breaks,
    assign_break_cohort_to_taxi,
    check_and_manage_breaks
)


class Taxi:
    """
    Represents a taxi in the simulation.

    Attributes
    ----------

    x : int
        horizontal grid coordinate

    y : int
        vertical grid coordinate

    taxi_id : int
        unique identifier of taxi

    available : bool
        flag that stores whether taxi is free

    to_request : bool
        flag that stores when taxi is moving towards a request
        but there is still no user sitting in it

    with_passenger : bool
        flag that stores when taxi is carrying a passenger

    actual_request_executing : int
        id of request that is being executed by the taxi

    requests_completed : list of ints
        list of requests completed by taxi

    time_waiting : int
        time spent with empty waiting

    time_serving : int
        time spent with carrying a passenger

    time_to_request : int
        time spent from call assignment to pickup, without a passenger

    time_cruising : int
        time spent with travelling empty with no assigned requests

    next_destination : Queue
        Queue that stores the path forward of the taxi

    safety_score : float
        score that measures safety of taxi driver at point in time

    home : tuple
         if config setting is "initial_conditions":"random", then this will be the home of the taxi instead of the base

    on_break : bool
        whether taxi is currently unavailable due to a break

    break_profile_id : str or None
        optional cohort id assigned for cohort-based break timing

    shift_duration_tu : float or None
        sampled shift duration in work-time units; end-of-shift fires when
        work accumulated since shift start reaches this value

    shift_ended : bool
        whether the end-of-shift break has been triggered for the current shift

    break_deferral_elapsed_tu : int
        accumulated delay while end-of-shift break is deferred during rush windows

    shift_start_work_time_tu : float
        cumulative work time when the current shift started; used to measure
        shift duration independent of short breaks taken during the shift

    intra_shift_break_after_work_tu : float or None
        per-driver work-time threshold for mandatory intra-shift break; None disables

    intra_shift_break_duration_tu : float or None
        per-driver duration of intra-shift (short) breaks; None disables

    demotivation_threshold_tu : float or None
        per-driver waiting-time threshold for demotivation break; None disables

    break_is_end_of_shift : bool
        True while driver is on inter-shift rest; False for intra-shift breaks

    route_length_pref : str
        soft route preference profile (`short_pref`, `neutral_pref`, `long_pref`)

    pref_strength_route : float
        strength of route preference in [0, 1]

    nonpreferred_accept_ceiling : float
        maximum acceptance probability for non-preferred routes

    breaks_started_today : int
        number of breaks the taxi has already started in the current simulation day

    breaks_day_index : int
        day index used to reset `breaks_started_today` when a new day starts

    scheduled_return_time_tu : int or None
        absolute simulation time for delayed break return (for next-day return policies)

    satisfaction_score : float
        bounded driver satisfaction state

    last_satisfaction_delta : float
        latest signed satisfaction update applied to the taxi

    break_start_safety_score : float
        safety score at break start used for non-linear break recovery

    initial_safety_score : float
        taxi-specific safety baseline sampled at simulation start; break recovery asymptotically returns to this value

    decline_count : int
        how often the Taxi declined a match

    total_declines : int
        total number of declines across the whole simulation (never reset)

    shift_start_income : float
        cumulative income (from eval_taxi_income) at the moment the current shift
        started; used to compute shift-relative income for the flexibility model.
        Reset whenever shift_start_work_time_tu is reset at a new shift start.

    """

    x: int
    y: int
    taxi_id: int
    available: bool
    to_request: bool
    with_passenger: bool
    on_break: bool
    actual_request_executing: int | None
    requests_completed: set[int]
    time_waiting: int
    time_waiting_since_last_trip: int
    time_on_break: int
    time_on_break_current: int
    time_serving: int
    time_cruising: int
    time_to_request: int
    work_time_at_last_break: int
    shift_start_work_time_tu: float
    break_profile_id: str | None
    shift_duration_tu: float | None
    shift_ended: bool
    break_deferral_elapsed_tu: int
    intra_shift_break_after_work_tu: float | None
    intra_shift_break_duration_tu: float | None
    demotivation_threshold_tu: float | None
    break_is_end_of_shift: bool
    route_length_pref: str
    pref_strength_route: float
    nonpreferred_accept_ceiling: float
    breaks_started_today: int
    breaks_day_index: int
    scheduled_return_time_tu: int | None
    satisfaction_score: float
    last_satisfaction_delta: float
    break_start_safety_score: float
    initial_safety_score: float
    safety_score: float
    next_destination: deque
    home: tuple[int, int] | None
    decline_count: int
    total_declines: int
    trip_lengths: list
    trip_count: int
    trip_length_sum: float
    trip_length_sum_sq: float
    shift_start_income: float

    def __init__(self, coords=None, taxi_id=None, safety_score=None):
        if coords is None:
            print("You have to put your taxi somewhere in the city!")
        elif taxi_id is None:
            print("Not a licenced taxi.")
        else:

            self.x = coords[0]
            self.y = coords[1]

            self.taxi_id = taxi_id

            self.available = True
            self.to_request = False
            self.with_passenger = False
            self.on_break = False

            self.actual_request_executing = None
            self.requests_completed = set()
            self.trip_lengths = []
            self.trip_count = 0
            self.trip_length_sum = 0.0
            self.trip_length_sum_sq = 0.0

            # types of time metrics to be stored
            self.time_waiting = 0
            self.time_waiting_since_last_trip = 0
            self.time_on_break = 0
            self.time_on_break_current = 0
            self.time_serving = 0
            self.time_cruising = 0
            self.time_to_request = 0

            self.work_time_at_last_break = 0
            self.shift_start_work_time_tu = 0.0

            # optional break-cohort state configured by Simulation
            self.break_profile_id = None
            self.shift_duration_tu = None
            self.shift_ended = False
            self.break_deferral_elapsed_tu = 0
            self.intra_shift_break_after_work_tu = None
            self.intra_shift_break_duration_tu = None
            self.demotivation_threshold_tu = None
            self.break_is_end_of_shift = False

            self.route_length_pref = "neutral_pref"
            self.pref_strength_route = 0.0
            self.nonpreferred_accept_ceiling = 0.0
            self.breaks_started_today = 0
            self.breaks_day_index = 0
            self.scheduled_return_time_tu = None
            self.satisfaction_score = 50.0
            self.last_satisfaction_delta = 0.0
            self.break_start_safety_score = 100.0

            # safety score starts variable across taxis; bounds are managed by Simulation
            if safety_score is None:
                self.safety_score = float(100)
            else:
                self.safety_score = float(safety_score)
            self.initial_safety_score = float(self.safety_score)
            self.break_start_safety_score = float(self.safety_score)

            # storing steps to take
            self.next_destination = deque()  # path to travel

            # this can only be filled if the city geometry is known
            self.home = None
            self.decline_count = 0
            self.total_declines = 0
            self.shift_start_income = 0.0

    def __str__(self):
        """
        Verbose string representation of taxi for debugging purposes.

        Returns
        -------
        string

        """
        s = [
            "Taxi ",
            str(self.taxi_id),
            ".\n\tPosition ",
            str(self.x) + "," + str(self.y) + "\n"
        ]
        if self.available:
            s += ["\tAvailable.\n"]
        elif self.to_request:
            s += ["\tTravelling towards request " + str(self.actual_request_executing) + ".\n"]
        elif self.with_passenger:
            s += ["\tCarrying the passenger of request " + str(self.actual_request_executing) + ".\n"]

        return "".join(s)

    def __iter__(self):
        """
        This method converts the class into a dict, with attributes as keys.
        """
        for attr, value in self.__dict__.items():
            yield attr, value


class Request:
    """
    Represents a request that is being made.
    
    Attributes
    ----------
    
    ox,oy : int
        grid coordinates of request origin
        
    dx,dy : int
        grid coordinates of request destination
        
    request_id : int
        unique id of request
    
    taxi_id : int
        id of taxi that serves the request

    mode : str
        current mode

    timestamps : dict
        stores simulation timestamps of the change of mode

    driver_safety_score_start : float or None
        taxi safety score when request is assigned

    driver_safety_score_end : float or None
        taxi safety score when request is completed or dropped

    average_safety_score : float or None
        mean safety score over all simulation steps while request is assigned
    """

    ox: int
    oy: int
    dx: int
    dy: int
    request_id: int
    taxi_id: int | None
    timestamps: dict[str, int | None]
    mode: str
    driver_safety_score_start: float | None
    driver_safety_score_end: float | None
    driver_safety_score_pickup: float | None
    safety_score_sum: float
    safety_score_count: int
    average_safety_score: float | None
    assigned_taxi_pos: list | None
    assigned_taxi_distance: int | None
    declined_taxi_ids: set[int]
    passenger_type: str
    safety_threshold: float
    w_dist: float
    w_safety: float
    w_wait: float
    cancellation_reason: str | None
    passenger_forced_accept: bool
    last_no_match_reason: str | None

    def __init__(self, ocoords=None, dcoords=None, request_id=None, timestamp=None):
        """
        
        """
        if (ocoords is None) or (dcoords is None):
            print("A request has to have a well-defined origin and destination.")
        elif request_id is None:
            print("Please identify each request uniquely.")
        elif timestamp is None:
            print("Please give a timestamp for the request!")
        else:
            # pickup coordinates
            self.ox = ocoords[0]
            self.oy = ocoords[1]

            # desired dropoff coordinates
            self.dx = dcoords[0]
            self.dy = dcoords[1]

            # id
            self.request_id = request_id

            # travel info, different time metrics
            self.taxi_id = None

            self.timestamps = {
                'request': timestamp,
                'assigned': None,
                'pickup': None,
                'dropoff': None
            }

            self.mode = 'pending'
            self.driver_safety_score_start = None
            self.driver_safety_score_end = None
            self.driver_safety_score_pickup = None
            self.safety_score_sum = 0.0
            self.safety_score_count = 0
            self.average_safety_score = None
            self.assigned_taxi_pos = None
            self.assigned_taxi_distance = None
            self.declined_taxi_ids = set()

            # passenger preference attributes (sampled by Simulation.add_request)
            self.passenger_type = 'safety_indifferent'
            self.safety_threshold = 0.0
            self.w_dist = 1.0
            self.w_safety = 0.5
            self.w_wait = 0.3
            self.cancellation_reason = None
            self.passenger_forced_accept = False
            self.last_no_match_reason = None

    def __str__(self):
        """
        Verbose string representation of request for debugging purposes.

        Returns
        -------
        string

        """
        s = [
            "Request ",
            str(self.request_id),
            ".\n\tOrigin ",
            str(self.ox) + "," + str(self.oy) + "\n",
            "\tDestination ",
            str(self.dx) + "," + str(self.dy) + "\n",
            "\tRequest timestamp ",
            str(self.timestamps['request']) + "\n"
        ]
        if self.taxi_id is not None:
            s += ["\tTaxi assigned ", str(self.taxi_id), ".\n"]
            s += ["\tWaiting since ", str(self.timestamps['request']), ".\n"]
            if self.mode != 'pending':
                s += ["\tPickup timestamp ", str(self.timestamps['pickup']), ".\n"]
                if self.timestamps['dropoff'] is not None:
                    s += ["\tDropoff timestamp ", str(self.timestamps['dropoff']), ".\n"]
        else:
            s += ["\tPending since ", str(self.timestamps['request']), ".\n"]

        return "".join(s)

    def __iter__(self):
        """
        This method converts the class into a dict, with attributes as keys.
        """
        for attr, value in self.__dict__.items():
            yield attr, value


class Simulation:
    """
    Class for containing the elements of the simulation.
    
    Attributes
    ----------
    time : int
        stores the time elapsed in the simulation
    
    num_taxis : int
        how many taxis there are
    
    request_rate : float
        base rate of requests per time unit (scaled by request_rate_schedule multiplier)

    request_rate_schedule : list of dict
        time-of-day windows with multipliers applied to request_rate;
        each entry has start, end (in time_of_day_tu), and multiplier (>= 0);
        gaps between windows use multiplier 1.0; empty list means constant base rate

    hard_limit : int
        max distance from which a taxi is still assigned to a request
    
    taxis : dict
        storing all Taxi() instances in a dict
        keys are `taxi_id`s
    
    latest_taxi_id : int
        shows latest given taxi_id
        used or generating new taxis
    
    taxis_available : list of int
        stores `taxi_id`s of available taxis
    
    taxis_to_request : list of int
        stores `taxi_id`s of taxis moving to serve a request
    
    taxis_to_destination : list of int
        stores `taxi_id`s of taxis with passenger
    
    requests : dict
        storing all Request() instances in a dict
        keys are `request_id`s
    
    latest_request_id : int
        shows latest given request_id
        used or generating new requests
    
    requests_pending : list of int
        requests waiting to be served
    
    requests_in_progress : list of int
        requests with assigned taxis
    
    requests_dropped : list of int
        unsuccessful requests
    
    city : City
        geometry of class City() underlying the simulation
        
    show_map_labels : bool

    driver_break_cohort_mix : dict or None
        optional cohort mix, keys are cohort IDs and values are selection weights

    shift_duration_tu_config : dict or None
        optional per-cohort specification for sampling shift duration targets

    use_break_cohorts : bool
        whether cohort-based break logic is enabled

    break_cohort_ids : list of str
        cohort IDs used for weighted sampling

    break_cohort_probs : list of float
        normalized cohort probabilities aligned with `break_cohort_ids`

    day_length_tu : int
        number of simulation time units that define one day for time-of-day logic

    rush_windows_tu : list of dict
        list of windows (`start`, `end`) where end-of-shift deferral can happen

    p_defer_end_of_shift_in_rush : float
        probability to defer end-of-shift break during a rush window

    max_break_deferral_tu : int
        maximum cumulative end-of-shift deferral in time units

    break_cohort_settings : dict[str, dict[str, object]]
        per-cohort break parameter specs: `inter_shift_rest_tu`, `intra_shift_break_after_work_tu`,
        `intra_shift_break_duration_tu`, `demotivation_threshold_tu`, `shift_start_offset_tu`

    safety_score_change_serving_rate : float
        signed safety score delta applied per step while assigned to a request

    safety_score_change_waiting_rate : float
        signed safety score delta applied per step while unassigned and not on break

    safety_score_break_recovery_constant : float
        positive constant C for non-linear break recovery R(t)=t/(t+C)

    initial_safety_score_min : float
        lower bound for initial taxi safety score sampling

    initial_safety_score_max : float
        upper bound for initial taxi safety score sampling

    driver_route_pref_mix : dict[str, float]
        route preference mix for sampling taxi preference cohorts

    route_pref_keys : list[str]
        available route preference IDs used for weighted sampling

    route_pref_probs : list[float]
        normalized probabilities aligned with `route_pref_ids`

    route_pref_strength_low, route_pref_strength_high : float
        lower and upper bounds for sampled route preference strength

    nonpreferred_accept_ceiling : float
        maximum probability of accepting non-preferred routes

    preference_base_acceptance_prob : float
        acceptance probability upper bound when preference match is high

    route_length_short_max : int
        maximum origin-destination length treated as short route

    route_length_medium_max : int
        maximum origin-destination length treated as medium route

    satisfaction_score_min : float
        lower clipping bound for satisfaction score

    satisfaction_score_max : float
        upper clipping bound for satisfaction score

    satisfaction_initial_min, satisfaction_initial_max : float
        initial satisfaction sampling bounds

    satisfaction_change_waiting_rate : float
        signed per-step satisfaction change while waiting (unassigned and not on break)

    satisfaction_income_weight : float
        multiplier for normalized trip income contribution

    satisfaction_income_ref : float
        reference scale for trip income normalization

    satisfaction_pref_match_delta : float
        signed satisfaction delta for preferred route assignment

    satisfaction_pref_mismatch_delta : float
        signed satisfaction delta for non-preferred route assignment

    """

    driver_break_cohort_mix: dict[str, float] | None
    shift_duration_tu_config: dict[str, object] | None
    use_break_cohorts: bool
    break_cohort_ids: list[str]
    break_cohort_probs: list[float]
    day_length_tu: int
    rush_windows_tu: list[dict[str, int]]
    p_defer_end_of_shift_in_rush: float
    max_break_deferral_tu: int
    break_cohort_settings: dict[str, dict[str, object]]
    safety_score_change_serving_rate: float
    safety_score_change_waiting_rate: float
    safety_score_break_recovery_constant: float
    safety_score_min: float
    safety_score_max: float
    initial_safety_score_min: float
    initial_safety_score_max: float
    driver_route_pref_mix: dict[str, float]
    route_pref_keys: list[str]
    route_pref_probs: list[float]
    route_pref_strength_low: float
    route_pref_strength_high: float
    nonpreferred_accept_ceiling: float
    preference_base_acceptance_prob: float
    route_length_short_max: int
    route_length_medium_max: int
    satisfaction_score_min: float
    satisfaction_score_max: float
    satisfaction_initial_min: float
    satisfaction_initial_max: float
    satisfaction_change_waiting_rate: float
    satisfaction_income_weight: float
    satisfaction_income_ref: float
    satisfaction_pref_match_delta: float
    satisfaction_pref_mismatch_delta: float
    regions: list[dict]
    region_popularity_grid: np.ndarray | None
    region_default_popularity: float
    region_match_mode: str
    max_declines: int | None
    passenger_pref_types: list[str]
    passenger_pref_probs: list[float]
    passenger_safety_threshold_config: dict
    passenger_preference_weights_config: dict
    passenger_score_temperature: float

    def __init__(self, **config):

        self.rng = np.random.default_rng()

        # initializing time
        self.time = 0

        if "num_taxis" not in config or "request_rate" not in config:
            print("'num_taxis' & 'request_rate' are required! set default to 0")
        self.num_taxis = config.get("num_taxis", 0)
        self.request_rate = config.get("request_rate", 0)

        # price that is used for every trip
        self.price_fixed = config.get("price_fixed", 0)

        # price per unit distance while carrying a passenger
        self.price_per_dist = config.get("price_per_dist", 1)

        # cost per unit distance (e.g. gas)
        self.cost_per_unit = config.get("cost_per_unit", 0)

        # cost per time (e.g. amortization)
        self.cost_per_time = config.get("cost_per_time", 0)

        self.matching = config.get("matching", None)

        self.batch_size = config.get("batch_size", 1)
        if "max_time" not in config:
            raise ValueError("Define duration of simulation with config 'max_time'")
        self.max_time = config["max_time"]
        self.num_iter = int(np.ceil(self.max_time / self.batch_size))

        self.max_request_waiting_time = config.get("max_request_waiting_time", 10000)

        self.behaviour = config.get("behaviour", "go_back")  # goback / stay / cruise
        self.initial_conditions = config.get("initial_conditions", "base")  # base / home

        self.reset_time = config.get("reset_time", self.max_time + 1)

        # break mechanism (including cohort handling) lives in break_mechanism.py
        configure_breaks(self, config)
        self._last_break_check_day = -1  # sentinel so day-0 reset fires on first step

        # time-of-day request rate schedule
        self.request_rate_schedule = self.configure_request_schedule(config.get("request_rate_schedule", []))

        # safety score parameters (signed deltas per simulation step)
        # applied while taxi is assigned to a request (to pickup or with passenger)
        self.safety_score_change_serving_rate = float(config.get("safety_score_change_serving_rate", -0.02))
        # applied while taxi is not assigned to a request and not on break
        self.safety_score_change_waiting_rate = float(config.get("safety_score_change_waiting_rate", -0.001))

        self.safety_score_break_recovery_constant = float(config.get("safety_score_break_recovery_constant", 60.0))
        if self.safety_score_break_recovery_constant <= 0.0:
            raise ValueError("safety_score_break_recovery_constant must be > 0.")

        self.safety_score_min = config.get("safety_score_min", 0)
        self.safety_score_max = config.get("safety_score_max", 100)

        # bounds for initial safety score sampling at taxi creation
        self.initial_safety_score_min = float(config.get("initial_safety_score_min", self.safety_score_min))
        self.initial_safety_score_max = float(config.get("initial_safety_score_max", self.safety_score_max))

        # validation
        if self.initial_safety_score_min < self.safety_score_min or self.initial_safety_score_max > self.safety_score_max:
            raise ValueError(
                f"Initial safety score bounds {self.initial_safety_score_min} - {self.initial_safety_score_max} "
                f"must be within [{self.safety_score_min}, {self.safety_score_max}]."
            )
        if self.initial_safety_score_min > self.initial_safety_score_max:
            raise ValueError(
                f"initial_safety_score_min ({self.initial_safety_score_min}) must be less than or equal to "
                f"initial_safety_score_max ({self.initial_safety_score_max})."
            )

        # soft route preference configuration for preference-aware matching
        route_pref_mix = config.get(
            "driver_route_pref_mix",
            {"short_pref": 1.0, "neutral_pref": 1.0, "long_pref": 1.0}
        )
        if not isinstance(route_pref_mix, dict) or len(route_pref_mix) == 0:
            raise ValueError("driver_route_pref_mix must be a non-empty dict {str: float}.")

        allowed_route_prefs = {"short_pref", "neutral_pref", "long_pref"}
        route_pref_keys = []
        route_pref_weights = []
        for pref_key, weight in route_pref_mix.items():
            if pref_key not in allowed_route_prefs:
                raise ValueError(f"Unsupported route preference key: {pref_key}")
            w = float(weight)
            if w < 0:
                raise ValueError("driver_route_pref_mix weights must be non-negative.")
            route_pref_keys.append(pref_key)
            route_pref_weights.append(w)

        route_pref_total = float(sum(route_pref_weights))
        if route_pref_total <= 0:
            raise ValueError("driver_route_pref_mix must sum to a positive value.")

        self.driver_route_pref_mix = {k: float(v) for k, v in route_pref_mix.items()}
        self.route_pref_keys = route_pref_keys
        self.route_pref_probs = list(np.array(route_pref_weights) / route_pref_total)

        nonpreferred_accept_ceiling_val = config.get("nonpreferred_accept_ceiling", 0.2)

        self.nonpreferred_accept_ceiling = float(nonpreferred_accept_ceiling_val)
        if not (0.0 <= self.nonpreferred_accept_ceiling <= 1.0):
            raise ValueError("nonpreferred_accept_ceiling must be in [0, 1].")

        self.preference_base_acceptance_prob = float(config.get("preference_base_acceptance_prob", 0.9))
        if not (0.0 <= self.preference_base_acceptance_prob <= 1.0):
            raise ValueError("preference_base_acceptance_prob must be in [0, 1].")
        if self.preference_base_acceptance_prob < self.nonpreferred_accept_ceiling:
            raise ValueError("preference_base_acceptance_prob must be >= nonpreferred_accept_ceiling.")

        route_pref_strength_range = config.get("route_pref_strength_range", {"low": 0.2, "high": 0.9})
        if not isinstance(route_pref_strength_range, dict):
            raise ValueError("route_pref_strength_range must be a dict with low/high.")
        self.route_pref_strength_low = float(route_pref_strength_range.get("low", 0.2))
        self.route_pref_strength_high = float(route_pref_strength_range.get("high", 0.9))
        if self.route_pref_strength_low < 0.0 or self.route_pref_strength_high > 1.0:
            raise ValueError("route_pref_strength_range values must be in [0, 1].")
        if self.route_pref_strength_low > self.route_pref_strength_high:
            raise ValueError("route_pref_strength_range low must be <= high.")

        # driver income-flexibility model
        # income_target_rate: expected income per work-time unit; None disables the model
        income_target_rate_val = config.get("income_target_rate", None)
        self.income_target_rate = float(income_target_rate_val) if income_target_rate_val is not None else None
        if self.income_target_rate is not None and self.income_target_rate <= 0.0:
            raise ValueError("income_target_rate must be > 0.")
        self.driver_flexibility_threshold = float(config.get("driver_flexibility_threshold", 0.3))
        if not (0.0 <= self.driver_flexibility_threshold <= 1.0):
            raise ValueError("driver_flexibility_threshold must be in [0, 1].")
        self.driver_flexibility_temperature = float(config.get("driver_flexibility_temperature", 0.15))
        if self.driver_flexibility_temperature <= 0.0:
            raise ValueError("driver_flexibility_temperature must be > 0.")

        if "route_length_class_thresholds" in config:
            route_length_thresholds = config["route_length_class_thresholds"]
            if not isinstance(route_length_thresholds, dict):
                raise ValueError("route_length_class_thresholds must be a dict with short_max/medium_max.")
            self.route_length_short_max = int(route_length_thresholds.get("short_max", 8))
            self.route_length_medium_max = int(route_length_thresholds.get("medium_max", 16))
        elif "avg_request_lengths" in config:
            avg_req_len = float(config["avg_request_lengths"])
            if avg_req_len <= 0:
                raise ValueError("avg_request_lengths must be > 0 for derived route classes.")
            self.route_length_short_max = max(1, int(round(0.75 * avg_req_len)))
            self.route_length_medium_max = max(self.route_length_short_max + 1, int(round(1.25 * avg_req_len)))
        else:
            self.route_length_short_max = 8
            self.route_length_medium_max = 16
        if self.route_length_short_max < 0 or self.route_length_medium_max < self.route_length_short_max:
            raise ValueError("route length thresholds must satisfy 0 <= short_max <= medium_max.")

        # satisfaction score configuration
        self.satisfaction_score_min = float(config.get("satisfaction_score_min", 0.0))
        self.satisfaction_score_max = float(config.get("satisfaction_score_max", 100.0))
        self.satisfaction_initial_min = float(config.get("satisfaction_initial_min", 45.0))
        self.satisfaction_initial_max = float(config.get("satisfaction_initial_max", 55.0))
        if self.satisfaction_score_min > self.satisfaction_score_max:
            raise ValueError("satisfaction_score_min must be <= satisfaction_score_max.")
        if self.satisfaction_initial_min > self.satisfaction_initial_max:
            raise ValueError("satisfaction_initial_min must be <= satisfaction_initial_max.")
        if self.satisfaction_initial_min < self.satisfaction_score_min or self.satisfaction_initial_max > self.satisfaction_score_max:
            raise ValueError(
                "Satisfaction initial range must be within [satisfaction_score_min, satisfaction_score_max].")

        self.satisfaction_change_waiting_rate = float(config.get("satisfaction_change_waiting_rate", -0.01))
        self.satisfaction_income_weight = float(config.get("satisfaction_income_weight", 0.5))
        self.satisfaction_income_ref = float(config.get("satisfaction_income_ref", 1000.0))
        if self.satisfaction_income_ref <= 0:
            raise ValueError("satisfaction_income_ref must be > 0.")

        self.satisfaction_pref_match_delta = float(config.get("satisfaction_pref_match_delta", 0.2))
        self.satisfaction_pref_mismatch_delta = float(config.get("satisfaction_pref_mismatch_delta", -0.3))

        # initializing counters
        self.latest_taxi_id = 0
        self.latest_request_id = 0

        # initializing object storage
        self.taxis = RandomDict()
        self.taxis_available = RandomDict()
        self.taxis_on_break = set()
        self.taxis_to_request = set()
        self.taxis_to_destination = set()

        self.requests = dict()
        self._requests_done_buffer = []
        self.requests_pending = set()

        # speeding up going through requests in the order of waiting times
        # they are pushed into a deque in the order of timestamps
        self.requests_pending_deque = deque()
        self.requests_pending_deque_batch = deque(maxlen=self.max_request_waiting_time)
        self.requests_pending_deque_temporary = deque()
        self.requests_in_progress = set()

        # city layout
        self.city = City(**config)
        # size random-number pool by peak effective rate so it doesn't exhaust early
        max_multiplier = max((w["multiplier"] for w in self.request_rate_schedule), default=1.0)
        self.city.length = int(min(self.max_time * self.request_rate * max_multiplier, 1e6))

        # whether to log all movements for debugging purposes
        self.log = config["log"]
        self.city.log = self.log
        # showing map of moving taxis in interactive jupyter notebook
        self.show_plot = config["show_plot"]

        # region config used for nearest_region_pref matching and per-region safety aggregation
        region_config = config.get("regions", None)
        if region_config is not None:
            self.region_default_popularity = float(region_config.get("default_popularity", 100.0))
            self.region_match_mode = region_config.get("match_mode", "origin")
            self.regions = region_config.get("regions", [])
            self._build_region_popularity_grid(region_config)
        else:
            self.region_popularity_grid = None
            self.region_default_popularity = 100.0
            self.region_match_mode = "origin"
            self.regions = []
        self.max_declines = config.get("max_declines", None)

        # passenger preference configuration
        passenger_pref_mix = config.get("passenger_preference_mix", {
            "safety_indifferent": 1.0,
            "safety_moderate": 1.0,
            "safety_strict": 1.0
        })
        if not isinstance(passenger_pref_mix, dict) or len(passenger_pref_mix) == 0:
            raise ValueError("passenger_preference_mix must be a non-empty dict.")
        pref_total = sum(passenger_pref_mix.values())
        if pref_total <= 0:
            raise ValueError("passenger_preference_mix weights must sum to a positive value.")
        self.passenger_pref_types = list(passenger_pref_mix.keys())
        self.passenger_pref_probs = [float(v) / pref_total for v in passenger_pref_mix.values()]

        self.passenger_safety_threshold_config = config.get("passenger_safety_threshold", {
            "safety_indifferent": {"mean": 0.0, "std": 0.0},
            "safety_moderate": {"mean": 55.0, "std": 10.0},
            "safety_strict": {"mean": 75.0, "std": 8.0}
        })

        self.passenger_preference_weights_config = config.get("passenger_preference_weights", {
            "w_dist": {"mean": 1.0, "std": 0.2},
            "w_safety": {"mean": 0.5, "std": 0.3},
            "w_wait": {"mean": 0.3, "std": 0.2}
        })

        self.passenger_score_temperature = float(config.get("passenger_score_temperature", 0.1))
        if self.passenger_score_temperature <= 0:
            raise ValueError("passenger_score_temperature must be > 0.")

        # initializing simulation with taxis
        for _ in range(self.num_taxis):
            self.add_taxi()

        # probably not used for anything but not sure so it will stay for now
        self.taxi_df = pd.DataFrame.from_records([dict(v) for k, v in self.taxis.items()], index='taxi_id')

        if self.show_plot:
            # plotting variables
            self.canvas = plt.figure()
            self.canvas_ax = self.canvas.add_subplot(1, 1, 1)
            self.canvas_ax.set_aspect('equal', 'box')
            self.cmap = plt.get_cmap('viridis')
            self.taxi_colors = list(np.linspace(0, 0.85, self.num_taxis))
            shuffle(self.taxi_colors)
            self.show_map_labels = config["show_map_labels"]
            self.show_pending = config["show_pending"]
            self.init_canvas()

    def configure_request_schedule(self, raw_config: list) -> list:
        if not isinstance(raw_config, list):
            raise ValueError("request_rate_schedule must be a list of {start, end, multiplier} dicts.")

        schedule = []
        for entry in raw_config:
            if not isinstance(entry, dict) or "start" not in entry or "end" not in entry or "multiplier" not in entry:
                raise ValueError("Each request_rate_schedule entry must have start, end, and multiplier.")
            start = int(entry["start"])
            end = int(entry["end"])
            multiplier = float(entry["multiplier"])
            if not (0 <= start < self.day_length_tu) or not (0 <= end <= self.day_length_tu):
                raise ValueError(
                    "request_rate_schedule start must be in [0, day_length_tu) "
                    "and end must be in [0, day_length_tu]."
                )
            if multiplier < 0:
                raise ValueError("request_rate_schedule multiplier must be >= 0.")
            schedule.append({"start": start, "end": end, "multiplier": multiplier})

        # Detect overlapping windows (convert to flat coverage set and check)
        covered = set()
        for entry in schedule:
            # remember: set & intersect, | union
            s, e = entry["start"], entry["end"]
            if s < e:
                span = set(range(s, e))
            else:
                span = set(range(s, self.day_length_tu)) | set(range(0, e))
            overlap = covered & span
            if overlap:
                raise ValueError(
                    f"request_rate_schedule has overlapping windows near time-of-day {min(overlap)}."
                )
            covered |= span

        if covered != set(range(0, self.day_length_tu)):
            print("Warning: request_rate_schedule has gaps in coverage. "
                  "Times not covered by any window will use the base request_rate.")

        return schedule

    def get_effective_request_rate(self) -> float:
        if self.request_rate_schedule:
            tod = self.time % self.day_length_tu
            for window in self.request_rate_schedule:
                s, e = window["start"], window["end"]
                in_window = (s <= tod < e) if s <= e else (tod >= s or tod < e)
                if in_window:
                    return self.request_rate * window["multiplier"]
        return self.request_rate

    def init_canvas(self):
        """
        Initialize plot.
        
        """
        self.canvas_ax.clear()
        self.canvas_ax.set_xlim(-0.5, self.city.n - 0.5)
        self.canvas_ax.set_ylim(-0.5, self.city.m - 0.5)

        self.canvas_ax.tick_params(length=0)
        self.canvas_ax.xaxis.set_ticks(list(range(self.city.n)))
        self.canvas_ax.yaxis.set_ticks(list(range(self.city.m)))
        if not self.show_map_labels:
            self.canvas_ax.xaxis.set_ticklabels([])
            self.canvas_ax.yaxis.set_ticklabels([])

        self.canvas_ax.set_aspect('equal', 'box')
        self.canvas.tight_layout()
        self.canvas_ax.grid()

    def add_taxi(self):
        """
        Create new taxi.
        
        """

        # adding home coordinates, starting taxi
        home = self.city.create_taxi_home_coords()
        initial_safety = self.rng.uniform(self.initial_safety_score_min, self.initial_safety_score_max)

        if self.initial_conditions == "base":
            # create a taxi at the base
            tx = Taxi(self.city.base_coords, self.latest_taxi_id, safety_score=initial_safety)
        elif self.initial_conditions == "home":
            # create a taxi at home
            tx = Taxi(home, self.latest_taxi_id, safety_score=initial_safety)
        else:
            raise NotImplementedError(f"behavior for adding taxi with initial_condition {self.initial_conditions} not defined")

        tx = assign_break_cohort_to_taxi(self, tx)
        tx.route_length_pref = str(self.rng.choice(self.route_pref_keys, p=self.route_pref_probs))
        tx.pref_strength_route = float(self.rng.uniform(self.route_pref_strength_low, self.route_pref_strength_high))
        tx.nonpreferred_accept_ceiling = self.nonpreferred_accept_ceiling
        tx.satisfaction_score = float(self.rng.uniform(self.satisfaction_initial_min, self.satisfaction_initial_max))
        tx.last_satisfaction_delta = 0.0
        tx.home = home

        # add to taxi storage
        self.taxis[self.latest_taxi_id] = tx
        # add to available taxi matrix
        self.city.A[self.city.coordinate_dict_ij_to_c[tx.x][tx.y]].add(self.latest_taxi_id)
        # add to available taxi storage
        self.taxis_available[self.latest_taxi_id] = tx
        # increase counter
        self.latest_taxi_id += 1

    def add_request(self):
        """
        Create new request.

        """
        # here we randomly choose a place for the request
        # the random coordinates are pre-stored in a deque for faster access
        # if there are no more pregenerated coordinates in the deque, we generate some more

        # origin and destination coordinates
        ox, oy, dx, dy = self.city.create_one_request_coord()
        r = Request([ox, oy], [dx, dy], self.latest_request_id, self.time)

        # sample passenger type and preference weights
        ptype = str(self.rng.choice(self.passenger_pref_types, p=self.passenger_pref_probs))
        r.passenger_type = ptype
        thresh_params = self.passenger_safety_threshold_config.get(ptype, {"mean": 0.0, "std": 0.0})
        r.safety_threshold = float(np.clip(
            self.rng.normal(thresh_params["mean"], thresh_params["std"]), 0.0, 100.0
        ))
        for wkey in ("w_dist", "w_safety", "w_wait"):
            params = self.passenger_preference_weights_config.get(wkey, {"mean": 1.0, "std": 0.0})
            setattr(r, wkey, float(max(0.0, self.rng.normal(params["mean"], params["std"]))))

        # add to request storage
        self.requests[self.latest_request_id] = r

        # add to free users
        self.requests_pending_deque.append(self.latest_request_id)
        # increase counter
        self.latest_request_id += 1

    def go_to_base(self, taxi_id, bcoords):
        """
        This function sends the taxi to the base rom wherever it is.
        """

        # fetch object
        t = self.taxis[taxi_id]

        # actual coordinates
        acoords = [t.x, t.y]
        # path between actual coordinates and destination
        path = self.city.create_path(acoords, bcoords)

        # erase path memory
        t.with_passenger = False
        t.to_request = False
        t.available = True
        #        print("Erasing path memory, Taxi "+str(taxi_id)+".")
        t.next_destination = deque()
        # put path into taxi path queue
        #        print("Filling path memory, Taxi "+str(taxi_id)+". Path ",path)
        t.next_destination.extend(path)

        # put object back to its place
        self.taxis[taxi_id] = t

    # TODO: make taxis drive home instead of teleporting (do not forget the going_home in dropoff_request() method!)
    def go_home_everybody(self):
        """
        Drop requests that are currently executing, and set taxis as available at their home locations.
        """
        for taxi_id in self.taxis:
            tx = self.taxis[taxi_id]

            if taxi_id in self.taxis_available:
                # (magic wand) Apparate taxi home!
                self.city.A[self.city.coordinate_dict_ij_to_c[tx.x][tx.y]].remove(taxi_id)
                self.city.A[self.city.coordinate_dict_ij_to_c[tx.home[0]][tx.home[1]]].add(taxi_id)
                tx.x, tx.y = tx.home

            if taxi_id in self.taxis_to_destination:
                # if somebody is sitting in it, finish request
                self.dropoff_request(tx.actual_request_executing, mode="going_home")

            if taxi_id in self.taxis_to_request:
                # if it was only going towards a request, cancel it
                self.dropoff_request(tx.actual_request_executing, mode="cancel")

            if taxi_id in self.taxis_on_break:
                # update position so the taxi returns from break at home, not mid-city
                tx.x, tx.y = tx.home

            self.taxis[taxi_id] = tx

    def cruise(self, taxi_id):
        return None

    def assign_request(self, request_id, taxi_id):
        """
        Given a request_id, taxi_id pair, this function makes the match.
        It sets new state variables for the request and the taxi, updates path of the taxi etc.
        """
        r = self.requests[request_id]
        t = self.taxis[taxi_id]

        # pair the match
        t.actual_request_executing = request_id
        r.taxi_id = taxi_id

        # remove taxi from the available ones
        self.city.A[self.city.coordinate_dict_ij_to_c[t.x][t.y]].remove(taxi_id)
        del self.taxis_available[taxi_id]
        t.with_passenger = False
        t.available = False
        t.to_request = True

        # reset waiting time since last trip (got a new assignment)
        t.time_waiting_since_last_trip = 0

        # mark taxi as moving to request
        self.taxis_to_request.add(taxi_id)

        # forget the path that has been assigned
        t.next_destination = deque()

        # create new path: to user, then to destination
        path = self.city.create_path([t.x, t.y], [r.ox, r.oy]) + \
               self.city.create_path([r.ox, r.oy], [r.dx, r.dy])[1:]
        t.next_destination.extend(path)

        # remove request from the pending ones, label it as "in progress"
        self.requests_in_progress.add(request_id)
        r.mode = 'waiting'
        r.timestamps['assigned'] = self.time
        r.driver_safety_score_start = float(t.safety_score)
        # record taxi position at assignment
        r.assigned_taxi_pos = [int(t.x), int(t.y)]
        try:
            r.assigned_taxi_distance = int(self.city.measure_distance([t.x, t.y], [r.ox, r.oy]))
        except Exception:
            r.assigned_taxi_distance = None

        route_class = self._classify_request_route(r)
        if self._is_preferred_route(t.route_length_pref, route_class):
            self._apply_satisfaction_delta(t, self.satisfaction_pref_match_delta)
        else:
            self._apply_satisfaction_delta(t, self.satisfaction_pref_mismatch_delta)

        # update taxi state in taxi storage
        self.taxis[taxi_id] = t
        # update request state
        self.requests[request_id] = r

        if self.log:
            print("\tM request " + str(request_id) + " taxi " + str(taxi_id))

    def matching_algorithm(self, mode="random_unlimited"):
        """
        This function contains the possible matching functions which are selected by the mode keyword.

        Parameters
        ----------

        mode : str, default baseline
            matching algorithm mode
                * random_unlimited : assigning a random taxi to the user
                * random_limited : assigning a random taxi to the user within the circle of a radius self.city.hard_limit
                * nearest : sending the nearest available taxi for the user from within the circle of a radius self.city.hard_limit
                * poorest : sending the least earning available taxi for the user from within the circle of a radius self.city.hard_limit
                * nearest_distance_pref : nearest matching with route-length preference-based acceptance; taxis that decline are excluded from future re-matches for the same request
                * nearest_region_pref : nearest matching with region-popularity-based acceptance; taxis that decline are excluded from future re-matches for the same request
                * safety_objective : objective safety-optimal matching; processes regions from least-safe to most-safe (by avg safety of available taxis at pickup), oldest request first per region, assigns the safest available taxi within hard_limit radius - no preferences applied
        """

        if len(self.requests_pending_deque) == 0:
            if self.log:
                print("\tNo pending requests.")
            return

        if self.log:
            print('\tMatching algorithm.')

        match mode:
            case "random_unlimited":
                while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                    # select a random taxi
                    taxi_id = self.taxis_available.random_key()
                    # select oldest request from deque
                    request_id = self.requests_pending_deque.popleft()
                    # make assignment
                    self.assign_request(request_id, taxi_id)

            case "random_limited":
                while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                    # select oldest request from deque
                    request_id = self.requests_pending_deque.popleft()
                    # fetch request
                    r = self.requests[request_id]
                    # search for nearest free taxis
                    possible_taxi_ids = self.city.find_nearest_available_taxis(
                        self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                        mode="circle",
                        radius=self.city.hard_limit
                    )
                    # if there were any taxis near
                    if len(possible_taxi_ids) > 0:
                        # select taxi
                        taxi_id = choice(possible_taxi_ids)
                        self.assign_request(request_id, taxi_id)
                    else:
                        # mark request as still pending
                        self.requests_pending_deque_temporary.append(request_id)

            case "nearest":
                while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                    # select oldest request from deque
                    request_id = self.requests_pending_deque.popleft()
                    # fetch request
                    r = self.requests[request_id]
                    # search for nearest free taxis
                    possible_taxi_ids = self.city.find_nearest_available_taxis(
                        self.city.coordinate_dict_ij_to_c[r.ox][r.oy])
                    # if there were any taxis near
                    if len(possible_taxi_ids) > 0:
                        # select taxi
                        taxi_id = choice(possible_taxi_ids)
                        self.assign_request(request_id, taxi_id)
                    else:
                        # mark request as still pending
                        self.requests_pending_deque_temporary.append(request_id)

            case "poorest":
                # always order taxi that has earned the least money so far
                # but choose only from the nearest ones
                # hard limiting: e.g. if there is no taxi within the radius, then quit

                # evaluate the earnings of the available taxis so far
                ta_list = list(self.taxis_available.keys)
                taxi_earnings = [self.eval_taxi_income(taxi_id) for taxi_id in ta_list]
                ta_list = list(np.array(ta_list)[np.argsort(taxi_earnings)])

                while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                    # select oldest request from deque
                    request_id = self.requests_pending_deque.popleft()
                    # fetch request
                    r = self.requests[request_id]
                    # find nearest vehicles in a radius
                    possible_taxi_ids = self.city.find_nearest_available_taxis(
                        self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                        mode="circle",
                        radius=self.city.hard_limit)
                    hit = 0
                    for t in ta_list:
                        if t in possible_taxi_ids:
                            # on first hit
                            # make assignment
                            self.assign_request(request_id, t)
                            hit = 1
                            break
                    if not hit:
                        self.requests_pending_deque_temporary.append(request_id)

            case "nearest_distance_pref":
                while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                    request_id = self.requests_pending_deque.popleft()
                    r = self.requests[request_id]

                    all_within = self.city.find_nearest_available_taxis(
                        self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                        mode="circle",
                        radius=self.city.hard_limit
                    )
                    possible_taxi_ids = sorted(
                        [t for t in all_within if t not in r.declined_taxi_ids],
                        key=lambda tid, req=r: abs(self.taxis[tid].x - req.ox) + abs(self.taxis[tid].y - req.oy)
                    )

                    if not possible_taxi_ids:
                        self.requests_pending_deque_temporary.append(request_id)
                        continue

                    route_class = self._classify_request_route(r)

                    assigned = False
                    for taxi_id in possible_taxi_ids:
                        taxi = self.taxis[taxi_id]
                        forced = (
                                self.max_declines is not None and
                                taxi.decline_count >= self.max_declines
                        )
                        p_accept = self._acceptance_probability(taxi, route_class)
                        if forced or self.rng.random() < p_accept:
                            taxi.decline_count = 0
                            self.assign_request(request_id, taxi_id)
                            assigned = True
                            break
                        else:
                            taxi.decline_count += 1
                            taxi.total_declines += 1
                            r.declined_taxi_ids.add(taxi_id)

                    if not assigned:
                        self.requests_pending_deque_temporary.append(request_id)

            case "nearest_region_pref":
                while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                    request_id = self.requests_pending_deque.popleft()
                    r = self.requests[request_id]

                    all_within = self.city.find_nearest_available_taxis(
                        self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                        mode="circle",
                        radius=self.city.hard_limit
                    )
                    # exclude taxis that already declined this request
                    possible_taxi_ids = sorted(
                        [t for t in all_within if t not in r.declined_taxi_ids],
                        key=lambda tid, req=r: abs(self.taxis[tid].x - req.ox) + abs(self.taxis[tid].y - req.oy)
                    )

                    if not possible_taxi_ids:
                        self.requests_pending_deque_temporary.append(request_id)
                        continue

                    p_base = self._region_acceptance_probability(r)

                    assigned = False
                    for taxi_id in possible_taxi_ids:
                        taxi = self.taxis[taxi_id]
                        # forced to accept when taxi has exhausted its decline budget
                        forced = (
                                self.max_declines is not None and
                                taxi.decline_count >= self.max_declines
                        )
                        flexibility = self._driver_income_flexibility(taxi)
                        p_accept = p_base + flexibility * (self.preference_base_acceptance_prob - p_base)
                        if forced or self.rng.random() < p_accept:
                            taxi.decline_count = 0
                            self.assign_request(request_id, taxi_id)
                            assigned = True
                            break
                        else:
                            taxi.decline_count += 1
                            taxi.total_declines += 1
                            r.declined_taxi_ids.add(taxi_id)

                    if not assigned:
                        self.requests_pending_deque_temporary.append(request_id)

            case "nearest_passenger_pref":
                # Nearest matching with passenger preference-weighted scoring
                # Passengers score each candidate taxi by proximity, safety, and pickup wait
                # acceptance probability decays the safety threshold as wait time grows, so any taxi is eventually accepted.
                while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                    request_id = self.requests_pending_deque.popleft()
                    r = self.requests[request_id]

                    all_within = self.city.find_nearest_available_taxis(
                        self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                        mode="circle",
                        radius=self.city.hard_limit
                    )

                    if not all_within:
                        r.last_no_match_reason = 'no_taxi_available'
                        self.requests_pending_deque_temporary.append(request_id)
                        continue

                    # Score all candidates; try in descending score order
                    scored = sorted(all_within, key=lambda tid, req=r: -self._passenger_score(req, self.taxis[tid]))

                    assigned = False
                    for taxi_id in scored:
                        score = self._passenger_score(r, self.taxis[taxi_id])
                        p_accept = self._passenger_acceptance_prob(r, score)
                        if self.rng.random() < p_accept:
                            self.assign_request(request_id, taxi_id)
                            assigned = True
                            break

                    if assigned:
                        r.last_no_match_reason = None
                    else:
                        r.last_no_match_reason = 'passenger_declined'
                        self.requests_pending_deque_temporary.append(request_id)

            case "nearest_two_sided_dist_pass_pref":
                # Two-sided preference matching:
                # driver route-length preference (nearest_distance_pref style) combined with passenger preference scoring.
                # Driver declines are permanent for this request; passenger rejections are not - the passenger threshold relaxes over time.
                while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                    request_id = self.requests_pending_deque.popleft()
                    r = self.requests[request_id]

                    all_within = self.city.find_nearest_available_taxis(
                        self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                        mode="circle",
                        radius=self.city.hard_limit
                    )
                    # Exclude driver-declined taxis; sort remaining by distance
                    candidate_ids = sorted(
                        [t for t in all_within if t not in r.declined_taxi_ids],
                        key=lambda tid, req=r: abs(self.taxis[tid].x - req.ox) + abs(self.taxis[tid].y - req.oy)
                    )

                    if not candidate_ids:
                        r.last_no_match_reason = 'no_taxi_available' if not all_within else 'driver_declined'
                        self.requests_pending_deque_temporary.append(request_id)
                        continue

                    route_class = self._classify_request_route(r)
                    assigned = False
                    any_driver_willing = False

                    for taxi_id in candidate_ids:
                        taxi = self.taxis[taxi_id]

                        # driver side: route distance pref
                        forced_driver = (
                                self.max_declines is not None and
                                taxi.decline_count >= self.max_declines
                        )
                        p_driver = self._acceptance_probability(taxi, route_class)
                        driver_accepts = forced_driver or self.rng.random() < p_driver
                        if not driver_accepts:
                            taxi.decline_count += 1
                            taxi.total_declines += 1
                            r.declined_taxi_ids.add(taxi_id)
                            continue

                        # passenger side: preference score
                        any_driver_willing = True
                        score = self._passenger_score(r, taxi)
                        p_passenger = self._passenger_acceptance_prob(r, score)
                        if self.rng.random() < p_passenger:
                            taxi.decline_count = 0
                            self.assign_request(request_id, taxi_id)
                            assigned = True
                            break
                        # Passenger declines: taxi stays available;
                        # not added to declined_taxi_ids so it can be re-evaluated next round with a relaxed threshold

                    if assigned:
                        r.last_no_match_reason = None
                    else:
                        if any_driver_willing:
                            r.last_no_match_reason = 'passenger_declined'
                        elif candidate_ids:
                            r.last_no_match_reason = 'driver_declined'
                        else:
                            r.last_no_match_reason = 'no_taxi_available'
                        self.requests_pending_deque_temporary.append(request_id)

            case "nearest_two_sided_region_pass_pref":
                # Two-sided preference matching:
                # driver region popularity preference (nearest_region_pref style) combined with passenger preference scoring.
                # Driver declines are permanent for this request; passenger rejections are not - the passenger threshold relaxes over time.
                while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                    request_id = self.requests_pending_deque.popleft()
                    r = self.requests[request_id]

                    all_within = self.city.find_nearest_available_taxis(
                        self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                        mode="circle",
                        radius=self.city.hard_limit
                    )
                    candidate_ids = sorted(
                        [t for t in all_within if t not in r.declined_taxi_ids],
                        key=lambda tid, req=r: abs(self.taxis[tid].x - req.ox) + abs(self.taxis[tid].y - req.oy)
                    )

                    if not candidate_ids:
                        r.last_no_match_reason = 'no_taxi_available' if not all_within else 'driver_declined'
                        self.requests_pending_deque_temporary.append(request_id)
                        continue

                    p_driver = self._region_acceptance_probability(r)
                    assigned = False
                    any_driver_willing = False

                    for taxi_id in candidate_ids:
                        taxi = self.taxis[taxi_id]

                        # driver side: region popularity
                        forced_driver = (
                                self.max_declines is not None and
                                taxi.decline_count >= self.max_declines
                        )
                        driver_accepts = forced_driver or self.rng.random() < p_driver
                        if not driver_accepts:
                            taxi.decline_count += 1
                            taxi.total_declines += 1
                            r.declined_taxi_ids.add(taxi_id)
                            continue

                        # passenger side: preference score
                        any_driver_willing = True
                        score = self._passenger_score(r, taxi)
                        p_passenger = self._passenger_acceptance_prob(r, score)
                        if self.rng.random() < p_passenger:
                            taxi.decline_count = 0
                            self.assign_request(request_id, taxi_id)
                            assigned = True
                            break

                    if assigned:
                        r.last_no_match_reason = None
                    else:
                        if any_driver_willing:
                            r.last_no_match_reason = 'passenger_declined'
                        elif candidate_ids:
                            r.last_no_match_reason = 'driver_declined'
                        else:
                            r.last_no_match_reason = 'no_taxi_available'
                        self.requests_pending_deque_temporary.append(request_id)

            case "safety_objective":
                # Objective safety-optimal matching
                # Process regions from least-safe to most-safe (lowest avg safety score of currently available taxis at pickup first)
                # Within each region, serve the oldest pending request first
                # Each request is matched with the safest available taxi within hard_limit radius - no preferences applied.

                # Drain the pending deque into a flat list preserving arrival order
                all_pending = []
                while self.requests_pending_deque:
                    all_pending.append(self.requests_pending_deque.popleft())

                if not self.regions:
                    # No region config: assign safest taxi within radius in arrival order
                    for request_id in all_pending:
                        r = self.requests[request_id]
                        possible_taxi_ids = self.city.find_nearest_available_taxis(
                            self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                            mode="circle",
                            radius=self.city.hard_limit
                        )
                        if not possible_taxi_ids:
                            self.requests_pending_deque_temporary.append(request_id)
                            continue
                        best_taxi_id = max(
                            possible_taxi_ids,
                            key=lambda tid: self.taxis[tid].safety_score
                        )
                        self.assign_request(request_id, best_taxi_id)
                else:
                    # Group requests by pickup region, preserving arrival order
                    region_requests = {region["id"]: [] for region in self.regions}
                    unregioned = []
                    for request_id in all_pending:
                        r = self.requests[request_id]
                        matched = None
                        for region in self.regions:
                            if (region["x_min"] <= r.ox <= region["x_max"] and
                                    region["y_min"] <= r.oy <= region["y_max"]):
                                matched = region["id"]
                                break
                        if matched is not None:
                            region_requests[matched].append(request_id)
                        else:
                            unregioned.append(request_id)

                    # Compute avg safety score of available taxis per region
                    region_safety = self.compute_region_safety_averages()

                    # Sort regions ascending by supply safety; None (no taxis) → least safe
                    def _supply_safety(region):
                        avg = region_safety.get(region["id"], {}).get("avg_safety_score")
                        return avg if avg is not None else float('-inf')

                    sorted_regions = sorted(self.regions, key=_supply_safety)

                    # Process regions from least-safe to most-safe
                    for region in sorted_regions:
                        for request_id in region_requests[region["id"]]:
                            r = self.requests[request_id]
                            possible_taxi_ids = self.city.find_nearest_available_taxis(
                                self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                                mode="circle",
                                radius=self.city.hard_limit
                            )
                            if not possible_taxi_ids:
                                self.requests_pending_deque_temporary.append(request_id)
                                continue
                            best_taxi_id = max(
                                possible_taxi_ids,
                                key=lambda tid: self.taxis[tid].safety_score
                            )
                            self.assign_request(request_id, best_taxi_id)

                    # Requests outside any region get the same safest-taxi-in-radius rule, served last
                    for request_id in unregioned:
                        r = self.requests[request_id]
                        possible_taxi_ids = self.city.find_nearest_available_taxis(
                            self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                            mode="circle",
                            radius=self.city.hard_limit
                        )
                        if not possible_taxi_ids:
                            self.requests_pending_deque_temporary.append(request_id)
                            continue
                        best_taxi_id = max(
                            possible_taxi_ids,
                            key=lambda tid: self.taxis[tid].safety_score
                        )
                        self.assign_request(request_id, best_taxi_id)

            case "safety_objective_two_sided":
                # System safety-objective matching with two-sided preferences (region driver + passenger).
                # System layer : regions ordered least-safe → most-safe; candidates sorted by safety
                #                score descending so the safest available taxi is tried first.
                # Individual layer : driver region-popularity preference (_region_acceptance_probability)
                #                    and passenger safety-score preference (_passenger_acceptance_prob)
                #                    applied within that ordering.
                # Driver declines are permanent for this request (added to declined_taxi_ids).
                # Passenger rejections are not permanent - the threshold relaxes each round.

                # Drain the pending deque into a flat list preserving arrival order
                all_pending = []
                while self.requests_pending_deque:
                    all_pending.append(self.requests_pending_deque.popleft())

                def _process_sot(request_id):
                    r = self.requests[request_id]
                    all_within = self.city.find_nearest_available_taxis(
                        self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                        mode="circle",
                        radius=self.city.hard_limit
                    )
                    candidate_ids = sorted(
                        [t for t in all_within if t not in r.declined_taxi_ids],
                        key=lambda tid: self.taxis[tid].safety_score,
                        reverse=True  # safest taxi offered first
                    )
                    if not candidate_ids:
                        r.last_no_match_reason = 'no_taxi_available' if not all_within else 'driver_declined'
                        self.requests_pending_deque_temporary.append(request_id)
                        return

                    p_driver = self._region_acceptance_probability(r)  # same for all taxis for this request
                    assigned = False
                    any_driver_willing = False

                    for taxi_id in candidate_ids:
                        taxi = self.taxis[taxi_id]

                        # driver side: region popularity preference
                        forced_driver = (
                            self.max_declines is not None and
                            taxi.decline_count >= self.max_declines
                        )
                        driver_accepts = forced_driver or self.rng.random() < p_driver
                        if not driver_accepts:
                            taxi.decline_count += 1
                            taxi.total_declines += 1
                            r.declined_taxi_ids.add(taxi_id)
                            continue

                        # passenger side: safety-score preference
                        any_driver_willing = True
                        score = self._passenger_score(r, taxi)
                        p_passenger = self._passenger_acceptance_prob(r, score)
                        if self.rng.random() < p_passenger:
                            taxi.decline_count = 0
                            self.assign_request(request_id, taxi_id)
                            assigned = True
                            break

                    if assigned:
                        r.last_no_match_reason = None
                    else:
                        if any_driver_willing:
                            r.last_no_match_reason = 'passenger_declined'
                        elif candidate_ids:
                            r.last_no_match_reason = 'driver_declined'
                        else:
                            r.last_no_match_reason = 'no_taxi_available'
                        self.requests_pending_deque_temporary.append(request_id)

                if not self.regions:
                    # No region config: process in arrival order
                    for request_id in all_pending:
                        _process_sot(request_id)
                else:
                    # Group requests by pickup region, preserving arrival order
                    region_requests = {region["id"]: [] for region in self.regions}
                    unregioned = []
                    for request_id in all_pending:
                        r = self.requests[request_id]
                        matched = None
                        for region in self.regions:
                            if (region["x_min"] <= r.ox <= region["x_max"] and
                                    region["y_min"] <= r.oy <= region["y_max"]):
                                matched = region["id"]
                                break
                        if matched is not None:
                            region_requests[matched].append(request_id)
                        else:
                            unregioned.append(request_id)

                    # Compute avg safety score of available taxis per region
                    region_safety = self.compute_region_safety_averages()

                    def _supply_safety_sot(region):
                        avg = region_safety.get(region["id"], {}).get("avg_safety_score")
                        return avg if avg is not None else float('-inf')

                    sorted_regions = sorted(self.regions, key=_supply_safety_sot)

                    # Process regions from least-safe to most-safe
                    for region in sorted_regions:
                        for request_id in region_requests[region["id"]]:
                            _process_sot(request_id)

                    # Requests outside any region served last
                    for request_id in unregioned:
                        _process_sot(request_id)

            case _:
                raise ValueError("I know of no such assignment mode! Please provide a valid one!")

    def _classify_request_route(self, request):
        route_length = int(self.city.measure_distance([request.ox, request.oy], [request.dx, request.dy]))
        if route_length <= self.route_length_short_max:
            return "short"
        if route_length <= self.route_length_medium_max:
            return "medium"
        return "long"

    @staticmethod
    def _route_match_score(route_pref, pref_strength, route_class):
        if route_pref == "neutral_pref":
            return 0.7
        if route_pref == "short_pref":
            return 1.0 if route_class == "short" else max(0.0, 1.0 - pref_strength)
        if route_pref == "long_pref":
            return 1.0 if route_class == "long" else max(0.0, 1.0 - pref_strength)
        return 1.0

    @staticmethod
    def _is_preferred_route(route_pref, route_class):
        if route_pref == "neutral_pref":
            return True
        if route_pref == "short_pref":
            return route_class == "short"
        if route_pref == "long_pref":
            return route_class == "long"
        return True

    def _driver_income_flexibility(self, taxi) -> float:
        """Income-pressure flexibility factor in [0, 1].

        Returns 0 when income is on target (preferences fully active) and
        approaches 1 when the driver's income significantly lags the expected
        pace for elapsed shift work time (preferences relax toward neutral).

        The shortfall fraction is:
            shortfall = max(0, expected_shift_income - actual_shift_income)
                        / (income_target_rate * shift_duration_tu)

        If shift_duration_tu is None the normalization falls back to the
        expected income accumulated so far (so flexibility still grows as the
        gap widens, just without an absolute shift-length reference).

        Note: shift_start_income must be reset alongside shift_start_work_time_tu
        at the start of each new shift so the computation stays shift-relative.
        """
        if self.income_target_rate is None:
            return 0.0
        work_time = float(
            taxi.time_serving + taxi.time_to_request +
            taxi.time_cruising + taxi.time_waiting
        )
        elapsed_shift_work = max(0.0, work_time - taxi.shift_start_work_time_tu)
        if elapsed_shift_work == 0.0:
            return 0.0
        expected_shift_income = self.income_target_rate * elapsed_shift_work
        actual_shift_income = self.eval_taxi_income(taxi.taxi_id) - taxi.shift_start_income
        if taxi.shift_duration_tu is not None:
            norm = self.income_target_rate * taxi.shift_duration_tu
        else:
            norm = max(1.0, expected_shift_income)
        shortfall = max(0.0, expected_shift_income - actual_shift_income) / max(1.0, norm)
        return float(1.0 / (1.0 + np.exp(
            -(shortfall - self.driver_flexibility_threshold) / self.driver_flexibility_temperature
        )))

    def _acceptance_probability(self, taxi, route_class):
        flexibility = self._driver_income_flexibility(taxi)
        effective_pref_strength = taxi.pref_strength_route * (1.0 - flexibility)
        ceiling = min(self.nonpreferred_accept_ceiling, taxi.nonpreferred_accept_ceiling)
        effective_ceiling = ceiling + flexibility * (self.preference_base_acceptance_prob - ceiling)
        match_score = self._route_match_score(taxi.route_length_pref, effective_pref_strength, route_class)
        if self._is_preferred_route(taxi.route_length_pref, route_class):
            return effective_ceiling + (self.preference_base_acceptance_prob - effective_ceiling) * match_score
        return effective_ceiling * match_score

    def _build_region_popularity_grid(self, region_config):
        self.region_popularity_grid = np.full(
            (self.city.n, self.city.m),
            self.region_default_popularity
        )
        for region in region_config.get("regions", []):
            x_min, x_max = region["x_min"], region["x_max"]
            y_min, y_max = region["y_min"], region["y_max"]
            self.region_popularity_grid[x_min:x_max + 1, y_min:y_max + 1] = float(region["popularity"])

    def _region_acceptance_probability(self, request):
        """Returns acceptance probability in [0, 1] based on region popularity."""
        if self.region_popularity_grid is None:
            return 1.0
        p_origin = self.region_popularity_grid[request.ox, request.oy] / 100.0
        p_dest = self.region_popularity_grid[request.dx, request.dy] / 100.0
        match self.region_match_mode:
            case "origin":
                return float(np.clip(p_origin, 0.0, 1.0))
            case "destination":
                return float(np.clip(p_dest, 0.0, 1.0))
            case "either":
                return float(np.clip(max(p_origin, p_dest), 0.0, 1.0))
            case "both":
                return float(np.clip((p_origin + p_dest) / 2.0, 0.0, 1.0))
        raise ValueError(
            f"The mode {self.region_match_mode} is unknown. Supported modes: 'orign', 'destination', 'either', 'both'")

    def _passenger_score(self, request, taxi) -> float:
        """
        Preference-weighted score of a taxi from the passenger's perspective.

        Combines proximity (w_dist), driver safety (w_safety), and pickup wait
        penalty (w_wait). All components are normalized to [0, 1].
        """
        dist = abs(taxi.x - request.ox) + abs(taxi.y - request.oy)
        norm_dist = 1.0 / (1.0 + dist)
        norm_safety = taxi.safety_score / max(1.0, self.safety_score_max)
        norm_wait = dist / max(1.0, float(self.city.hard_limit))
        return float(request.w_dist * norm_dist + request.w_safety * norm_safety - request.w_wait * norm_wait)

    def _passenger_acceptance_prob(self, request, score: float) -> float:
        """
        Sigmoid acceptance probability for a passenger given a taxi's preference score.

        The threshold decays linearly with waiting time so that passengers become
        progressively less selective as they approach their patience limit.
        """
        waiting_time = self.time - request.timestamps['request']
        decay = max(0.0, 1.0 - waiting_time / max(1, self.max_request_waiting_time))
        current_threshold = (request.safety_threshold / 100.0) * decay
        return float(1.0 / (1.0 + np.exp(-(score - current_threshold) / self.passenger_score_temperature)))

    def _apply_safety_score_delta(self, taxi, delta):
        taxi.safety_score = min(self.safety_score_max, max(self.safety_score_min, taxi.safety_score + delta))

    def _apply_satisfaction_delta(self, taxi, delta):
        taxi.satisfaction_score = min(self.satisfaction_score_max, max(self.satisfaction_score_min, taxi.satisfaction_score + delta))
        taxi.last_satisfaction_delta = float(delta)

    def pickup_request(self, request_id):
        """
        Pick up passenger.
        
        Parameters
        ----------
        
        request_id : int
        """

        # mark pickup timestamp
        r = self.requests[request_id]
        t = self.taxis[r.taxi_id]

        self.taxis_to_request.remove(r.taxi_id)
        self.taxis_to_destination.add(r.taxi_id)

        # change taxi state to with passenger
        t.to_request = False
        t.with_passenger = True
        t.available = False
        t.actual_request_executing = request_id

        r.timestamps['pickup'] = self.time
        r.mode = 'serving'
        # record taxi safety at pickup time
        r.driver_safety_score_pickup = float(t.safety_score)

        # update request and taxi instances
        self.requests[request_id] = r
        self.taxis[r.taxi_id] = t
        if self.log:
            print('\tP ' + "request " + str(request_id) + ' taxi ' + str(t.taxi_id))

    def dropoff_request(self, request_id, mode="simple"):
        """
        Drop off passenger, when taxi reached request destination.
        
        """

        r = self.requests[request_id]
        t = self.taxis[r.taxi_id]

        if r.safety_score_count > 0:
            r.average_safety_score = r.safety_score_sum / r.safety_score_count
        else:
            r.average_safety_score = None

        if mode == "simple" or mode == "going_home":
            # mark request as done
            r.timestamps['dropoff'] = self.time
            r.mode = 'done'
            r.driver_safety_score_end = float(t.safety_score)
            self.requests_in_progress.remove(request_id)
            t.requests_completed.add(request_id)
            # remove taxi from to_destination list
            self.taxis_to_destination.remove(r.taxi_id)

            trip_len = float(abs(r.dy - r.oy) + abs(r.dx - r.ox))
            t.trip_lengths.append(trip_len)
            t.trip_count += 1
            t.trip_length_sum += trip_len
            t.trip_length_sum_sq += trip_len * trip_len
            total_trip_time = 0
            if r.timestamps['assigned'] is not None:
                total_trip_time = max(0, self.time - r.timestamps['assigned'])
            trip_income = (
                    self.price_fixed +
                    trip_len * self.price_per_dist -
                    trip_len * self.cost_per_unit -
                    total_trip_time * self.cost_per_time
            )
            income_component = np.tanh(trip_income / self.satisfaction_income_ref)
            self._apply_satisfaction_delta(t, self.satisfaction_income_weight * income_component)
        elif mode == "cancel":
            # mark request as dropped
            r.mode = 'dropped'
            r.driver_safety_score_end = float(t.safety_score)
            # remove request from progressing ones
            self.requests_in_progress.remove(request_id)
            # clear taxi path
            t.next_destination = deque()
            # remove taxi from to_request list
            self.taxis_to_request.remove(r.taxi_id)

        # update taxi lists
        if mode == "going_home":
            # (magic wand) Apparate taxi home!
            # TODO: taxis are teleported...
            t.x, t.y = t.home

        # update global availability containers
        self.taxis_available[r.taxi_id] = t
        self.city.A[self.city.coordinate_dict_ij_to_c[t.x][t.y]].add(r.taxi_id)

        # update taxi internal states
        t.with_passenger = False
        t.available = True
        t.actual_request_executing = None

        # update request and taxi instances in global containers
        self.requests[request_id] = r
        self.taxis[r.taxi_id] = t

        if r.mode in ('done', 'dropped'):
            self._requests_done_buffer.append(r)

        if self.log:
            print("\tD request " + str(request_id) + ' taxi ' + str(t.taxi_id))

    def compute_region_safety_averages(self, regions=None):
        """
        Computes the average safety score of all taxis currently located in each region.

        Parameters
        ----------
        regions : list of dict, optional
            Each dict must have keys: id, x_min, x_max, y_min, y_max.
            May include additional keys (name, popularity, etc.) which are ignored.
            Defaults to self.regions when not provided.

        Returns
        -------
        dict[str, dict]
            Maps region id to {"avg_safety_score": float | None, "taxi_count": int}.
            avg_safety_score is None when no taxis are present in that region.
        """
        if regions is None:
            regions = self.regions
        result = {}
        for region in regions:
            region_id = region["id"]
            x_min, x_max = region["x_min"], region["x_max"]
            y_min, y_max = region["y_min"], region["y_max"]
            scores = []
            for taxi_id in self.taxis:
                taxi = self.taxis[taxi_id]
                if x_min <= taxi.x <= x_max and y_min <= taxi.y <= y_max:
                    scores.append(taxi.safety_score)
            result[region_id] = {
                "avg_safety_score": float(np.mean(scores)) if scores else None,
                "taxi_count": len(scores),
            }
        return result

    def eval_taxi_income(self, taxi_id):
        """

        Parameters
        ----------
        taxi_id : int
            select taxi from self.taxis with id

        Returns
        -------
        price : int
            evaulated earnnigs of the taxi based on config

        """

        t = self.taxis[taxi_id]

        price = \
            len(t.requests_completed) * self.price_fixed + \
            int(not t.available) * self.price_fixed + \
            t.time_serving * self.price_per_dist - \
            (t.time_cruising + t.time_serving + t.time_to_request) * self.cost_per_unit - \
            (t.time_serving + t.time_cruising + t.time_to_request + t.time_waiting) * self.cost_per_time

        return price

    def plot_simulation(self):
        """
        Draws current state of the simulation on the predefined grid of the class.
        
        Is based on the taxis and requests and their internal states.
        """

        self.init_canvas()

        for taxi_id, i in self.taxis.keys.items():
            t = self.taxis[taxi_id]

            # plot a circle at the place of the taxi
            self.canvas_ax.plot(t.x, t.y, 'o', ms=10, c=self.cmap(self.taxi_colors[i]))

            if self.show_map_labels:
                self.canvas_ax.annotate(
                    str(i),
                    xy=(t.x, t.y),
                    xytext=(t.x, t.y),
                    ha='center',
                    va='center',
                    color='white'
                )

            # if the taxi has a path ahead of it, plot it
            if len(t.next_destination) > 0:
                path = np.array([[t.x, t.y]] + list(t.next_destination))
                if len(path) > 1:
                    xp, yp = path.T
                    # plot path
                    self.canvas_ax.plot(
                        xp,
                        yp,
                        '-',
                        c=self.cmap(self.taxi_colors[i])
                    )
                    # plot a star at taxi destination
                    self.canvas_ax.plot(
                        path[-1][0],
                        path[-1][1],
                        '*',
                        ms=5,
                        c=self.cmap(self.taxi_colors[i])
                    )

            # if a taxi serves a request, put request on the map
            request_id = t.actual_request_executing
            if (request_id is not None) and (not t.with_passenger):
                r = self.requests[request_id]
                self.canvas_ax.plot(
                    r.ox,
                    r.oy,
                    'ro',
                    ms=3
                )
                if self.show_map_labels:
                    self.canvas_ax.annotate(
                        request_id,
                        xy=(r.ox, r.oy),
                        xytext=(r.ox - 0.2, r.oy - 0.2),
                        ha='center',
                        va='center'
                    )

        # plot taxi base
        self.canvas_ax.plot(
            self.city.base_coords[0],
            self.city.base_coords[1],
            'ks',
            ms=15
        )

        # plot pending requests
        if self.show_pending:
            for request_id in self.requests_pending_deque:
                self.canvas_ax.plot(
                    self.requests[request_id].ox,
                    self.requests[request_id].oy,
                    'ro',
                    ms=3,
                    alpha=0.5
                )

        self.canvas.show()

    def move_taxi(self, taxi_id):
        """
        Move a taxi one step forward according to its path queue.

        Update taxi position on availablity grid, if necessary.

        Parameters
        ----------
        plt.ion()
        taxi_id : int
            unique id of taxi that we want to move
        """
        t = self.taxis[taxi_id]

        if t.on_break:
            t.time_on_break += 1
            t.  time_on_break_current += 1

            # Safety score break
            # Non-linear recovery: fast early gain, then diminishing returns.
            elapsed = float(t.time_on_break_current)
            c = self.safety_score_break_recovery_constant
            recovery_fraction = elapsed / (elapsed + c)
            # t.initial_safety_score is recovery ceiling
            target = t.break_start_safety_score + (
                        t.initial_safety_score - t.break_start_safety_score) * recovery_fraction
            t.safety_score = min(t.initial_safety_score, max(self.safety_score_min, target))

            self.taxis[taxi_id] = t
            return

        try:
            # move taxi one step forward
            move = t.next_destination.popleft()

            old_x = t.x
            old_y = t.y

            t.x = move[0]
            t.y = move[1]

            if t.with_passenger:
                t.time_serving += 1
            else:
                if t.available:
                    t.time_cruising += 1
                else:
                    t.time_to_request += 1

            # move available taxis on availability grid
            if t.available:
                self.city.A[self.city.coordinate_dict_ij_to_c[old_x][old_y]].remove(taxi_id)
                self.city.A[self.city.coordinate_dict_ij_to_c[t.x][t.y]].add(taxi_id)
            if self.log:
                print("\tF moved taxi " + str(taxi_id) + " remaining path ", list(t.next_destination), "\n", end="")
        except IndexError:
            t.time_waiting += 1
            if t.available:
                t.time_waiting_since_last_trip += 1

        is_serving = t.to_request or t.with_passenger

        safety_delta = self.safety_score_change_serving_rate if is_serving else self.safety_score_change_waiting_rate
        self._apply_safety_score_delta(t, safety_delta)

        if not is_serving:
            self._apply_satisfaction_delta(t, self.satisfaction_change_waiting_rate)

        self.taxis[taxi_id] = t

    def run_batch(self, run_id, data_path='results'):
        """
        Create a batch run, where metrics are evaluated at every batch step and at the end.
        
        Parameters
        ----------
        
        run_id : str
            id that stands for simulation

        data_path : str
            where to save the results
        """

        if not os.path.exists(data_path):
            os.mkdir(data_path)

        measurement = Measurements(self)

        if self.num_iter is None:
            print("No batch run parameters were defined in the config file, please add them!")
            return

        print("Running simulation with run_id " + run_id + ".")
        print("Batch time " + str(self.batch_size) + ".")
        print("Number of items " + str(self.num_iter) + ".")
        print("Total time simulated " + str(self.batch_size * self.num_iter) + ".")
        print("Starting...")

        results = []
        region_safety_rows = []

        time1 = time()
        for i in range(self.num_iter):
            # tick the clock
            for _ in range(self.batch_size):
                self.step_time("")

            ptm = measurement.read_per_taxi_metrics()

            if i == 0:
                # clearing future output files
                open(data_path + '/run_' + run_id + '_per_taxi_metrics.json', 'w').close()
                open(data_path + '/run_' + run_id + '_per_request_metrics.json', 'w').close()

                # adding taxi homes to output
                ptm['taxi_homes'] = [[int(self.taxis[t].home[0]), int(self.taxis[t].home[1])] for t in self.taxis]

                # write one-time static taxi attributes (preferences, shift cohort)
                taxi_static = {
                    "taxi_ids": [t for t in self.taxis],
                    "route_length_pref": [self.taxis[t].route_length_pref for t in self.taxis],
                    "pref_strength_route": [round(self.taxis[t].pref_strength_route, 4) for t in self.taxis],
                    "nonpreferred_accept_ceiling": [round(self.taxis[t].nonpreferred_accept_ceiling, 4) for t in self.taxis],
                    "break_profile_id": [self.taxis[t].break_profile_id for t in self.taxis],
                    "shift_duration_tu": [self.taxis[t].shift_duration_tu for t in self.taxis],
                    "initial_safety_score": [round(self.taxis[t].initial_safety_score, 4) for t in self.taxis],
                }
                with open(data_path + '/run_' + run_id + '_taxi_static.json', 'w') as f:
                    json.dump(taxi_static, f)

            region_safety = self.compute_region_safety_averages() if self.regions else None

            # dumping per taxi metrics out (per batch)
            with open(data_path + '/run_' + run_id + '_per_taxi_metrics.json', 'a') as f:
                json.dump(ptm, f)
                f.write('\n')
            results.append(measurement.read_aggregated_metrics(ptm, region_safety=region_safety))

            # flush completed/dropped requests to disk and free their memory
            if self._requests_done_buffer:
                chunk = {
                    "timestamp": self.time,
                    "requests": [Measurements._serialize_request(r) for r in self._requests_done_buffer],
                }
                with open(data_path + '/run_' + run_id + '_per_request_metrics.json', 'a') as f:
                    json.dump(chunk, f)
                    f.write('\n')
                for r in self._requests_done_buffer:
                    del self.requests[r.request_id]
                self._requests_done_buffer.clear()

            if region_safety is not None:
                row = {"timestamp": self.time}
                for region_id, data in region_safety.items():
                    avg = data["avg_safety_score"]
                    row[f"{region_id}_avg_safety_score"] = avg if avg is not None else np.nan
                    row[f"{region_id}_taxi_count"] = data["taxi_count"]
                region_safety_rows.append(row)

            time2 = time()
            print('Simulation batch ' + str(i + 1) + '/' + str(self.num_iter) + ' , %.2f sec/batch.' % (time2 - time1))

            time1 = time2

        # dumping batch results
        with open(data_path + '/run_' + run_id + '_aggregates.csv', 'w', newline='') as f:
            pd.DataFrame.from_records(results).to_csv(f, float_format="%.4f")

        # dumping per-region safety averages (only when regions are configured)
        if region_safety_rows:
            with open(data_path + '/run_' + run_id + '_region_safety_averages.csv', 'w', newline='') as f:
                pd.DataFrame.from_records(region_safety_rows).to_csv(f, float_format="%.4f")

        # dump any requests still in-flight at simulation end (pending/assigned/serving) (not dumped already)
        prm = measurement.read_per_request_metrics()
        with open(data_path + '/run_' + run_id + '_per_request_metrics.json', 'a') as f:
            json.dump(prm, f)
            f.write('\n')

        # compressing written objects
        files_to_compress = ['_per_taxi_metrics.json', '_per_request_metrics.json', '_aggregates.csv', '_taxi_static.json']
        if region_safety_rows:
            files_to_compress.append('_region_safety_averages.csv')
        for file in files_to_compress:
            with open(data_path + '/run_' + run_id + file, 'rb') as f1:
                with gzip.open(data_path + '/run_' + run_id + file + '.gz', 'wb') as f2:
                    shutil.copyfileobj(f1, f2)
            os.remove(data_path + '/run_' + run_id + file)

        print("Done.")

    def step_time(self, handler):
        """
        Ticks simulation time by 1.
        """

        if self.log:
            print("\n")
            print("Timestamp " + str(self.time))
            print("Taxis:\n")
            print("\t Available: " + str(len(self.taxis_available)))
            print("\t To request: " + str(len(self.taxis_to_request)))
            print("\t To destination: " + str(len(self.taxis_to_destination)))
            print("\n")
            req_counter = {'TOTAL': self.latest_request_id}
            for request_id in self.requests:
                r = self.requests[request_id]
                if r.mode in req_counter:
                    req_counter[r.mode] += 1
                else:
                    req_counter[r.mode] = 1
            print('Requests:')
            for mode in req_counter:
                print('\t' + mode + ': ' + str(req_counter[mode]))
            print("\n")
            print("Requests pending: ")
            print('\t', self.requests_pending)
            print('\t', self.requests_pending_deque)
            print('\t', self.requests_pending_deque_temporary)
            print("Requests in progress: ")
            print('\t', self.requests_in_progress)
            print("All requests: ")
            print('\t', [(k, str(v)) for k, v in self.requests.items()])
        if (self.time > 0) and (self.time % self.reset_time == 0):
            # print("Going home!")
            self.go_home_everybody()
        else:
            # move every taxi one step towards its destination
            for taxi_id in self.taxis:
                self.move_taxi(taxi_id)

                t = self.taxis[taxi_id]

                if t.actual_request_executing is not None and not t.available:
                    r = self.requests.get(t.actual_request_executing)
                    if r is not None:
                        r.safety_score_sum += float(t.safety_score)
                        r.safety_score_count += 1
                        self.requests[r.request_id] = r

                # if a taxi can pick up its passenger, do it
                if taxi_id in self.taxis_to_request:
                    r = self.requests[t.actual_request_executing]
                    if (t.x == r.ox) and (t.y == r.oy):
                        try:
                            self.pickup_request(t.actual_request_executing)
                        except KeyError:
                            print(t)
                            print(r)
                # if a taxi can drop off its passenger, do it
                elif taxi_id in self.taxis_to_destination:
                    r = self.requests[t.actual_request_executing]
                    if (t.x == r.dx) and (t.y == r.dy):
                        self.dropoff_request(r.request_id)
                        if self.behaviour == "go_back":
                            if self.initial_conditions == "base":
                                self.go_to_base(taxi_id, self.city.base_coords)
                            elif self.initial_conditions == "home":
                                self.go_to_base(taxi_id, t.home)
                        elif self.behaviour == "stay":
                            pass
                        elif self.behaviour == "cruise":
                            self.cruise(taxi_id)

        # check and manage taxi breaks
        check_and_manage_breaks(self)

        # make matchings
        self.matching_algorithm(mode=self.matching)

        # reunite pending requests
        self.requests_pending_deque_temporary.reverse()
        self.requests_pending_deque.extendleft(self.requests_pending_deque_temporary)
        self.requests_pending_deque_temporary = deque()
        # delete old requests from pending ones
        if self.time > self.max_request_waiting_time and len(self.requests_pending_deque) > 0:
            while len(self.requests_pending_deque) > 0 and (
                    self.requests_pending_deque[0] in self.requests_pending_deque_batch[0]):
                request_id = self.requests_pending_deque.popleft()
                r = self.requests[request_id]
                r.mode = 'dropped'
                r.cancellation_reason = r.last_no_match_reason if r.last_no_match_reason else 'patience_exceeded'
                self.requests[request_id] = r
                self._requests_done_buffer.append(r)

        self.requests_pending = set(self.requests_pending_deque)

        # generate requests
        new_requests = set()
        rfrac, rint = np.modf(self.get_effective_request_rate())
        for _ in range(int(rint)):
            self.add_request()

            new_requests.add(self.latest_request_id)
        if rfrac > 1e-3:
            try:
                p = self.city.request_p.pop()
            except IndexError:
                self.city.request_p.extend(self.rng.random(self.city.length))
                p = self.city.request_p.pop()
            if p < rfrac:
                self.add_request()
                new_requests.add(self.latest_request_id)

        # this automatically pushes out requests that have been waiting for too long
        self.requests_pending_deque_batch.append(new_requests)

        if self.show_plot:
            self.plot_simulation()

        # step time
        self.time += 1


class Measurements:

    def __init__(self, simulation):
        self.simulation = simulation

    @staticmethod
    def _serialize_request(r):
        return {
            "request_id": r.request_id,
            "mode": r.mode,
            "origin": (r.ox, r.oy),
            "destination": (r.dx, r.dy),
            "taxi_id": r.taxi_id,
            "timestamp": r.timestamps["request"],
            "assignment": r.timestamps["assigned"],
            "pickup": r.timestamps["pickup"],
            "dropoff": r.timestamps["dropoff"],
            "driver_safety_score_start": getattr(r, "driver_safety_score_start", None),
            "driver_safety_score_end": getattr(r, "driver_safety_score_end", None),
            "driver_average_safety_score": getattr(r, "average_safety_score", None),
            "driver_safety_score_pickup": getattr(r, "driver_safety_score_pickup", None),
            "assigned_taxi_pos": getattr(r, "assigned_taxi_pos", None),
            "assigned_taxi_distance": getattr(r, "assigned_taxi_distance", None),
            "passenger_type": getattr(r, "passenger_type", None),
            "safety_threshold": getattr(r, "safety_threshold", None),
            "w_dist": getattr(r, "w_dist", None),
            "w_safety": getattr(r, "w_safety", None),
            "w_wait": getattr(r, "w_wait", None),
            "cancellation_reason": getattr(r, "cancellation_reason", None),
            "passenger_forced_accept": getattr(r, "passenger_forced_accept", False),
        }

    def read_per_taxi_metrics(self):
        """
        Returns metrics for taxis.

        Outputs a dictionary that stores these metrics in lists and the timestamp of the call.

        Output
        ------

        timestamp: int
            the timestamp of the measurement

        trip_avg_length: list of floats
            average trip lengths per taxi

        trip_std_length: list of floats
            standard deviation of trip lengths per taxi

        trip_income: list of floats
            average trip income per taxi

        trip_num_completed: int
            number of trips completed by the taxi

        time_serving: list of floats
            time of useful travel time from overall time per taxi

        time_to_request: list of floats
            time of empty travel time (from assignment to pickup)

        time_cruising: list of floats
            time of travelling with no assigned request

        time_waiting: list of floats
            time of standing in place with no assigned request

        position: (int,int)
            current taxi position on grid
        """

        # for the taxis

        # average trip lengths per taxi
        trip_avg_length = []
        trip_std_length = []
        incomes = []
        trip_num_completed = []

        time_serving = []
        time_to_request = []
        time_cruising = []
        time_waiting = []
        time_on_break = []
        time_on_break_current = []
        on_break = []
        breaks_started_today = []

        position = []
        safety_scores = []
        satisfaction_scores = []
        satisfaction_deltas = []
        total_declines = []

        for taxi_id in self.simulation.taxis:
            taxi = self.simulation.taxis[taxi_id]
            n = taxi.trip_count
            trip_num_completed.append(n)
            if n > 0:
                mean = taxi.trip_length_sum / n
                variance = taxi.trip_length_sum_sq / n - mean * mean
                trip_avg_length.append(round(mean, 4))
                trip_std_length.append(round(variance ** 0.5 if variance > 0 else 0.0, 4))
            else:
                trip_avg_length.append(0)
                trip_std_length.append(np.nan)
            incomes.append(self.simulation.eval_taxi_income(taxi_id))

            s = taxi.time_serving
            w = taxi.time_waiting
            r = taxi.time_to_request
            c = taxi.time_cruising

            time_serving.append(s)
            time_cruising.append(c)
            time_waiting.append(w)
            time_to_request.append(r)
            time_on_break.append(int(getattr(taxi, "time_on_break", 0)))
            time_on_break_current.append(int(getattr(taxi, "time_on_break_current", 0)))
            on_break.append(bool(getattr(taxi, "on_break", False)))
            breaks_started_today.append(int(getattr(taxi, "breaks_started_today", 0)))

            position.append([int(taxi.x), int(taxi.y)])
            safety_scores.append(round(taxi.safety_score, 4))
            satisfaction_scores.append(round(getattr(taxi, "satisfaction_score", np.nan), 4))
            satisfaction_deltas.append(round(getattr(taxi, "last_satisfaction_delta", 0.0), 4))
            total_declines.append(int(getattr(taxi, "total_declines", 0)))

        return {
            "timestamp": self.simulation.time,
            "trip_avg_length": trip_avg_length,
            "trip_std_length": trip_std_length,
            "trip_income": incomes,
            "trip_num_completed": trip_num_completed,
            "time_serving": time_serving,
            "time_cruising": time_cruising,
            "time_waiting": time_waiting,
            "time_to_request": time_to_request,
            "time_on_break": time_on_break,
            "time_on_break_current": time_on_break_current,
            "on_break": on_break,
            "breaks_started_today": breaks_started_today,
            "position": position,
            "safety_score": safety_scores,
            "satisfaction_score": satisfaction_scores,
            "satisfaction_delta": satisfaction_deltas,
            "total_declines": total_declines
        }

    def read_per_request_metrics(self):
        """
        Returns request dict.

        Output
        -------

        timestamp: int
            the timestamp of the measurement


        """

        # for the requests

        output_dict = {
            "timestamp": self.simulation.time,
            "requests": []
        }

        for request_id in self.simulation.requests:
            r = self.simulation.requests[request_id]
            output_dict["requests"].append(self._serialize_request(r))

        return output_dict

    @staticmethod
    def read_aggregated_metrics(per_taxi_metrics, region_safety=None):

        metrics = {"timestamp": per_taxi_metrics["timestamp"]}

        for k in per_taxi_metrics:
            if k[0:6] == 'trip_i' or k[0:4] == 'time' or k[0:12] == 'satisfaction' or k == 'total_declines':
                metrics['avg_' + k] = np.nanmean(per_taxi_metrics[k])
                metrics['std_' + k] = np.nanstd(per_taxi_metrics[k])

        if region_safety:
            for region_id, data in region_safety.items():
                avg = data["avg_safety_score"]
                metrics[f"region_safety_avg_{region_id}"] = avg if avg is not None else np.nan
                metrics[f"region_safety_count_{region_id}"] = data["taxi_count"]

        return metrics
