# Taxi Simulation

Agent-based simulation of a taxi–passenger city system on a discrete grid. Taxis and passengers have heterogeneous
preferences and get matched by algorithms that range from simple nearest-neighbor to two-sided preference-weighted
scoring.

---

## Core mechanics

- Taxis move on an `n × m` square grid at one cell per time unit (TU).
- **1 TU = 10 seconds** (grid spacing 100 m, speed 10 m/s).
- Requests are generated each step with rate `request_rate` (possibly regulated by a time-of-day schedule). Origins and
  destinations are sampled from configurable mixtures of 2-D Gaussians, not uniformly, to model spatial demand patterns.
- A matching algorithm pairs pending requests to available taxis. If matched, the taxi travels to the origin (pickup),
  then to the destination (dropoff).
- Requests that have waited longer than `max_request_waiting_time` TU are dropped.
- After dropoff, the taxi either stays at the dropoff location (`stay`) or returns to base (`go_back`), depending on
  `behaviour`. (`cruise` behavior is planned but not yet implemented.)

---

## Matching algorithms

| config `matching` value              | Description                                                                                                                                                                                                                                                                                                                                                                                |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nearest`                            | Assign the nearest available taxi within `hard_limit`.                                                                                                                                                                                                                                                                                                                                     |
| `nearest_distance_pref`              | Nearest matching; taxis probabilistically decline based on route-length preference. Declines are tracked per request so declined taxis are skipped on retry.                                                                                                                                                                                                                               |
| `nearest_region_pref`                | Nearest matching; acceptance probability is proportional to the region popularity score of the request origin/destination.                                                                                                                                                                                                                                                                 |
| `nearest_passenger_pref`             | Passenger preference scoring: all taxis in radius are scored by proximity, safety, and pickup wait. The best-scoring taxi is accepted with a sigmoid probability whose threshold decays with waiting time - every passenger eventually accepts.                                                                                                                                            |
| `nearest_two_sided_dist_pass_pref`   | Two-sided matching: driver route-length preference checked first (permanent per-request decline); then passenger preference score checked. Passenger rejections are not permanent - threshold relaxes over time.                                                                                                                                                                           |
| `nearest_two_sided_region_pass_pref` | Two-sided matching: driver region popularity preference checked first (permanent per-request decline); then passenger preference score checked. Requires a `regions` config block.                                                                                                                                                                                                         |
| `safety_objective`                   | Objective safety-optimal baseline: sorts regions by ascending avg safety score of available taxis at pickup (least-safe region first), then within each region serves the oldest request first, always assigning the globally most-safe available taxi. No preferences applied. Requires a `regions` config block for region-aware ordering; falls back to global arrival-order otherwise. |
| `safety_objective_two_sided`         | Region-aware safety objective combined with two-sided preferences: same region-prioritised, safest-taxi-first dispatching as `safety_objective`, plus driver region-popularity preference and passenger safety-score preference layered on top. Requires a `regions` config block.                                                                                                         |
| `poorest`                            | Assign the taxi with the lowest cumulative income within `hard_limit`.                                                                                                                                                                                                                                                                                                                     |
| `random_limited`                     | Random taxi within `hard_limit`.                                                                                                                                                                                                                                                                                                                                                           |
| `random_unlimited`                   | Fully random taxi from the entire grid.                                                                                                                                                                                                                                                                                                                                                    |

---

## Configuration

All parameters are stored in `.conf` files in the `configs/` directory (JSON format). A `ConfigGenerator` in
`generate_configs.py` programmatically creates config files for parameter sweeps. Below is a full reference for every
supported key.

### Basic simulation parameters

```json
{
  "n": 100,
  "m": 100,
  "base_coords": [
    50,
    50
  ],
  "hard_limit": 20,
  "length": 200000,
  "num_taxis": 100,
  "request_rate": 10.0,
  "matching": "nearest",
  "max_request_waiting_time": 300,
  "max_time": 432000,
  "batch_size": 4320,
  "behaviour": "stay",
  "initial_conditions": "base",
  "reset_time": 10000000,
  "price_fixed": 450,
  "price_per_dist": 140,
  "cost_per_unit": 13,
  "cost_per_time": 0,
  "log": false,
  "show_plot": false,
  "show_map_labels": false,
  "show_pending": false
}
```

| Key                        | Type         | Description                                                                                               |
|----------------------------|--------------|-----------------------------------------------------------------------------------------------------------|
| `n`, `m`                   | `int`        | Grid dimensions (width × height).                                                                         |
| `base_coords`              | `[int, int]` | Taxi base / depot location.                                                                               |
| `hard_limit`               | `int`        | Search radius for limited matching algorithms.                                                            |
| `length`                   | `int`        | Pre-generation buffer size for random coordinates.                                                        |
| `num_taxis`                | `int`        | Fixed fleet size.                                                                                         |
| `request_rate`             | `float`      | Base request arrivals per TU (before time-of-day scaling).                                                |
| `matching`                 | `str`        | Matching algorithm name (see table above).                                                                |
| `max_request_waiting_time` | `int`        | TU after which unmatched requests are dropped.                                                            |
| `max_time`                 | `int`        | Total simulation duration in TU.                                                                          |
| `batch_size`               | `int`        | Measurement interval; metrics are recorded every `batch_size` TU.                                         |
| `behaviour`                | `str`        | Post-dropoff taxi behaviour: `go_back`, `stay`, `cruise`.                                                 |
| `initial_conditions`       | `str`        | Initial taxi placement: `base` (all at depot) or `home` (random home coords).                             |
| `reset_time`               | `int`        | Periodic reset interval; taxis are teleported home at multiples of this value. Set very large to disable. |
| `price_fixed`              | `float`      | Flat fare per trip.                                                                                       |
| `price_per_dist`           | `float`      | Fare per grid-cell of trip length.                                                                        |
| `cost_per_unit`            | `float`      | Operating cost per grid-cell travelled.                                                                   |
| `cost_per_time`            | `float`      | Time-based operating cost per TU.                                                                         |

---

### Request origin and destination distributions

Requests are sampled from a mixture of 2-D Gaussians. Two distribution lists are required: one for origins, one for
destinations. The geometry is also embedded in a compact `geom_specification_compact.json` format used by
`ConfigGenerator`.

```json
{
  "request_origin_distributions": [
    {
      "location": [
        30,
        30
      ],
      "strength": 5,
      "sigma": 8
    },
    {
      "location": [
        70,
        70
      ],
      "strength": 3,
      "sigma": 5
    }
  ],
  "request_destination_distributions": [
    {
      "location": [
        50,
        50
      ],
      "strength": 10,
      "sigma": 15
    }
  ],
  "avg_request_lengths": 18.4,
  "R": 0.5,
  "d": 258,
  "geom": 10
}
```

`avg_request_lengths`, `R` and `d` are auto-computed by `ConfigGenerator` and embedded in generated configs for
reference.

---

### Time-of-day request rate schedule

Multiplies `request_rate` by a window-specific factor to model morning/evening rush hours and nighttime rest.
Windows must be non-overlapping; any time-of-day not covered uses a multiplier of 1.0.

```json
{
  "day_length_tu": 8640,
  "request_rate_schedule": [
    {
      "start": 0,
      "end": 1080,
      "multiplier": 0.25
    },
    {
      "start": 1080,
      "end": 2520,
      "multiplier": 0.70
    },
    {
      "start": 2520,
      "end": 3600,
      "multiplier": 1.80
    },
    {
      "start": 3600,
      "end": 5760,
      "multiplier": 1.00
    },
    {
      "start": 5760,
      "end": 6840,
      "multiplier": 2.00
    },
    {
      "start": 6840,
      "end": 8640,
      "multiplier": 0.50
    }
  ]
}
```

| Key                     | Default         | Description                                                                                                 |
|-------------------------|-----------------|-------------------------------------------------------------------------------------------------------------|
| `day_length_tu`         | `8640`          | Length of one logical day in TU (8640 TU = 24 h). Used for time-of-day window logic and break day tracking. |
| `request_rate_schedule` | `[]` (constant) | List of `{start, end, multiplier}` windows. Empty list means constant rate.                                 |

---

### Driver safety score

Each taxi maintains a `safety_score ∈ [safety_score_min, safety_score_max]`. It evolves every TU:

- **While serving** (en route to pickup or carrying a passenger): `safety_score += safety_score_change_serving_rate` (
  typically negative).
- **While waiting** (available, not on break): `safety_score += safety_score_change_waiting_rate` (typically a small
  negative value; slight fatigue even while idle).
- **While on break**: non-linear recovery towards the taxi's `initial_safety_score` ceiling -
  `target(t) = break_start_safety_score + (initial_safety_score - break_start_safety_score) × t/(t + C)`, where
  `C = safety_score_break_recovery_constant`. Recovery is fast initially and diminishes over time.

Each taxi is initialized with a score drawn from `Uniform(initial_safety_score_min, initial_safety_score_max)`, which
also becomes the personal recovery ceiling (inherent safety) .

```json
{
  "safety_score_change_serving_rate": -0.02,
  "safety_score_change_waiting_rate": -0.001,
  "safety_score_break_recovery_constant": 180.0,
  "safety_score_min": 0,
  "safety_score_max": 100,
  "initial_safety_score_min": 20,
  "initial_safety_score_max": 80
}
```

| Key                                                    | Default          | Description                                                                                                         |
|--------------------------------------------------------|------------------|---------------------------------------------------------------------------------------------------------------------|
| `safety_score_change_serving_rate`                     | `-0.02`          | Per-TU delta while assigned (to pickup or with passenger). Negative = fatigue.                                      |
| `safety_score_change_waiting_rate`                     | `-0.001`         | Per-TU delta while unassigned and not on break. Typically a small negative value.                                   |
| `safety_score_break_recovery_constant`                 | `180.0`          | Half-recovery time C in TU: after C TU on break, score is halfway to ceiling.                                       |
| `safety_score_min`, `safety_score_max`                 | `0`, `100`       | Hard clipping bounds.                                                                                               |
| `initial_safety_score_min`, `initial_safety_score_max` | equal to min/max | Per-taxi initialization range and personal recovery ceiling. Must be within `[safety_score_min, safety_score_max]`. |

---

### Driver satisfaction score

Each taxi also maintains a `satisfaction_score ∈ [satisfaction_score_min, satisfaction_score_max]`. It changes at three
events:

- **Each TU while waiting**: `+= satisfaction_change_waiting_rate` (typically negative).
- **At assignment**: `+= satisfaction_pref_match_delta` if the trip's Manhattan length matches the driver's
  `route_length_pref` profile (`short_pref`, `neutral_pref`, `long_pref`), otherwise
  `+= satisfaction_pref_mismatch_delta`. This fires for **every matching algorithm** regardless of what criterion the
  algorithm used to accept the match - route-length satisfaction is a property of the driver, not of the algorithm.
  `neutral_pref` drivers always count as matched.
- **At dropoff**: `+= satisfaction_income_weight × tanh(trip_income / satisfaction_income_ref)`.

```json
{
  "satisfaction_score_min": 0.0,
  "satisfaction_score_max": 100.0,
  "satisfaction_initial_min": 45.0,
  "satisfaction_initial_max": 55.0,
  "satisfaction_change_waiting_rate": -0.01,
  "satisfaction_income_weight": 0.5,
  "satisfaction_income_ref": 1000.0,
  "satisfaction_pref_match_delta": 0.2,
  "satisfaction_pref_mismatch_delta": -0.3
}
```

| Key                                | Default        | Description                                          |
|------------------------------------|----------------|------------------------------------------------------|
| `satisfaction_score_min/max`       | `0.0`, `100.0` | Clipping bounds.                                     |
| `satisfaction_initial_min/max`     | `45.0`, `55.0` | Per-taxi initialization range.                       |
| `satisfaction_change_waiting_rate` | `-0.01`        | Per-TU delta while waiting (not on break).           |
| `satisfaction_income_weight`       | `0.5`          | Weight on `tanh(income/ref)` at dropoff.             |
| `satisfaction_income_ref`          | `1000.0`       | Income normalization scale.                          |
| `satisfaction_pref_match_delta`    | `0.2`          | Added at assignment of a preferred-length route.     |
| `satisfaction_pref_mismatch_delta` | `-0.3`         | Added at assignment of a non-preferred-length route. |

---

### Driver route-length preferences

Each taxi is assigned a preference profile at initialization: `short_pref`, `neutral_pref`, or `long_pref`. Requests are
classified as `short`, `medium`, or `long` based on Manhattan trip distance.

The **acceptance probability** for `nearest_distance_pref` is:

```
preferred route:     p = ceiling + (base_acceptance - ceiling) × match_score
non-preferred route: p = ceiling × match_score
```

where `match_score ∈ [0, 1]` scales with how well the route fits the preference and preference strength.

```json
{
  "driver_route_pref_mix": {
    "short_pref": 0.35,
    "neutral_pref": 0.30,
    "long_pref": 0.35
  },
  "route_pref_strength_range": {
    "low": 0.2,
    "high": 0.9
  },
  "route_length_class_thresholds": {
    "short_max": 8,
    "medium_max": 16
  },
  "preference_base_acceptance_prob": 0.9,
  "nonpreferred_accept_ceiling": 0.25,
  "max_declines": 3
}
```

| Key                               | Default                 | Description                                                                                                                  |
|-----------------------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `driver_route_pref_mix`           | equal weights           | Cohort weights for `short_pref`, `neutral_pref`, `long_pref`.                                                                |
| `route_pref_strength_range`       | `{low:0.2, high:0.9}`   | Per-taxi preference strength sampled from this uniform range.                                                                |
| `route_length_class_thresholds`   | derived from avg length | Explicit distance cutoffs for short/medium/long trip classes. If absent, derived as 0.75 × and 1.25 × `avg_request_lengths`. |
| `preference_base_acceptance_prob` | `0.9`                   | Upper acceptance probability for strongly preferred routes.                                                                  |
| `nonpreferred_accept_ceiling`     | `0.25`                  | Maximum acceptance probability for non-preferred routes.                                                                     |
| `max_declines`                    | `null`                  | Per-request forced-accept threshold: after this many declines by a single taxi, it must accept. `null` disables.             |

---

### Driver income flexibility

A driver who is falling behind on earnings becomes less picky and accepts trips they would otherwise decline. Optional:
disabled unless `income_target_rate` is set, in which case preferences behave exactly as described above.

`income_target_rate` is a target earning *pace* (income per unit of work time). At each match attempt, a driver is
compared against how much they should have earned for the time worked so far this shift:

```
target_so_far = income_target_rate × work_time_this_shift
behind        = max(0, target_so_far − earned_this_shift) / full_shift_target
flexibility   = sigmoid((behind − driver_flexibility_threshold) / driver_flexibility_temperature)
```

`behind` is how far the driver trails target, as a fraction of a whole shift's target income
(`full_shift_target = income_target_rate × shift_duration_tu`). It is 0 at the start of a shift and grows only if
earnings keep lagging. `flexibility` runs from 0 (on target, preferences fully active) to 1 (far behind, preferences
ignored). `driver_flexibility_threshold` is the gap at which relaxation kicks in; `driver_flexibility_temperature` sets
how abruptly. (When breaks are disabled there is no shift, so `behind` is measured over the taxi's whole run instead.)

`flexibility` then eases the driver's acceptance toward `preference_base_acceptance_prob`, the same cap a fully matched
trip uses:

```
route-length pref:   effective_pref_strength = pref_strength × (1 − flexibility)
                     effective_ceiling       = ceiling + flexibility × (base_acceptance − ceiling)
region pref:         p_accept = p_region + flexibility × (base_acceptance − p_region)
```

This applies to every algorithm with a driver preference (`nearest_distance_pref`, `nearest_region_pref`,
`nearest_two_sided_dist_pass_pref`, `nearest_two_sided_region_pass_pref`, and the region layer of
`safety_objective_two_sided`), but not to plain `safety_objective`, which has no driver preference.

| Key                              | Default | Description                                                                                     |
|----------------------------------|---------|-------------------------------------------------------------------------------------------------|
| `income_target_rate`             | `null`  | Expected income per work-time unit. `null` disables the whole model. Must be `> 0` when set.    |
| `driver_flexibility_threshold`   | `0.3`   | Shortfall fraction at the sigmoid midpoint; below it preferences stay active, above they relax. |
| `driver_flexibility_temperature` | `0.15`  | Sigmoid steepness for the transition; smaller is sharper. Must be `> 0`.                        |

---

### Break and shift system

Breaks are controlled by a cohort system. Every taxi is assigned a cohort at initialization; cohort weights and
per-cohort parameters govern shift length and break timing.

#### Cohort assignment

```json
{
  "driver_break_cohort_mix": {
    "short_shift": 0.50,
    "mid_shift": 0.30,
    "long_shift": 0.20
  },
  "shift_duration_tu": {
    "short_shift": {
      "dist": "uniform",
      "low": 900,
      "high": 1440
    },
    "mid_shift": {
      "dist": "uniform",
      "low": 1440,
      "high": 2160
    },
    "long_shift": {
      "dist": "uniform",
      "low": 2520,
      "high": 3240
    }
  }
}
```

`shift_duration_tu` specs support `"dist": "uniform"` (with `low`/`high`) or `"dist": "fixed"` (with `value`), or a bare
numeric value.

#### Per-cohort break parameters

Configured in `break_cohort_settings` (one entry per cohort):

```json
{
  "break_cohort_settings": {
    "short_shift": {
      "inter_shift_rest_tu": {
        "dist": "uniform",
        "low": 1080,
        "high": 2160
      },
      "intra_shift_break_after_work_tu": {
        "dist": "uniform",
        "low": 540,
        "high": 720
      },
      "intra_shift_break_duration_tu": {
        "dist": "uniform",
        "low": 60,
        "high": 120
      },
      "demotivation_threshold_tu": {
        "dist": "uniform",
        "low": 240,
        "high": 480
      },
      "shift_start_offset_tu": {
        "dist": "uniform",
        "low": 1,
        "high": 900
      }
    }
  }
}
```

| Per-cohort key                    | Required                        | Description                                                                                                         |
|-----------------------------------|---------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `inter_shift_rest_tu`             | Yes                             | Duration of the inter-shift rest period (end-of-shift break). Sampled per break event.                              |
| `intra_shift_break_after_work_tu` | No                              | Accumulated work time (serving + en route) that triggers an intra-shift break.                                      |
| `intra_shift_break_duration_tu`   | Required if intra-shift enabled | Duration of intra-shift (short mid-shift) breaks.                                                                   |
| `demotivation_threshold_tu`       | No                              | Accumulated waiting-only time that triggers a demotivation break. Requires `intra_shift_break_duration_tu`.         |
| `shift_start_offset_tu`           | No (default `0`)                | Staggered start offset: subtracts from the taxi's initial work-time counter to desynchronize fleet-wide shift ends. |

#### Break triggers (evaluated every TU)

Three break types fire in priority order:

1. **End-of-shift break** - fires when accumulated work time since shift start ≥ `shift_duration_tu`. The taxi takes
   `inter_shift_rest_tu` rest; after returning, a new shift begins.
2. **Intra-shift break** - fires when work time since last break ≥ `intra_shift_break_after_work_tu`. Duration is
   `intra_shift_break_duration_tu`.
3. **Demotivation break** - fires when continuous waiting time (no assignment) ≥ `demotivation_threshold_tu`. Duration
   is `intra_shift_break_duration_tu`.

All breaks only fire when the taxi is **available** (not currently serving a request).

#### Rush-window deferral

End-of-shift breaks can be deferred during rush hours to keep supply available:

```json
{
  "rush_windows_tu": [
    {
      "start": 2520,
      "end": 3600
    },
    {
      "start": 5760,
      "end": 6840
    }
  ],
  "p_defer_end_of_shift_in_rush": 0.3,
  "max_break_deferral_tu": 180
}
```

| Key                            | Default | Description                                                                                             |
|--------------------------------|---------|---------------------------------------------------------------------------------------------------------|
| `rush_windows_tu`              | `[]`    | List of `{start, end}` time-of-day windows (in TU mod `day_length_tu`) during which deferral can occur. |
| `p_defer_end_of_shift_in_rush` | `0.0`   | Probability of deferring the end-of-shift break on each eligible TU.                                    |
| `max_break_deferral_tu`        | `0`     | Cumulative deferral cap per break event; deferral stops once this is reached.                           |

---

### Passenger preferences

Every request is assigned preference attributes at creation time, drawn from configurable distributions. These
attributes drive the `nearest_passenger_pref`, `nearest_two_sided_dist_pass_pref`, and
`nearest_two_sided_region_pass_pref` matching algorithms.

#### Passenger type mix

```json
{
  "passenger_preference_mix": {
    "safety_indifferent": 0.40,
    "safety_moderate": 0.40,
    "safety_strict": 0.20
  },
  "passenger_safety_threshold": {
    "safety_indifferent": {
      "mean": 0.0,
      "std": 0.0
    },
    "safety_moderate": {
      "mean": 55.0,
      "std": 10.0
    },
    "safety_strict": {
      "mean": 75.0,
      "std": 8.0
    }
  }
}
```

Each request draws a type from the weighted mix and samples its `safety_threshold` from a Normal distribution clipped to
`[0, 100]`.

#### Preference weights

```json
{
  "passenger_preference_weights": {
    "w_dist": {
      "mean": 1.0,
      "std": 0.2
    },
    "w_safety": {
      "mean": 0.5,
      "std": 0.3
    },
    "w_wait": {
      "mean": 0.3,
      "std": 0.2
    }
  },
  "passenger_score_temperature": 0.1
}
```

Each request samples weights `w_dist`, `w_safety`, `w_wait` per-passenger from truncated Normal distributions (clipped
to ≥ 0).

#### Scoring and acceptance

For a candidate taxi, the passenger computes:

```
score = w_dist × 1/(1+dist) + w_safety × (safety_score/safety_max) − w_wait × (dist/hard_limit)
```

The **acceptance probability** uses a sigmoid with a decaying threshold:

```
threshold(t) = (safety_threshold / 100) × max(0, 1 − waiting_time / max_request_waiting_time)
P(accept)    = sigmoid((score − threshold(t)) / temperature)
```

The threshold decays linearly to 0 as the request approaches its patience limit, so every passenger eventually accepts
any positively-scoring taxi.

| Key                            | Default                       | Description                                                        |
|--------------------------------|-------------------------------|--------------------------------------------------------------------|
| `passenger_preference_mix`     | equal weights for three types | Weighted cohort mix for passenger type sampling.                   |
| `passenger_safety_threshold`   | per-type Normal params        | Per-type mean/std for `safety_threshold` sampling.                 |
| `passenger_preference_weights` | see above                     | Per-weight `{mean, std}` specs for `w_dist`, `w_safety`, `w_wait`. |
| `passenger_score_temperature`  | `0.1`                         | Sigmoid temperature; lower = sharper accept/reject boundary.       |

#### Cancellation reasons

When a request is dropped (patience expired), its `cancellation_reason` records why it failed to match in the last
attempt:

| Value                | Meaning                                                                                                                                      |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `patience_exceeded`  | Request was never attempted (no taxis available globally) or expired naturally.                                                              |
| `no_taxi_available`  | Taxis exist in the fleet but none were within `hard_limit` at the last match attempt.                                                        |
| `driver_declined`    | Taxis were in range but all declined due to driver preferences (`nearest_two_sided_dist_pass_pref` or `nearest_two_sided_region_pass_pref`). |
| `passenger_declined` | Willing taxis were available but the passenger score was too low.                                                                            |

---

### Region preferences

Used by `nearest_region_pref`. Defines spatial regions with popularity scores that modulate driver acceptance
probability.

```json
{
  "regions": {
    "default_popularity": 100,
    "match_mode": "origin",
    "regions": [
      {
        "id": "center",
        "x_min": 40,
        "x_max": 60,
        "y_min": 40,
        "y_max": 60,
        "popularity": 100
      },
      {
        "id": "outskirt",
        "x_min": 0,
        "x_max": 20,
        "y_min": 0,
        "y_max": 20,
        "popularity": 30
      }
    ]
  },
  "max_declines": 3
}
```

| Key                  | Default    | Description                                                                                            |
|----------------------|------------|--------------------------------------------------------------------------------------------------------|
| `default_popularity` | `100`      | Popularity for cells not covered by any defined region.                                                |
| `match_mode`         | `"origin"` | Which coordinate drives acceptance: `"origin"`, `"destination"`, `"either"` (max), `"both"` (average). |
| `max_declines`       | `null`     | Force-accept after this many declines by a single taxi per request.                                    |

---

## Config generation

`generate_configs.py` generates sweeps of config files from a base config. Usage:

```
python generate_configs.py <mode> [arguments]
```

Supported modes:

| Mode                         | Description                                                                                                                                                                                  | Usage                                                                                                                                                                                                 |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `simple`                     | Single config: d=225, R=0.5, `nearest`, stay+base behaviour.                                                                                                                                 | `python generate_configs.py simple <base> [days=1] [geom=0]`                                                                                                                                          |
| `sweep`                      | Full grid: all geometries (0–6), all behaviour types (0–4), R ∈ linspace(0.05, 1, 20), d ∈ linspace(50, 400, 11), all algorithms.                                                            | `python generate_configs.py sweep <base>`                                                                                                                                                             |
| `passenger_pref`             | Sweeps R=[0.2, 0.5, 1.0] for `nearest_passenger_pref`; behaviour types 0–3.                                                                                                                  | `python generate_configs.py passenger_pref <base> [days=5] [geom=10] [max_declines=off]`                                                                                                              |
| `region_pref`                | Sweeps R=[0.2, 0.5, 1.0] for `nearest_region_pref` with a given regions file; behaviours 0–3.                                                                                                | `python generate_configs.py region_pref <base> <regions_file> [days=5] [geom=10] [max_declines=off]`                                                                                                  |
| `distance_pref`              | Sweeps R=[0.2, 0.5, 1.0] for `nearest_distance_pref`; behaviours 0–3.                                                                                                                        | `python generate_configs.py distance_pref <base> [days=5] [geom=10] [max_declines=off]`                                                                                                               |
| `two_sided`                  | Sweeps R=[0.2, 0.5, 1.0] for the chosen two-sided algorithm; behaviour types 0–3. The `region` variant requires a regions file.                                                              | `python generate_configs.py two_sided <base> dist [days=5] [geom=10] [max_declines=off]`<br>`python generate_configs.py two_sided <base> region <regions_file> [days=5] [geom=10] [max_declines=off]` |
| `safety_objective`           | Sweeps R=[0.2, 0.5, 1.0] for `safety_objective`; behaviours 0–3. Passing a `regions_file` enables region-aware safety prioritisation; omitting it falls back to global arrival-order.        | `python generate_configs.py safety_objective <base> [regions_file] [days=5] [geom=10]`                                                                                                                |
| `safety_objective_two_sided` | Sweeps R=[0.2, 0.5, 1.0] for `safety_objective_two_sided` (region-aware safety objective + driver region preference + passenger safety preference); requires a regions file; behaviours 0–3. | `python generate_configs.py safety_objective_two_sided <base> <regions_file> [days=5] [geom=10] [max_declines=off]`                                                                                   |
| `passenger_fairness`         | Fixed: `passenger_fairness/test.conf`, d≈258 (ρ=15/km²), R=[0.2, 0.5, 1.0], geoms [0,1,2,3,6], `nearest`, stay+base. No extra arguments.                                                     | `python generate_configs.py passenger_fairness`                                                                                                                                                       |
| `multiple_runs`              | Multi-run averaging for paper figures (hardcoded to `2019_05_19_base.conf`). No extra arguments.                                                                                             | `python generate_configs.py multiple_runs`                                                                                                                                                            |
| `long_run`                   | Single 100-day run (hardcoded: `2019_02_14_base.conf`, d=225, R=0.5, `nearest`, geom 0, stay+base). No extra arguments.                                                                      | `python generate_configs.py long_run`                                                                                                                                                                 |
| `missing`                    | Fills in missing R ranges from a previous sweep (hardcoded to `2019_05_06_base.conf`). No extra arguments.                                                                                   | `python generate_configs.py missing`                                                                                                                                                                  |

### Argument reference

| Argument         | Required by                                                                                                                                            | Default                            | Description                                                                                                                                                                                                                 |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `<base>`         | `simple`, `sweep`, `nearest_baseline`, `passenger_pref`, `region_pref`, `distance_pref`, `two_sided`, `safety_objective`, `safety_objective_two_sided` | -                                  | Base `.conf` filename in `configs/` (include the extension, e.g. `big_city_base.conf`).                                                                                                                                     |
| `<regions_file>` | required: `region_pref`, `two_sided region`, `safety_objective_two_sided`; optional: `safety_objective`                                                | -                                  | Path to a regions JSON file **relative to** `configs/` (e.g. `regions_big_city_balanced.json`). Its content is embedded under the `regions` key. For `safety_objective`, auto-detected when the argument ends with `.json`. |
| `<dist\|region>` | `two_sided`                                                                                                                                            | -                                  | Variant: `dist` → `nearest_two_sided_dist_pass_pref`; `region` → `nearest_two_sided_region_pass_pref`.                                                                                                                      |
| `[days]`         | optional for parameterised modes                                                                                                                       | `1` (`simple`) / `5` (all others)  | Simulation length in real days; scales `max_time` and `batch_size` (48 samples per day).                                                                                                                                    |
| `[geom]`         | optional for parameterised modes                                                                                                                       | `0` (`simple`) / `10` (all others) | Geometry index - selects a row from `configs/geom_specification_compact.json`.                                                                                                                                              |
| `[max_declines]` | optional for `passenger_pref`, `region_pref`, `distance_pref`, `two_sided`, `safety_objective_two_sided`                                               | off (`null`)                       | Per-request forced-accept threshold: a taxi that has declined the same request this many times must accept it next time. Omit to leave the feature off.                                                                     |

Every generated config automatically includes all default values for breaks, safety, satisfaction, driver preferences,
and passenger preferences, so base configs only need to specify what differs from defaults.

---

## Running simulations

**Single run:**

```
python run.py <config_name_without_extension>
```

Results are written to `results/` as gzipped files.

**Parallel batch:**

```
python batch_run.py configs/passenger_pref/
```

Runs all `.conf` files in the given directory in parallel across available CPU cores.

**Shell-based batch (SLURM / local):**

```
bash batch_run.sh ...
```

---

## Output files

Each run produces 4-5 gzipped files in `results/`:

| File                                     | Content                                                                                                                                          |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `run_<id>_aggregates.csv.gz`             | Per-batch aggregate metrics (means/stds across taxis, region safety if configured). One row per batch step.                                      |
| `run_<id>_per_taxi_metrics.json.gz`      | Per-batch snapshot of per-taxi metrics: trip lengths, income, time breakdowns, safety score, satisfaction score, decline counts, break state.    |
| `run_<id>_per_request_metrics.json.gz`   | End-of-simulation dump of all requests: mode, timestamps, assigned taxi, safety scores, passenger type, preference weights, cancellation reason. |
| `run_<id>_taxi_static.json.gz`           | One-time static attributes of every taxi (route preference, accept ceiling, shift cohort, initial safety). See field table below.                |
| `run_<id>_region_safety_averages.csv.gz` | Per-batch regional average safety scores (only when `regions` is configured).                                                                    |

### Key per-request fields

| Field                                  | Description                                                      |
|----------------------------------------|------------------------------------------------------------------|
| `mode`                                 | Final state: `done`, `dropped`, `pending`, `waiting`, `serving`. |
| `passenger_type`                       | Sampled passenger segment.                                       |
| `safety_threshold`                     | Personal safety score threshold.                                 |
| `w_dist`, `w_safety`, `w_wait`         | Sampled preference weights.                                      |
| `cancellation_reason`                  | Why the request was dropped (see table above).                   |
| `driver_safety_score_start/end/pickup` | Driver safety at assignment, pickup, and dropoff.                |
| `average_safety_score`                 | Mean driver safety score over the trip duration.                 |
| `assigned_taxi_distance`               | Manhattan distance from taxi to pickup at assignment.            |

---

## Analysis notebooks

| Notebook                        | Purpose                                           |
|---------------------------------|---------------------------------------------------|
| `notebooks/distributions.ipynb` | Request and trip length distributions.            |
| `notebooks/figures.ipynb`       | Main result figures (service rate, waiting time). |

---

## Debugging with interactive visualization

For debugging and fun purposes, display the map of the simulation with the taxis and requests color-coded and moving.
Must be run from a Jupyter Notebook.

```jupyter
%matplotlib
notebook
from city_model import Simulation
from ipywidgets import Button
import json

config = json.load(open('configs/simple.conf'))
s = Simulation(**config)  # create a Simulation instance
b = Button(description="Step")  # create clickable time tick button
b.on_click(s.step_time)  # assign time ticker function to button callback
b
```

Inspect individual objects:

```jupyter
print(s.requests[3])  # Request state and timestamps
print(s.taxis[1])  # Taxi position, flags, scores
```
