#!/home/bokanyie/anaconda3/bin/python

# Usage:
# python generate_configs.py 0711_base.conf
#
# This script generates config files for runs with different parameters. It creates a sweep of parameters R and d based
# on the geometry settings. The geometries are defined in the geom_specification_compact.json file, where each line
# defines the distribution of request origins and destinations as the superposition of 2D Gaussian distributions.

import sys
import json
import numpy as np

from city_model import City

CONFIGS_PATH_PREFIX = "configs/"


class ConfigGenerator:
    """
    Generate parameterized simulation configurations.

    Parameters
    ----------
    base : str
        file name for common configuration parameters (loaded from configs/ directory)
    days : int, optional
        number of real days the simulation should run (default: 1)
    """

    def __init__(self, base: str, days: float | int = 1) -> None:

        self.base = base
        self.days = days

        # different matching algorithms
        alg1 = "random_unlimited"
        alg2 = "random_limited"
        alg3 = "nearest"
        alg4 = "poorest"
        alg5 = "nearest_distance_pref"
        alg6 = "nearest_region_pref"
        alg7 = "nearest_passenger_pref"
        alg8 = "nearest_two_sided_dist_pass_pref"
        alg9 = "nearest_two_sided_region_pass_pref"
        alg10 = "safety_objective"
        alg11 = "safety_objective_two_sided"
        self.alg_list = [alg1, alg2, alg3, alg4, alg5, alg6, alg7, alg8, alg9, alg10, alg11]

        # different geometries
        geom_dict_all = {i: json.loads(geom.strip('\n')) for i, geom
                         in enumerate(open("configs/geom_specification_compact.json").readlines())}

        self.geom_dict = {i: geom_dict_all[i] for i in geom_dict_all.keys()}

        # common parameters
        with open(CONFIGS_PATH_PREFIX + base) as f:
            self.common = json.load(f)

        # =====================
        # global parameters
        # =====================

        # 1 distance unit in meters
        self.scale = 100

        # system volume
        self.system_square_m = self.common['n'] * self.common['m'] * self.scale ** 2

        # velocity of taxis in distance unit per time unit
        # should correspond to 36 km/h!!! (19 m/s = 36 km/h)
        self.velocity = 1

        # time unit in seconds
        # time unit is 100m / 10m/s * 1 grid/s
        self.tu = self.scale / 10 * self.velocity

        # simulation time:
        # days * 24h * 3600 s/h / time unit
        # days * 24 * 3600 / 10
        simulation_time = int(round(days * 24 * 3600 / self.tu, -2))
        self.common['max_time'] = simulation_time
        self.common['batch_size'] = int(simulation_time / (days * 48))  # 48 sample points in each day => every 30 min

        # reset taxi positions after an 8-hour shift
        self.reset_time = round(0.01 * 8 * 3600 / self.tu, 0) * 100

        # what to do after request, where to start and head back to, reset
        self.behav_types = [("go_back", "base", "false"),
                            ("stay", "base", "false"),
                            ("go_back", "home", "false"),
                            ("stay", "home", "false"),
                            ("stay", "home", "true")]

        self.len_dict = {}

    @staticmethod
    def avg_length(conf: dict) -> float:
        """
        Given a configuration dictionary, calculates average request length in a geometry.

        Parameters
        ----------
        conf : dict
            configuration dictionary with city geometry parameters

        Returns
        -------
        float
            average Euclidean distance of requests (rounded to 1 decimal place)
        """

        c = City(**conf)
        tt = [c.create_one_request_coord() for _ in range(c.length)]

        templ = []
        for i in range(int(len(tt) / 2)):
            templ.append(np.abs(tt[2 * i][0] - tt[2 * i + 1][0]) + np.abs(tt[2 * i][1] - tt[2 * i + 1][1]))

        return round(np.mean(templ), 1)

    def generate_config(self, d: float, R: float, alg: int | str, geom: int, behav_type: int,
                        regions_file: str | None = None,
                        no_breaks: bool = False,
                        constant_rate: bool = False) -> dict | None:
        """
        Generate a simulation configuration with specified parameters.

        Parameters
        ----------
        d : float
            characteristic distance (supply density); defines taxi spatial density as N = V / d^2
        R : float
            demand-to-supply ratio; request rate is computed as lambda = N * v * R / avg_request_length
        alg : int or str
            matching algorithm; can be algorithm index (int) or algorithm name (str).
            Valid indices reference self.alg_list; valid names are from the list.
        geom : int
            geometry index; selects a request distribution pattern from self.geom_dict
        behav_type : int
            taxi behavior type index; selects from self.behav_types (e.g., 0='go_back'+base, 1=stay+base)
        regions_file : str, optional
            path to a regions JSON file; embeds its content under
            ``regions`` in the config and sets default ``max_declines`` if not already present
        no_breaks : bool, optional
            if True, remove all break-related parameters from the config so drivers never take
            breaks; intended for calibration runs (Layers 1-3) where breaks would confound
            supply/demand or safety measurements
        constant_rate : bool, optional
            if True, remove ``request_rate_schedule`` so demand is constant throughout the
            simulation; intended for calibration runs where time-varying demand would obscure
            steady-state utilization or safety measurements

        Returns
        -------
        dict or None
            configuration dictionary if successful; None if request rate is zero or invalid

        """

        conf = dict(self.common)

        conf.pop('request_origin_distributions', None)
        conf.pop('request_destination_distributions', None)

        conf.update(self.geom_dict[geom])
        conf['geom'] = geom

        # parameters
        conf['R'] = round(R, 2)
        conf['d'] = round(d, 0)

        # d = \sqrt(V/Nt)
        N = int(round(self.system_square_m / d ** 2))
        conf['num_taxis'] = N

        if conf['geom'] in self.len_dict:
            conf['avg_request_lengths'] = self.len_dict[geom]
        else:
            conf['avg_request_lengths'] = self.avg_length(conf)
            self.len_dict[conf['geom']] = conf['avg_request_lengths']

        llambda = N * self.velocity * R / conf['avg_request_lengths']
        if llambda == 0:
            return None

        conf['request_rate'] = llambda

        if isinstance(alg, int):
            conf['matching'] = self.alg_list[alg]
        else:
            conf["matching"] = alg

        behaviour, ic, reset = self.behav_types[behav_type]
        conf.update({"behaviour": behaviour, "initial_conditions": ic})
        conf['reset'] = reset
        if reset == 'true':
            conf.update({"reset_time": self.reset_time})
        else:
            conf.pop("reset_time", None)

        if no_breaks:
            for key in ["driver_break_cohort_mix", "shift_duration_tu", "break_cohort_settings",
                        "day_length_tu", "rush_windows_tu", "p_defer_end_of_shift_in_rush",
                        "max_break_deferral_tu"]:
                conf.pop(key, None)
        else:
            if "driver_break_cohort_mix" not in conf:
                conf["driver_break_cohort_mix"] = {
                    "short_shift": 0.5,
                    "mid_shift": 0.3,
                    "long_shift": 0.2
                }
            if "shift_duration_tu" not in conf:
                conf["shift_duration_tu"] = {
                    "short_shift": {"dist": "uniform", "low": 900, "high": 1440},
                    "mid_shift": {"dist": "uniform", "low": 1440, "high": 2160},
                    "long_shift": {"dist": "uniform", "low": 2520, "high": 3240}
                }
            if "day_length_tu" not in conf:
                conf["day_length_tu"] = 8640
            if "rush_windows_tu" not in conf:
                conf["rush_windows_tu"] = [{"start": 2520, "end": 3600}, {"start": 5760, "end": 6840}]
            if "p_defer_end_of_shift_in_rush" not in conf:
                conf["p_defer_end_of_shift_in_rush"] = 0.3
            if "max_break_deferral_tu" not in conf:
                conf["max_break_deferral_tu"] = 180
            if "break_cohort_settings" not in conf:
                conf["break_cohort_settings"] = {
                    "short_shift": {
                        "inter_shift_rest_tu": {"dist": "uniform", "low": 1080, "high": 2160},
                        "intra_shift_break_after_work_tu": {"dist": "uniform", "low": 540, "high": 720},
                        "intra_shift_break_duration_tu": {"dist": "uniform", "low": 60, "high": 120},
                        "demotivation_threshold_tu": {"dist": "uniform", "low": 240, "high": 480},
                        "shift_start_offset_tu": {"dist": "uniform", "low": 1, "high": 900}
                    },
                    "mid_shift": {
                        "inter_shift_rest_tu": {"dist": "uniform", "low": 2160, "high": 4320},
                        "intra_shift_break_after_work_tu": {"dist": "uniform", "low": 720, "high": 1080},
                        "intra_shift_break_duration_tu": {"dist": "uniform", "low": 60, "high": 150},
                        "demotivation_threshold_tu": {"dist": "uniform", "low": 300, "high": 600},
                        "shift_start_offset_tu": {"dist": "uniform", "low": 1, "high": 1440}
                    },
                    "long_shift": {
                        "inter_shift_rest_tu": {"dist": "uniform", "low": 5400, "high": 8640},
                        "intra_shift_break_after_work_tu": {"dist": "uniform", "low": 1080, "high": 1440},
                        "intra_shift_break_duration_tu": {"dist": "uniform", "low": 90, "high": 180},
                        "demotivation_threshold_tu": {"dist": "uniform", "low": 360, "high": 720},
                        "shift_start_offset_tu": {"dist": "uniform", "low": 1, "high": 2520}
                    }
                }

        if constant_rate:
            conf.pop("request_rate_schedule", None)
        else:
            if "request_rate_schedule" not in conf:
                conf["request_rate_schedule"] = [
                    {"start": 0, "end": 1080, "multiplier": 0.25},
                    {"start": 1080, "end": 2520, "multiplier": 0.7},
                    {"start": 2520, "end": 3600, "multiplier": 1.8},
                    {"start": 3600, "end": 5760, "multiplier": 1.0},
                    {"start": 5760, "end": 6840, "multiplier": 2.0},
                    {"start": 6840, "end": 8640, "multiplier": 0.5}
                ]

        # Add safety score parameters if not already in base config
        if "safety_score_change_serving_rate" not in conf:
            conf["safety_score_change_serving_rate"] = -0.02
        if "safety_score_change_waiting_rate" not in conf:
            conf["safety_score_change_waiting_rate"] = -0.001
        if "safety_score_break_recovery_constant" not in conf:
            conf["safety_score_break_recovery_constant"] = 180.0
        if "safety_score_min" not in conf:
            conf["safety_score_min"] = 0
        if "safety_score_max" not in conf:
            conf["safety_score_max"] = 100
        if "initial_safety_score_min" not in conf:
            conf["initial_safety_score_min"] = 20
        if "initial_safety_score_max" not in conf:
            conf["initial_safety_score_max"] = 80

        # Add preference-aware matching parameters
        if "driver_route_pref_mix" not in conf:
            conf["driver_route_pref_mix"] = {
                "short_pref": 0.35,
                "neutral_pref": 0.30,
                "long_pref": 0.35
            }
        if "route_pref_strength_range" not in conf:
            conf["route_pref_strength_range"] = {"low": 0.2, "high": 0.9}
        if "preference_base_acceptance_prob" not in conf:
            conf["preference_base_acceptance_prob"] = 0.9
        if "nonpreferred_accept_ceiling" not in conf:
            conf["nonpreferred_accept_ceiling"] = 0.25

        # Add satisfaction parameters
        if "satisfaction_score_min" not in conf:
            conf["satisfaction_score_min"] = 0.0
        if "satisfaction_score_max" not in conf:
            conf["satisfaction_score_max"] = 100.0
        if "satisfaction_initial_min" not in conf:
            conf["satisfaction_initial_min"] = 45.0
        if "satisfaction_initial_max" not in conf:
            conf["satisfaction_initial_max"] = 55.0
        if "satisfaction_change_waiting_rate" not in conf:
            conf["satisfaction_change_waiting_rate"] = -0.01
        if "satisfaction_income_weight" not in conf:
            conf["satisfaction_income_weight"] = 0.5
        if "satisfaction_income_ref" not in conf:
            conf["satisfaction_income_ref"] = 1000.0
        if "satisfaction_pref_match_delta" not in conf:
            conf["satisfaction_pref_match_delta"] = 0.2
        if "satisfaction_pref_mismatch_delta" not in conf:
            conf["satisfaction_pref_mismatch_delta"] = -0.3

        # =====================
        # Passenger preference parameters
        # =====================
        if "passenger_preference_mix" not in conf:
            conf["passenger_preference_mix"] = {
                "safety_indifferent": 0.40,
                "safety_moderate": 0.40,
                "safety_strict": 0.20
            }
        if "passenger_safety_threshold" not in conf:
            conf["passenger_safety_threshold"] = {
                "safety_indifferent": {"mean": 0.0, "std": 0.0},
                "safety_moderate": {"mean": 55.0, "std": 10.0},
                "safety_strict": {"mean": 75.0, "std": 8.0}
            }
        if "passenger_preference_weights" not in conf:
            conf["passenger_preference_weights"] = {
                "w_dist": {"mean": 1.0, "std": 0.2},
                "w_safety": {"mean": 0.5, "std": 0.3},
                "w_wait": {"mean": 0.3, "std": 0.2}
            }
        if "passenger_score_temperature" not in conf:
            conf["passenger_score_temperature"] = 0.1

        # Embed region config when a file is provided
        if regions_file is not None:
            with open("configs/" + regions_file) as f:
                conf["regions"] = json.load(f)
            if "max_declines" not in conf:
                conf["max_declines"] = None

        return conf

    def dump_config(self, config_dict: dict | None, run: int | None = None) -> tuple[str, str] | None:
        """
        Generate filename and JSON content for a configuration.

        Parameters
        ----------
        config_dict : dict or None
            configuration dictionary to dump; None returns None
        run : int, optional
            run index; if provided, appended to filename as '_run_<run>'

        Returns
        -------
        tuple[str, str] or None
            (filename, content) pair if config_dict is not None; None otherwise
        """
        if config_dict is None:
            print("Request rate too low.")
            return None

        # pop non JSON-serializable element
        for k in ["request_origin_distributions", "request_destination_distributions"]:
            if k in config_dict:
                for elem in config_dict[k]:
                    elem.pop("cdf_inv", None)

        # request rate
        R_string = ('%.2f' % config_dict['R']).replace('.', '_')
        d_string = '%d' % config_dict['d']

        # filename
        fname = self.base.split('.')[0] + \
                '_days_' + ('%g' % self.days).replace('.', '_') + \
                '_d_' + d_string + \
                '_R_' + R_string + \
                '_alg_' + config_dict['matching'] + \
                '_geom_' + str(config_dict['geom']) + \
                '_behav_' + config_dict['behaviour'] + \
                '_ic_' + config_dict['initial_conditions'] + \
                '_reset_' + config_dict['reset']
        if run is not None:
            fname += '_run_' + str(run) + '.conf'
        else:
            fname += '.conf'

        content = json.dumps(config_dict, indent=4, separators=(',', ': ')) + '\n'

        return fname, content


if __name__ == '__main__':
    supported_modes = ["sweep", "long_run", "new_geoms", "multiple_runs", "figure2", "missing", "simple",
                       "passenger_fairness", "nearest_baseline",
                       "region_pref", "distance_pref", "passenger_pref", "two_sided",
                       "safety_objective", "safety_objective_two_sided",
                       "calibrate_supply", "calibrate_safety", "calibrate_flexibility", "calibrate_region"]
    if len(sys.argv) < 2 or sys.argv[1] not in supported_modes:
        print(f"Please give a mode argument: {', '.join(supported_modes)}")
        sys.exit(1)
    mode = sys.argv[1]

    if mode == "sweep":

        "Generating configs for all possible config combinations for exploration purposes."

        if len(sys.argv) < 3:
            print("Please give a base config file as second argument!")
            sys.exit(1)
        g = ConfigGenerator(sys.argv[2])

        # ====================================================
        # generate configs corresponding to parameter matrix
        # ====================================================

        # different Gaussian geoms
        for geom in range(7):
            for behav_type in range(len(g.behav_types)):
                # sweeping through a range of R and d systematically
                d_list = list(np.linspace(50, 400, 11))
                for d in d_list:
                    # different ratios
                    R_list = list(np.linspace(0.05, 1, 20))
                    for R in R_list:
                        # inserting different algorithms
                        for alg in g.alg_list:
                            conf = g.generate_config(d, R, alg, geom, behav_type)
                            fname, content = g.dump_config(conf)

                            # dump
                            with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                                f.write(content)

    elif mode == "long_run":
        gen = ConfigGenerator('2019_02_14_base.conf', days=100)
        conf = gen.generate_config(225, 0.5, 'nearest', 0, 1)
        fname, content = gen.dump_config(conf)
        fname = fname.split('.')[0] + '_long_run.conf'
        # dump
        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
            f.write(content)

    elif mode == "new_geoms":
        gen = ConfigGenerator('2019_02_14_base.conf')
        geoms = [0, 7, 8, 9]
        for g in geoms:
            print(g)
            conf = gen.generate_config(225, 0.5, 'nearest', g, 1)
            fname, content = gen.dump_config(conf)
            fname = fname.split('.')[0] + '_new_geoms.conf'
            # dump
            with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                f.write(content)

    elif mode == "multiple_runs":
        gen = ConfigGenerator('2019_05_19_base.conf')

        # simplest geom
        # behaviour = ic: base, behav: stay, reset: false
        # run 10/20 times to take average

        # Figure 1

        taxi_density = np.array([5, 15, 25])
        d_list = np.sqrt(1e6 / taxi_density)
        R_list = np.linspace(0.06, 1.02, 17)
        for d in d_list:
            for R in R_list:
                conf = gen.generate_config(d, R, 'nearest', 0, 1)
                for r in range(10):
                    if conf is not None:
                        fname, content = gen.dump_config(conf, run=r)
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

        # Figure 2

        taxi_density = np.linspace(3, 30, 10)  # with rho = N/A [1/km^2]
        d_list = np.sqrt(1e6 / taxi_density)
        R_list = [0.2, 0.4, 0.6]
        for d in d_list:
            for R in R_list:
                conf = gen.generate_config(d, R, 'nearest', 0, 1)
                for r in range(20):
                    if conf is not None:
                        fname, content = gen.dump_config(conf, run=r)
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

        # Figure 4

        taxi_density = 15
        d = np.sqrt(1e6 / taxi_density)
        R_list = np.linspace(0.06, 1.02, 17)
        geom_list = [0, 1, 2, 3, 6]
        for R in R_list:
            for g in geom_list:
                conf = gen.generate_config(d, R, 'nearest', g, 1)
                for r in range(10):
                    if conf is not None:
                        fname, content = gen.dump_config(conf, run=r)
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

        # Figure 5

        R = 0.4
        for g in geom_list:
            for behav in [0, 1]:
                conf = gen.generate_config(d, R, 'nearest', g, behav)
                for r in range(10):
                    if conf is not None:
                        fname, content = gen.dump_config(conf, run=r)
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

        # Figure 6

        taxi_density = 15
        d = np.sqrt(1e6 / taxi_density)
        R = 0.4

        geom_list = [0, 1, 2, 3, 6]
        algs = ['nearest', 'random_limited', 'poorest']

        for a in algs:
            for g in geom_list:
                conf = gen.generate_config(d, R, a, g, 1)
                for r in range(10):
                    if conf is not None:
                        fname, content = gen.dump_config(conf, run=r)
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

    elif mode == "figure2":
        gen = ConfigGenerator('2019_05_06_base.conf')
        taxi_density = np.linspace(3, 30, 40)  # with rho = N/A [1/km^2]
        d_list = np.sqrt(1e6 / taxi_density)
        R_list = [0.2, 0.4, 0.6]
        for d in d_list:
            for R in R_list:
                conf = gen.generate_config(d, R, 'nearest', 0, 1)
                for r in range(10):
                    if conf is not None:
                        fname, content = gen.dump_config(conf, run=r)
                        fname = fname.split('.')[0] + '_question.conf'
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

    elif mode == "missing":
        gen = ConfigGenerator('2019_05_06_base.conf')

        # Figure 1

        # missing parameter range for high R
        taxi_density = np.array([5, 15, 25])
        d_list = np.sqrt(1e6 / taxi_density)
        R_list = np.linspace(0.66, 1.02, 7)
        for d in d_list:
            for R in R_list:
                conf = gen.generate_config(d, R, 'nearest', 0, 1)
                for r in range(10):
                    if conf is not None:
                        fname, content = gen.dump_config(conf, run=r)
                        fname = fname.split('.')[0] + '_missing.conf'
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

        # Figure 2

        # more runs for averaging small R better
        taxi_density = np.linspace(3, 30, 10)  # with rho = N/A [1/km^2]
        d_list = np.sqrt(1e6 / taxi_density)
        R_list = [0.2, 0.4, 0.6]
        for d in d_list:
            for R in R_list:
                conf = gen.generate_config(d, R, 'nearest', 0, 1)
                for r in range(10, 20):
                    if conf is not None:
                        fname, content = gen.dump_config(conf, run=r)
                        fname = fname.split('.')[0] + '_missing.conf'
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

        # Figure 6

        taxi_density = 15
        d = np.sqrt(1e6 / taxi_density)
        R = 0.4

        geom_list = [0, 1, 2, 3, 6]
        algs = ['nearest', 'random', 'poorest']

        for a in algs:
            for g in geom_list:
                conf = gen.generate_config(d, R, a, g, 1)
                for r in range(10):
                    if conf is not None:
                        fname, content = gen.dump_config(conf, run=r)
                        fname = fname.split('.')[0] + '_missing.conf'
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

    elif mode == "passenger_fairness":
        gen = ConfigGenerator('passenger_fairness/test.conf', 1)
        taxi_density = 15
        d = np.sqrt(1e6 / taxi_density)
        R_list = [0.2, 0.5, 1]
        geom_list = [0, 1, 2, 3, 6]

        for R in R_list:
            for g in geom_list:
                conf = gen.generate_config(d, R, 'nearest', g, 1)
                fname, content = gen.dump_config(conf)
                with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                    f.write(content)
                print(f"Successfully wrote {fname}")

    elif mode == "nearest_baseline":
        # Baseline nearest-matching sweep over both taxi density (d) and demand ratio (R), with breaks enabled.
        # Use this to establish a preference-free reference before comparing preference-aware algorithms.
        # Usage: python generate_configs.py nearest_baseline <base_config> [days] [geom]
        # Example: python generate_configs.py nearest_baseline big_city_base_balanced_calibrated.conf 5 10
        # Defaults: days=5, geom=10
        if len(sys.argv) < 3:
            print("Usage: python generate_configs.py nearest_baseline <base_config> [days] [geom]")
            print("Example: python generate_configs.py nearest_baseline big_city_base_balanced_calibrated.conf 5 10")
            sys.exit(1)

        base_config = sys.argv[2]
        days = float(sys.argv[3]) if len(sys.argv) > 3 else 5
        geom = int(sys.argv[4]) if len(sys.argv) > 4 else 10

        gen = ConfigGenerator(base_config, days=days)
        taxi_densities = [5, 10, 15, 20, 25]  # taxis/km²
        d_list = [np.sqrt(1e6 / rho) for rho in taxi_densities]
        R_list = [0.2, 0.5, 1.0]

        for d in d_list:
            for R in R_list:
                conf = gen.generate_config(d, R, "nearest", geom, 1)
                if conf is not None:
                    fname, content = gen.dump_config(conf)
                    with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                        f.write(content)
                    print(f"Successfully wrote {fname}")

    elif mode == "region_pref":
        # Usage: python generate_configs.py region_pref <base_config> <regions_file> [days] [geom] [max_declines]
        # Generates configs sweeping R for nearest_region_pref algorithm.
        if len(sys.argv) < 4:
            print(
                "Usage: python generate_configs.py region_pref <base_config> <regions_file> [days] [geom] [max_declines]")
            print(
                "Example: python generate_configs.py region_pref big_city_base.conf configs/regions_big_city_imbalanced.json 5 10 3")
            sys.exit(1)

        base_config = sys.argv[2]
        regions_file = sys.argv[3]
        days = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        geom = int(sys.argv[5]) if len(sys.argv) > 5 else 10
        max_declines = int(sys.argv[6]) if len(sys.argv) > 6 else None

        gen = ConfigGenerator(base_config, days=days)
        taxi_densities = [5, 10, 15, 20, 25]  # taxis/km²
        d_list = [np.sqrt(1e6 / rho) for rho in taxi_densities]
        R_list = [0.2, 0.5, 1.0]

        for d in d_list:
            for R in R_list:
                for behav in range(4):
                    conf = gen.generate_config(d, R, "nearest_region_pref", geom, behav, regions_file=regions_file)
                    if conf is not None:
                        if max_declines is not None:
                            conf["max_declines"] = max_declines
                        fname, content = gen.dump_config(conf)
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

    elif mode == "distance_pref":
        # Usage: python generate_configs.py distance_pref <base_config> [days] [geom] [max_declines]
        # Generates configs sweeping R for nearest_distance_pref algorithm.
        if len(sys.argv) < 3:
            print("Usage: python generate_configs.py distance_pref <base_config> [days] [geom] [max_declines]")
            print("Example: python generate_configs.py distance_pref big_city_base.conf 5 10")
            sys.exit(1)

        base_config = sys.argv[2]
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        geom = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        max_declines = int(sys.argv[5]) if len(sys.argv) > 5 else None

        gen = ConfigGenerator(base_config, days=days)
        taxi_densities = [5, 10, 15, 20, 25]  # taxis/km²
        d_list = [np.sqrt(1e6 / rho) for rho in taxi_densities]
        R_list = [0.2, 0.5, 1.0]

        for d in d_list:
            for R in R_list:
                for behav in range(4):
                    conf = gen.generate_config(d, R, "nearest_distance_pref", geom, behav)
                    if conf is not None:
                        if max_declines is not None:
                            conf["max_declines"] = max_declines
                        fname, content = gen.dump_config(conf)
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

    elif mode == "passenger_pref":
        # Passenger preference mode: sweeps R across three algorithms to compare
        # baseline (nearest), passenger-only preference, and two-sided preference.
        # Usage: python generate_configs.py passenger_pref <base_config> [days] [geom] [max_declines]
        # Example: python generate_configs.py passenger_pref big_city_base.conf 5 10 3
        if len(sys.argv) < 3:
            print("Usage: python generate_configs.py passenger_pref <base_config> [days] [geom] [max_declines]")
            print("Example: python generate_configs.py passenger_pref big_city_base.conf 5 10 3")
            sys.exit(1)

        base_config = sys.argv[2]
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        geom = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        max_declines = int(sys.argv[5]) if len(sys.argv) > 5 else None

        gen = ConfigGenerator(base_config, days=days)
        taxi_densities = [5, 10, 15, 20, 25]  # taxis/km²
        d_list = [np.sqrt(1e6 / rho) for rho in taxi_densities]
        R_list = [0.2, 0.5, 1.0]
        # Compare baseline, passenger-only, and both two-sided variants
        algs = ["nearest", "nearest_passenger_pref", "nearest_two_sided_dist_pass_pref",
                "nearest_two_sided_region_pass_pref"]

        for d in d_list:
            for R in R_list:
                for alg in algs:
                    conf = gen.generate_config(d, R, alg, geom, 1)
                    if conf is not None:
                        if max_declines is not None:
                            conf["max_declines"] = max_declines
                        fname, content = gen.dump_config(conf)
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

    elif mode == "two_sided":
        # Usage: python generate_configs.py two_sided <base_config> <variant> [days] [geom] [max_declines]
        # variant: "dist" -> nearest_two_sided_dist_pass_pref
        #          "region" -> nearest_two_sided_region_pass_pref
        # Generates configs sweeping R for the chosen two-sided matching algorithm.
        valid_variants = {"dist": "nearest_two_sided_dist_pass_pref", "region": "nearest_two_sided_region_pass_pref"}
        if len(sys.argv) < 4 or sys.argv[3] not in valid_variants:
            print("Usage: python generate_configs.py two_sided <base_config> <variant> [days] [geom] [max_declines]")
            print("  variant: 'dist'   -> nearest_two_sided_dist_pass_pref")
            print("           'region' -> nearest_two_sided_region_pass_pref")
            print("Example: python generate_configs.py two_sided big_city_base.conf dist 5 10 3")
            sys.exit(1)

        base_config = sys.argv[2]
        variant = sys.argv[3]
        days = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        geom = int(sys.argv[5]) if len(sys.argv) > 5 else 10
        max_declines = int(sys.argv[6]) if len(sys.argv) > 6 else None

        alg = valid_variants[variant]
        gen = ConfigGenerator(base_config, days=days)
        taxi_densities = [5, 10, 15, 20, 25]  # taxis/km²
        d_list = [np.sqrt(1e6 / rho) for rho in taxi_densities]
        R_list = [0.2, 0.5, 1.0]

        for d in d_list:
            for R in R_list:
                conf = gen.generate_config(d, R, alg, geom, 1)
                if conf is not None:
                    if max_declines is not None:
                        conf["max_declines"] = max_declines
                    fname, content = gen.dump_config(conf)
                    with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                        f.write(content)
                    print(f"Successfully wrote {fname}")

    elif mode == "safety_objective":
        # Usage: python generate_configs.py safety_objective <base_config> [days] [geom]
        # Generates configs sweeping R for the safety_objective matching algorithm.
        if len(sys.argv) < 3:
            print("Usage: python generate_configs.py safety_objective <base_config> [days] [geom]")
            print("Example: python generate_configs.py safety_objective big_city_base.conf 5 10")
            sys.exit(1)

        base_config = sys.argv[2]
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        geom = int(sys.argv[4]) if len(sys.argv) > 4 else 10

        gen = ConfigGenerator(base_config, days=days)
        taxi_densities = [5, 10, 15, 20, 25]  # taxis/km²
        d_list = [np.sqrt(1e6 / rho) for rho in taxi_densities]
        R_list = [0.2, 0.5, 1.0]

        for d in d_list:
            for R in R_list:
                for behav in range(4):
                    conf = gen.generate_config(d, R, "safety_objective", geom, behav)
                    if conf is not None:
                        fname, content = gen.dump_config(conf)
                        with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                            f.write(content)
                        print(f"Successfully wrote {fname}")

    elif mode == "simple":
        # Simple mode: creates a single config file with default parameters
        # Usage: python generate_configs.py simple <base_config> [days] [geom]

        if len(sys.argv) < 3:
            print("Usage: python generate_configs.py simple <base_config> [days] [geom]")
            print("Example: python generate_configs.py simple 2019_05_06_base.conf 5 10")
            print("Defaults: days=1, geom=0, d=225, R=0.5, algorithm=nearest, behavior=stay")
            sys.exit(1)

        base_config = sys.argv[2]
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        geom = int(sys.argv[4]) if len(sys.argv) > 4 else 0

        gen = ConfigGenerator(base_config, days=days)
        # Generate with default parameters: d=225, R=0.5, nearest algorithm, stay with base behavior
        conf = gen.generate_config(225, 0.5, 'nearest', geom, 1)

        if conf is not None:
            fname, content = gen.dump_config(conf)
            with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                f.write(content)
            print(f"Successfully wrote {fname}")
            print(f"  Simulation duration: {days} days")
            print("  Taxi density (d): 225")
            print("  Demand ratio (R): 0.5")
            print("  Algorithm: nearest")
            print(f"  Geometry: {geom}")
            print("  Behavior: stay")
        else:
            print("Failed to generate config")
            sys.exit(1)

    elif mode == "safety_objective_two_sided":
        # System safety-objective matching combined with two-sided preferences:
        # region-popularity driver preference + passenger safety-score preference.
        # Requires a regions file so driver region-popularity is meaningful.
        # Usage: python generate_configs.py safety_objective_two_sided <base_config> <regions_file> [days] [geom] [max_declines]
        # Example: python generate_configs.py safety_objective_two_sided big_city_base_balanced_calibrated.conf regions_big_city_balanced.json 5 10
        if len(sys.argv) < 4:
            print("Usage: python generate_configs.py safety_objective_two_sided <base_config> <regions_file> [days] [geom] [max_declines]")
            print("Example: python generate_configs.py safety_objective_two_sided big_city_base_balanced_calibrated.conf regions_big_city_balanced.json 5 10")
            sys.exit(1)

        base_config = sys.argv[2]
        regions_file = sys.argv[3]
        days = float(sys.argv[4]) if len(sys.argv) > 4 else 5
        geom = int(sys.argv[5]) if len(sys.argv) > 5 else 10
        max_declines = int(sys.argv[6]) if len(sys.argv) > 6 else None

        gen = ConfigGenerator(base_config, days=days)
        taxi_densities = [5, 10, 15, 20, 25]  # taxis/km²
        d_list = [np.sqrt(1e6 / rho) for rho in taxi_densities]
        R_list = [0.2, 0.5, 1.0]

        for d in d_list:
            for R in R_list:
                conf = gen.generate_config(d, R, "safety_objective_two_sided", geom, 1,
                                           regions_file=regions_file)
                if conf is not None:
                    if max_declines is not None:
                        conf["max_declines"] = max_declines
                    fname, content = gen.dump_config(conf)
                    with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                        f.write(content)
                    print(f"Successfully wrote {fname}")

    elif mode == "calibrate_supply":
        # Calibration Layer 1/2: nearest algorithm, no breaks, constant demand rate.
        # Produces a utilization and income baseline for the chosen supply/demand ratio.
        # Usage: python generate_configs.py calibrate_supply <base_config> [days] [geom]
        # Example: python generate_configs.py calibrate_supply big_city_base.conf 0.25 0
        # Defaults: days=0.25 (~6 simulated hours), geom=0
        if len(sys.argv) < 3:
            print("Usage: python generate_configs.py calibrate_supply <base_config> [days] [geom]")
            print("Example: python generate_configs.py calibrate_supply big_city_base.conf 0.25 0")
            sys.exit(1)

        base_config = sys.argv[2]
        days = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
        geom = int(sys.argv[4]) if len(sys.argv) > 4 else 0

        gen = ConfigGenerator(base_config, days=days)
        d = np.sqrt(1e6 / 15)  # taxi density ~15/km²
        R_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

        for R in R_list:
            conf = gen.generate_config(d, R, "nearest", geom, 1,
                                       no_breaks=True, constant_rate=True)
            if conf is not None:
                fname, content = gen.dump_config(conf)
                with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                    f.write(content)
                print(f"Successfully wrote {fname}")

    elif mode == "calibrate_safety":
        # Calibration Layer 3: nearest algorithm, no breaks, constant demand rate.
        # Isolates the safety score degradation rate over approximately one long-shift duration.
        # days=0.25 produces ~2200 time units, spanning the long-shift reference length.
        # Usage: python generate_configs.py calibrate_safety <base_config> [days] [geom]
        # Example: python generate_configs.py calibrate_safety big_city_base.conf 0.25 0
        if len(sys.argv) < 3:
            print("Usage: python generate_configs.py calibrate_safety <base_config> [days] [geom]")
            print("Example: python generate_configs.py calibrate_safety big_city_base.conf 0.25 0")
            sys.exit(1)

        base_config = sys.argv[2]
        days = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
        geom = int(sys.argv[4]) if len(sys.argv) > 4 else 0

        gen = ConfigGenerator(base_config, days=days)
        d = np.sqrt(1e6 / 15)
        R = 0.5  # mid-utilization

        conf = gen.generate_config(d, R, "nearest", geom, 1,
                                   no_breaks=True, constant_rate=True)
        if conf is not None:
            fname, content = gen.dump_config(conf)
            with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                f.write(content)
            print(f"Successfully wrote {fname}")
            print(f"  max_time: {conf['max_time']} TU  (long-shift reference: ~2880 TU)")

    elif mode == "calibrate_flexibility":
        # Calibration Layer 5: nearest_distance_pref algorithm with income flexibility active.
        # Samples the shortfall distribution across acceptance decisions at multiple R values.
        # Requires income_target_rate in the base config (from Layer 2).
        # Usage: python generate_configs.py calibrate_flexibility <base_config> [days] [geom]
        # Example: python generate_configs.py calibrate_flexibility big_city_base.conf 1 0
        if len(sys.argv) < 3:
            print("Usage: python generate_configs.py calibrate_flexibility <base_config> [days] [geom]")
            print("Example: python generate_configs.py calibrate_flexibility big_city_base.conf 1 0")
            print("Note: income_target_rate must be set in the base config (from Layer 2).")
            sys.exit(1)

        base_config = sys.argv[2]
        days = float(sys.argv[3]) if len(sys.argv) > 3 else 1
        geom = int(sys.argv[4]) if len(sys.argv) > 4 else 0

        gen = ConfigGenerator(base_config, days=days)
        d = np.sqrt(1e6 / 15)
        R_list = [0.3, 0.5, 0.7]

        for R in R_list:
            conf = gen.generate_config(d, R, "nearest_distance_pref", geom, 1)
            if conf is not None:
                fname, content = gen.dump_config(conf)
                with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                    f.write(content)
                print(f"Successfully wrote {fname}")

    elif mode == "calibrate_region":
        # Calibration Layer 7: paired nearest baseline and nearest_region_pref configs at the same R values.
        # Produces per-region service-rate data for comparing geographic acceptance against the baseline.
        # Usage: python generate_configs.py calibrate_region <base_config> <regions_file> [days] [geom]
        # Example: python generate_configs.py calibrate_region big_city_base.conf regions.json 0.25 0
        if len(sys.argv) < 4:
            print("Usage: python generate_configs.py calibrate_region <base_config> <regions_file> [days] [geom]")
            print("Example: python generate_configs.py calibrate_region big_city_base.conf regions.json 0.25 0")
            sys.exit(1)

        base_config = sys.argv[2]
        regions_file = sys.argv[3]
        days = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
        geom = int(sys.argv[5]) if len(sys.argv) > 5 else 0

        gen = ConfigGenerator(base_config, days=days)
        d = np.sqrt(1e6 / 15)
        R_list = [0.3, 0.5, 0.7]

        for R in R_list:
            conf_base = gen.generate_config(d, R, "nearest", geom, 1, constant_rate=True)
            if conf_base is not None:
                fname, content = gen.dump_config(conf_base)
                with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                    f.write(content)
                print(f"Successfully wrote {fname}")

            conf_region = gen.generate_config(d, R, "nearest_region_pref", geom, 1,
                                              regions_file=regions_file, constant_rate=True)
            if conf_region is not None:
                fname, content = gen.dump_config(conf_region)
                with open(CONFIGS_PATH_PREFIX + fname, 'w') as f:
                    f.write(content)
                print(f"Successfully wrote {fname}")
