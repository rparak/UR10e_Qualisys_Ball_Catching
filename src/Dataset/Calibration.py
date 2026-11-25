## =========================================================================== ##
# MIT License
# Copyright (c) 2026 Roman Parak
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
## =========================================================================== ##
# Author   : Roman Parak
# File Name: ../Dataset/Calibration.py
## =========================================================================== ##

# CSV (CSV file reading and writing)
import csv
# Math (Mathematical functions)
import math
# Dataclasses (Data Classes)
from dataclasses import dataclass, field, replace
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the configurations' structures
from Configuration.Core import Configuration_Str, Dataset_Configuration_Str
#   ../Geometry/Core - the throw cuboid
from Geometry.Core import Cuboid_Cls
#   ../Flight/Core - the flight model
from Flight.Core import Flight_Model_Str, Simulate
#   ../Recording/Core, Segmentation, Identification - the transform, the flights and their fit
from Recording.Core import Transform_To_Robot
from Recording.Segmentation import Flight_Str
from Recording.Identification import Fit_Initial_States, Fitted_Flight_Str
#   ../Observation/Core, Estimation - the error model, the patterns and their statistics
from Observation.Core import Assign_Pools, Donor_Str, Draw_Pattern, Observation_Error, Observation_Model_Str, CONST_DONOR_RATES
from Observation.Estimation import (Aggregate_Error, Aggregate_Pattern, Bootstrap_Bounds, Error_Parts, Estimate_Observation_Model,
                                    Pattern_Parts, Residual_Extremes, Residual_Str, CONST_ERROR_DIAGNOSTICS, CONST_ERROR_STATISTICS,
                                    CONST_EXTREMES, CONST_PATTERN_STATISTICS)
#   ../Dataset/Core - the throws, the target area, the floor time
from Dataset.Core import Draw_Throws, Floor_Time, Target_Area

"""
Description:
    Initialization of constants.

    The calibration of the generation models on the 45 identification flights (SPEC §7.5, §7.6, §9.9): the error model's
    estimate and its spread, the error and
    pattern statistics of the real flights against generated ones with their bootstrap bounds, the
    release distribution, the throw-shape statistics, and the coverage of the identification releases
    by the generated throws (§2.13.6). Only identification flights are passed in; the validation flights
    are never read (SPEC §2.14).
"""
# A release whose fitted apex is less than this above it is a late start (the release not observed) and
# is left out of the release distribution (the frame decision: 100 Hz ON
# Experiment 6, 0.24 m; the next smallest 0.93 m). *Assumed*.
#   Unit [float, m]
CONST_LATE_START = 0.5
# The seed of every synthetic check, and the number of parametric-bootstrap estimates.
#   Unit [int]
CONST_CALIBRATION_SEED = 20261111
CONST_SPREAD_REPEATS = 5
# The synthetic seeds of the error-statistics check: each generated statistic is the median over this many
# synthetic sets, its verdict against the bound taken on the median (a single
# set's count of statistics outside ranged 8-13 for the same parameters).
#   Unit [int]
CONST_CHECK_SEEDS = 5
# The generated throws of the throw-shape check (F6 Q9) and the composed patterns per rate.
#   Unit [int]
CONST_SHAPE_THROWS = 10000
CONST_PATTERN_DRAWS = 1000
# The height above which no recorded sample exists (F6 Part A); the share of throws above it is reported.
#   Unit [float, m]
CONST_CAMERA_CEILING = 2.75
# The quantiles of the throw-shape statistics and the quantities (F6 Q9).
#   Unit [Tuple<int>, %], [Tuple<string>]
CONST_SHAPE_QUANTILES = (10, 50, 90)
CONST_SHAPE_QUANTITIES = ('speed', 'elevation', 'apex', 'floor_time')
# The grid on which the apex is found, and the longest flight considered.
#   Unit [float, s]
CONST_APEX_STEP = 1.0 / 600.0
CONST_LONGEST_FLIGHT = 2.5
# The time step of the coverage check's search for a target-area crossing.
#   Unit [float, s]
CONST_COVERAGE_STEP = 0.001
# The error model's bounds from the identification flights' largest values: the value
# times this margin, rounded up to this step. The margin is *assumed*: without one the largest recorded
# residual would be reached by 8 of 100 synthetic sets of the 45 flights, with it by 17 (Part A, 2026-11-15).
#   Unit [float, -], [float, m]
CONST_BOUND_MARGIN = 1.25
CONST_BOUND_STEP = 0.005


class Calibration_Error_Cls(ValueError):
    """
    Description:
        The calibration cannot run: a transform that turns the vertical (the error model's axes and the
        flight model assume a rotation about z only). The message names the value.
    """


@dataclass
class Calibration_Str:
    """
    Description:
        Everything scripts/catch.py observation reports (SPEC §9.9). A bound is a (low, high) bootstrap
        interval over the identification flights; 'generated' values come from the configured models.
    """

    # The estimated error model and, per parameter, the (min, max) of CONST_SPREAD_REPEATS re-estimates on
    # synthetic flights generated from it (a parametric bootstrap).
    #   Unit [Observation_Model_Str], [Dict<string, Tuple<float, float>>]
    Estimate: Observation_Model_Str = field(default_factory=Observation_Model_Str)
    Spread: tp.Dict[str, tp.Tuple[float, float]] = field(default_factory=dict)
    # The largest values of the identification residuals (Observation.Estimation.CONST_EXTREMES) and the bounds
    # derived from them (Derived_Bound): the offset's from the largest residual, the white noise's from the
    # largest step inside a run.
    #   Unit [Dict<string, float>, m], [float, m], [float, m]
    Extremes: tp.Dict[str, float] = field(default_factory=dict)
    Offset_Bound: float = 0.0
    White_Bound: float = 0.0
    # The error statistics: real, bounds, generated with the configured model (the median over
    # CONST_CHECK_SEEDS synthetic sets) and the (min, max) over those sets.
    #   Unit [Dict<string, float>], [Dict<string, Tuple<float, float>>], [Dict<string, float>], [Dict<string, Tuple<float, float>>]
    Error_Real: tp.Dict[str, float] = field(default_factory=dict)
    Error_Bounds: tp.Dict[str, tp.Tuple[float, float]] = field(default_factory=dict)
    Error_Generated: tp.Dict[str, float] = field(default_factory=dict)
    Error_Generated_Range: tp.Dict[str, tp.Tuple[float, float]] = field(default_factory=dict)
    # The pattern statistics per simulated rate: (real, bounds, generated, the real rate compared against).
    #   Unit [Dict<float, Tuple>]
    Patterns: tp.Dict[float, tp.Tuple[tp.Dict[str, float], tp.Dict[str, tp.Tuple[float, float]], tp.Dict[str, float], float]] = \
        field(default_factory=dict)
    # The release distribution: the releases used, those left out, mean and sd per axis, apex mean and sd.
    #   Unit [int], [List<string>], [Vector<float> 1x3, m], [Vector<float> 1x3, m], [float, m], [float, m]
    Releases: int = 0
    Excluded: tp.List[str] = field(default_factory=list)
    Release_Mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Release_Std: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Apex_Mean: float = 0.0
    Apex_Std: float = 0.0
    # The throw-shape statistics ('<quantity>_p<q>'): real, bounds, generated; the share of generated throws
    # above CONST_CAMERA_CEILING and the real flights above it.
    #   Unit [Dict<string, float>], [Dict<string, Tuple<float, float>>], [Dict<string, float>], [float, -], [int], [int]
    Shape_Real: tp.Dict[str, float] = field(default_factory=dict)
    Shape_Bounds: tp.Dict[str, tp.Tuple[float, float]] = field(default_factory=dict)
    Shape_Generated: tp.Dict[str, float] = field(default_factory=dict)
    Above_Share: float = 0.0
    Above_Real: int = 0
    Above_Real_Of: int = 0
    # The coverage (§2.13.6): per identification flight, its name and why it is outside ('' inside).
    #   Unit [List<Tuple<string, string>>]
    Coverage: tp.List[tp.Tuple[str, str]] = field(default_factory=list)


def _Check_Transform(configuration: Configuration_Str) -> None:
    """
    Description:
        The transform rotates about z only (SPEC §7.3): then the QTM and robot axes coincide up to sign,
        the identification residual's per-axis variances hold in the robot frame, and gravity stays
        along -z.

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.
    """

    alpha, beta, _ = configuration.Transform.Orientation
    if abs(alpha) > 1e-12 or abs(beta) > 1e-12 or abs(math.sin(2.0 * configuration.Transform.Orientation[2])) > 1e-12:
        raise Calibration_Error_Cls('Transform.Orientation %s rad is not a multiple of 90 deg about z: the error model per axis and '
                                    'the flight model assume the vertical is kept. Revise the calibration first (SPEC §7.3).'
                                    % (list(configuration.Transform.Orientation),))


def Derived_Bound(largest: float) -> float:
    """
    Description:
        A bound of the error model from the largest value the identification flights show: the value times
        CONST_BOUND_MARGIN, rounded up to a whole multiple of CONST_BOUND_STEP (a value
        already on a multiple stays).

    Args:
        (1) largest [float]: The largest value [m], >= 0.

    Returns:
        (1) parameter [float]: The bound [m].
    """

    return round(math.ceil(largest * CONST_BOUND_MARGIN / CONST_BOUND_STEP - 1e-9) * CONST_BOUND_STEP, 9)


def Robot_States(fitted: tp.List[Fitted_Flight_Str], configuration: Configuration_Str) -> np.ndarray:
    """
    Description:
        The fitted initial states in the robot frame by the assumed transform (SPEC §7.3): the position
        transformed, the velocity rotated.

    Args:
        (1) fitted [List<Fitted_Flight_Str>]: The fitted flights (QTM frame).
        (2) configuration [Configuration_Str(object)]: The system configuration.

    Returns:
        (1) parameter [Matrix<float> Fx6]: The states [m, m/s].
    """

    x = np.array([f.X_0 for f in fitted])
    origin = Transform_To_Robot(np.zeros((1, 3)), Transform_Str=configuration.Transform)
    return np.hstack((Transform_To_Robot(x[:, :3], Transform_Str=configuration.Transform),
                      Transform_To_Robot(x[:, 3:], Transform_Str=configuration.Transform) - origin))


def Shape_Values(state: np.ndarray, *, Flight_Model_Str: Flight_Model_Str) -> tp.Dict[str, np.ndarray]:
    """
    Description:
        The shape of throws from their release states (F6 Q9): the release speed [m/s], the elevation of
        the release velocity [deg], the apex height (the largest z on the grid of CONST_APEX_STEP) [m] and
        the flight time from the release to the floor (Dataset.Core.Floor_Time) [s].

    Args:
        (1) state [Matrix<float> Bx6]: The release states, robot frame [m, m/s].
        (2) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Dict<string, Vector<float>>]: Per quantity of CONST_SHAPE_QUANTITIES, one value per throw.
    """

    grid = np.arange(0.0, CONST_LONGEST_FLIGHT, CONST_APEX_STEP)
    apex, apex_time = np.zeros(len(state)), np.zeros(len(state))
    for start in range(0, len(state), 1000):
        z = Simulate(state[start:start + 1000], grid, Flight_Model_Str=Flight_Model_Str)[..., 2]
        apex[start:start + 1000] = z.max(axis=1)
        apex_time[start:start + 1000] = grid[np.argmax(z, axis=1)]
    speed = np.linalg.norm(state[:, 3:], axis=1)
    return {'speed': speed, 'elevation': np.degrees(np.arcsin(state[:, 5] / speed)), 'apex': apex,
            'floor_time': Floor_Time(state, apex_time, Flight_Model_Str=Flight_Model_Str)}


def _Shape_Statistics(values: tp.Dict[str, np.ndarray]) -> tp.Dict[str, float]:
    """
    Description:
        The quantiles CONST_SHAPE_QUANTILES of each shape quantity.

    Args:
        (1) values [Dict<string, Vector<float>>]: Shape_Values.

    Returns:
        (1) parameter [Dict<string, float>]: '<quantity>_p<q>' -> value.
    """

    return {'%s_p%d' % (name, q): float(np.percentile(values[name], q)) for name in CONST_SHAPE_QUANTITIES for q in CONST_SHAPE_QUANTILES}


def Coverage(names: tp.List[str], state: np.ndarray, configuration: Configuration_Str, dataset: Dataset_Configuration_Str, *,
             Flight_Model_Str: Flight_Model_Str) -> tp.List[tp.Tuple[str, str]]:
    """
    Description:
        Whether each release state lies in the support of the generated throws (SPEC §2.13.6): its position
        in Areas.Throw, its speed in Throw.Release_Speed, and its trajectory inside the target area
        (Dataset.Core.Target_Area: Throw.Target_Area or Throw.Target_Catch_Area) at a time in Throw.Flight_Time
        (searched on a grid of CONST_COVERAGE_STEP). Every reason a state is outside is named.

    Args:
        (1) names [List<string>]: The flights' names.
        (2) state [Matrix<float> Fx6]: Their release states, robot frame [m, m/s].
        (3) configuration [Configuration_Str(object)]: The system configuration.
        (4) dataset [Dataset_Configuration_Str(object)]: The dataset configuration.
        (5) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [List<Tuple<string, string>>]: (name, reasons; '' when inside).
    """

    throw = Cuboid_Cls(configuration.Areas.Throw.Centre, configuration.Areas.Throw.Size)
    target = Target_Area(dataset, configuration)
    low, high = dataset.Throw.Flight_Time
    times = np.arange(low, high + 0.5 * CONST_COVERAGE_STEP, CONST_COVERAGE_STEP)
    path = Simulate(state, times, Flight_Model_Str=Flight_Model_Str)[..., :3]
    out = []
    for i, name in enumerate(names):
        reasons = []
        if not throw.Is_Point_Inside(state[i, :3]):
            reasons.append('release outside Areas.Throw')
        speed = float(np.linalg.norm(state[i, 3:]))
        if not dataset.Throw.Release_Speed[0] <= speed <= dataset.Throw.Release_Speed[1]:
            reasons.append('release speed %.2f m/s outside Throw.Release_Speed' % speed)
        if not any(target.Is_Point_Inside(p) for p in path[i]):
            reasons.append('not in the target area during Throw.Flight_Time')
        out.append((name, '; '.join(reasons)))
    return out


def Synthetic_Residuals(flights: tp.List[Flight_Str], fitted: tp.List[Fitted_Flight_Str], seed: tp.List[int], *,
                        Flight_Model_Str: Flight_Model_Str, Observation_Model_Str: Observation_Model_Str) -> tp.List[Residual_Str]:
    """
    Description:
        The residuals of synthetic flights made like the real ones: each identification flight's own
        frames, the truth from its fitted initial state, plus the error model's error; refitted with the
        identification's fit, so the synthetic residual passes through the same fit as the real one.

    Args:
        (1) flights [List<Flight_Str>]: The trimmed identification flights.
        (2) fitted [List<Fitted_Flight_Str>]: Their fits.
        (3) seed [List<int>]: The generator's seed.
        (4) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.
        (5) Observation_Model_Str [Observation_Model_Str(object)]: REQUIRED, keyword-only: the error model.

    Returns:
        (1) parameter [List<Residual_Str>]: The synthetic residuals.
    """

    rng = np.random.default_rng(seed)
    synthetic = []
    for flight, fit in zip(flights, fitted):
        truth = Simulate(fit.X_0, flight.Time, Flight_Model_Str=Flight_Model_Str)[:, :3]
        error = Observation_Error(flight.Time, flight.Rate, Observation_Model_Str=Observation_Model_Str, scale=1.0, rng=rng)
        synthetic.append(replace(flight, Position=truth + error))
    return [Residual_Str(Frame=f.Flight.Frame, Rate=f.Flight.Rate, Residual=f.Residual)
            for f in Fit_Initial_States(synthetic, Flight_Model_Str=Flight_Model_Str)]


def Calibrate(flights: tp.List[Flight_Str], configuration: Configuration_Str, dataset: Dataset_Configuration_Str, *,
              Flight_Model_Str: Flight_Model_Str, Observation_Model_Str: Observation_Model_Str,
              progress: tp.Optional[tp.Callable[[str], None]] = None) -> Calibration_Str:
    """
    Description:
        The calibration (SPEC §9.9) on the trimmed identification flights: (1) fit them under the flight
        model; (2) estimate the error model and its spread, and take the residuals' largest values and the bounds
        derived from them; (3) the error statistics, real against the
        configured model (Observation_Model_Str); (4) the pattern statistics per rate of CONST_DONOR_RATES,
        real against CONST_PATTERN_DRAWS composed patterns of the real floor times (370 Hz against the real
        300 Hz); (5) the release distribution (late starts left out); (6) the throw-shape statistics,
        real against CONST_SHAPE_THROWS throws drawn with the dataset configuration; (7) the coverage of the
        identification releases. Every synthetic draw has a seed under CONST_CALIBRATION_SEED.

    Args:
        (1) flights [List<Flight_Str>]: The trimmed identification flights (Dataset.Core.Identification_Flights).
        (2) configuration [Configuration_Str(object)]: The system configuration.
        (3) dataset [Dataset_Configuration_Str(object)]: The dataset configuration (the throw distribution).
        (4) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.
        (5) Observation_Model_Str [Observation_Model_Str(object)]: REQUIRED, keyword-only: the configured error model.
        (6) progress [Callable[[string], None]]: Called with a step's name, or None.

    Returns:
        (1) parameter [Calibration_Str(object)]: The calibration.
    """

    def Step(name: str) -> None:
        """
        Description:
            Report a step.

        Args:
            (1) name [string]: The step.
        """

        if progress is not None:
            progress(name)

    _Check_Transform(configuration)
    seed = CONST_CALIBRATION_SEED
    names = ['%s/Experiment_%d.txt' % (flight.Folder, flight.Experiment) for flight in flights]
    Step('fitting the identification flights')
    fitted = Fit_Initial_States(flights, Flight_Model_Str=Flight_Model_Str)
    residuals = [Residual_Str(Frame=f.Flight.Frame, Rate=f.Flight.Rate, Residual=f.Residual) for f in fitted]

    Step('estimating the error model and its spread')
    estimate = Estimate_Observation_Model(residuals)
    extremes = Residual_Extremes(residuals)
    # The estimator gives no bound: the synthetic flights of the spread carry the configured ones, as the datasets do.
    bounded = replace(estimate, Offset_Bound=Observation_Model_Str.Offset_Bound, White_Bound=Observation_Model_Str.White_Bound)
    repeats = [Estimate_Observation_Model(Synthetic_Residuals(flights, fitted, [seed, 1, r], Flight_Model_Str=Flight_Model_Str,
                                                              Observation_Model_Str=bounded)) for r in range(CONST_SPREAD_REPEATS)]
    spread = {}
    for name in ('White_Std', 'White_Dof', 'Drift_Std', 'Drift_Time', 'Offset_Std', 'Offset_Time', 'Offset_Dof'):
        values = np.array([np.atleast_1d(getattr(m, name)) for m in repeats])
        for axis in range(values.shape[1]):
            key = name if values.shape[1] == 1 else '%s_%s' % (name, 'xyz'[axis])
            spread[key] = (float(values[:, axis].min()), float(values[:, axis].max()))

    Step('error statistics')
    parts = Error_Parts(residuals)
    error_real, error_bounds = Aggregate_Error(parts), Bootstrap_Bounds(parts, Aggregate_Error, seed=seed)
    sets = [Aggregate_Error(Error_Parts(Synthetic_Residuals(flights, fitted, [seed, 2, k], Flight_Model_Str=Flight_Model_Str,
                                                            Observation_Model_Str=Observation_Model_Str))) for k in range(CONST_CHECK_SEEDS)]
    error_generated = {key: float(np.median([s[key] for s in sets])) for key in sets[0]}
    error_range = {key: (float(min(s[key] for s in sets)), float(max(s[key] for s in sets))) for key in sets[0]}

    Step('pattern statistics')
    donors = Assign_Pools([Donor_Str(Name=name, Rate=flight.Rate, Time=flight.Time) for name, flight in zip(names, flights)])
    floor = Floor_Time(np.array([f.X_0 for f in fitted]), np.array([f.Flight.Time[-1] for f in fitted]), Flight_Model_Str=Flight_Model_Str)
    patterns = {}
    for rate, donor_rate in CONST_DONOR_RATES.items():
        rows = [i for i, flight in enumerate(flights) if flight.Rate == donor_rate]
        real_parts = Pattern_Parts([flights[i].Frame for i in rows], [donor_rate] * len(rows))
        rng = np.random.default_rng([seed, 3, int(rate)])
        frames = []
        for _ in range(CONST_PATTERN_DRAWS):
            present, _ = Draw_Pattern(donors, rate, float(floor[rows[int(rng.integers(len(rows)))]]), 'composed', 'all', rng)
            frames.append(np.flatnonzero(present))
        patterns[rate] = (Aggregate_Pattern(real_parts), Bootstrap_Bounds(real_parts, Aggregate_Pattern, seed=seed),
                          Aggregate_Pattern(Pattern_Parts(frames, [rate] * len(frames))), donor_rate)

    Step('release distribution and throw shape')
    state = Robot_States(fitted, configuration)
    shape = Shape_Values(state, Flight_Model_Str=Flight_Model_Str)
    kept = [i for i in range(len(flights)) if shape['apex'][i] - state[i, 2] >= CONST_LATE_START]
    excluded = [names[i] for i in range(len(flights)) if i not in kept]
    real_values = {key: value[kept] for key, value in shape.items()}
    releases = state[kept, :3]
    shape_parts = [{key: value[i] for key, value in real_values.items()} for i in range(len(kept))]

    def Aggregate_Shape(sample: tp.List[tp.Dict[str, float]]) -> tp.Dict[str, float]:
        """
        Description:
            The throw-shape statistics of a resample of releases.

        Args:
            (1) sample [List<Dict<string, float>>]: Per release, its shape values.

        Returns:
            (1) parameter [Dict<string, float>]: '<quantity>_p<q>' -> value.
        """

        return _Shape_Statistics({key: np.array([s[key] for s in sample]) for key in CONST_SHAPE_QUANTITIES})

    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 4]).spawn(CONST_SHAPE_THROWS)]
    throws = Draw_Throws(rngs, dataset, configuration, Flight_Model_Str=Flight_Model_Str)
    generated = Shape_Values(throws['release_state'], Flight_Model_Str=Flight_Model_Str)

    Step('coverage of the identification releases')
    coverage = Coverage(names, state, configuration, dataset, Flight_Model_Str=Flight_Model_Str)
    return Calibration_Str(Estimate=estimate, Spread=spread, Extremes=extremes, Offset_Bound=Derived_Bound(extremes['residual_max']),
                           White_Bound=Derived_Bound(extremes['step_max']),
                           Error_Real=error_real, Error_Bounds=error_bounds, Error_Generated=error_generated,
                           Error_Generated_Range=error_range,
                           Patterns=patterns, Releases=len(kept), Excluded=excluded, Release_Mean=releases.mean(axis=0),
                           Release_Std=releases.std(axis=0), Apex_Mean=float(real_values['apex'].mean()),
                           Apex_Std=float(real_values['apex'].std()), Shape_Real=_Shape_Statistics(real_values),
                           Shape_Bounds=Bootstrap_Bounds(shape_parts, Aggregate_Shape, seed=seed), Shape_Generated=_Shape_Statistics(generated),
                           Above_Share=float(np.mean(generated['apex'] > CONST_CAMERA_CEILING)),
                           Above_Real=int(np.sum(real_values['apex'] > CONST_CAMERA_CEILING)), Above_Real_Of=len(kept), Coverage=coverage)


def Summary_Rows(calibration: Calibration_Str) -> tp.List[tp.List[str]]:
    """
    Description:
        The rows of the observation summary (SPEC §9.9; the section 'bound' holds the residuals' largest values
        and the two derived bounds in the column 'real'): section, statistic, rate, real, low, high,
        generated, inside ('true' / 'false'; '' where no bound applies), generated_min, generated_max (the
        error statistics' range over the synthetic sets; '' elsewhere). Floats %.9g, SI units.

    Args:
        (1) calibration [Calibration_Str(object)]: The calibration.

    Returns:
        (1) parameter [List<List<string>>]: The rows, without the header.
    """

    def F(value: tp.Any) -> str:
        """
        Description:
            A float as %.9g, '' for None.

        Args:
            (1) value [Any]: The value.

        Returns:
            (1) parameter [string]: The text.
        """

        return '' if value is None else '%.9g' % value

    def Inside(value: float, bound: tp.Tuple[float, float]) -> str:
        """
        Description:
            Whether a value lies inside its bound, as text.

        Args:
            (1) value [float]: The value.
            (2) bound [Tuple<float, float>]: (low, high).

        Returns:
            (1) parameter [string]: 'true', 'false', or '' when either is NaN.
        """

        if not (np.isfinite(value) and np.isfinite(bound[0]) and np.isfinite(bound[1])):
            return ''
        return 'true' if bound[0] <= value <= bound[1] else 'false'

    rows = []
    estimate = calibration.Estimate
    for name in ('White_Std', 'White_Dof', 'Drift_Std', 'Drift_Time', 'Offset_Std', 'Offset_Time', 'Offset_Dof'):
        values = np.atleast_1d(getattr(estimate, name))
        for axis, value in enumerate(values):
            key = name if values.size == 1 else '%s_%s' % (name, 'xyz'[axis])
            low, high = calibration.Spread[key]
            rows.append(['parameter', key, '', F(value), F(low), F(high), '', '', '', ''])
    for key in CONST_EXTREMES:
        rows.append(['bound', key, '', F(calibration.Extremes.get(key)), '', '', '', '', '', ''])
    rows.append(['bound', 'Offset_Bound', '', F(calibration.Offset_Bound), '', '', '', '', '', ''])
    rows.append(['bound', 'White_Bound', '', F(calibration.White_Bound), '', '', '', '', '', ''])
    for key in CONST_ERROR_STATISTICS + CONST_ERROR_DIAGNOSTICS:
        section = 'error' if key in CONST_ERROR_STATISTICS else 'error_diagnostic'
        bound = calibration.Error_Bounds[key]
        spread = calibration.Error_Generated_Range.get(key, (None, None))
        rows.append([section, key, '', F(calibration.Error_Real[key]), F(bound[0]), F(bound[1]), F(calibration.Error_Generated[key]),
                     Inside(calibration.Error_Generated[key], bound), F(spread[0]), F(spread[1])])
    for rate, (real, bounds, generated, against) in calibration.Patterns.items():
        for key in CONST_PATTERN_STATISTICS:
            rows.append(['pattern', key, '%g/%g' % (rate, against), F(real[key]), F(bounds[key][0]), F(bounds[key][1]), F(generated[key]),
                         Inside(generated[key], bounds[key]), '', ''])
    for axis in range(3):
        rows.append(['release', 'Release_Mean_%s' % 'xyz'[axis], '', F(calibration.Release_Mean[axis]), '', '', '', '', '', ''])
        rows.append(['release', 'Release_Std_%s' % 'xyz'[axis], '', F(calibration.Release_Std[axis]), '', '', '', '', '', ''])
    rows.append(['release', 'Apex_Mean', '', F(calibration.Apex_Mean), '', '', '', '', '', ''])
    rows.append(['release', 'Apex_Std', '', F(calibration.Apex_Std), '', '', '', '', '', ''])
    rows.append(['release', 'releases', '', '%d' % calibration.Releases, '', '', '', '', '', ''])
    for name in calibration.Excluded:
        rows.append(['release', 'excluded (late start): %s' % name, '', '', '', '', '', '', '', ''])
    for key in calibration.Shape_Real:
        bound = calibration.Shape_Bounds[key]
        rows.append(['shape', key, '', F(calibration.Shape_Real[key]), F(bound[0]), F(bound[1]), F(calibration.Shape_Generated[key]),
                     Inside(calibration.Shape_Generated[key], bound), '', ''])
    rows.append(['shape', 'share_above_%gm' % CONST_CAMERA_CEILING, '', F(calibration.Above_Real / calibration.Above_Real_Of), '', '',
                 F(calibration.Above_Share), '', '', ''])
    for name, reason in calibration.Coverage:
        rows.append(['coverage', name, '', '', '', '', '', 'true' if reason == '' else 'false: %s' % reason, '', ''])
    return rows


def Write_Observation_Summary(path: str, calibration: Calibration_Str) -> None:
    """
    Description:
        Write the observation summary (SPEC §9.9): comma-separated, a header line, LF, no timestamp.

    Args:
        (1) path [string]: The file.
        (2) calibration [Calibration_Str(object)]: The calibration.
    """

    with open(path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['section', 'statistic', 'rate', 'real', 'low', 'high', 'generated', 'inside', 'generated_min', 'generated_max'])
        writer.writerows(Summary_Rows(calibration))


def Summary_Lines(calibration: Calibration_Str) -> tp.List[str]:
    """
    Description:
        The report of scripts/catch.py observation (SPEC §9.9), for stdout: the estimate and its spread, the
        error statistics, the pattern statistics per rate, the release distribution, the throw shape, and
        the coverage; lengths in [mm], times in [ms] or [s]; every value outside its bound marked OUTSIDE.

    Args:
        (1) calibration [Calibration_Str(object)]: The calibration.

    Returns:
        (1) parameter [List<string>]: The lines.
    """

    def Mark(value: float, bound: tp.Tuple[float, float]) -> str:
        """
        Description:
            '' inside its bound, 'OUTSIDE' outside, '-' without a verdict.

        Args:
            (1) value [float]: The value.
            (2) bound [Tuple<float, float>]: (low, high).

        Returns:
            (1) parameter [string]: The mark.
        """

        if not (np.isfinite(value) and np.isfinite(bound[0]) and np.isfinite(bound[1])):
            return '-'
        return '' if bound[0] <= value <= bound[1] else 'OUTSIDE'

    lines = ['The error model M4, estimated on the 45 identification flights (spread: min-max of %d re-estimates):' % CONST_SPREAD_REPEATS]
    for row in Summary_Rows(calibration):
        if row[0] == 'parameter':
            lines.append('  %-14s %-14s [%s, %s]' % (row[1], row[3], row[4], row[5]))
    if calibration.Extremes:
        lines.append('Largest values on the identification flights [mm]: residual %.1f, run mean %.1f, jump across a dropout %.1f, step '
                     'inside a run %.1f' % tuple(1e3 * calibration.Extremes[key] for key in CONST_EXTREMES))
        lines.append('Bounds derived from them (x %g, rounded up to %g mm): Offset_Bound %g m (the residual), White_Bound %g m (the step)'
                     % (CONST_BOUND_MARGIN, 1e3 * CONST_BOUND_STEP, calibration.Offset_Bound, calibration.White_Bound))
    lines.append('Error statistics [mm, share]: real, its bootstrap 95 %% bound, generated with the configured model (the median of %d '
                 'synthetic sets, their range in brackets):' % CONST_CHECK_SEEDS)
    for key in CONST_ERROR_STATISTICS + CONST_ERROR_DIAGNOSTICS:
        scale = 1.0 if key in ('offset_share', 'shuffled_offset_share', 'kurtosis_d2_y', 'share_above_20mm') else 1e3
        real, bound, generated = calibration.Error_Real[key], calibration.Error_Bounds[key], calibration.Error_Generated[key]
        low, high = calibration.Error_Generated_Range[key]
        lines.append('  %-22s %9.3f  [%8.3f, %8.3f]  %9.3f  [%8.3f, %8.3f]  %s' % (key, real * scale, bound[0] * scale, bound[1] * scale,
                                                                                generated * scale, low * scale, high * scale, Mark(generated, bound)))
    for rate, (real, bounds, generated, against) in calibration.Patterns.items():
        lines.append('Dropout patterns at %g Hz (composed) against the real %g Hz flights:%s'
                     % (rate, against, ' 370 Hz is an extrapolation (assumed).' if rate != against else ''))
        for key in CONST_PATTERN_STATISTICS:
            lines.append('  %-16s %8.3f  [%7.3f, %7.3f]  %8.3f  %s' % (key, real[key], bounds[key][0], bounds[key][1], generated[key],
                                                                     Mark(generated[key], bounds[key])))
    lines.append('Releases: %d (left out as late starts: %s); mean %s m, sd %s m; apex %.4f +- %.4f m'
                 % (calibration.Releases, ', '.join(calibration.Excluded) or 'none', np.round(calibration.Release_Mean, 4),
                    np.round(calibration.Release_Std, 4), calibration.Apex_Mean, calibration.Apex_Std))
    lines.append('Throw shape (%d generated throws) against the releases:' % CONST_SHAPE_THROWS)
    for key in calibration.Shape_Real:
        bound = calibration.Shape_Bounds[key]
        lines.append('  %-16s %8.4f  [%8.4f, %8.4f]  %8.4f  %s' % (key, calibration.Shape_Real[key], bound[0], bound[1],
                                                                 calibration.Shape_Generated[key], Mark(calibration.Shape_Generated[key], bound)))
    lines.append('  above z = %.2f m: %.1f %% of the generated throws, %d of %d real releases (%.1f %%)'
                 % (CONST_CAMERA_CEILING, 100.0 * calibration.Above_Share, calibration.Above_Real, calibration.Above_Real_Of,
                    100.0 * calibration.Above_Real / calibration.Above_Real_Of))
    inside = [name for name, reason in calibration.Coverage if reason == '']
    lines.append('Coverage (SPEC §2.13.6, identification flights): %d of %d inside the support of the generated throws'
                 % (len(inside), len(calibration.Coverage)))
    for name, reason in calibration.Coverage:
        if reason:
            lines.append('  outside: %s: %s' % (name, reason))
    return lines
