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
# File Name: ../Evaluation/Core.py
## =========================================================================== ##

# Time (the inference time)
import time
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the robot area the catch surface is derived from
from Configuration.Core import Configuration_Str
#   ../Flight/Core - the one flight model (the truth of a simulated flight)
from Flight.Core import Flight_Model_Str
#   ../Dataset/Core - a generated dataset
from Dataset.Core import Dataset_Str
#   ../Prediction/Core - the predictors' error and the trajectory a metric is computed on
from Prediction.Core import Prediction_Error_Cls, Trajectory_From_State, Trajectory_Str

"""
Description:
    Initialization of constants.

    The metrics of SPEC §7.12, each computed by one function, per flight and then as distributions
    (§2.13.11): the position error at fixed horizons (simulated flights, exact truth) and in horizon ranges at
    the sample times (simulated: exact truth and the observed positions; real: the recorded samples), the
    remainder RMS, the end mismatch, and on simulated flights only the catch-point and time-to-catch errors
    and the crossing category at the catch surface (§2.14) and the velocity's direction and speed errors at
    the fixed horizons and at the catch surface. Past a trajectory's predicted end every
    metric uses the state held at t_end. A predictor is a function of a prefix
    giving a trajectory (numerical or learned), so this module imports no predictor and no heavy runtime. Pure
    computation: no file, no printing.
"""
# The fixed horizons from t_0 (SPEC §7.12); 'end' is the truth's end.
#   Unit [Tuple<float>, s]
CONST_HORIZONS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
# The horizon ranges from t_0, [low, high), the last closed (SPEC §7.12).
#   Unit [Tuple<Tuple<float, float>>, s]
CONST_HORIZON_RANGES = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0))
# A horizon range with fewer flights lists its values and reports no statistics. *Assumed*.
#   Unit [int]
CONST_MINIMUM_FLIGHTS = 10
# The prefix-duration groups of the summary, [low, high) (F7 Part A).
#   Unit [Tuple<float>, s]
CONST_PREFIX_BINS = (0.0, 0.05, 0.1, 0.2, 0.4, np.inf)
# The bisection steps of a crossing time on its grid interval (1/600 s / 2^60).
#   Unit [int]
CONST_CROSSING_BISECTIONS = 60
# The crossing categories at the catch surface: both cross after t_0 (the
# errors computed); one does and the other does not; the truth crossed before t_0 (no metric); neither. And the
# outcome of a prediction the predictor refused, on any flight: no metric, its
# reason kept, counted; it never stops an evaluation.
#   Unit [Tuple<string>]
CONST_CROSSINGS = ('both', 'disagree', 'passed', 'neither')
CONST_REFUSED = 'refused'
# The key of the velocity errors at the catch surface, beside the fixed horizons' ('0.05' ... '0.8').
#   Unit [string]
CONST_CATCH = 'catch'


@dataclass
class Flight_Result_Str:
    """
    Description:
        The metrics of one flight and one N (SPEC §7.12, §9.7). Real flights carry no catch metric (SPEC §2.14):
        Crossing '' and NaN errors.
    """

    # The flight (a dataset index, or a recording's name), its partition ('validation', 'test' or 'real'), its
    # dataset (the dataset seed's label; '' for a real flight) and the training seed of the predictor ('' for
    # a numerical predictor, which has none).
    #   Unit [string]
    Flight: str = ''
    Partition: str = ''
    Dataset: str = ''
    Seed: str = ''
    # N, the prefix duration from the first sample to t_0, and the rate.
    #   Unit [int], [float, s], [float, Hz]
    N: int = 0
    Prefix_Duration: float = 0.0
    Rate: float = 0.0
    # The remainder RMS and the end mismatch (predicted minus true end).
    #   Unit [float, m], [float, s]
    Remainder_RMS: float = float('nan')
    End_Mismatch: float = float('nan')
    # The catch-point error, the time-to-catch error (predicted minus true) and the crossing category.
    #   Unit [float, m], [float, s], [string]
    Catch_Point_Error: float = float('nan')
    Time_To_Catch_Error: float = float('nan')
    Crossing: str = ''
    # The inference time: the prediction and its queries.
    #   Unit [float, s]
    Inference_Time: float = float('nan')
    # Why the predictor refused the prediction (Crossing CONST_REFUSED); '' when it predicted.
    #   Unit [string]
    Refusal: str = ''
    # The position error at each fixed horizon reached ('0.05' ... '0.8', 'end').
    #   Unit [Dict<string, float>, m]
    Fixed: tp.Dict[str, float] = field(default_factory=dict)
    # The velocity's errors at each fixed horizon reached and at the catch surface ('0.05' ... '0.8', 'catch';
    # simulated flights only): the angle between the predicted and the true velocity, and the predicted speed
    # minus the true one.
    #   Unit [Dict<string, float>, rad], [Dict<string, float>, m/s]
    Direction: tp.Dict[str, float] = field(default_factory=dict)
    Speed: tp.Dict[str, float] = field(default_factory=dict)
    # Per horizon range reached: (range index, the median error of its samples, the samples), against the
    # exact truth (simulated) or the recording (real), and against the observed positions (simulated only).
    #   Unit [List<Tuple<int, float, int>>, -, m, -]
    Ranges: tp.List[tp.Tuple[int, float, int]] = field(default_factory=list)
    Ranges_Observed: tp.List[tp.Tuple[int, float, int]] = field(default_factory=list)


def Catch_Height(configuration: Configuration_Str) -> float:
    """
    Description:
        The catch surface's height: the horizontal plane z = Areas.Robot centre + r/2 (0.8607 m, the transform's
        reference height, SPEC §7.3), derived, not configured.

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.

    Returns:
        (1) parameter [float]: The height [m].
    """

    return float(configuration.Areas.Robot.Centre[2] + 0.5 * configuration.Areas.Robot.Radius)


def Held_State(trajectory: tp.Any, t: np.ndarray) -> np.ndarray:
    """
    Description:
        The trajectory's state at each time, the state at t_end for times after it.

    Args:
        (1) trajectory [trajectory]: The trajectory (numerical or learned).
        (2) t [Vector<float> 1xM]: The times, each >= T_0 [s].

    Returns:
        (1) parameter [Matrix<float> Mx6]: The states [m, m/s].
    """

    return trajectory.State(np.minimum(np.asarray(t, dtype=np.float64).reshape(-1), trajectory.T_End))


def Velocity_Errors(predicted: np.ndarray, true: np.ndarray) -> tp.Tuple[float, float]:
    """
    Description:
        The errors of a predicted velocity (the cup is aligned with the ball's velocity, SPEC
        §2.9): the angle between the two velocities, and the predicted speed minus the true one.

    Args:
        (1) predicted [Vector<float> 1x3]: The predicted velocity [m/s].
        (2) true [Vector<float> 1x3]: The true velocity [m/s].

    Returns:
        (1) parameter 1 [float]: The direction error [rad], in [0, pi]; NaN when a speed is 0.
        (2) parameter 2 [float]: The speed error [m/s].
    """

    a, b = float(np.linalg.norm(predicted)), float(np.linalg.norm(true))
    if a == 0.0 or b == 0.0:
        return float('nan'), a - b
    return float(np.arccos(np.clip(np.dot(predicted, true) / (a * b), -1.0, 1.0))), a - b


def Crossing_Time(trajectory: tp.Any, height: float, after: float) -> tp.Optional[float]:
    """
    Description:
        The first time at or after `after` at which the trajectory descends through the height: its scan times
        in range (a numerical trajectory's grid states; a query-time trajectory's own scan) searched for
        z >= height > z next, then bisection on the trajectory's height.

    Args:
        (1) trajectory [Trajectory_Str(object)]: The trajectory.
        (2) height [float]: The height [m].
        (3) after [float]: The earliest time, clipped to [T_0, T_End] [s].

    Returns:
        (1) parameter [float]: The time [s], or None when it does not descend through the height in range.
    """

    low = min(max(after, trajectory.T_0), trajectory.T_End)
    t = trajectory.Scan_Times(low)
    z = trajectory.State(t)[:, 2]
    hit = np.where((z[:-1] >= height) & (z[1:] < height))[0]
    if not hit.size:
        return None
    a, b = t[hit[0]], t[hit[0] + 1]
    for _ in range(CONST_CROSSING_BISECTIONS):
        middle = 0.5 * (a + b)
        if trajectory.State([middle])[0, 2] >= height:
            a = middle
        else:
            b = middle
    return float(a)


def _Ranges(h: np.ndarray, error: np.ndarray) -> tp.List[tp.Tuple[int, float, int]]:
    """
    Description:
        One value per horizon range reached: the median of the errors of its samples (SPEC §7.12).

    Args:
        (1) h [Vector<float> 1xM]: The horizons of the samples from t_0 [s].
        (2) error [Vector<float> 1xM]: Their errors [m].

    Returns:
        (1) parameter [List<Tuple<int, float, int>>]: (range index, median error [m], samples).
    """

    out = []
    for r, (low, high) in enumerate(CONST_HORIZON_RANGES):
        last = r == len(CONST_HORIZON_RANGES) - 1
        inside = (h >= low) & ((h <= high) if last else (h < high))
        if inside.any():
            out.append((r, float(np.median(error[inside])), int(inside.sum())))
    return out


def Simulated_Result(trajectory: tp.Any, truth: Trajectory_Str, truth_end: float, frame_time: np.ndarray, frame_position: np.ndarray,
                     after_index: np.ndarray, after_position: np.ndarray, height: float) -> Flight_Result_Str:
    """
    Description:
        The metrics of a simulated flight (SPEC §7.12): the fixed horizons and the truth's end against the exact
        truth, with the velocity's direction and speed errors at the fixed horizons; the horizon ranges at the
        observed sample times after the prefix against the exact truth and the observed positions; the remainder
        RMS over every frame after t_0 to the floor; the end mismatch; the crossing of the catch surface
        (Crossing_Time of both trajectories), with the velocity's errors there (the predicted velocity at the
        predicted crossing against the true one at the true crossing). The caller fills the identity fields and
        the inference time.

    Args:
        (1) trajectory [Trajectory_Str(object)]: The prediction.
        (2) truth [Trajectory_Str(object)]: The exact truth (Prediction.Core.Trajectory_From_State of the release
                                            state from t = 0).
        (3) truth_end [float]: The truth's floor time (the dataset's) [s].
        (4) frame_time [Vector<float> 1xK]: The time of every frame from the release to the floor [s].
        (5) frame_position [Matrix<float> Kx3]: The true positions there [m].
        (6) after_index [Vector<int> 1xM]: The frames of the observed samples after the prefix.
        (7) after_position [Matrix<float> Mx3]: Their observed positions [m].
        (8) height [float]: The catch surface's height (Catch_Height) [m].

    Returns:
        (1) parameter [Flight_Result_Str(object)]: The metrics.
    """

    t_0 = trajectory.T_0
    result = Flight_Result_Str(End_Mismatch=trajectory.T_End - truth_end)
    for h in CONST_HORIZONS:
        if t_0 + h <= truth_end:
            predicted, true = Held_State(trajectory, [t_0 + h])[0], truth.State([min(t_0 + h, truth.T_End)])[0]
            result.Fixed['%g' % h] = float(np.linalg.norm(predicted[:3] - true[:3]))
            result.Direction['%g' % h], result.Speed['%g' % h] = Velocity_Errors(predicted[3:], true[3:])
    result.Fixed['end'] = float(np.linalg.norm(Held_State(trajectory, [truth_end])[0, :3] - truth.State([min(truth_end, truth.T_End)])[0, :3]))
    if len(after_index):
        t = frame_time[after_index]
        predicted = Held_State(trajectory, t)[:, :3]
        result.Ranges = _Ranges(t - t_0, np.linalg.norm(predicted - frame_position[after_index], axis=1))
        result.Ranges_Observed = _Ranges(t - t_0, np.linalg.norm(predicted - after_position, axis=1))
    later = frame_time > t_0
    if later.any():
        error = Held_State(trajectory, frame_time[later])[:, :3] - frame_position[later]
        result.Remainder_RMS = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
    true_crossing = Crossing_Time(truth, height, 0.0)
    if true_crossing is not None and true_crossing < t_0:
        result.Crossing = 'passed'
        return result
    predicted_crossing = Crossing_Time(trajectory, height, t_0)
    if true_crossing is None and predicted_crossing is None:
        result.Crossing = 'neither'
    elif true_crossing is None or predicted_crossing is None:
        result.Crossing = 'disagree'
    else:
        result.Crossing = 'both'
        predicted, true = trajectory.State([predicted_crossing])[0], truth.State([true_crossing])[0]
        result.Catch_Point_Error = float(np.linalg.norm(predicted[:3] - true[:3]))
        result.Time_To_Catch_Error = predicted_crossing - true_crossing
        result.Direction[CONST_CATCH], result.Speed[CONST_CATCH] = Velocity_Errors(predicted[3:], true[3:])
    return result


def Real_Result(trajectory: tp.Any, after_time: np.ndarray, after_position: np.ndarray) -> Flight_Result_Str:
    """
    Description:
        The metrics of a real flight (SPEC §2.14, §7.12): the horizon ranges and the remainder RMS against the
        recorded samples after the prefix, the end mismatch against the last recorded sample (a soft end); no
        fixed horizon, no catch metric, no velocity metric (a recording holds positions only). The caller fills
        the identity fields and the inference time.

    Args:
        (1) trajectory [Trajectory_Str(object)]: The prediction.
        (2) after_time [Vector<float> 1xM]: The recorded sample times after the prefix [s].
        (3) after_position [Matrix<float> Mx3]: Their positions [m].

    Returns:
        (1) parameter [Flight_Result_Str(object)]: The metrics.
    """

    result = Flight_Result_Str()
    if len(after_time):
        error = np.linalg.norm(Held_State(trajectory, after_time)[:, :3] - after_position, axis=1)
        result.Ranges = _Ranges(after_time - trajectory.T_0, error)
        result.Remainder_RMS = float(np.sqrt(np.mean(error ** 2)))
        result.End_Mismatch = trajectory.T_End - float(after_time[-1])
    return result


def Statistics(values: tp.Sequence[float]) -> tp.Tuple[int, float, float, float, float]:
    """
    Description:
        The count, mean, median, p90 (linear interpolation) and max of the finite values; NaN statistics when
        there is none.

    Args:
        (1) values [Sequence<float>]: The values.

    Returns:
        (1) parameter [Tuple<int, float, float, float, float>]: count, mean, median, p90, max.
    """

    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if not v.size:
        return 0, float('nan'), float('nan'), float('nan'), float('nan')
    return int(v.size), float(v.mean()), float(np.median(v)), float(np.percentile(v, 90)), float(v.max())


def _Metric_Values(results: tp.List[Flight_Result_Str]) -> tp.Dict[str, tp.List[float]]:
    """
    Description:
        Every summarised metric of a group of results, in the summary's units: '<metric>' -> per-flight values.

    Args:
        (1) results [List<Flight_Result_Str>]: The results.

    Returns:
        (1) parameter [Dict<string, List<float>>]: The values [mm], [ms] or [-].
    """

    metrics: tp.Dict[str, tp.List[float]] = {'remainder_rms_mm': [1e3 * r.Remainder_RMS for r in results],
                                             'end_mismatch_ms': [1e3 * r.End_Mismatch for r in results],
                                             'end_mismatch_abs_ms': [1e3 * abs(r.End_Mismatch) for r in results],
                                             'inference_ms': [1e3 * r.Inference_Time for r in results]}
    for h in [('%g' % h) for h in CONST_HORIZONS] + ['end']:
        values = [1e3 * r.Fixed[h] for r in results if h in r.Fixed]
        if values:
            metrics['fixed_%s_mm' % h] = values
    for h in [('%g' % h) for h in CONST_HORIZONS] + [CONST_CATCH]:
        if any(h in r.Direction for r in results):
            metrics['direction_%s_deg' % h] = [float(np.degrees(r.Direction[h])) for r in results if h in r.Direction]
            metrics['speed_%s_m_s' % h] = [r.Speed[h] for r in results if h in r.Speed]
            metrics['speed_abs_%s_m_s' % h] = [abs(r.Speed[h]) for r in results if h in r.Speed]
    for r, (low, high) in enumerate(CONST_HORIZON_RANGES):
        for name, attribute in (('range', 'Ranges'), ('range_observed', 'Ranges_Observed')):
            values = [1e3 * value for result in results for index, value, _ in getattr(result, attribute) if index == r]
            if values:
                metrics['%s_%g_%g_mm' % (name, low, high)] = values
    metrics['refused'] = [1.0 for r in results if r.Crossing == CONST_REFUSED]
    metrics['refused_share'] = [1.0 if r.Crossing == CONST_REFUSED else 0.0 for r in results]
    crossed = [r for r in results if r.Crossing in ('both', 'disagree')]
    if any(r.Crossing and r.Crossing != CONST_REFUSED for r in results):
        metrics['catch_point_mm'] = [1e3 * r.Catch_Point_Error for r in results if r.Crossing == 'both']
        metrics['time_to_catch_ms'] = [1e3 * r.Time_To_Catch_Error for r in results if r.Crossing == 'both']
        metrics['time_to_catch_abs_ms'] = [1e3 * abs(r.Time_To_Catch_Error) for r in results if r.Crossing == 'both']
        metrics['crossing_disagreement'] = [1.0 if r.Crossing == 'disagree' else 0.0 for r in crossed]
        for category in ('passed', 'neither'):
            metrics['crossing_%s' % category] = [1.0 if r.Crossing == category else 0.0 for r in results]
    return metrics


def Summary_Rows(results: tp.List[Flight_Result_Str]) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Description:
        The distributions of SPEC §2.13.11: per partition, dataset, training seed and N, over all flights, per rate
        and per prefix-duration group (CONST_PREFIX_BINS), each metric's count, mean, median, p90 and max; a
        horizon range with fewer than CONST_MINIMUM_FLIGHTS flights reports its count and no statistics (its
        values are in per_horizon.csv). Then the spread of each statistic: over the datasets of a partition with 2
        or more (per seed; rows with dataset 'spread_min', 'spread_max' and 'spread_sd', the sample sd), and
        over the training seeds of a dataset with 2 or more (rows with seed 'spread_min', 'spread_max' and
        'spread_sd'). A crossing share's mean is the rate (e.g. the crossing-disagreement rate among the flights
        where one of the two crosses). Refused predictions are counted by the metric 'refused'
        (its count) and 'refused_share' (its mean, over every result); they carry no other metric.

    Args:
        (1) results [List<Flight_Result_Str>]: Every flight's result.

    Returns:
        (1) parameter [List<Dict<string, Any>>]: The rows: partition, dataset, seed, n, group, value, metric,
                                                 count, mean, median, p90, max.
    """

    def Group_Value(result: Flight_Result_Str, group: str) -> str:
        """
        Description:
            The value of a result in a group: '' for all flights, the rate, or the prefix-duration bin.

        Args:
            (1) result [Flight_Result_Str(object)]: The result.
            (2) group [string]: 'all', 'rate' or 'prefix'.

        Returns:
            (1) parameter [string]: The group value.
        """

        if group == 'all':
            return ''
        if group == 'rate':
            return '%g' % result.Rate
        i = int(np.searchsorted(CONST_PREFIX_BINS, result.Prefix_Duration, side='right')) - 1
        return '[%g, %g)' % (CONST_PREFIX_BINS[i], CONST_PREFIX_BINS[i + 1])

    def Spread(rows: tp.List[tp.Dict[str, tp.Any]], over: str, kept: str) -> tp.List[tp.Dict[str, tp.Any]]:
        """
        Description:
            The spread rows of each statistic over one axis (the datasets or the seeds), per value of the other.

        Args:
            (1) rows [List<Dict<string, Any>>]: The rows of the single evaluations.
            (2) over [string]: The axis spread over, 'dataset' or 'seed'.
            (3) kept [string]: The other axis.

        Returns:
            (1) parameter [List<Dict<string, Any>>]: The rows, `over` holding 'spread_min', 'spread_max', 'spread_sd'.
        """

        groups: tp.Dict[tp.Tuple, tp.List[tp.Dict[str, tp.Any]]] = {}
        members: tp.Dict[tp.Tuple[str, str], tp.Set[str]] = {}
        for row in rows:
            members.setdefault((row['partition'], row[kept]), set()).add(row[over])
            groups.setdefault((row['partition'], row[kept], row['n'], row['group'], row['value'], row['metric']), []).append(row)
        out = []
        for (partition, other, n, group, value, metric), group_rows in groups.items():
            if len(members[(partition, other)]) < 2 or len(group_rows) < 2:
                continue
            for name, function in (('spread_min', np.min), ('spread_max', np.max), ('spread_sd', lambda a: np.std(a, ddof=1))):
                row = {'partition': partition, over: name, kept: other, 'n': n, 'group': group, 'value': value, 'metric': metric}
                for statistic in ('count', 'mean', 'median', 'p90', 'max'):
                    column = np.array([r[statistic] for r in group_rows], dtype=np.float64)
                    row[statistic] = float(function(column)) if np.all(np.isfinite(column)) else float('nan')
                out.append(row)
        return out

    rows = []
    keys: tp.Dict[tp.Tuple[str, str, str, int, str, str], tp.List[Flight_Result_Str]] = {}
    for result in results:
        for group in ('all', 'rate', 'prefix'):
            keys.setdefault((result.Partition, result.Dataset, result.Seed, result.N, group, Group_Value(result, group)), []).append(result)
    for key in sorted(keys, key=lambda k: (k[0], k[1], k[2], k[3], ('all', 'rate', 'prefix').index(k[4]), k[5])):
        partition, dataset, seed, n, group, value = key
        for metric, values in _Metric_Values(keys[key]).items():
            count, mean, median, p90, largest = Statistics(values)
            if metric.startswith('range') and count < CONST_MINIMUM_FLIGHTS:
                mean = median = p90 = largest = float('nan')
            rows.append({'partition': partition, 'dataset': dataset, 'seed': seed, 'n': n, 'group': group, 'value': value, 'metric': metric,
                         'count': count, 'mean': mean, 'median': median, 'p90': p90, 'max': largest})
    return rows + Spread(rows, 'dataset', 'seed') + Spread(rows, 'seed', 'dataset')


def _Refused(error: Prediction_Error_Cls) -> Flight_Result_Str:
    """
    Description:
        The result of a refused prediction: the outcome CONST_REFUSED with the
        predictor's reason, every metric empty; the caller fills the identity fields.

    Args:
        (1) error [Prediction_Error_Cls(object)]: The predictor's refusal.

    Returns:
        (1) parameter [Flight_Result_Str(object)]: The result.
    """

    return Flight_Result_Str(Crossing=CONST_REFUSED, Refusal=str(error))


def _Timed_Prediction(predictor: tp.Callable[[np.ndarray, np.ndarray, float], tp.Any], time_: np.ndarray, position: np.ndarray, rate: float,
                      queries: np.ndarray) -> tp.Tuple[tp.Any, float, np.ndarray]:
    """
    Description:
        A prediction and its queries (the held states at the query times), timed together: the inference time
        of SPEC §7.12, on the CPU this runs on.

    Args:
        (1) predictor [Callable[[Vector, Matrix, float], trajectory]]: The predictor: a prefix's times [s],
                                                                       positions [m] and rate [Hz] -> its trajectory.
        (2) time_ [Vector<float> 1xN]: The prefix times [s].
        (3) position [Matrix<float> Nx3]: The prefix positions [m].
        (4) rate [float]: The rate [Hz].
        (5) queries [Vector<float> 1xQ]: The query times, each >= the last prefix time [s].

    Returns:
        (1) parameter 1 [trajectory]: The trajectory.
        (2) parameter 2 [float]: The inference time [s].
        (3) parameter 3 [Matrix<float> Qx6]: The held states at the queries [m, m/s].
    """

    start = time.perf_counter()
    trajectory = predictor(time_, position, rate)
    states = Held_State(trajectory, queries) if len(queries) else np.zeros((0, 6))
    return trajectory, time.perf_counter() - start, states


def Evaluate_Simulated(data: Dataset_Str, indices: np.ndarray,
                       predictors: tp.List[tp.Tuple[str, tp.Callable[[np.ndarray, np.ndarray, float], tp.Any]]],
                       n_values: tp.List[int], partition: str, dataset: str, height: float, *,
                       Flight_Model_Str: Flight_Model_Str,
                       progress: tp.Optional[tp.Callable[[int, int], None]] = None) -> tp.Tuple[tp.List[Flight_Result_Str], tp.List[tp.Any]]:
    """
    Description:
        Predict and score the given flights of a dataset for every N and every predictor (SPEC §7.12): the prefix
        is the first N observed samples (a flight with fewer is left out of that N); the truth is the release
        state's trajectory under the flight model and the stored frames. Reads only the flights given.

    Args:
        (1) data [Dataset_Str(object)]: The dataset.
        (2) indices [Vector<int> 1xP]: The flights (one partition, Dataset.Core.Partition_Indices).
        (3) predictors [List<Tuple<string, Callable>>]: Per predictor its training seed's label ('' for a numerical
                                                        predictor) and the function of a prefix (times [s],
                                                        positions [m], rate [Hz]) giving its trajectory.
        (4) n_values [List<int>]: The values of N.
        (5) partition [string]: The partition's name, written with each result.
        (6) dataset [string]: The dataset's label, written with each result.
        (7) height [float]: The catch surface's height [m].
        (8) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the dataset's flight model (the truth).
        (9) progress [Callable[[int, int], None]]: Called with (flights done, flights) every 100 flights, or None.

    Returns:
        (1) parameter 1 [List<Flight_Result_Str>]: The results, by flight, then N, then predictor.
        (2) parameter 2 [List<Tuple>]: Per result (trajectory, query times, query states), for predictions.npz; None for a
                                       refused prediction (Crossing CONST_REFUSED, its reason in Refusal: never dropped,
                                       never stopping the evaluation).
    """

    results, predictions = [], []
    for done, i in enumerate(indices):
        i = int(i)
        truth = Trajectory_From_State(data.Release_State[i], 0.0, 0.0, Flight_Model_Str=Flight_Model_Str)
        frame_time, frame_position = data.Truth(i)
        observed_time, observed_position = data.Observations(i)
        frames = data.Observed_Frame[data.Observed_Offset[i]:data.Observed_Offset[i + 1]]
        for n in n_values:
            if len(observed_time) < n:
                continue
            queries = observed_time[n:]
            for seed, predictor in predictors:
                try:
                    trajectory, inference, states = _Timed_Prediction(predictor, observed_time[:n], observed_position[:n], float(data.Rate[i]),
                                                                      queries)
                    result = Simulated_Result(trajectory, truth, float(data.Floor_Time[i]), frame_time, frame_position, frames[n:],
                                              observed_position[n:], height)
                    result.Inference_Time = inference
                    predictions.append((trajectory, queries, states))
                except Prediction_Error_Cls as error:
                    result = _Refused(error)
                    predictions.append(None)
                result.Flight, result.Partition, result.Dataset, result.Seed, result.N = '%d' % i, partition, dataset, seed, n
                result.Prefix_Duration = float(observed_time[n - 1] - observed_time[0])
                result.Rate = float(data.Rate[i])
                results.append(result)
        if progress is not None and (done + 1) % 100 == 0:
            progress(done + 1, len(indices))
    return results, predictions


def Evaluate_Real(flights: tp.List[tp.Tuple[str, float, np.ndarray, np.ndarray]],
                  predictors: tp.List[tp.Tuple[str, tp.Callable[[np.ndarray, np.ndarray, float], tp.Any]]],
                  n_values: tp.List[int]) -> tp.Tuple[tp.List[Flight_Result_Str], tp.List[tp.Any]]:
    """
    Description:
        Predict and score real flights for every N and every predictor (SPEC §2.14, §7.12): the prefix is the first
        N recorded samples (the caller has trimmed the first 40 ms and put the flight in the robot frame), the
        truth the recorded samples after it; a flight with fewer than N + 1 samples is left out of that N.

    Args:
        (1) flights [List<Tuple<string, float, Vector, Matrix>>]: Per flight its name, rate [Hz], times from its
                                                                  first kept sample [s] and positions [m].
        (2) predictors [List<Tuple<string, Callable>>]: Per predictor its training seed's label and the function
                                                        of a prefix giving its trajectory (Evaluate_Simulated).
        (3) n_values [List<int>]: The values of N.

    Returns:
        (1) parameter 1 [List<Flight_Result_Str>]: The results, by flight, then N, then predictor.
        (2) parameter 2 [List<Tuple>]: Per result (trajectory, query times, query states), for predictions.npz; None for a
                                       refused prediction (Crossing CONST_REFUSED, its reason in Refusal: never dropped,
                                       never stopping the evaluation).
    """

    results, predictions = [], []
    for flight, rate, t, p in flights:
        for n in n_values:
            if len(t) < n + 1:
                continue
            for seed, predictor in predictors:
                try:
                    trajectory, inference, states = _Timed_Prediction(predictor, t[:n], p[:n], rate, t[n:])
                    result = Real_Result(trajectory, t[n:], p[n:])
                    result.Inference_Time = inference
                    predictions.append((trajectory, t[n:], states))
                except Prediction_Error_Cls as error:
                    result = _Refused(error)
                    predictions.append(None)
                result.Flight, result.Partition, result.Seed, result.N, result.Rate = flight, 'real', seed, n, rate
                result.Prefix_Duration = float(t[n - 1] - t[0])
                results.append(result)
    return results, predictions
