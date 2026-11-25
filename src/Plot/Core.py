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
# File Name: ../Plot/Core.py
## =========================================================================== ##

# OS (Operating system interfaces)
import os
# CSV (the captions' file)
import csv
# Dataclasses (Data classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation; the cup's mouth at a recorded pose) [pip3 install mujoco]
import mujoco
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Dataset/Core, Manifest - a dataset, its partitions, its manifest
from Dataset.Core import Read_Dataset, Dataset_Str, CONST_PARTITION_VALIDATION
from Dataset.Manifest import Read_Manifest
#   ../Learning/Core - a training seed's epochs
from Learning.Core import Read_Epochs
#   ../Robot/Core - the cup's mouth at a joint vector
from Robot.Core import Robot_Str, Forward_Kinematics
#   ../Catch/Decision, Interception, Physical - the stored results of catch, read back
from Catch.Decision import Read_Summary as Read_Decision_Summary
from Catch.Interception import Read_Summary as Read_Interception_Summary, Read_Per_Throw as Read_Interception_Rows
from Catch.Interception import CONST_PER_THROW_COLUMNS as CONST_INTERCEPTION_COLUMNS, CONST_MOVING_COLUMNS as CONST_INTERCEPTION_MOVING_COLUMNS
from Catch.Physical import (Read_Summary as Read_Catch_Summary, CONST_PER_THROW_COLUMNS as CONST_CATCH_COLUMNS,
                            CONST_MOVING_COLUMNS as CONST_CATCH_MOVING_COLUMNS)
#   ../Evaluation/Core, Report - a trajectory's state; the stored evaluations' and comparisons' files
from Evaluation.Core import Held_State
from Evaluation.Report import (Read_Comparison_Table, Read_Summary as Read_Evaluation_Summary, Read_Predictions, Read_Per_Flight,
                               CONST_COMPARISON_SUMMARY_COLUMNS, CONST_COMPARISON_PAIRED_COLUMNS)
#   ../View/Core - a throw's partition from the dataset's sizes; one recorded throw
from View.Core import Partition_Code, Throw_Record_Str

"""
Description:
    Initialization of constants.

    The figures' data (the visualisation, delivery 2; SPEC §7.14): what each figure of scripts/plots.py shows, built from
    the stored files the commands of scripts/catch.py write, by those files' own readers, and from the predictions and
    the recorded throws the script computes. Every plotted number comes from here: none is typed into a figure. A test
    evaluation's per-flight files (per_flight.csv, per_horizon.csv, predictions.npz) are never opened: the accuracy on
    the simulated test partitions and on the 45 real validation flights comes from compare's tables (Final_Test,
    Final_Validation). No matplotlib: the drawing is Plot.Figures'.
"""
# The comparison's sets by their keys, in the order a figure shows them (Plot.Figures, Plot.Style): the simulated validation,
# the simulated held-out (test) partition and the real validation flights; the script names each one's table and rows.
#   Unit [Tuple<string>]
CONST_COMPARISON_SETS = ('validation', 'held_out', 'real')
# The sets the speed-up is taken over: the simulated test partition and the real flights (the comparison's bars).
#   Unit [Tuple<string>]
CONST_SPEEDUP_SETS = ('held_out', 'real')
# The catch's classes in the order a stacked bar shows them (summary.csv's count columns; SPEC §9.14).
#   Unit [Tuple<string>]
CONST_CLASS_COLUMNS = ('caught', 'bounced_out', 'rim', 'hit_robot', 'missed', 'hit_stand', 'late', 'no_go_catchable', 'no_go')
# The share at which the restitution's cliff is read: where the retained share of the goes crosses one half.
#   Unit [float]
CONST_CLIFF_SHARE = 0.5
# The largest gap between a throw's observed samples the prediction figure's rule accepts: a stored prediction exists at
# the observed times only, so a longer dropout would be drawn as a straight segment (the review of 2026-11-24; 109 of the
# 1 500 validation throws of Default qualify).
#   Unit [float, s]
CONST_MAX_GAP = 0.1
# The step the motion figure's states are taken at: the scene's control period (Catch.Physical: 8 steps of 0.00025 s).
#   Unit [float, s]
CONST_MOTION_STEP = 0.002
# The columns of captions.csv.
#   Unit [Tuple<string>]
CONST_CAPTION_COLUMNS = ('file', 'caption')


class Plot_Error_Cls(ValueError):
    """
    Description:
        A figure's data cannot be built: a stored file is missing or does not hold what the figure shows. The message
        names the file and what is missing; the script prints it verbatim as an [ERROR] line.
    """


@dataclass
class Comparison_Str:
    """
    Description:
        The comparison of the two predictors (Load_Comparison): the median remainder RMS of each set, the paired
        differences, the inference times.
    """

    # The values of N.
    #   Unit [Vector<int> 1xK]
    N: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # Per (method, set) the value, the lowest and the highest over the training seeds, per N.
    #   Unit [Dict<Tuple<string, string>, Matrix<float> 3xK, mm>]
    Accuracy: tp.Dict[tp.Tuple[str, str], np.ndarray] = field(default_factory=dict)
    # Per set the learned predictor's median paired difference against the numerical one and its 95 % interval, per N.
    #   Unit [Dict<string, Matrix<float> 3xK, mm>]
    Paired: tp.Dict[str, np.ndarray] = field(default_factory=dict)
    # Per (method, set) the median and the 90th percentile of the inference time, per N.
    #   Unit [Dict<Tuple<string, string>, Matrix<float> 2xK, ms>]
    Inference: tp.Dict[tp.Tuple[str, str], np.ndarray] = field(default_factory=dict)
    # The number of flights of each set (at the smallest N).
    #   Unit [Dict<string, int>]
    Flights: tp.Dict[str, int] = field(default_factory=dict)
    # How many times faster the learned predictor predicts: the geometric mean, over the N values and the sets of
    # CONST_SPEEDUP_SETS, of the numerical predictor's median inference time over the learned one's.
    #   Unit [float, -]
    Speedup: float = float('nan')


@dataclass
class Horizon_Str:
    """
    Description:
        The position error at the fixed horizons of the simulated validation (Load_Horizons).
    """

    # The horizons and the values of N.
    #   Unit [Vector<float> 1xH, s], [Vector<int> 1xK]
    Horizon: np.ndarray = field(default_factory=lambda: np.zeros(0))
    N: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # Per (method, N) the median over the training seeds of each seed's median error, and its lowest and highest, per horizon.
    #   Unit [Dict<Tuple<string, int>, Matrix<float> 3xH, mm>]
    Error: tp.Dict[tp.Tuple[str, int], np.ndarray] = field(default_factory=dict)


@dataclass
class Throw_Str:
    """
    Description:
        One simulated validation throw with the stored predictions of both predictors (Load_Throw).
    """

    # The dataset's label and the throw's index.
    #   Unit [string], [int]
    Dataset: str = ''
    Flight: int = 0
    # The true path at every frame and the observed samples.
    #   Unit [Vector<float> 1xT, s], [Matrix<float> Tx3, m], [Vector<float> 1xM, s], [Matrix<float> Mx3, m]
    Truth_Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Truth_Position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    Observed_Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Observed_Position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # Per (method, N) the stored prediction's query times and predicted positions, and the true position there.
    #   Unit [Dict<Tuple<string, int>, Tuple<Vector, Matrix, Matrix>>: s, m, m]
    Predicted: tp.Dict[tp.Tuple[str, int], tp.Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass
class Real_Str:
    """
    Description:
        One real identification flight with both predictors' predictions (Real_Predictions), in the robot frame by the
        assumed transform.
    """

    # The flight's name, its rate, its recorded samples (the first 40 ms trimmed) and the N of the paths shown.
    #   Unit [string], [float, Hz], [Vector<float> 1xM, s], [Matrix<float> Mx3, m], [int]
    Name: str = ''
    Rate: float = 0.0
    Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    Shown_N: int = 0
    # Per method the predicted path from the first Shown_N samples to its floor time: times and positions.
    #   Unit [Dict<string, Tuple<Vector, Matrix>>: s, m]
    Path: tp.Dict[str, tp.Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    # Per (method, N) the horizon of every recorded sample after the prefix and the position error there.
    #   Unit [Dict<Tuple<string, int>, Tuple<Vector, Vector>>: s, mm]
    Error: tp.Dict[tp.Tuple[str, int], tp.Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass
class Share_Str:
    """
    Description:
        The retained share of the goes against the restitution (Load_Restitution), per mode, dataset and acceleration.
    """

    # Per (mode, dataset, acceleration) the restitution of each contact, the retained share of the goes and its 95 %
    # Clopper-Pearson interval, sorted by the restitution.
    #   Unit [Dict<Tuple<string, string, int>, Matrix<float> 4xL>: -]
    Share: tp.Dict[tp.Tuple[str, str, int], np.ndarray] = field(default_factory=dict)
    # Per (mode, dataset, acceleration) the restitution where the share crosses CONST_CLIFF_SHARE (NaN if it does not).
    #   Unit [Dict<Tuple<string, string, int>, float>: -]
    Cliff: tp.Dict[tp.Tuple[str, str, int], float] = field(default_factory=dict)
    # Per mode the contacts its stored catch was simulated at (time constant, damping ratio) and their restitutions.
    #   Unit [Dict<string, List<Tuple<float, float, float>>>: s, -, -]
    Contacts: tp.Dict[str, tp.List[tp.Tuple[float, float, float]]] = field(default_factory=dict)


@dataclass
class Motion_Str:
    """
    Description:
        One throw caught in motion and the static catch of the same throw (Motion_From_Records), against the time from the
        release.
    """

    # The times; per mode ('moving', 'static') the cup's speed along the ball's direction at its first cup contact, the
    # ball's speed and the distance from the cup's mouth to the ball's centre.
    #   Unit [Dict<string, Vector<float> 1xS>: s], [Dict<string, Vector<float> 1xS>: m/s, m/s, m]
    Time: tp.Dict[str, np.ndarray] = field(default_factory=dict)
    Cup_Speed: tp.Dict[str, np.ndarray] = field(default_factory=dict)
    Ball_Speed: tp.Dict[str, np.ndarray] = field(default_factory=dict)
    Distance: tp.Dict[str, np.ndarray] = field(default_factory=dict)
    # The moving catch's phases: (name, start, end) from the release.
    #   Unit [List<Tuple<string, float, float>>: s]
    Phases: tp.List[tp.Tuple[str, float, float]] = field(default_factory=list)
    # The moving catch's first cup contact and the relative speed there (its catch row), the part of the cup it meets
    # first (its catch row: cup_bottom, cup_wall or cup_lip), and its class.
    #   Unit [float, s], [float, m/s], [string], [string]
    Contact: float = float('nan')
    Relative_Speed: float = float('nan')
    Contact_Part: str = ''
    Class: str = ''
    # Per mode at its own first cup contact: the time of the last state before it, and the ball's speed, the cup's speed
    # along the ball's direction and their relative speed by the backward difference over the two states before it (one
    # simulation step, 0.00025 s), from the record's states.
    #   Unit [Dict<string, Tuple<float, float, float, float>>: s, m/s, m/s, m/s]
    At_Contact: tp.Dict[str, tp.Tuple[float, float, float, float]] = field(default_factory=dict)
    # The time the ball's centre passes nearest the moving cup's mouth before the contact.
    #   Unit [float, s]
    Mouth_Pass: float = float('nan')
    # The moving catch's joints at each time.
    #   Unit [Matrix<float> Sx6, rad], [Matrix<float> Sx6, rad/s]
    Joints: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    Joint_Speeds: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))


def _Rows(path: str, reader: tp.Callable[[str], tp.List[tp.Dict[str, str]]]) -> tp.List[tp.Dict[str, str]]:
    """
    Description:
        A stored file's rows by its own reader; a missing file is a Plot_Error_Cls naming it.

    Args:
        (1) path [string]: The file.
        (2) reader [Callable[[string], List<Dict<string, string>>]]: Its reader.

    Returns:
        (1) parameter [List<Dict<string, string>>]: The rows.
    """

    if not os.path.isfile(path):
        raise Plot_Error_Cls('%s does not exist; the figure is drawn from it. Run the command that writes it first (SPEC §9).' % path)
    return reader(path)


def _Number(text: str) -> float:
    """
    Description:
        A stored number, NaN for an empty field.

    Args:
        (1) text [string]: The field.

    Returns:
        (1) parameter [float]: The number.
    """

    return float(text) if text != '' else float('nan')


def Training_Curves(run: str) -> tp.Dict[int, np.ndarray]:
    """
    Description:
        The learning curves of a learned run: every training seed's epochs.csv (seed_<s>/), by seed.

    Args:
        (1) run [string]: The run folder.

    Returns:
        (1) parameter [Dict<int, Matrix<float> Ex4>]: Per seed its epochs (Learning.Core.Read_Epochs).
    """

    seeds = sorted(int(name[5:]) for name in os.listdir(run) if name.startswith('seed_') and name[5:].isdigit()) if os.path.isdir(run) else []
    if not seeds:
        raise Plot_Error_Cls('%s holds no seed_<s>/epochs.csv; the learning curves are those of a learned run written by train (SPEC §9.6).' % run)
    return {seed: Read_Epochs(os.path.join(run, 'seed_%d' % seed, 'epochs.csv')) for seed in seeds}


def Load_Comparison(comparisons: str, methods: tp.Dict[str, str], sets: tp.Dict[str, tp.Tuple[str, str, str]]) -> Comparison_Str:
    """
    Description:
        The two predictors compared, from compare's tables only (the sets of CONST_COMPARISON_SETS, each from the table,
        partition and dataset the caller names; the support 'all'): the median
        remainder RMS with its range over the seeds, the learned run's paired difference against the numerical run with
        its bootstrap interval, the inference time's median and 90th percentile.

    Args:
        (1) comparisons [string]: The comparisons' parent folder (Data/Generated/Comparisons).
        (2) methods [Dict<string, string>]: Per method (Model.Name) its run's name (Run.Name), the learned predictor first; the
                                            paired difference is the first's against the second's.
        (3) sets [Dict<string, Tuple<string, string, string>>]: Per set of CONST_COMPARISON_SETS its comparison's name, the
                                                                partition and the dataset of its rows ('' for the real flights).

    Returns:
        (1) parameter [Comparison_Str(object)]: The comparison, by method.
    """

    out = Comparison_Str()
    (learned, learned_run), (_, numerical_run) = list(methods.items())[:2]
    tables: tp.Dict[str, tp.Tuple[tp.List[tp.Dict[str, str]], tp.List[tp.Dict[str, str]]]] = {}
    missing = [key for key in CONST_COMPARISON_SETS if key not in sets]
    if missing:
        raise Plot_Error_Cls('The comparison names no table for the sets %s.' % ', '.join(missing))
    for key in CONST_COMPARISON_SETS:
        name = sets[key][0]
        if name not in tables:
            folder = os.path.join(comparisons, name)
            tables[name] = (_Rows(os.path.join(folder, 'summary.csv'), lambda p: Read_Comparison_Table(p, CONST_COMPARISON_SUMMARY_COLUMNS)),
                            _Rows(os.path.join(folder, 'paired.csv'), lambda p: Read_Comparison_Table(p, CONST_COMPARISON_PAIRED_COLUMNS)))
    n_values = sorted({int(row['n']) for summary, _ in tables.values() for row in summary})
    out.N = np.array(n_values, dtype=np.int64)
    for key in CONST_COMPARISON_SETS:
        name, partition, dataset = sets[key]
        summary, paired = tables[name]
        for method, run in methods.items():
            rows = {(int(r['n']), r['metric']): r for r in summary
                    if r['run'] == run and r['partition'] == partition and r['dataset'] == dataset and r['support'] == 'all'}
            missing = [n for n in n_values if (n, 'remainder_rms_median_mm') not in rows]
            if missing:
                raise Plot_Error_Cls('%s/summary.csv holds no remainder_rms_median_mm of %s on %s %s at N %s.'
                                     % (os.path.join(comparisons, name), run, partition, dataset or '(real)', missing))
            values = [rows[(n, 'remainder_rms_median_mm')] for n in n_values]
            out.Accuracy[(method, key)] = np.array([[_Number(r['value']) for r in values], [_Number(r['seed_min']) for r in values],
                                                    [_Number(r['seed_max']) for r in values]])
            out.Inference[(method, key)] = np.array([[_Number(rows[(n, 'inference_median_ms')]['value']) for n in n_values],
                                                     [_Number(rows[(n, 'inference_p90_ms')]['value']) for n in n_values]])
            out.Flights[key] = int(values[0]['flights'])
        rows = {int(r['n']): r for r in paired if r['run'] == learned_run and r['against'] == numerical_run and r['partition'] == partition
                and r['dataset'] == dataset and r['support'] == 'all'}
        if sorted(rows) != n_values:
            raise Plot_Error_Cls('%s/paired.csv holds no difference of %s against %s on %s %s at every N.'
                                 % (os.path.join(comparisons, name), learned_run, numerical_run, partition, dataset or '(real)'))
        out.Paired[key] = np.array([[_Number(rows[n][column]) for n in n_values] for column in ('median_mm', 'low_mm', 'high_mm')])
    learned_name, numerical_name = list(methods)[:2]
    ratios = [out.Inference[(numerical_name, key)][0] / out.Inference[(learned_name, key)][0] for key in CONST_SPEEDUP_SETS]
    out.Speedup = float(np.exp(np.mean(np.log(np.concatenate(ratios)))))
    return out


def Load_Horizon_Ranges(evaluations: tp.Dict[str, str], sets: tp.Dict[str, tp.Tuple[str, str]], n: int) -> tp.Dict[tp.Tuple[str, str], np.ndarray]:
    """
    Description:
        The position error against the prediction horizon from the stored summaries of the final (test) evaluations of
        F14 (summary.csv; never a per-flight file): per set and method the horizon ranges of SPEC §7.12 (the metrics
        range_<a>_<b>_mm, against the exact truth for simulated flights and the recorded samples for real ones) at one N,
        each range's median and 90th percentile (the median over the training seeds of each seed's value; the summaries
        hold no quartiles) and its count. A range without a median (fewer than 10 flights) is left out.

    Args:
        (1) evaluations [Dict<string, string>]: Per method its test evaluation's folder.
        (2) sets [Dict<string, Tuple<string, string>>]: Per set key the partition and the dataset of its rows.
        (3) n [int]: The N.

    Returns:
        (1) parameter [Dict<Tuple<string, string>, Matrix<float> 5xR>]: Per (set, method) the ranges' lower and upper
                                                                       bounds [s], the median and the 90th percentile
                                                                       [mm] and the count, sorted by the range.
    """

    out = {}
    for method, folder in evaluations.items():
        rows = _Rows(os.path.join(folder, 'summary.csv'), Read_Evaluation_Summary)
        for key, (partition, dataset) in sets.items():
            values: tp.Dict[tp.Tuple[float, float], tp.List[tp.Tuple[float, float, float]]] = {}
            for row in rows:
                metric = row['metric']
                if row['partition'] != partition or row['dataset'] != dataset or int(row['n']) != n or row['group'] != 'all' \
                        or not (row['seed'] == '' or row['seed'].isdigit()) or not metric.startswith('range_') \
                        or metric.startswith('range_observed_') or row['median'] == '':
                    continue
                low, high = (float(v) for v in metric[len('range_'):-len('_mm')].split('_'))
                values.setdefault((low, high), []).append((float(row['median']), float(row['p90']), float(row['count'])))
            if not values:
                raise Plot_Error_Cls('%s/summary.csv holds no horizon range of %s %s at N %d.' % (folder, partition, dataset or '(real)', n))
            out[(key, method)] = np.array([[a, b, float(np.median([v[0] for v in values[(a, b)]])), float(np.median([v[1] for v in values[(a, b)]])),
                                            float(min(v[2] for v in values[(a, b)]))] for a, b in sorted(values)]).T
    return out


def Throw_Errors(evaluations: tp.Dict[str, tp.Tuple[str, str]], dataset: str, data: Dataset_Str
                 ) -> tp.Tuple[tp.Dict[str, tp.Dict[tp.Tuple[int, int], float]], tp.Dict[str, tp.Dict[tp.Tuple[int, int], float]], tp.Dict[int,
                                                                                                                                           float]]:
    """
    Description:
        What the prediction figure's rule chooses its throw by, from the validation evaluations: per method (its seed shown)
        and (flight, N) the remainder RMS (per_flight.csv) and the error at the stored prediction's last query, the end of
        the figure's curve (predictions.npz against the dataset's truth at that time) [mm]; per flight the largest gap
        between its observed samples [s].

    Args:
        (1) evaluations [Dict<string, Tuple<string, string>>]: Per method its validation evaluation's folder and the seed's
                                                               label shown.
        (2) dataset [string]: The dataset's label.
        (3) data [Dataset_Str(object)]: The dataset (its validation flights' truth is read).

    Returns:
        (1) parameter 1 [Dict<string, Dict<Tuple<int, int>, float>>]: The remainder RMS.
        (2) parameter 2 [Dict<string, Dict<Tuple<int, int>, float>>]: The error at the last query.
        (3) parameter 3 [Dict<int, float>]: The largest gap per flight.
    """

    rms: tp.Dict[str, tp.Dict[tp.Tuple[int, int], float]] = {}
    end: tp.Dict[str, tp.Dict[tp.Tuple[int, int], float]] = {}
    for method, (folder, seed) in evaluations.items():
        rms[method] = {(int(r['flight']), int(r['n'])): float(r['remainder_rms_mm'])
                       for r in _Rows(os.path.join(folder, 'per_flight.csv'), Read_Per_Flight)
                       if r['partition'] == 'validation' and r['dataset'] == dataset and r['seed'] == seed and r['remainder_rms_mm'] != ''}
        end[method] = {}
        for r in _Rows(os.path.join(folder, 'predictions.npz'), lambda path: Read_Predictions(path, dataset=dataset)):
            if r.Seed != seed or r.Partition != 'validation' or not len(r.Query_Time):
                continue
            t, x = data.Truth(int(r.Flight))
            truth = np.array([np.interp(r.Query_Time[-1], t, x[:, k]) for k in range(3)])
            end[method][(int(r.Flight), r.N)] = float(1e3 * np.linalg.norm(r.Query_State[-1, :3] - truth))
    gaps = {}
    for flight in {f for per in end.values() for f, _ in per}:
        t, _ = data.Observations(flight)
        gaps[flight] = float(np.max(np.diff(t))) if len(t) > 1 else float('inf')
    return rms, end, gaps


def Choose_Throw(rms: tp.Dict[str, tp.Dict[tp.Tuple[int, int], float]], end: tp.Dict[str, tp.Dict[tp.Tuple[int, int], float]],
                 gaps: tp.Dict[int, float], medians: tp.Dict[str, tp.Dict[int, float]], n_values: tp.Sequence[int], dataset: str
                 ) -> tp.Tuple[int, float]:
    """
    Description:
        The simulated validation throw the prediction figure shows, by a stated rule: among the throws without a gap between
        their observed samples longer than CONST_MAX_GAP and whose remainder RMS and whose error at the end of the figure's
        curve (Throw_Errors) both fall from each N to the next for every method, the one nearest the simulated test
        medians: the smallest sum over the methods and the N values of |ln(remainder RMS / the test median)|; a tie to
        the smaller index.

    Args:
        (1) rms [Dict<string, Dict<Tuple<int, int>, float>>]: Per method and (flight, N) the remainder RMS [mm].
        (2) end [Dict<string, Dict<Tuple<int, int>, float>>]: Per method and (flight, N) the error at the curve's end [mm].
        (3) gaps [Dict<int, float>]: Per flight the largest gap between its observed samples [s].
        (4) medians [Dict<string, Dict<int, float>>]: Per method the simulated test median of the remainder RMS per N [mm].
        (5) n_values [Sequence<int>]: The N values, in increasing order.
        (6) dataset [string]: The dataset's label, for the message.

    Returns:
        (1) parameter 1 [int]: The throw's index.
        (2) parameter 2 [float]: Its score.
    """

    best: tp.Optional[tp.Tuple[float, int]] = None
    for flight in sorted({flight for flight, _ in next(iter(rms.values()))}):
        score, ordered = 0.0, gaps.get(flight, float('inf')) <= CONST_MAX_GAP
        for method in rms:
            keys = [(flight, n) for n in n_values]
            if not all(k in rms[method] and k in end[method] for k in keys):
                ordered = False
                break
            r, e = [rms[method][k] for k in keys], [end[method][k] for k in keys]
            ordered &= all(a > b for a, b in zip(r, r[1:])) and all(a > b for a, b in zip(e, e[1:]))
            score += float(np.sum(np.abs(np.log(np.array(r) / np.array([medians[method][n] for n in n_values])))))
        if ordered and (best is None or score < best[0]):
            best = (score, int(flight))
    if best is None:
        raise Plot_Error_Cls('No validation throw of %s without a gap over %g s has its errors falling from each N to the next for every method; '
                             'name one with --index.' % (dataset, CONST_MAX_GAP))
    return best[1], best[0]


def Load_Horizons(evaluations: tp.Dict[str, str], dataset: str) -> Horizon_Str:
    """
    Description:
        The position error at the fixed horizons (SPEC §7.12) of each method's simulated validation on one dataset, from
        its evaluation's summary.csv (the metrics fixed_<h>_mm of the group 'all'): per N the median over the seeds of each
        seed's median, with the lowest and the highest (the per-seed rows only: the summary's spread_* rows are aggregates).

    Args:
        (1) evaluations [Dict<string, string>]: Per method its validation evaluation's folder (<Evaluations>/<Run.Name>/validation).
        (2) dataset [string]: The dataset's label.

    Returns:
        (1) parameter [Horizon_Str(object)]: The errors.
    """

    out = Horizon_Str()
    for method, folder in evaluations.items():
        values: tp.Dict[tp.Tuple[int, float], tp.List[float]] = {}
        for row in _Rows(os.path.join(folder, 'summary.csv'), Read_Evaluation_Summary):
            metric = row['metric']
            per_seed = row['seed'] == '' or row['seed'].isdigit()
            if row['partition'] == 'validation' and row['dataset'] == dataset and row['group'] == 'all' and per_seed \
                    and metric.startswith('fixed_') \
                    and metric.endswith('_mm') and metric != 'fixed_end_mm' and row['median'] != '':
                values.setdefault((int(row['n']), float(metric[6:-3])), []).append(float(row['median']))
        if not values:
            raise Plot_Error_Cls('%s/summary.csv holds no fixed-horizon error of the dataset %s.' % (folder, dataset))
        out.Horizon = np.array(sorted({h for _, h in values}))
        out.N = np.array(sorted({n for n, _ in values}), dtype=np.int64)
        for n in out.N:
            per = [values.get((int(n), float(h)), [np.nan]) for h in out.Horizon]
            out.Error[(method, int(n))] = np.array([[np.median(v) for v in per], [np.min(v) for v in per], [np.max(v) for v in per]])
    return out


def Load_Throw(dataset: str, index: int, evaluations: tp.Dict[str, tp.Tuple[str, str]], data: tp.Optional[Dataset_Str] = None) -> Throw_Str:
    """
    Description:
        One simulated validation throw and both predictors' stored predictions of it: the partition comes from the
        dataset's manifest (partition_sizes), so a throw outside the validation partition is refused before flights.npz is
        opened (a test throw is never read); then the true path and the observed samples, and each method's records of
        its validation evaluation's predictions.npz for this flight and its seed, with the true position at each query.

    Args:
        (1) dataset [string]: The dataset's folder.
        (2) index [int]: The throw's index.
        (3) evaluations [Dict<string, Tuple<string, string>>]: Per method its validation evaluation's folder and the seed's
                                                               label shown ('' for a numerical predictor).
        (4) data [Dataset_Str(object) or None]: The dataset already read (Validation_Data), or None to read it here.

    Returns:
        (1) parameter [Throw_Str(object)]: The throw.
    """

    label = os.path.basename(os.path.normpath(dataset))
    try:
        sizes = [int(value) for value in Read_Manifest(os.path.join(dataset, 'manifest.json'))['partition_sizes']]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise Plot_Error_Cls('%s is not a dataset folder (%s): it holds manifest.json written by scripts/catch.py dataset (SPEC §9.5).'
                             % (dataset, error)) from error
    if Partition_Code(index, sizes) != CONST_PARTITION_VALIDATION:
        raise Plot_Error_Cls('The throw %d of %s is not in its validation partition (the flights %d to %d); the predictions are stored for '
                             'the validation partition only, and a test throw is never read.' % (index, label, sizes[0], sizes[0] + sizes[1] - 1))
    data = Read_Dataset(os.path.join(dataset, 'flights.npz')) if data is None else data
    out = Throw_Str(Dataset=label, Flight=index)
    out.Truth_Time, out.Truth_Position = data.Truth(index)
    out.Observed_Time, out.Observed_Position = data.Observations(index)
    for method, (folder, seed) in evaluations.items():
        records = [r for r in _Rows(os.path.join(folder, 'predictions.npz'), lambda p: Read_Predictions(p, dataset=label, flight='%d' % index))
                   if r.Seed == seed and r.Partition == 'validation']
        if not records:
            raise Plot_Error_Cls('%s/predictions.npz holds no prediction of the throw %d of %s with the seed %r.' % (folder, index, label, seed))
        for r in records:
            truth = np.column_stack([np.interp(r.Query_Time, out.Truth_Time, out.Truth_Position[:, k]) for k in range(3)])
            out.Predicted[(method, r.N)] = (r.Query_Time, r.Query_State[:, :3], truth)
    return out


def Real_Predictions(name: str, rate: float, time: np.ndarray, position: np.ndarray,
                     predictors: tp.Dict[str, tp.Callable[[np.ndarray, np.ndarray, float], tp.Any]], n_values: tp.Sequence[int],
                     shown: int, points: int = 200) -> Real_Str:
    """
    Description:
        Both predictors on one real flight, as the final evaluation predicts a real flight (Evaluation.Core.Evaluate_Real):
        the prefix is the first N recorded samples, the truth the recorded samples after it; the path shown is the
        prediction from the first `shown` samples, from its t_0 to its floor time; a refused prediction is an error.

    Args:
        (1) name [string]: The flight's name.
        (2) rate [float]: Its rate [Hz].
        (3) time [Vector<float> 1xM]: Its sample times from its first kept sample [s].
        (4) position [Matrix<float> Mx3]: Its positions in the robot frame [m].
        (5) predictors [Dict<string, Callable>]: Per method the function of a prefix giving its trajectory.
        (6) n_values [Sequence<int>]: The values of N of the error.
        (7) shown [int]: The N of the path shown.
        (8) points [int]: The points of the path shown.

    Returns:
        (1) parameter [Real_Str(object)]: The predictions.
    """

    if len(time) < max(list(n_values) + [shown]) + 1:
        raise Plot_Error_Cls('The flight %s has %d samples; the figure needs more than %d.' % (name, len(time), max(list(n_values) + [shown])))
    out = Real_Str(Name=name, Rate=float(rate), Time=np.asarray(time), Position=np.asarray(position), Shown_N=int(shown))
    for method, predictor in predictors.items():
        for n in sorted(set(n_values) | {shown}):
            trajectory = predictor(out.Time[:n], out.Position[:n], out.Rate)
            if n in n_values:
                error = np.linalg.norm(Held_State(trajectory, out.Time[n:])[:, :3] - out.Position[n:], axis=1)
                out.Error[(method, int(n))] = (out.Time[n:] - trajectory.T_0, 1e3 * error)
            if n == shown:
                t = np.linspace(trajectory.T_0, trajectory.T_End, points)
                out.Path[method] = (t, Held_State(trajectory, t)[:, :3])
    return out


def Load_Restitution(summaries: tp.Dict[str, str], restitution: tp.Callable[[float, float], float]) -> Share_Str:
    """
    Description:
        The retained share of the goes against the restitution of each contact simulated (the levels, on the first seed
        only), per mode ('static', 'moving', 'markers': the moving catch with the ball's markers as collision shapes),
        dataset and acceleration, from the stored catches' summary.csv; the restitution of a contact (time constant,
        damping ratio) is computed by the function given (Catch.Physical.Rebound at plot time, Q14), never typed. The
        cliff is read where the share first falls below CONST_CLIFF_SHARE, linear between the two contacts around it.

    Args:
        (1) summaries [Dict<string, string>]: The validation summary.csv of the static and of the moving catch.
        (2) restitution [Callable[[float, float], float]]: The restitution of a contact.

    Returns:
        (1) parameter [Share_Str(object)]: The shares.
    """

    out = Share_Str()
    groups: tp.Dict[tp.Tuple[str, str, int], tp.List[tp.Dict[str, str]]] = {}
    for mode, path in summaries.items():
        rows = _Rows(path, Read_Catch_Summary)
        first = min(int(r['seed']) for r in rows) if rows else 0
        for r in rows:
            if int(r['seed']) != first:
                continue
            variant = 'markers' if r.get('markers') == 'true' else mode
            groups.setdefault((variant, r['dataset'], int(float(r['acceleration_deg_s2']))), []).append(r)
    cache: tp.Dict[tp.Tuple[float, float], float] = {}
    for key, rows in groups.items():
        values = []
        for r in rows:
            contact = (float(r['time_constant_s']), float(r['damping_ratio']))
            if contact not in cache:
                cache[contact] = float(restitution(*contact))
            values.append((cache[contact], _Number(r['retained_of_go']), _Number(r['retained_of_go_lower']), _Number(r['retained_of_go_upper'])))
            known = out.Contacts.setdefault(key[0], [])
            if (contact[0], contact[1], cache[contact]) not in known:
                known.append((contact[0], contact[1], cache[contact]))
                known.sort(key=lambda c: c[2])
        share = np.array(sorted(values)).T
        out.Share[key] = share
        below = np.flatnonzero(share[1] < CONST_CLIFF_SHARE)
        cliff = float('nan')
        if len(below) and below[0] > 0:
            k = below[0]
            e0, e1, s0, s1 = share[0, k - 1], share[0, k], share[1, k - 1], share[1, k]
            cliff = float(e0 + (s0 - CONST_CLIFF_SHARE) * (e1 - e0) / (s0 - s1))
        out.Cliff[key] = cliff
    return out


def Load_Classes(summaries: tp.Dict[tp.Tuple[str, str], str], nominal: tp.Tuple[float, float]) -> tp.Dict[tp.Tuple[str, str, str, int], np.ndarray]:
    """
    Description:
        The catch's classes with the nominal contact (no markers), summed over the seeds, from the stored catches'
        summary.csv, per partition, mode, dataset and acceleration.

    Args:
        (1) summaries [Dict<Tuple<string, string>, string>]: Per (partition, mode) the summary.csv.
        (2) nominal [Tuple<float, float>]: The nominal contact's time constant [s] and damping ratio [-] (Contact).

    Returns:
        (1) parameter [Dict<Tuple<string, string, string, int>, Vector<int> 1xC>]: The counts, in CONST_CLASS_COLUMNS' order.
    """

    out: tp.Dict[tp.Tuple[str, str, str, int], np.ndarray] = {}
    for (partition, mode), path in summaries.items():
        for r in _Rows(path, Read_Catch_Summary):
            if (float(r['time_constant_s']), float(r['damping_ratio'])) != tuple(nominal) or r.get('markers') == 'true':
                continue
            key = (partition, mode, r['dataset'], int(float(r['acceleration_deg_s2'])))
            out[key] = out.get(key, np.zeros(len(CONST_CLASS_COLUMNS), dtype=np.int64)) + [int(r[c]) for c in CONST_CLASS_COLUMNS]
    return out


def Load_Sweep(path: str) -> tp.Dict[tp.Tuple[str, str, int], np.ndarray]:
    """
    Description:
        The robustness sweep (F11, static, as recorded): the retained share of the catchable throws and its interval
        against the injected error's scale, per error source ('real', 'simulated'; the run's own error as 'run', its scale
        NaN), dataset and acceleration, sorted by the scale.

    Args:
        (1) path [string]: The sweep's summary.csv.

    Returns:
        (1) parameter [Dict<Tuple<string, string, int>, Matrix<float> 4xS>]: The scale, the share, its lower and upper bound.
    """

    groups: tp.Dict[tp.Tuple[str, str, int], tp.List[tp.Tuple[float, ...]]] = {}
    for r in _Rows(path, Read_Catch_Summary):
        if 'error_source' not in r:
            raise Plot_Error_Cls('%s is not the sweep\'s summary.csv (it has no error_source column; SPEC §9.14).' % path)
        groups.setdefault((r['error_source'], r['dataset'], int(float(r['acceleration_deg_s2']))), []).append(
            (_Number(r['error_scale']), _Number(r['retained_of_catchable']), _Number(r['retained_of_catchable_lower']),
             _Number(r['retained_of_catchable_upper'])))
    return {key: np.array(sorted(values, key=lambda v: (np.isnan(v[0]), v[0]))).T for key, values in groups.items()}


def Load_Decision(decisions: tp.Dict[str, str], interceptions: tp.Dict[str, str]) -> tp.Dict[tp.Tuple[str, str, str, int], np.ndarray]:
    """
    Description:
        The decision's errors per partition, dataset and acceleration: F9's stage rule (the decision's summary.csv) and
        F10's decision at every sample (the interception's), the false-no-go rate per seed and the false goes with their
        95 % upper bound per seed.

    Args:
        (1) decisions [Dict<string, string>]: Per partition the decision's summary.csv.
        (2) interceptions [Dict<string, string>]: Per partition the static interception's summary.csv.

    Returns:
        (1) parameter [Dict<Tuple<string, string, string, int>, Matrix<float> 3xS>]: Per (stage 'stages' or 'samples',
                                                                                     partition, dataset, acceleration) per seed
                                                                                     the false-no-go rate, the false goes and
                                                                                     the false-go rate's upper bound.
    """

    out: tp.Dict[tp.Tuple[str, str, str, int], tp.List[tp.Tuple[float, float, float]]] = {}
    for stage, paths, reader in (('stages', decisions, Read_Decision_Summary), ('samples', interceptions, Read_Interception_Summary)):
        for partition, path in paths.items():
            for r in _Rows(path, reader):
                out.setdefault((stage, partition, r['dataset'], int(float(r['acceleration_deg_s2']))), []).append(
                    (_Number(r['false_no_go_rate']), _Number(r['false_go']), _Number(r['false_go_upper'])))
    return {key: np.array(values).T for key, values in out.items()}


def Load_Interception(per_throw: tp.Dict[str, str], dataset: str) -> tp.Dict[tp.Tuple[str, int], tp.Tuple[np.ndarray, np.ndarray]]:
    """
    Description:
        The interception of one dataset's goes per mode and acceleration, from the stored per_throw.csv: the lateral miss
        at the cup's mouth of every go with a crossing, and the accepted plans of every go (the re-plans plus one).

    Args:
        (1) per_throw [Dict<string, string>]: Per mode ('static', 'moving') the validation interception's per_throw.csv.
        (2) dataset [string]: The dataset's label.

    Returns:
        (1) parameter [Dict<Tuple<string, int>, Tuple<Vector, Vector>>]: The lateral misses [mm] and the plans [-].
    """

    out = {}
    for mode, path in per_throw.items():
        groups: tp.Dict[int, tp.Tuple[tp.List[float], tp.List[float]]] = {}
        for r in _Rows(path, Read_Interception_Rows):
            if r['dataset'] != dataset or r['decision'] != 'go':
                continue
            lateral, plans = groups.setdefault(int(float(r['acceleration_deg_s2'])), ([], []))
            if r['lateral_m'] != '':
                lateral.append(1e3 * float(r['lateral_m']))
            plans.append(float(r['plans'] or 0))
        for acceleration, (lateral, plans) in groups.items():
            out[(mode, acceleration)] = (np.sort(lateral), np.array(plans))
    return out


def Motion_From_Records(records: tp.Dict[str, Throw_Record_Str], model: mujoco.MjModel, data: mujoco.MjData, robot: Robot_Str,
                        moving: str = 'moving') -> Motion_Str:
    """
    Description:
        The motion figure's data from the records of one throw (View.Core.Record_Throw) caught in motion and statically:
        at every CONST_MOTION_STEP the cup's mouth (Robot.Core.Forward_Kinematics at the recorded joints) and the ball's
        centre, their velocities by central differences, the cup's speed along the ball's direction just before the moving
        catch's first cup contact (the same direction for both modes), the ball's speed and their distance; the moving
        catch's phases from its interception and catch rows: the observation from the release to the first plan (the robot
        at the ready pose while the decision waits), the move from the ready pose to the stroke's start (the first plan's
        start to the arrival), the wait, the stroke to the cup's largest speed along the ball, the braking to the stroke's
        end and the rest; per mode the speeds at its own first cup contact; the time the ball passes nearest the mouth.

    Args:
        (1) records [Dict<string, Throw_Record_Str>]: Per mode ('moving', 'static') the record of the same throw.
        (2) model [mujoco.MjModel]: A compiled scene with the robot.
        (3) data [mujoco.MjData]: Its data (the robot's joints are overwritten).
        (4) robot [Robot_Str(object)]: The robot in the model.
        (5) moving [string]: The key of the moving record.

    Returns:
        (1) parameter [Motion_Str(object)]: The motion.
    """

    record = records[moving]
    if record.Decision != 'go' or not record.Stroke:
        raise Plot_Error_Cls('The throw %d of %s is %s in the moving catch; the motion figure shows a go caught with a stroke. Choose another '
                             'throw (--index).' % (record.Flight, record.Dataset, 'a no-go' if record.Decision != 'go' else 'a go without a stroke'))
    interception = dict(zip(CONST_INTERCEPTION_COLUMNS + CONST_INTERCEPTION_MOVING_COLUMNS, record.Interception_Row))
    catch = dict(zip(CONST_CATCH_COLUMNS + CONST_CATCH_MOVING_COLUMNS, record.Catch_Row))
    out = Motion_Str(Contact=float(catch['cup_contact_s']), Relative_Speed=float(catch['contact_speed_m_s']),
                     Contact_Part=str(catch['first_contact']),
                     Class=str(record.Class))
    direction = None
    for mode in (moving,) + tuple(m for m in records if m != moving):
        r = records[mode]
        if r.Decision != 'go':
            raise Plot_Error_Cls('The throw %d of %s is a no-go in the %s catch; the motion figure compares two goes.' % (r.Flight, r.Dataset, mode))
        every = max(1, int(round(CONST_MOTION_STEP / float(np.median(np.diff(r.State_Time))))))
        t = r.State_Time[::every]
        q = r.State_Position[::every][:, r.Robot_Address]
        mouth = np.array([Forward_Kinematics(model, data, joints, Robot_Str=robot)[0] for joints in q])
        ball = r.State_Position[::every][:, r.Ball_Address:r.Ball_Address + 3]
        cup_velocity, ball_velocity = np.gradient(mouth, t, axis=0), np.gradient(ball, t, axis=0)
        if direction is None:
            k = max(0, int(np.searchsorted(t, out.Contact)) - 2)
            direction = ball_velocity[k] / np.linalg.norm(ball_velocity[k])
            out.Joints, out.Joint_Speeds = q, np.gradient(q, t, axis=0)
        out.Time[mode], out.Cup_Speed[mode] = t, cup_velocity @ direction
        out.Ball_Speed[mode], out.Distance[mode] = np.linalg.norm(ball_velocity, axis=1), np.linalg.norm(ball - mouth, axis=1)
        row = dict(zip(CONST_CATCH_COLUMNS, r.Catch_Row))
        j = max(2, int(np.searchsorted(r.State_Time, float(row['cup_contact_s']))))
        pair = [r.State_Position[i] for i in (j - 2, j - 1)]
        step = float(r.State_Time[j - 1] - r.State_Time[j - 2])
        ends = [Forward_Kinematics(model, data, q[r.Robot_Address], Robot_Str=robot)[0] for q in pair]
        ball_v = (pair[1][r.Ball_Address:r.Ball_Address + 3] - pair[0][r.Ball_Address:r.Ball_Address + 3]) / step
        cup_v = (ends[1] - ends[0]) / step
        out.At_Contact[mode] = (float(r.State_Time[j - 1]), float(np.linalg.norm(ball_v)), float(cup_v @ direction),
                                float(np.linalg.norm(ball_v - cup_v)))
    start, arrival = float(record.Plan_Time[0]), float(interception['arrival_s'])
    stroke_start, stroke_end = float(interception['stroke_start_s']), float(interception['stroke_end_s'])
    t, speed = out.Time[moving], out.Cup_Speed[moving]
    inside = (t >= stroke_start) & (t <= stroke_end)
    peak = float(t[inside][np.argmax(speed[inside])]) if inside.any() else stroke_start
    before = t <= out.Contact
    out.Mouth_Pass = float(t[before][np.argmin(out.Distance[moving][before])])
    out.Phases = [('observe', float(t[0]), start), ('move', start, arrival), ('wait', arrival, stroke_start), ('stroke', stroke_start, peak),
                  ('braking', peak, stroke_end), ('rest', stroke_end, float(t[-1]))]
    return out


def Read_Captions(path: str) -> tp.List[tp.Tuple[str, str]]:
    """
    Description:
        captions.csv (SPEC §9.16; Write_Captions) read back, so that a later call keeps the captions of the files it does
        not write again; a file with another header is refused.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [List<Tuple<string, string>>]: The rows: the file and its caption.
    """

    with open(path, 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        if tuple(next(reader, None) or ()) != CONST_CAPTION_COLUMNS:
            raise Plot_Error_Cls('%s does not have the columns %s (SPEC §9.16); move it out of the figures\' folder.'
                                 % (path, ', '.join(CONST_CAPTION_COLUMNS)))
        return [(row[0], row[1]) for row in reader]


def Write_Captions(path: str, rows: tp.Iterable[tp.Tuple[str, str]]) -> None:
    """
    Description:
        captions.csv (SPEC §9.16): one row per figure written, its file (relative to the figures' folder) and its caption;
        comma-separated, a header line, LF. Read back by Read_Captions only (the README's captions are written from it by hand, F13).

    Args:
        (1) path [string]: The file.
        (2) rows [Iterable<Tuple<string, string>>]: The rows.
    """

    with open(path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(CONST_CAPTION_COLUMNS)
        writer.writerows(rows)
