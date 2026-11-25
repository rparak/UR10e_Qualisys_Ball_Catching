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
# File Name: ../scripts/catch.py
## =========================================================================== ##

# System (Default)
import sys
# OS (Operating system interfaces)
import os
# The repository root, built from this file's location, so the script runs from any working
# directory.
#   Unit [string]
CONST_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
#   Add access if it is not in the system path.
if os.path.join(CONST_REPO_ROOT, 'src') not in sys.path:
    sys.path.append(os.path.join(CONST_REPO_ROOT, 'src'))
# Typing (Support for type hints)
import typing as tp
# Argparse (Command-line parsing)
import argparse
# Shutil (High-level file operations; the dataset's temporary folder)
import shutil
# Tempfile (Temporary folders; a dataset is written to one and renamed)
import tempfile
# Time (the generation time printed)
import time
# Subprocess (the git commit and dirty flag of the manifests and the ledgers)
import subprocess
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Lib.:
#   ../Configuration/Core - the only reader of the configuration
from Configuration.Core import (Load_Configuration, Load_Dataset_Configuration, Load_Training_Configuration, Configuration_Error_Cls,
                                Configuration_Str, CONST_MODEL_NAMES, CONST_LEARNED_MODEL_NAMES, CONST_NAME_PATTERN, CONST_CATCH_MODES)
#   ../Recording/Core, Segmentation - the recordings and their flights (the real validation flights of 'evaluate
#   --partition test'; the dropout donors of 'dataset')
from Recording.Core import Load_Recordings, Recording_Error_Cls
from Recording.Segmentation import Segment_Flights
#   ../Recording/Identification - a flight's name, its trim and its fitted state (the real flights' coverage)
from Recording.Identification import Validate_Flight_Model, Flight_Name, Trim_Flight, CONST_IDENTIFICATION_TRIM
from Recording.Core import Transform_To_Robot
#   ../Recording/Core - the set names
from Recording.Core import CONST_SET_VALIDATION
#   ../Flight/Core, ../Observation/Core - the models built from the configuration
from Flight.Core import Flight_Model_Str
from Observation.Core import Observation_Model_Str
#   ../Dataset/Core, Calibration - the generator, a dataset and its partitions (F6); the real flights' coverage
from Dataset.Core import (Dataset_Error_Cls, Generate_Dataset, Load_Donors, Partition_Sizes, Write_Dataset, Read_Dataset,
                          Partition_Indices, CONST_PARTITION_TRAIN, CONST_PARTITION_VALIDATION, CONST_PARTITION_HELD_OUT)
from Dataset.Manifest import Build_Manifest, Model_Hash, Read_Manifest, Sha256_Bytes, Sha256_File, Write_Manifest
from Observation.Core import Observation_Error_Cls
from Dataset.Calibration import Coverage, Robot_States
#   ../Prediction/Core - the numerical predictors, their error and their trajectory (F7)
from Prediction.Core import Prediction_Error_Cls, Trajectory_Str
#   ../Learning/Core, Predictor - a trained learned predictor at work, in NumPy (F8); the training's runtime is
#   imported only by 'train' for a learned predictor
from Learning.Core import Write_Normalisation
from Learning.Predictor import Run_Predictor, Write_Weights, CONST_WEIGHTS_FILE, CONST_NORMALISATION_FILE
#   ../Evaluation/Core, Report - the metrics, the evaluation's files and the test ledger (F7)
from Evaluation.Core import Catch_Height, Evaluate_Real, Evaluate_Simulated, Flight_Result_Str, Summary_Rows
from Evaluation.Report import (Append_Test_Ledger, Prediction_Record_Str, Write_Coverage, Write_Per_Flight, Write_Per_Horizon, Write_Predictions,
                               Write_Summary, Read_Per_Flight, Read_Per_Horizon, Read_Coverage, Write_Comparison_Table, Report_Error_Cls,
                               CONST_COMPARISON_SUMMARY_COLUMNS, CONST_COMPARISON_PAIRED_COLUMNS, CONST_COMPARISON_SIM_TO_REAL_COLUMNS)
#   ../Evaluation/Comparison - stored evaluations compared (F14)
from Evaluation.Comparison import (Build_Table, Comparison_Summary, Comparison_Paired, Comparison_Sim_To_Real, Comparison_Error_Cls,
                                   Report_Lines, CONST_REAL, CONST_BOOTSTRAP_SEED, CONST_BOOTSTRAP_RESAMPLES)

"""
Description:
    Initialization of constants.

    The user's entry point (SPEC §6): one sub-command per task of a user who generates data,
    trains, evaluates, compares, catches and watches: dataset, train, evaluate, compare, catch, view. The project's
    derivations (recordings, identify, observation, workspace, scene, dataset-check, sweep) are the sub-commands of
    scripts/Maintenance/derive.py, which loads this script's helpers (one direction only: this script never loads it).
"""
# The shipped configuration (SPEC §8).
#   Unit [string]
CONST_CONFIG_FILE = os.path.join(CONST_REPO_ROOT, 'Config', 'ball_catching.yaml')
# The recordings (SPEC §9.2), read only.
#   Unit [string]
CONST_RECORDINGS_DIR = os.path.join(CONST_REPO_ROOT, 'Data', 'Recordings')
# The shipped dataset configuration (SPEC §8.2).
#   Unit [string]
CONST_DATASET_CONFIG_FILE = os.path.join(CONST_REPO_ROOT, 'Config', 'dataset.yaml')
# The shipped training configuration (SPEC §8.3; F7).
#   Unit [string]
CONST_TRAINING_CONFIG_FILE = os.path.join(CONST_REPO_ROOT, 'Config', 'training.yaml')
# The parent folder of the evaluations (SPEC §9.7); --out DIR replaces it.
#   Unit [string]
CONST_EVALUATIONS_DIR = os.path.join(CONST_REPO_ROOT, 'Data', 'Generated', 'Evaluations')
# The parent folder of the comparisons (SPEC §9.7; F14); --out DIR replaces it.
#   Unit [string]
CONST_COMPARISONS_DIR = os.path.join(CONST_REPO_ROOT, 'Data', 'Generated', 'Comparisons')
# The partitions an evaluation reads, by name (SPEC §9.5, §9.7).
#   Unit [Dict<string, int>]
CONST_PARTITION_CODES = {'validation': CONST_PARTITION_VALIDATION, 'test': CONST_PARTITION_HELD_OUT}
# The test ledger's file name in the evaluations folder (SPEC §9.7).
#   Unit [string]
CONST_TEST_LEDGER = 'test_ledger.csv'
# The project's safety sentence (SPEC §1), printed with every result.
#   Unit [string]
CONST_SAFETY = 'Outputs are simulation results; nothing is run on, or validated on, the real UR10e.'
# The throws of one task of 'catch' (a worker's unit; the outputs do not depend on it), and how often a run reports
# its progress, in tasks.
#   Unit [int]
CONST_CHUNK_THROWS = 50
CONST_PROGRESS_TASKS = 20
# The folders of the three stages of 'catch' under --out DIR, in the stages' order.
#   Unit [Dict<string, string>]
CONST_STAGE_FOLDERS = {'decision': 'Decisions', 'interception': 'Interceptions', 'catch': 'Catches'}


def _Catch_Criteria(configuration: Configuration_Str) -> tp.Any:
    """
    Description:
        The catch pose's criteria from the configuration (Catch, Robot.Joint_Speed).

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.

    Returns:
        (1) parameter [Criteria_Str(object)]: The criteria.
    """

    #   ../Catch/Core - the criteria's structure
    from Catch.Core import Criteria_Str
    c = configuration.Catch
    return Criteria_Str(Joint_Margin=c.Joint_Margin, Clearance=c.Clearance, Speed_Margin=c.Speed_Margin,
                        Joint_Speed=np.array(configuration.Robot.Joint_Speed))


def _Decision_Datasets(configuration: Configuration_Str) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Description:
        The decision's datasets (Decision.Datasets; SPEC §7.11), checked: each dataset's flights.npz has the configured
        hash and its manifest's, and its manifest's model hash is that of the system configuration given, so the
        stored configuration copies need not be read (KNOWN_ISSUES). The data itself is not read here.

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.

    Returns:
        (1) parameter [List<Dict<string, Any>>]: Per dataset: label, path, folder, hash, calibration (bool).
    """

    model_sha256 = Model_Hash(Flight_Model_Str=_Flight_Model(configuration), Observation_Model_Str=_Observation_Model(configuration))
    out, problems = [], []
    for item in configuration.Decision.Datasets:
        folder = os.path.join(CONST_REPO_ROOT, item.Path)
        try:
            manifest = Read_Manifest(os.path.join(folder, 'manifest.json'))
            data_hash = Sha256_File(os.path.join(folder, 'flights.npz'))
        except (OSError, ValueError) as error:
            problems.append('the dataset %s cannot be read (%s); generate it with scripts/catch.py dataset (SPEC §9.5)' % (item.Path, error))
            continue
        if data_hash != item.Hash or manifest.get('sha256', {}).get('data') != item.Hash:
            problems.append('the dataset %s has flights.npz %s and manifest %s, not Decision.Datasets\' %s (SPEC §8.1)'
                            % (item.Path, data_hash, manifest.get('sha256', {}).get('data'), item.Hash))
            continue
        if manifest.get('model_sha256') != model_sha256:
            problems.append('the dataset %s was generated with the models %s, not this configuration\'s %s; one flight model drives the '
                            'datasets and the decision (SPEC §2.5)' % (item.Path, manifest.get('model_sha256'), model_sha256))
            continue
        out.append({'label': os.path.basename(os.path.normpath(item.Path)), 'path': item.Path, 'folder': folder, 'hash': data_hash,
                    'calibration': item.Path in configuration.Decision.Calibration})
    if problems:
        raise Run_Error_Cls('The decision\'s datasets cannot be used (%d problem(s)):\n  %s' % (len(problems), '\n  '.join(problems)))
    return out


def _Decision_Throws(data: tp.Any, indices: np.ndarray, stages: tp.List[int], predictors: tp.List[tp.Tuple[str, tp.Callable[..., tp.Any]]],
                     times: np.ndarray, states: np.ndarray, decision_time: np.ndarray, calibrate: bool, area: tp.Tuple[tp.Any, tp.Any],
                     progress: tp.Callable[[int, int], None]) -> tp.Tuple[tp.Dict[tp.Tuple[str, int], tp.Any], tp.Dict[str, tp.List[tp.List[str]]],
                                                                         tp.Dict[tp.Tuple[str, int], tp.List[tp.Any]]]:
    """
    Description:
        The predictions of a dataset's throws at each stage, sampled on the truth's grid (every CONST_SAMPLE_STEP from
        the release) after the decision time: their candidates (Catch.Decision.Candidates) and, to calibrate the
        margins, their errors against the truth (Catch.Decision.Margin_Errors). A refused prediction is a no-go at its
        stage, its N recorded.

    Args:
        (1) data [Dataset_Str(object)]: The dataset.
        (2) indices [Vector<int> 1xF]: Its throws of the partition.
        (3) stages [List<int>]: The stages' N.
        (4) predictors [List<Tuple<string, Callable>>]: Per seed the function of a prefix (_Predictors).
        (5) times [Vector<float> 1xT]: The truth's grid [s] (Catch.Workspace.Flight_Samples).
        (6) states [Tensor<float> FxTx6]: The true states on it [m, m/s].
        (7) decision_time [Matrix<float> FxK]: The decision time per throw and stage [s], NaN without one.
        (8) calibrate [bool]: Collect the errors.
        (9) area [Tuple<Vector<float> 1x3>]: Areas.Catch's centre and size [m].
        (10) progress [Callable[[int, int], None]]: Called every 1000 throws with (done, all).

    Returns:
        (1) parameter [Dict]: Per (seed, stage index) the candidates' throws, horizons and positions.
        (2) parameter [Dict]: Per seed and throw the N whose prediction was refused.
        (3) parameter [Dict]: Per (seed, stage index) the errors (bins, cross-track, early).
    """

    #   ../Catch/Decision - the candidates and the errors
    from Catch.Decision import Candidates, Margin_Errors
    #   ../Catch/Workspace - the truth's grid step
    from Catch.Workspace import CONST_SAMPLE_STEP

    keys = [(seed, k) for seed, _ in predictors for k in range(len(stages))]
    candidates: tp.Dict[tp.Tuple[str, int], tp.Any] = {key: ([], [], []) for key in keys}
    errors: tp.Dict[tp.Tuple[str, int], tp.List[tp.Any]] = {key: [] for key in keys}
    refused = {seed: [[] for _ in indices] for seed, _ in predictors}
    for row, i in enumerate(indices):
        observed_time, observed_position = data.Observations(int(i))
        for k, n in enumerate(stages):
            if not np.isfinite(decision_time[row, k]):
                continue
            for seed, predictor in predictors:
                try:
                    trajectory = predictor(observed_time[:n], observed_position[:n], float(data.Rate[i]))
                except Prediction_Error_Cls:
                    refused[seed][row].append('%d' % n)
                    continue
                first = int(np.floor(trajectory.T_0 / CONST_SAMPLE_STEP)) + 1
                last = min(int(np.floor(trajectory.T_End / CONST_SAMPLE_STEP + 1e-9)), len(times) - 1)
                while last >= first and times[last] > trajectory.T_End:
                    last -= 1
                if last < first:
                    continue
                grid = times[first:last + 1]
                predicted = trajectory.State(grid)
                h, position = Candidates(grid, predicted, decision_time[row, k], area[0], area[1])
                rows, horizons, positions = candidates[(seed, k)]
                rows.append(np.full(len(h), row, dtype=np.int64))
                horizons.append(h)
                positions.append(position)
                if calibrate:
                    errors[(seed, k)].append(Margin_Errors(predicted, states[row, first:last + 1], grid - decision_time[row, k]))
        if (row + 1) % 1000 == 0:
            progress(row + 1, len(indices))
    return candidates, refused, errors


def _Decision_Stage(configuration: Configuration_Str, training: tp.Any, run: str, partition: str, datasets: tp.List[tp.Dict[str, tp.Any]],
                    context: tp.Any, grids: tp.List[tp.Any], margins: tp.Dict[tp.Tuple[str, int], tp.Tuple[np.ndarray, np.ndarray]],
                    started: float) -> tp.Tuple[tp.List[tp.Tuple[tp.Any, ...]], tp.List[tp.Tuple[tp.Any, ...]], tp.List[tp.Tuple[tp.Any, ...]],
                                                tp.Dict[str, tp.Any]]:
    """
    Description:
        The first stage of 'catch', the decision (F9; SPEC §2.8, §7.11, §9.12): the catch decision of a run's predictor on
        the simulated throws of Decision.Datasets, one partition. Per throw the truth (Catch.Decision: catchable from the
        decision at each of Decision.Stages, at both Robot.Joint_Accelerations) and the decision (go at the first stage
        whose predicted trajectory passes the margins). On the validation partition the margins are calibrated on the
        validation partitions of Decision.Calibration at Decision.Quantile and the throws decided with them (the rates
        are in-sample for the margins); on the test partition the margins given (the stored validation decision's) are
        used. Nothing is written here, and no recording is read.

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.
        (2) training [Training_Configuration_Str(object)]: The run's training configuration.
        (3) run [string]: The run folder.
        (4) partition [string]: 'validation' or 'test'.
        (5) datasets [List<Dict<string, Any>>]: The decision's datasets (_Decision_Datasets).
        (6) context [Catch_Context_Str(object)]: The catch-pose context of the scene.
        (7) grids [List<Tuple>]: The area's reach per acceleration (_Interception_Setup: Interception_Setup_Str.Grids).
        (8) margins [Dict<Tuple<string, int>, Tuple>]: The stored margins per seed and N (the test partition); empty on
                                                       the validation partition.
        (9) started [float]: The command's start (time.perf_counter), for the progress lines [s].

    Returns:
        (1) parameter 1 [List<Tuple>]: The rows of per_throw.csv.
        (2) parameter 2 [List<Tuple>]: The rows of summary.csv.
        (3) parameter 3 [List<Tuple>]: The rows of margins.csv (the validation partition; else empty).
        (4) parameter 4 [Dict<string, Any>]: The manifest's fields of this stage: seeds, throws, truth_samples.
    """

    #   ../Catch/Core, Workspace, Decision - the joints used, the truth's samples, the truth, the margins, the decision
    from Catch.Core import CONST_JOINTS_USED
    from Catch.Workspace import Flight_Samples, Joint_Wrap
    from Catch.Decision import (Truth_Area, Truth_Samples, Exact_Reach, Catchable, Margins, Decide, Stage_Rule, Rates, CONST_HORIZON_BIN,
                                CONST_HORIZON_BINS)

    decision, catch, robot_limits = configuration.Decision, configuration.Catch, configuration.Robot
    catch_area = configuration.Areas.Catch
    truth_area = Truth_Area(catch_area.Centre, catch_area.Size, catch.Workspace_Step)
    area = (catch_area.Centre, catch_area.Size)
    step = catch.Workspace_Step
    start = np.array(robot_limits.Ready_Pose[:CONST_JOINTS_USED])
    speed, wrap = np.array(robot_limits.Joint_Speed[:CONST_JOINTS_USED]), Joint_Wrap(context)
    accelerations = list(robot_limits.Joint_Accelerations)
    flight_model = _Flight_Model(configuration)
    predictors = _Predictors(training, {'flight': flight_model, 'observation': _Observation_Model(configuration)}, run)
    seeds, stages = [label for label, _ in predictors], decision.Stages
    margins = dict(margins)

    # 1\ Per dataset: the truth, and the predictions' candidates and errors.
    per_dataset = []
    pooled: tp.Dict[tp.Tuple[str, int], tp.List[tp.Any]] = {(seed, k): [] for seed in seeds for k in range(len(stages))}
    for d in datasets:
        data = Read_Dataset(os.path.join(d['folder'], 'flights.npz'))
        indices = Partition_Indices(data, CONST_PARTITION_CODES[partition])
        decision_time = np.full((len(indices), len(stages)), np.nan)
        for row, i in enumerate(indices):
            observed = data.Observed_Time[data.Observed_Offset[i]:data.Observed_Offset[i + 1]]
            for k, n in enumerate(stages):
                if len(observed) >= n:
                    decision_time[row, k] = observed[n - 1]
        sample = Truth_Samples(data.Release_State[indices], data.Floor_Time[indices], truth_area[0], truth_area[1], Flight_Model_Str=flight_model)
        print('[INFO] %s: %d %s throws, %d true samples inside the truth area (Areas.Catch + one step): their catch poses, then %d '
              'predictor(s) at N = %s.'
              % (d['label'], len(indices), partition, len(sample[0]), len(predictors), ', '.join('%d' % n for n in stages)), file=sys.stderr)
        reach = Exact_Reach(sample[2], sample[3], start, speed, accelerations, wrap, Catch_Context_Str=context)
        truth = [[Catchable(sample[0], sample[1], reach[:, a], decision_time[:, k]) for k in range(len(stages))] for a in range(len(accelerations))]
        times, states, _ = Flight_Samples(data.Release_State[indices], data.Floor_Time[indices], Flight_Model_Str=flight_model)
        calibrate = d['calibration'] and partition == 'validation'
        candidates, refused, errors = _Decision_Throws(
            data, indices, stages, predictors, times, states, decision_time, calibrate, area,
            lambda done, total: print('[INFO]   %d of %d throws (%.0f s)' % (done, total, time.perf_counter() - started), file=sys.stderr))
        for key, parts in errors.items():
            pooled[key] += parts
        per_dataset.append((d, indices, truth, candidates, refused, len(sample[0])))

    # 2\ The margins: calibrated on the validation partitions of Decision.Calibration, or the stored ones.
    margin_rows = []
    if partition == 'validation':
        for (seed, k), parts in pooled.items():
            if not parts:
                raise Run_Error_Cls('No prediction of seed %r at N = %d to calibrate the margins on (Decision.Calibration).' % (seed, stages[k]))
            bins, cross, early = (np.concatenate([part[c] for part in parts]) for c in range(3))
            cross_margin, time_margin, counts = Margins(bins, cross, early, decision.Quantile)
            margins[(seed, stages[k])] = (cross_margin, time_margin)
            margin_rows += [(seed, stages[k], b, b * CONST_HORIZON_BIN, int(counts[b]), *cross_margin[b], time_margin[b])
                            for b in range(CONST_HORIZON_BINS)]
    missing = [(seed, n) for seed in seeds for n in stages if (seed, n) not in margins]
    if missing:
        raise Run_Error_Cls('The stored margins hold no seed and N %s.' % missing)

    # 3\ The decisions, their outcomes and the rates.
    per_throw, summary, caches = [], [], [{} for _ in accelerations]
    for d, indices, truth, candidates, refused, _ in per_dataset:
        F = len(indices)
        for seed in seeds:
            for a, acceleration in enumerate(accelerations):
                go = []
                for k, n in enumerate(stages):
                    rows, horizons, positions = candidates[(seed, k)]
                    joined = (np.concatenate(rows), np.concatenate(horizons), np.concatenate(positions)) if rows \
                        else (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros((0, 3)))
                    go.append(Decide(*joined, F, *margins[(seed, n)], grids[a][0], grids[a][1], step, area[0], area[1], bounds=caches[a]))
                made, stage, judged = Stage_Rule(go, truth[a])
                rates = Rates(made, judged)
                label = '%.0f' % np.degrees(acceleration)
                summary.append((d['label'], partition, seed, label, rates['throws'], rates['catchable'], rates['not_catchable'], rates['go'],
                                rates['false_go'], rates['false_go_rate'], rates['false_go_lower'], rates['false_go_upper'], rates['false_no_go'],
                                rates['false_no_go_rate'], rates['false_no_go_lower'], rates['false_no_go_upper'], rates['false_go_of_go'],
                                sum(1 for r in refused[seed] if r), ';'.join('%d:%d' % (n, int(np.sum(stage == k))) for k, n in enumerate(stages))))
                for row in range(F):
                    outcome = ('true_go' if judged[row] else 'false_go') if made[row] else ('false_no_go' if judged[row] else 'true_no_go')
                    per_throw.append((d['label'], '%d' % indices[row], seed, label, 'true' if judged[row] else 'false',
                                      'go' if made[row] else 'no-go', '%d' % stages[stage[row]] if made[row] else '', outcome,
                                      ';'.join(refused[seed][row])))
    return per_throw, summary, margin_rows, {'seeds': seeds, 'throws': [int(len(entry[1])) for entry in per_dataset],
                                             'truth_samples': [int(entry[5]) for entry in per_dataset]}


def _Map(function: tp.Callable[[tp.Any], tp.Any], tasks: tp.List[tp.Any], workers: int,
         progress: tp.Callable[[int, int], None]) -> tp.List[tp.Any]:
    """
    Description:
        The results of a function over tasks, in the tasks' order: in this process with one worker, otherwise in a pool
        of `workers` spawned processes (each task computed on its own, so the results do not depend on the count).

    Args:
        (1) function [Callable]: A module-level function of src (a worker imports it by name).
        (2) tasks [List<Any>]: The tasks, picklable.
        (3) workers [int]: The processes, >= 1.
        (4) progress [Callable[[int, int], None]]: Called after every CONST_PROGRESS_TASKS tasks with (done, all).

    Returns:
        (1) parameter [List<Any>]: The results.
    """

    #   Multiprocessing (the worker pool)
    import multiprocessing
    results = []
    if workers == 1:
        iterator = (function(task) for task in tasks)
        pool = None
    else:
        pool = multiprocessing.get_context('spawn').Pool(workers)
        iterator = pool.imap(function, tasks, chunksize=1)
    try:
        for result in iterator:
            results.append(result)
            if len(results) % CONST_PROGRESS_TASKS == 0 or len(results) == len(tasks):
                progress(len(results), len(tasks))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return results


def _Interception_Setup(configuration: Configuration_Str, training: tp.Any, run: str, context: tp.Any
                        ) -> tp.Tuple[tp.Any, tp.List[str], tp.List[float]]:
    """
    Description:
        The interception's setup for a run (shared by 'catch' and the sweep of scripts/Maintenance/derive.py): the area's map
        (every lattice point of Areas.Catch must qualify), its reach at both accelerations, the run's seeds and the setup without its margins.

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.
        (2) training [Training_Configuration_Str(object)]: The run's training configuration.
        (3) run [string]: The run folder.
        (4) context [Catch_Context_Str(object)]: The catch-pose context of the scene.

    Returns:
        (1) parameter [Interception_Setup_Str(object)]: The setup, its margins not set.
        (2) parameter [List<string>]: The seeds' labels ('' for a numerical predictor).
        (3) parameter [List<float>]: The accelerations [rad/s^2].
    """

    #   ../Catch/Core, Workspace, Decision, Interception - the lattice, the area's map, the reach, the setup
    from Catch.Core import Arrival_Axes, CONST_JOINTS_USED
    from Catch.Workspace import Derive_Workspace, Joint_Wrap
    from Catch.Decision import Area_Lattice, Reach_Grid, Truth_Area
    from Catch.Interception import Interception_Setup_Str

    decision, interception, catch = configuration.Decision, configuration.Interception, configuration.Catch
    truth_area = Truth_Area(configuration.Areas.Catch.Centre, configuration.Areas.Catch.Size, catch.Workspace_Step)
    area = (configuration.Areas.Catch.Centre, configuration.Areas.Catch.Size)
    points, index = Area_Lattice(area[0], area[1], catch.Workspace_Step)
    axes = Arrival_Axes(catch.Arrival_Elevation, catch.Arrival_Azimuth)
    print('[INFO] Checking the %d lattice points of Areas.Catch for %d arrival directions.' % (len(points), len(axes)), file=sys.stderr)
    area_map = Derive_Workspace(points, index, catch.Workspace_Step, axes, Catch_Context_Str=context)
    if not area_map.Qualifies.all():
        raise Run_Error_Cls('%d of the %d lattice points of Areas.Catch do not qualify (the first at %s m); the area comes from '
                            'scripts/Maintenance/derive.py workspace (SPEC §7.11).' % (int((~area_map.Qualifies).sum()), len(points),
                                                         points[np.flatnonzero(~area_map.Qualifies)[0]].tolist()))
    robot_limits = configuration.Robot
    ready, speed = np.array(robot_limits.Ready_Pose), np.array(robot_limits.Joint_Speed)
    accelerations = list(robot_limits.Joint_Accelerations)
    wrap = Joint_Wrap(context)
    grids = [Reach_Grid(area_map, ready[:CONST_JOINTS_USED], speed[:CONST_JOINTS_USED], a, wrap) for a in accelerations]
    seeds = ['%d' % seed for seed in training.Seeds] if training.Model_Name in CONST_LEARNED_MODEL_NAMES else ['']
    setup = Interception_Setup_Str(Areas=configuration.Areas, Truth_Centre=truth_area[0], Truth_Size=truth_area[1],
                                   Criteria=_Catch_Criteria(configuration), Flight=_Flight_Model(configuration),
                                   Observation=_Observation_Model(configuration), Model_Name=training.Model_Name, Run=os.path.abspath(run),
                                   Ready=ready, Speed=speed, Accelerations=accelerations, Grids=grids, Step=catch.Workspace_Step,
                                   Stages=list(interception.Stages), Decision_Stages=list(decision.Stages), Latency=interception.Latency,
                                   Compute_Budget=interception.Compute_Budget, Guard=interception.Guard, Path_Step=interception.Path_Step,
                                   Angle_Tolerance=interception.Angle_Tolerance, Catch_Mode=interception.Catch_Mode,
                                   Stroke_Scales=list(interception.Stroke_Scales), Stroke_Budget=interception.Stroke_Compute_Budget)
    return setup, seeds, accelerations


def _Moving(configuration: Configuration_Str) -> bool:
    """
    Description:
        Whether the configuration catches in motion (Interception.Catch_Mode 'moving'; F15).

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.

    Returns:
        (1) parameter [bool]: Moving.
    """

    return configuration.Interception.Catch_Mode == CONST_CATCH_MODES[1]


def _Run_Folder(configuration: Configuration_Str, parent: str, run_name: str) -> str:
    """
    Description:
        A run's folder of results under a stage's parent folder: <parent>/<Run.Name> for the static catch (where the
        results of F9, F10 and F11 are), <parent>/<Run.Name>/moving for a moving catch, so
        that neither mode writes into, or reads, the other's results. A partition's results are in <folder>/<partition>.

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.
        (2) parent [string]: The stage's parent folder.
        (3) run_name [string]: Run.Name.

    Returns:
        (1) parameter [string]: The folder.
    """

    return os.path.join(parent, run_name, CONST_CATCH_MODES[1]) if _Moving(configuration) else os.path.join(parent, run_name)


def _Decision_Columns_Equal(path: str, rows: tp.Sequence[tp.Sequence[tp.Any]]) -> tp.Optional[bool]:
    """
    Description:
        The reading R1 of a moving catch (F15): whether the decision's columns of the
        interception's rows (Catch.Interception.Decision_Lines: the dataset to the catch time) are, row for row and in
        the same order, the leading fields of the lines of the static catch's stored per_throw.csv of the same partition.
        The decision is the same for both catches, so a difference means the two runs did not decide alike. The stored
        file is read as text, never written; without it the reading is not applied.

    Args:
        (1) path [string]: The static catch's stored per_throw.csv of the interception.
        (2) rows [Sequence<Sequence<Any>>]: The moving catch's interception rows.

    Returns:
        (1) parameter [bool or None]: True equal, False different, None when the file does not exist.
    """

    #   ../Catch/Interception - the decision's columns as the file holds them
    from Catch.Interception import Decision_Lines, CONST_DECISION_COLUMNS

    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='ascii', newline='') as file:
        stored = [','.join(line.rstrip('\n').split(',')[:CONST_DECISION_COLUMNS]) for line in file][1:]
    return stored == Decision_Lines(rows)


def _Catch_Targets(configuration: Configuration_Str, run_name: str, out: tp.Optional[str]) -> tp.Tuple[tp.Dict[str, str], str]:
    """
    Description:
        Where 'catch' writes (SPEC §9.12-9.14): per stage the run's folder, <Decision.Output_Folder>/<Run.Name>,
        <Interception.Output_Folder>/<Run.Name> and <Contact.Output_Folder>/<Run.Name>, and the ledger
        Contact.Ledger_File; with --out DIR the folders DIR/Decisions/<Run.Name>, DIR/Interceptions/<Run.Name> and
        DIR/Catches/<Run.Name> and the ledger's file name in DIR. A partition's results are
        in <folder>/<partition>. A moving catch's folders are <...>/<Run.Name>/moving (_Run_Folder; F15 Q6); the
        ledger is the same.

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.
        (2) run_name [string]: Run.Name.
        (3) out [string]: --out DIR, or None.

    Returns:
        (1) parameter 1 [Dict<string, string>]: Per stage of CONST_STAGE_FOLDERS the run's folder.
        (2) parameter 2 [string]: The ledger's file.
    """

    configured = {'decision': configuration.Decision.Output_Folder, 'interception': configuration.Interception.Output_Folder,
                  'catch': configuration.Contact.Output_Folder}
    if out is None:
        return ({stage: _Run_Folder(configuration, os.path.join(CONST_REPO_ROOT, configured[stage]), run_name) for stage in CONST_STAGE_FOLDERS},
                os.path.join(CONST_REPO_ROOT, configuration.Contact.Ledger_File))
    return ({stage: _Run_Folder(configuration, os.path.join(out, folder), run_name) for stage, folder in CONST_STAGE_FOLDERS.items()},
            os.path.join(out, os.path.basename(configuration.Contact.Ledger_File)))


def _Stage_Fields(configuration: Configuration_Str, config_bytes: bytes, datasets: tp.List[tp.Dict[str, tp.Any]],
                  system_sha256: str) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    """
    Description:
        Per stage of 'catch' the fields of its manifest that a stored validation result must share with a call on the
        test partition (SPEC §9.12-9.14): the decision's (the run's configuration, the system configuration, the
        datasets, the truth area), the interception's (those, the ladder, its calibration, the quantile and the
        planner's values, _Planner_Fields) and the catch's (the planner's fields, the system configuration and the
        contact's values).

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.
        (2) config_bytes [bytes]: The run's training configuration copy.
        (3) datasets [List<Dict<string, Any>>]: The decision's datasets (_Decision_Datasets).
        (4) system_sha256 [string]: The SHA-256 of the system configuration file.

    Returns:
        (1) parameter [Dict<string, Dict<string, Any>>]: Per stage of CONST_STAGE_FOLDERS the fields by their names.
    """

    contact = configuration.Contact
    planner = _Planner_Fields(configuration, config_bytes, datasets)
    decision = {'run_configuration': planner['run_configuration'], 'system_configuration': system_sha256, 'datasets': planner['datasets'],
                'truth_area': planner['truth_area']}
    return {'decision': decision, 'interception': dict(decision, **planner),
            'catch': dict(planner, system_configuration=system_sha256, step_s=contact.Step, time_constant_s=contact.Time_Constant,
                          damping_ratio=contact.Damping_Ratio, friction=list(contact.Friction), settle_time_s=contact.Settle_Time,
                          rest_speed_m_s=contact.Rest_Speed, rest_time_s=contact.Rest_Time)}


def _Catch(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'catch' (F9, F10, F11; SPEC §2.8-2.10, §7.11, §9.12-9.14): the whole chain of
        a run's predictor on the simulated throws of Decision.Datasets, one partition, in three stages. (1) The
        decision (_Decision_Stage; F9). (2) The interception (F10): per throw the truth, the decision at every observed
        sample from Interception.Stages' first with the margins of the largest stage <= N, F9's stages with the same
        margins (the baseline), and per acceleration with a go the re-planned motion, its result at the true crossing
        of the cup's mouth plane, its path checked by exact distances and its tracking by the servos. (3) The physical
        catch (F11): every go simulated in the scene with the configured contact and classified. Stages 2 and 3 are one
        pass over the throws (Catch.Physical.Catch_Chunk gives the interception's rows and the catch's), so the
        interception is computed once.
        --partition validation (the default) calibrates the decision's margins on the validation partitions of
        Decision.Calibration and the ladder's margins on the TRAIN partitions of Interception.Calibration, both at
        Decision.Quantile, and simulates the nominal contact for every seed and Contact.Levels for the first; a moving
        catch also simulates the first seed's contacts a second time with the ball's markers as collision shapes (the
        markers variant, in memory).
        --partition test takes the margins of the stored validation decision and interception, simulates the nominal
        contact only, refuses a tree or a run with uncommitted changes and a missing or differing validation result of
        any stage, and first appends one row to Contact.Ledger_File (every call counted).
        R2: every throw has exactly one class and the robot touches nothing but the ball. R3: the retained shares with
        their 95 % intervals are descriptive. A moving catch (Interception.Catch_Mode 'moving';
        F15) plans every go with a stroke along the ball's path where one is in time and certified, writes into
        <folder>/moving/<partition>/ with its further columns, takes on the test partition the stored validation results
        of the same mode, and is read by R1 (the decision's columns of every row equal those of the static catch's stored
        rows of the same partition, <folder>/<partition>/per_throw.csv of the interception; not applied without them)
        and by the interception's R2 for a stroke (within the limits, certified to its end at rest, at rest at the
        stroke's start before it begins, the crossing inside the stroke). --workers N runs the throws of stages 2 and 3
        in N processes; the outputs are the same for every N. Writes, at the end, <folder>/<partition>/ per stage (_Catch_Targets): the decision's
        per_throw.csv, summary.csv, margins.csv (validation) and manifest.json; the interception's the same four; the
        catch's per_throw.csv, summary.csv and manifest.json, which holds the SHA-256 of the interception's rows and of
        the margins. An existing folder of any stage is refused before anything is computed. No recording is read.

    Args:
        (1) args [argparse.Namespace]: --run, --partition, --config, --out, --workers.

    Returns:
        (1) parameter [int]: 0 written and R2 (a moving catch: and R1) holds; 1 refused (nothing written; a test-partition
                             call already counted stays counted), or written with a reading violated (named, and in the
                             manifests).
    """

    #   ../Scene/Core - the scene and its robot
    from Scene.Core import Build_Scene, Scene_Robot, Scene_Error_Cls
    #   ../Scene/Ball - the ball's error
    from Scene.Ball import Ball_Error_Cls
    #   ../Robot/Core, Motion - the robot's and the motion's errors
    from Robot.Core import Robot_Error_Cls
    from Robot.Motion import Motion_Error_Cls
    #   ../Catch/Core, Decision, Interception, Physical - the catch pose, the decision's and the margins' files, the
    #   ladder's calibration and the interception's files, the physical catch
    from Catch.Core import Catch_Context, Catch_Error_Cls
    from Catch.Decision import (Read_Margins, Write_Margins, Write_Per_Throw as Write_Decision, Write_Summary as Write_Decision_Summary,
                                CONST_HORIZON_BIN, CONST_HORIZON_BINS)
    from Catch.Interception import (Calibrate_Chunk, Ladder_Margins, Summary_Rows as Interception_Summary,
                                    Write_Per_Throw as Write_Interception, Write_Summary as Write_Interception_Summary,
                                    CONST_SUMMARY_COLUMNS as CONST_INTERCEPTION_SUMMARY_COLUMNS, CONST_MOVING_SUMMARY_COLUMNS)
    from Catch.Physical import (Catch_Chunk, Contact_Str, Summary_Rows as Catch_Summary, Write_Per_Throw, Write_Summary, CONST_CLASSES,
                                CONST_PER_THROW_COLUMNS, CONST_SUMMARY_COLUMNS)

    started = time.perf_counter()
    partition = args.partition
    try:
        configuration = Load_Configuration(args.config)
        moving = _Moving(configuration)
        if args.workers < 1:
            raise Run_Error_Cls('--workers must be an integer >= 1, found %d.' % args.workers)
        decision, interception, contact = configuration.Decision, configuration.Interception, configuration.Contact
        training, run_manifest, config_bytes = _Load_Run_Copy(args.run)
        if run_manifest.get('device') not in (None, 'cpu'):
            raise Run_Error_Cls('The run %s was trained on the device %s; every reported number comes from the CPU (SPEC §7.8). Train it with '
                                'Run.Device cpu.' % (args.run, run_manifest.get('device')))
        datasets = _Decision_Datasets(configuration)
        folders, ledger = _Catch_Targets(configuration, training.Name, args.out)
        targets = {stage: os.path.join(folder, partition) for stage, folder in folders.items()}
        existing = [target for target in targets.values() if os.path.exists(target)]
        if existing:
            raise Run_Error_Cls('The folder %s exists; a decision, an interception and a catch are never overwritten. Use --out or remove '
                                'it.' % ', '.join(existing))
        system_sha256, commit, dirty = Sha256_File(args.config), *_Git_State()
        fields = _Stage_Fields(configuration, config_bytes, datasets, system_sha256)
        stored_margins = {}
        decision_margins: tp.Dict[tp.Tuple[str, int], tp.Tuple[np.ndarray, np.ndarray]] = {}
        margins: tp.Dict[tp.Tuple[str, int], tp.Tuple[np.ndarray, np.ndarray]] = {}
        margins_file = os.path.join(targets['interception'], 'margins.csv')
        if partition == 'test':
            if run_manifest.get('git_dirty') is not False or dirty:
                raise Run_Error_Cls('catch --partition test reads the simulated test partitions: it runs only from a committed tree on a '
                                    'run of a committed tree (git_dirty: the tree %s, the run %s). Commit first (owner, 2026-11-22, F11 Q9).'
                                    % (dirty, run_manifest.get('git_dirty')))
            margins_file = os.path.join(folders['interception'], 'validation', 'margins.csv')
            for stage in CONST_STAGE_FOLDERS:
                source = os.path.join(folders[stage], 'validation')
                try:
                    stored = Read_Manifest(os.path.join(source, 'manifest.json'))
                    if stage == 'decision':
                        decision_margins = Read_Margins(os.path.join(source, 'margins.csv'))
                    elif stage == 'interception':
                        margins = Read_Margins(margins_file)
                    if stage != 'catch':
                        stored_margins[os.path.join(source, 'margins.csv')] = Sha256_File(os.path.join(source, 'margins.csv'))
                except (OSError, ValueError, Catch_Error_Cls) as error:
                    raise Run_Error_Cls('The test partition is caught with the margins of the validation partition, after its catch: the '
                                        'validation %s %s cannot be read (%s). Run catch on the validation partition first.'
                                        % (stage, source, error)) from error
                differing = [key for key, value in fields[stage].items() if stored.get(key) != value]
                if differing:
                    raise Run_Error_Cls('The stored validation %s %s differs from this call in %s; run catch on the validation partition '
                                        'again with this configuration and run.' % (stage, source, ', '.join(differing)))
            os.makedirs(os.path.dirname(ledger) or '.', exist_ok=True)
            counted = Append_Test_Ledger(ledger, training.Name, training.Model_Name, system_sha256, fields['catch']['datasets'], commit, dirty)
            print('[INFO] Catch test read %d recorded in %s (%s).' % (counted, ledger, commit[:12]), file=sys.stderr)
        model = Build_Scene(Areas_Str=configuration.Areas)
        context = Catch_Context(model, Scene_Robot(model), Criteria_Str=_Catch_Criteria(configuration))
        setup, seeds, accelerations = _Interception_Setup(configuration, training, args.run, context)
    except (Configuration_Error_Cls, Run_Error_Cls, Dataset_Error_Cls, Scene_Error_Cls, Robot_Error_Cls, Catch_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    for level, message in configuration.Notes + training.Notes:
        print('[%s] %s' % (level, message), file=sys.stderr)
    report = (lambda what: (lambda done, total: print('[INFO]   %s: %d of %d tasks (%.0f s)' % (what, done, total, time.perf_counter() - started),
                                                      file=sys.stderr)))
    nominal = (contact.Time_Constant, contact.Damping_Ratio)
    settings = (lambda levels, markers: Contact_Str(Step=contact.Step, Friction=list(contact.Friction), Settle_Time=contact.Settle_Time,
                                                    Rest_Speed=contact.Rest_Speed, Rest_Time=contact.Rest_Time, Levels=levels,
                                                    Markers=markers))
    try:
        # 1\ The decision (F9), with the reach of the area's map that the setup holds.
        print('[INFO] Stage 1 of 3, the decision at N = %s.' % ', '.join('%d' % n for n in decision.Stages), file=sys.stderr)
        decided = _Decision_Stage(configuration, training, args.run, partition, datasets, context, setup.Grids, decision_margins, started)
        # 2\ The ladder's margins: calibrated on the train partitions of Interception.Calibration, or the stored ones.
        margin_rows = []
        if partition == 'validation':
            tasks = []
            for d in datasets:
                if d['path'] not in interception.Calibration:
                    continue
                data = Read_Dataset(os.path.join(d['folder'], 'flights.npz'))
                indices = Partition_Indices(data, CONST_PARTITION_TRAIN)
                tasks += [(seed, (setup, os.path.join(d['folder'], 'flights.npz'), indices[k:k + CONST_CHUNK_THROWS], seed))
                          for seed in seeds for k in range(0, len(indices), CONST_CHUNK_THROWS)]
            print('[INFO] Stage 2 of 3, the interception: calibrating the ladder N = %s on the train partitions of %s (%d tasks, %d worker(s)).'
                  % (', '.join('%d' % n for n in interception.Stages), ', '.join(interception.Calibration), len(tasks), args.workers),
                  file=sys.stderr)
            results = _Map(Calibrate_Chunk, [task for _, task in tasks], args.workers, report('calibration'))
            for seed in seeds:
                pooled = {n: [] for n in interception.Stages}
                for (owner, _), errors in zip(tasks, results):
                    if owner == seed:
                        for n in interception.Stages:
                            pooled[n] += errors[n]
                for n, (cross_margin, time_margin, counts) in Ladder_Margins(pooled, decision.Quantile).items():
                    margins[(seed, n)] = (cross_margin, time_margin)
                    margin_rows += [(seed, n, b, b * CONST_HORIZON_BIN, int(counts[b]), *cross_margin[b], time_margin[b])
                                    for b in range(CONST_HORIZON_BINS)]
        missing = [(seed, n) for seed in seeds for n in interception.Stages if (seed, n) not in margins]
        if missing:
            raise Run_Error_Cls('The margins hold no seed and N %s.' % missing)
        setup.Margins = margins
        # 3\ The throws of the partition, per dataset, seed and chunk: the interception and the catch of every go.
        tasks, throws = [], []
        for d in datasets:
            data = Read_Dataset(os.path.join(d['folder'], 'flights.npz'))
            indices = Partition_Indices(data, CONST_PARTITION_CODES[partition])
            throws.append(int(len(indices)))
            for seed in seeds:
                varied = partition == 'validation' and seed == seeds[0]
                levels = [nominal] + (list(contact.Levels) if varied else [])
                for k in range(0, len(indices), CONST_CHUNK_THROWS):
                    tasks.append((setup, d['label'], os.path.join(d['folder'], 'flights.npz'), indices[k:k + CONST_CHUNK_THROWS], seed,
                                  settings(levels, moving and varied), None))
        print('[INFO] Stages 2 and 3 of 3, the interception and the catch of the %s partitions of %s: %s throws, %d seed(s), %d tasks, %d '
              'worker(s).' % (partition, ', '.join(d['label'] for d in datasets), ' + '.join('%d' % n for n in throws), len(seeds), len(tasks),
                              args.workers), file=sys.stderr)
        results = _Map(Catch_Chunk, tasks, args.workers, report('catch'))
    except (Run_Error_Cls, Motion_Error_Cls, Catch_Error_Cls, Dataset_Error_Cls, Ball_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    intercepted = [row for rows, _ in results for row in rows]
    intercepted_summary = Interception_Summary(intercepted, partition, decision.Stages)
    caught = [row for _, rows in results for row in rows]
    per_throw = [tuple(row) for row in caught]
    summary = [tuple(row) for row in Catch_Summary(caught, partition)]

    # 4\ The reading fixed before the data. R2: every throw has exactly one class, and the robot touched nothing but the ball.
    column = {key: k for k, key in enumerate(CONST_PER_THROW_COLUMNS)}
    expected = sum(len(task[3]) * len(accelerations) * len(task[5].Levels) * (2 if task[5].Markers else 1) for task in tasks)
    unclassified = sum(1 for row in per_throw if row[column['class']] not in CONST_CLASSES)
    touching = [row for row in per_throw if row[column['robot_contact']]]
    r2 = len(per_throw) == expected and unclassified == 0 and not touching
    if not r2:
        print('[ERROR] R2 is violated: %d rows for %d throws, accelerations and contacts; %d rows without one of the classes; %d goes whose '
              'robot touched something but the ball (the first: %s).'
              % (len(per_throw), expected, unclassified, len(touching), touching[0][column['robot_contact']] if touching else '-'),
              file=sys.stderr)
    # A moving catch's readings (F15). R1: the decision's columns of every row are those of the static
    # catch's stored rows of the same partition (read as text, never written). The interception's R2: no violation.
    r1: tp.Optional[bool] = None
    violations = sum(int(row[CONST_INTERCEPTION_SUMMARY_COLUMNS.index('r2_violations')]) for row in intercepted_summary)
    static_rows = os.path.join(os.path.dirname(folders['interception']), partition, 'per_throw.csv')
    if moving:
        try:
            r1 = _Decision_Columns_Equal(static_rows, intercepted)
        except OSError as error:
            print('[ERROR] The static catch\'s stored rows %s cannot be read (%s).' % (static_rows, error), file=sys.stderr)
            return 1
        if r1 is False:
            print('[ERROR] R1 is violated: the decision\'s columns of the %d rows differ from those of the static catch\'s stored rows %s; '
                  'the decision is the same for both catches (SPEC §9.13).' % (len(intercepted), static_rows), file=sys.stderr)
        if violations:
            print('[ERROR] The interception\'s R2 is violated by %d planned goes (summary.csv, r2_violations).' % violations, file=sys.stderr)

    # 5\ The files of the three stages.
    run_sha256 = Sha256_File(os.path.join(args.run, 'manifest.json'))
    shared = dict(run_name=training.Name, model=training.Model_Name, partition=partition, dataset_paths=[d['path'] for d in datasets],
                  accelerations_deg_s2=[float(np.degrees(a)) for a in accelerations])

    def Write_Stage(folder: str, files: tp.List[tp.Tuple[str, tp.Callable[[str], None]]], manifest: tp.Dict[str, tp.Any]) -> None:
        """
        Description:
            A stage's files and its manifest, which holds each file's SHA-256.

        Args:
            (1) folder [string]: The folder.
            (2) files [List<Tuple<string, Callable>>]: Per file its name and the function that writes it to a path.
            (3) manifest [Dict<string, Any>]: The stage's manifest.
        """

        for name, write in files:
            write(os.path.join(folder, name))
        manifest['sha256'] = dict(manifest['sha256'], **{name: Sha256_File(os.path.join(folder, name)) for name, _ in files})
        Write_Manifest(os.path.join(folder, 'manifest.json'), manifest)

    decision_files = [('per_throw.csv', lambda path: Write_Decision(path, decided[0])),
                      ('summary.csv', lambda path: Write_Decision_Summary(path, decided[1]))]
    if decided[2]:
        decision_files.append(('margins.csv', lambda path: Write_Margins(path, decided[2])))
    decision_manifest = Build_Manifest('decision', {'run_manifest': run_sha256},
                                       dict(fields['decision'], seeds=decided[3]['seeds'], stages=list(decision.Stages),
                                            quantile=decision.Quantile, calibration=list(decision.Calibration), throws=decided[3]['throws'],
                                            truth_samples=decided[3]['truth_samples'], **shared), commit, dirty)
    interception_files = [('per_throw.csv', lambda path: Write_Interception(path, intercepted, moving)),
                          ('summary.csv', lambda path: Write_Interception_Summary(path, intercepted_summary, moving))]
    if margin_rows:
        interception_files.append(('margins.csv', lambda path: Write_Margins(path, margin_rows)))
    readings = dict(r1_decision_columns_equal_static=r1, r2_violations=violations) if moving else {}
    interception_manifest = Build_Manifest('interception', {'run_manifest': run_sha256}, dict(fields['interception'], seeds=seeds, throws=throws,
                                                                                               **readings, **shared), commit, dirty)
    try:
        _Write_Folder(targets['decision'], lambda folder: Write_Stage(folder, decision_files, decision_manifest))
        _Write_Folder(targets['interception'], lambda folder: Write_Stage(folder, interception_files, interception_manifest))
        catch_manifest = Build_Manifest('catch', {'run_manifest': run_sha256, 'interception_margins': Sha256_File(margins_file),
                                                  'interception_per_throw': Sha256_File(os.path.join(targets['interception'], 'per_throw.csv'))},
                                        dict(fields['catch'], seeds=seeds, throws=throws,
                                             levels=[list(level) for level in contact.Levels] if partition == 'validation' else [],
                                             **(dict(markers_variant=partition == 'validation') if moving else {}),
                                             r2_one_class_and_no_robot_contact=r2, **shared), commit, dirty)
        catch_files = [('per_throw.csv', lambda path: Write_Per_Throw(path, per_throw, False, moving)),
                       ('summary.csv', lambda path: Write_Summary(path, summary, False, moving))]
        _Write_Folder(targets['catch'], lambda folder: Write_Stage(folder, catch_files, catch_manifest))
        changed = [path for path, digest in stored_margins.items() if Sha256_File(path) != digest]
        if changed:
            raise Run_Error_Cls('The stored margins %s changed while the catch ran; they are read, never written.' % ', '.join(changed))
    except (Run_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1

    # 6\ The report: the three tables, then the readings.
    stages = decision.Stages
    print('%-11s %-5s %4s %5s %6s %7s  %-34s %-34s %s' % ('dataset', 'seed', 'acc', 'N', 'throws', 'go', 'false go [95 % CP]',
                                                        'false no-go [95 % CP]', 'go by stage'))
    for row in decided[1]:
        false_go = '%d/%d %.3f%% [%.3f, %.3f]' % (row[8], row[6], 100 * row[9], 100 * row[10], 100 * row[11])
        false_no_go = '%d/%d %.2f%% [%.2f, %.2f]' % (row[12], row[5], 100 * row[13], 100 * row[14], 100 * row[15])
        print('%-11s %-5s %4s %5s %6d %7d  %-34s %-34s %s' % (row[0], row[2] or '-', row[3], '/'.join('%d' % n for n in stages), row[4], row[7],
                                                            false_go, false_no_go, row[18]))
    print('%-11s %-5s %4s %6s %6s  %-30s %-30s %-9s %-30s %s' % ('dataset', 'seed', 'acc', 'throws', 'go', 'false go [95 % CP]',
                                                              'false no-go [95 % CP]', 'F9 rule', 'intercepted of catchable', 'R2 viol.'))
    for row in intercepted_summary:
        print('%-11s %-5s %4s %6d %6d  %-30s %-30s %-9s %-30s %d' % (
            row[0], row[2] or '-', row[3], row[4], row[7], '%d/%d %.3f%% [%.3f, %.3f]' % (row[8], row[6], 100 * row[9], 100 * row[10], 100 * row[11]),
            '%d/%d %.2f%% [%.2f, %.2f]' % (row[12], row[5], 100 * row[13], 100 * row[14], 100 * row[15]), '%.2f%%' % (100 * row[16]),
            '%d %.2f%% [%.2f, %.2f]' % (row[18], 100 * row[19], 100 * row[20], 100 * row[21]), row[25]))
    key = {key: k for k, key in enumerate(CONST_SUMMARY_COLUMNS)}
    print('%-11s %-4s %4s %-14s %6s %6s %6s  %-28s %-28s %7s %s' % ('dataset', 'seed', 'acc', 'contact', 'throws', 'go', 'f. go',
                                                                   'retained of the goes [95 %]', 'retained of catchable [95 %]', 'at rest',
                                                                   'bounced / rim / missed / hit'))
    for row in summary:
        get = (lambda name: row[key[name]])
        print('%-11s %-4s %4s %-14s %6d %6d %6d  %-28s %-28s %7d %d / %d / %d / %d' % (
            get('dataset'), get('seed') or '-', get('acceleration_deg_s2'),
            '%g, %g%s' % (get('time_constant_s'), get('damping_ratio'), ' +m' if moving and row[-1] == 'true' else ''),
            get('throws'), get('go'), get('false_go'),
            '%d %.1f%% [%.1f, %.1f]' % (get('retained'), 100 * get('retained_of_go'), 100 * get('retained_of_go_lower'),
                                       100 * get('retained_of_go_upper')),
            '%d %.1f%% [%.1f, %.1f]' % (get('retained_catchable'), 100 * get('retained_of_catchable'), 100 * get('retained_of_catchable_lower'),
                                       100 * get('retained_of_catchable_upper')),
            get('at_rest'), get('bounced_out'), get('rim'), get('missed'), get('hit_robot') + get('hit_stand')))
    if moving:
        extra = {key: len(CONST_INTERCEPTION_SUMMARY_COLUMNS) + k for k, key in enumerate(CONST_MOVING_SUMMARY_COLUMNS)}
        print('%-11s %-5s %4s %6s %8s %11s %-22s %s' % ('dataset', 'seed', 'acc', 'go', 'strokes', 'mean scale', 'cup speed p50 [m/s]',
                                                      'relative speed p50 / p95 [m/s]'))
        for row in intercepted_summary:
            print('%-11s %-5s %4s %6d %8d %11.2f %-22.2f %.2f / %.2f' % (row[0], row[2] or '-', row[3], row[7], row[extra['strokes']],
                                                                       row[extra['stroke_scale_mean']], row[extra['cup_speed_p50_m_s']],
                                                                       row[extra['relative_speed_p50_m_s']], row[extra['relative_speed_p95_m_s']]))
        print('[INFO] A moving catch (Interception.Catch_Mode). R1 (the decision\'s columns equal the static catch\'s stored rows %s): %s. The '
              'interception\'s R2 (within the limits, certified to the stroke\'s end, at rest at its start, the crossing inside it): %s.'
              % (static_rows, {None: 'not applied, no stored static rows', True: 'holds', False: 'VIOLATED'}[r1],
                 'holds' if not violations else 'VIOLATED by %d goes' % violations), file=sys.stderr)
    print('[INFO] R2 (every throw has one class; the robot touches nothing but the ball): %s. R3: the retained shares are descriptive. The '
          'interception\'s rows and the catch come from one pass (their SHA-256 is in the catch\'s manifest).' % ('holds' if r2 else 'VIOLATED'),
          file=sys.stderr)
    print('[INFO] Wrote %s in %.0f s (%s partition%s). The ball\'s contact is assumed (SPEC §4.14). Simulated throws only (SPEC §2.14). %s'
          % (', '.join(targets[stage] for stage in CONST_STAGE_FOLDERS), time.perf_counter() - started, partition,
             ', counted' if partition == 'test' else ', the decision\'s margins in-sample, the ladder\'s from the train partitions', CONST_SAFETY),
          file=sys.stderr)
    return 0 if r2 and r1 is not False and not (moving and violations) else 1


def _Planner_Fields(configuration: Configuration_Str, config_bytes: bytes, datasets: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
    """
    Description:
        What a stored interception must agree with before its margins or rows are used (by 'catch' and 'view'): the run's
        configuration, the datasets, the truth area, the ladder, the calibration, the quantile, the decision's stages and
        the planner's values, as the interception's manifest holds them (SPEC §9.13). A moving catch's also hold the
        mode, the stroke's scales and its compute budget (F15); the static catch's are those of F10, unchanged.

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.
        (2) config_bytes [bytes]: The run's training configuration copy.
        (3) datasets [List<Dict<string, Any>>]: The decision's datasets (_Decision_Datasets).

    Returns:
        (1) parameter [Dict<string, Any>]: The fields by their names in the manifest.
    """

    #   ../Catch/Decision - the truth area
    from Catch.Decision import Truth_Area

    decision, interception, catch = configuration.Decision, configuration.Interception, configuration.Catch
    truth_area = Truth_Area(configuration.Areas.Catch.Centre, configuration.Areas.Catch.Size, catch.Workspace_Step)
    fields = {'run_configuration': Sha256_Bytes(config_bytes), 'datasets': [d['hash'] for d in datasets],
              'truth_area': [truth_area[0], truth_area[1]], 'stages': list(interception.Stages),
              'calibration': list(interception.Calibration), 'quantile': decision.Quantile, 'decision_stages': list(decision.Stages),
              'latency_s': interception.Latency, 'compute_budget_s': interception.Compute_Budget, 'guard_s': interception.Guard,
              'path_step_m': interception.Path_Step, 'angle_tolerance_deg': float(np.degrees(interception.Angle_Tolerance))}
    if _Moving(configuration):
        fields.update(catch_mode=interception.Catch_Mode, stroke_scales=list(interception.Stroke_Scales),
                      stroke_compute_budget_s=interception.Stroke_Compute_Budget)
    return fields


def _Stored_Seeds() -> tp.Dict[int, str]:
    """
    Description:
        The seeds of the stored datasets (Data/Generated/Datasets/*/manifest.json), each with its dataset's folder: a new
        throw of 'view' never takes one of them, so that it is never a stored throw.

    Returns:
        (1) parameter [Dict<int, string>]: The folder's name by seed.
    """

    seeds: tp.Dict[int, str] = {}
    parent = os.path.join(CONST_REPO_ROOT, 'Data', 'Generated', 'Datasets')
    for name in sorted(os.listdir(parent)) if os.path.isdir(parent) else []:
        try:
            seeds[int(Read_Manifest(os.path.join(parent, name, 'manifest.json'))['seed'])] = name
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return seeds


def _New_Throw(configuration: Configuration_Str, dataset_config: str, seed: int, aim: str, folder: str) -> str:
    """
    Description:
        A new simulated throw for 'view': one flight (index 0) generated as 'dataset' generates
        (Dataset.Core.Generate_Dataset: the throw, M4's error and the composed dropouts from the identification flights'
        donors; the validation recordings are not read) from the dataset configuration given with the seed given, aimed
        into the catch area (Throw.Target_Catch_Area with margin 0, the target of Catch_In) or anywhere (the file's own
        Throw.Target_Area), and written as flights.npz into `folder`. The same seed gives the same throw. A seed of a
        stored dataset is refused: with it the throw would be that dataset's flight 0.

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.
        (2) dataset_config [string]: The dataset configuration file.
        (3) seed [int]: The seed.
        (4) aim [string]: 'catch' or 'anywhere'.
        (5) folder [string]: The folder flights.npz is written into.

    Returns:
        (1) parameter [string]: The file.
    """

    stored = _Stored_Seeds()
    if seed in stored:
        raise Run_Error_Cls('--new %d is the seed of the stored dataset %s: the throw would be that dataset\'s flight 0, not a new one. '
                            'Choose another seed, or show the stored throw with --dataset and --index.' % (seed, stored[seed]))
    dataset = Load_Dataset_Configuration(dataset_config)
    if aim == 'catch':
        dataset.Throw.Target_Shape, dataset.Throw.Target_Margin = 'catch_area', 0.0
    elif dataset.Throw.Target_Shape != 'quarter_sphere':
        raise Run_Error_Cls('--aim anywhere needs a dataset configuration with Throw.Target_Area (the quarter sphere); %s gives '
                            'Throw.Target_Catch_Area. Name another file with --dataset-config, or use --aim catch.' % dataset_config)
    dataset.Seed, dataset.Flights = int(seed), 1
    data = Generate_Dataset(dataset, configuration, Load_Donors(CONST_RECORDINGS_DIR), Flight_Model_Str=_Flight_Model(configuration),
                            Observation_Model_Str=_Observation_Model(configuration))
    path = os.path.join(folder, 'flights.npz')
    Write_Dataset(path, data)
    return path


def _Stored_Line(path: str, prefix: str) -> tp.Optional[str]:
    """
    Description:
        The line of a stored per_throw.csv that begins with a prefix (a throw's dataset, flight, seed, acceleration and,
        for a catch, contact), without its line end; None when the file holds none.

    Args:
        (1) path [string]: The file.
        (2) prefix [string]: The line's first fields, each followed by a comma.

    Returns:
        (1) parameter [string or None]: The line.
    """

    with open(path, 'r', encoding='ascii', newline='') as file:
        for line in file:
            if line.startswith(prefix):
                return line.rstrip('\n')
    return None


def _Stored_Check(record: tp.Any, interceptions: str, catches: str) -> tp.Tuple[tp.Optional[bool], str]:
    """
    Description:
        Compare a recorded throw with the stored results of the same run ('view'; SPEC §7.13): the interception's row and
        the catch's row with the nominal contact, written by their own writers, against the lines of the stored validation
        per_throw.csv of the interception and of the catch (written by 'catch') for the same dataset, flight, seed and
        acceleration, byte for byte. The stored files are read, never written. Without stored lines for the throw (a train
        throw, a new throw, a dataset outside Decision.Datasets, no stored result) nothing is compared.

    Args:
        (1) record [Throw_Record_Str(object)]: The record.
        (2) interceptions [string]: The stored validation interception's folder.
        (3) catches [string]: The stored validation catch's folder.

    Returns:
        (1) parameter [bool or None]: True equal, False different, None not compared.
        (2) parameter [string]: What was found.
    """

    #   ../Catch/Interception, Physical - the two files' writers
    from Catch.Interception import Write_Per_Throw as Write_Interception, CONST_PER_THROW_COLUMNS as CONST_INTERCEPTION_COLUMNS
    from Catch.Physical import Write_Per_Throw as Write_Catch

    found = []
    moving = len(record.Interception_Row) > len(CONST_INTERCEPTION_COLUMNS)
    with tempfile.TemporaryDirectory() as folder:
        for name, write, row, stored, fields in (('interception', lambda path, rows: Write_Interception(path, rows, moving),
                                                  record.Interception_Row, interceptions, 4),
                                                 ('catch', lambda path, rows: Write_Catch(path, rows, False, moving), record.Catch_Row, catches, 6)):
            path = os.path.join(stored, 'per_throw.csv')
            if not os.path.isfile(path):
                return None, 'no stored %s (%s is absent)' % (name, path)
            write(os.path.join(folder, name + '.csv'), [row])
            with open(os.path.join(folder, name + '.csv'), 'r', encoding='ascii', newline='') as file:
                mine = file.read().splitlines()[1]
            theirs = _Stored_Line(path, ','.join(mine.split(',')[:fields]) + ',')
            if theirs is None:
                return None, 'the stored %s holds no row of this throw (%s)' % (name, path)
            if mine != theirs:
                differing = [k for k, (a, b) in enumerate(zip(mine.split(','), theirs.split(','))) if a != b]
                return False, 'the %s\'s row differs from the stored one in %s in the field(s) %s:\n  now    %s\n  stored %s' % (
                    name, path, differing, mine, theirs)
            found.append(name)
    return True, 'the interception\'s row and the catch\'s row of this throw equal the stored validation rows byte for byte'


def _View_Frames(record: tp.Any, configuration: Configuration_Str, args: argparse.Namespace, size: tp.Tuple[int, int]) -> int:
    """
    Description:
        Write a recorded throw as frames ('view --frames DIR'; SPEC §9.15): DIR/frame_<number>.png, rendered offscreen at
        --size from the --camera preset in the --theme, one per 1 / --fps of video time at --speed, from --from to --to,
        and DIR/frames.csv with what each frame shows. The frames are clean (the scene, the zones, the trajectory and its
        labels); --overlay burns in the text overlay and the legend; --markers draws the ball's marked mesh instead of
        its collision sphere. DIR is written through a temporary folder; an existing DIR is refused.

    Args:
        (1) record [Throw_Record_Str(object)]: The record.
        (2) configuration [Configuration_Str(object)]: The system configuration (the areas).
        (3) args [argparse.Namespace]: The options.
        (4) size [Tuple<int, int>]: The frames' width and height [px].

    Returns:
        (1) parameter [int]: The number of frames written.
    """

    #   MuJoCo (the model's data shown) [pip3 install mujoco]
    import mujoco
    #   ../View/Core, Theme - the frames, the decorations, the offscreen renderer, the two files; the themed scene
    from View.Core import (Frame_At, Frame_Times, Frame_Row, Overlay_Lines, Legend, Decorations, Ghost_Positions, Offscreen_Cls, Write_Png,
                           Write_Frames, CONST_THEMES, CONST_CAMERAS)
    from View.Theme import Themed_Scene

    theme, camera = CONST_THEMES[args.theme], CONST_CAMERAS[args.camera]
    times = Frame_Times(record, args.fps, args.speed, args.start, args.end)
    model = Themed_Scene(Areas_Str=configuration.Areas, Theme_Str=theme, width=size[0], height=size[1], markers=args.markers)
    data = mujoco.MjData(model)

    def Write(folder: str) -> None:
        """
        Description:
            The frames and frames.csv.

        Args:
            (1) folder [string]: The folder.
        """

        offscreen = Offscreen_Cls(model, size[0], size[1])
        rows = []
        try:
            for number, t in enumerate(times):
                frame = Frame_At(record, float(t))
                data.qpos[:] = record.State_Position[frame.State]
                mujoco.mj_forward(model, data)
                decorations = Decorations(record, frame, Theme_Str=theme, Areas_Str=configuration.Areas, Camera_Str=camera,
                                          zones=not args.no_zones, labels=not args.no_labels)
                image = offscreen.Render(data, camera, decorations, ghost=Ghost_Positions(record, frame) if args.ghost else None,
                                         texts=Overlay_Lines(record, frame) if args.overlay else None,
                                         legend=Legend(record, Theme_Str=theme))
                name = 'frame_%06d.png' % number
                Write_Png(os.path.join(folder, name), image)
                rows.append(Frame_Row(record, frame, number, name))
                if (number + 1) % 60 == 0 or number + 1 == len(times):
                    print('[INFO]   %d of %d frames' % (number + 1, len(times)), file=sys.stderr)
        finally:
            offscreen.Close()
        Write_Frames(os.path.join(folder, 'frames.csv'), rows)

    _Write_Folder(args.frames, Write)
    return len(times)


def _View_Window(record: tp.Any, configuration: Configuration_Str, args: argparse.Namespace) -> None:
    """
    Description:
        Play a recorded throw back in MuJoCo's interactive viewer ('view' without --frames; SPEC §7.13): the passive viewer
        shows the themed scene at the record's positions, the decorations in its user scene, the text overlay and the
        legend; nothing is simulated here. The keys are ones the viewer itself does not use in MuJoCo 3.16.0
        (mujoco.mjVISSTRING, mjRNDSTRING, the geom groups 0-5 and the viewer's help; its own keys keep working): SPACE
        pause or resume, UP / DOWN one control period forward / back, - / = half / twice the speed, ENTER from the
        release again, 6 the next camera preset, 7 the zones, 8 the ghost pose, 9 the loop. The window is not tested: the
        owner checks it by the list in SPEC §10.

    Args:
        (1) record [Throw_Record_Str(object)]: The record.
        (2) configuration [Configuration_Str(object)]: The system configuration (the areas).
        (3) args [argparse.Namespace]: The options.
    """

    #   MuJoCo and its viewer (the window) [pip3 install mujoco]
    import mujoco
    import mujoco.viewer
    #   ../View/Core, Theme - the frames, the decorations, the legend, the camera presets; the themed scene
    from View.Core import (Frame_At, Overlay_Lines, Legend, Legend_Figure, Decorations, Draw, Add_Ghost, Ghost_Positions, Set_Camera,
                           CONST_THEMES, CONST_CAMERAS, CONST_SAFETY as CONST_VIEW_SAFETY)
    from View.Theme import Themed_Scene

    theme, cameras = CONST_THEMES[args.theme], list(CONST_CAMERAS)
    model = Themed_Scene(Areas_Str=configuration.Areas, Theme_Str=theme, markers=args.markers)
    data = mujoco.MjData(model)
    first, last = float(record.State_Time[0]), float(record.State_Time[-1])
    state = {'time': first if args.start is None else args.start, 'paused': False, 'speed': args.speed, 'camera': cameras.index(args.camera),
             'camera_set': False, 'zones': not args.no_zones, 'ghost': args.ghost, 'loop': args.loop}
    tick = float(model.opt.timestep)

    def On_Key(key: int) -> None:
        """
        Description:
            A key pressed in the window (GLFW's key codes).

        Args:
            (1) key [int]: The key.
        """

        if key == 32:
            state['paused'] = not state['paused']
        elif key in (265, 264):
            state['paused'], state['time'] = True, min(max(state['time'] + (tick if key == 265 else -tick), first), last)
        elif key in (45, 61):
            state['speed'] = min(max(state['speed'] * (0.5 if key == 45 else 2.0), 1.0 / 64.0), 4.0)
        elif key == 257:
            state['time'], state['paused'] = first, False
        elif key == 54:
            state['camera'], state['camera_set'] = (state['camera'] + 1) % len(cameras), False
        elif key == 55:
            state['zones'] = not state['zones']
        elif key == 56:
            state['ghost'] = not state['ghost']
        elif key == 57:
            state['loop'] = not state['loop']

    entries = Legend(record, Theme_Str=theme)
    legend = Legend_Figure(entries)
    with mujoco.viewer.launch_passive(model, data, key_callback=On_Key, show_left_ui=False, show_right_ui=False) as viewer:
        before = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            if not state['paused']:
                state['time'] += state['speed'] * (now - before)
                if state['time'] >= last:
                    state['time'], state['paused'] = (first, False) if state['loop'] else (last, True)
            before = now
            frame = Frame_At(record, state['time'])
            camera = CONST_CAMERAS[cameras[state['camera']]]
            with viewer.lock():
                if not state['camera_set']:
                    Set_Camera(model, viewer.cam, viewer.opt, Camera_Str=camera)
                    state['camera_set'] = True
                data.qpos[:] = record.State_Position[frame.State]
                mujoco.mj_forward(model, data)
                viewer.user_scn.ngeom = 0
                ghost = Ghost_Positions(record, frame) if state['ghost'] else None
                if ghost is not None:
                    Add_Ghost(model, ghost, viewer.opt, viewer.user_scn)
                Draw(viewer.user_scn, Decorations(record, frame, Theme_Str=theme, Areas_Str=configuration.Areas, Camera_Str=camera,
                                                  zones=state['zones'], labels=not args.no_labels))
            left, right = Overlay_Lines(record, frame)
            status = 'speed %gx%s%s | camera %s' % (state['speed'], ' | paused' if state['paused'] else '', ' | loop' if state['loop'] else '',
                                                   cameras[state['camera']])
            viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
                              (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, CONST_VIEW_SAFETY, ''),
                              (None, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, status, '')])
            viewport = viewer.viewport
            if viewport is not None and viewport.width > 0:
                height = 40 + 28 * len(entries)
                viewer.set_figures((mujoco.MjrRect(viewport.width - 310, viewport.height - height - 10, 300, height), legend))
            viewer.sync()
            time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - now)))


def _View(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'view' (SPEC §2.11, §7.13, §9.15): one simulated throw computed with the code
        of 'catch' (the decision at every sample, the re-planned motion, the catch simulated with the nominal contact;
        View.Core.Record_Throw) and played back, in MuJoCo's interactive viewer (macOS: start it with .venv/bin/mjpython)
        or, with --frames DIR, as offscreen PNG frames with frames.csv. The throw: a stored one (--dataset DIR, the first
        of Decision.Datasets by default, with --index I, or --random with --pick-seed S) of the train or the validation
        partition, a test partition's refused before the dataset's file is opened; or a new one (--new SEED, --aim
        catch|anywhere). The run's predictor drives the robot (--seed, --acceleration) with the ladder's margins of the
        run's stored validation interception; --compare RUN ... shows further runs' predictions beside it, for the
        display only. A stored validation throw is compared with the stored interception and catch (printed); a
        difference is an error. No test partition and no validation recording is read; nothing is counted.

    Args:
        (1) args [argparse.Namespace]: The options of 'view'.

    Returns:
        (1) parameter [int]: 0 shown or written; 1 refused, or the throw differs from its stored result.
    """

    #   ../Scene/Core - the scene and its robot
    from Scene.Core import Build_Scene, Scene_Robot, Scene_Error_Cls
    #   ../Scene/Ball - the ball's error
    from Scene.Ball import Ball_Error_Cls
    #   ../Robot/Core, Motion - the robot's and the motion's errors
    from Robot.Core import Robot_Error_Cls
    from Robot.Motion import Motion_Error_Cls
    #   ../Catch/Core, Decision, Physical - the catch pose, the stored margins, the catch's settings
    from Catch.Core import Catch_Context, Catch_Error_Cls
    from Catch.Decision import Read_Margins
    from Catch.Physical import Contact_Str
    #   ../View/Core - the throw's choice and its record
    from View.Core import (Pick_Throw, Partition_Code, Record_Throw, View_Error_Cls, CONST_CAMERAS, CONST_THEMES,
                           CONST_SAFETY as CONST_VIEW_SAFETY)

    started = time.perf_counter()
    try:
        # 1\ The options, the launcher and the target, before anything is computed or read.
        if (args.index is not None) + bool(args.random) + (args.new is not None) != 1:
            raise Run_Error_Cls('view shows one throw: give exactly one of --index I (a stored throw), --random (a stored throw picked with '
                                '--pick-seed) and --new SEED (a new throw).')
        if args.pick_seed is not None and not args.random:
            raise Run_Error_Cls('--pick-seed picks a stored throw with --random; it has no effect with --index or --new.')
        if args.new is not None and args.dataset is not None:
            raise Run_Error_Cls('--new generates a throw from --dataset-config; --dataset names a stored dataset for --index or --random.')
        try:
            size = tuple(int(value) for value in args.size.lower().split('x'))
        except ValueError:
            size = ()
        if len(size) != 2 or not (16 <= size[0] <= 3840 and 16 <= size[1] <= 2160):
            raise Run_Error_Cls('--size is WIDTHxHEIGHT in pixels, from 16x16 to 3840x2160, found %r.' % args.size)
        if args.theme not in CONST_THEMES or args.camera not in CONST_CAMERAS:
            raise Run_Error_Cls('--theme is one of %s and --camera one of %s, found %r and %r.'
                                % (', '.join(CONST_THEMES), ', '.join(CONST_CAMERAS), args.theme, args.camera))
        if not (args.speed > 0.0 and args.fps > 0.0 and args.every >= 1):
            raise Run_Error_Cls('--speed and --fps must be above 0 and --every at least 1, found %r, %r and %r.' % (args.speed, args.fps, args.every))
        if args.frames is None and sys.platform == 'darwin' and 'MJPYTHON_BIN' not in os.environ:
            raise Run_Error_Cls('On macOS MuJoCo\'s interactive viewer runs under its own launcher: start view with .venv/bin/mjpython '
                                'instead of .venv/bin/python (.venv/bin/mjpython scripts/catch.py view ...), or write frames with --frames DIR.')
        if args.frames is not None and os.path.exists(args.frames):
            raise Run_Error_Cls('The folder %s exists; frames are never overwritten. Choose another --frames folder or remove it.' % args.frames)
        configuration = Load_Configuration(args.config)
        # 2\ The stored throw's dataset, before the run is looked at: its manifest first, so that a test partition's throw is
        #    refused before its file is opened; then the file's hash and the models it was generated with.
        model_sha256 = Model_Hash(Flight_Model_Str=_Flight_Model(configuration), Observation_Model_Str=_Observation_Model(configuration))
        partition, index, label, path, pick_seed = '', 0, '', '', args.pick_seed
        if args.new is None:
            folder = os.path.join(CONST_REPO_ROOT, configuration.Decision.Datasets[0].Path) if args.dataset is None \
                else os.path.abspath(args.dataset)
            label = os.path.basename(os.path.normpath(folder))
            try:
                manifest = Read_Manifest(os.path.join(folder, 'manifest.json'))
                sizes = [int(value) for value in manifest['partition_sizes']]
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise Run_Error_Cls('%s is not a dataset folder (%s): it holds flights.npz and manifest.json written by scripts/catch.py '
                                    'dataset (SPEC §9.5).' % (folder, error)) from error
            if args.random and pick_seed is None:
                pick_seed = int(np.random.SeedSequence().generate_state(1)[0] % 1000000)
            index = Pick_Throw(sizes, index=args.index, pick_seed=pick_seed)
            partition = ('train', 'validation')[Partition_Code(index, sizes)]
            path = os.path.join(folder, 'flights.npz')
            if Sha256_File(path) != manifest.get('sha256', {}).get('data'):
                raise Run_Error_Cls('The dataset %s has a flights.npz that differs from its manifest; generate it again with scripts/catch.py '
                                    'dataset (SPEC §9.5).' % folder)
            if manifest.get('model_sha256') != model_sha256:
                raise Run_Error_Cls('The dataset %s was generated with the models %s, not this configuration\'s %s; one flight model drives '
                                    'the datasets and the catch (SPEC §2.5).' % (folder, manifest.get('model_sha256'), model_sha256))
        interception, contact = configuration.Interception, configuration.Contact
        training, run_manifest, config_bytes = _Load_Run_Copy(args.run)
        if run_manifest.get('device') not in (None, 'cpu'):
            raise Run_Error_Cls('The run %s was trained on the device %s; every reported number comes from the CPU (SPEC §7.8). Train it with '
                                'Run.Device cpu.' % (args.run, run_manifest.get('device')))
        learned = training.Model_Name in CONST_LEARNED_MODEL_NAMES
        seeds = ['%d' % seed for seed in training.Seeds] if learned else ['']
        seed = seeds[0] if args.seed is None else '%d' % args.seed
        if seed not in seeds:
            raise Run_Error_Cls('--seed %s is not a training seed of the run %s (%s).' % (args.seed, args.run, ', '.join(seeds) or 'a numerical '
                                                                                         'predictor has none'))
        degrees = [float(np.degrees(a)) for a in configuration.Robot.Joint_Accelerations]
        if args.acceleration is not None and not any(abs(args.acceleration - value) < 1e-6 for value in degrees):
            raise Run_Error_Cls('--acceleration %g is not one of Robot.Joint_Accelerations (%s deg/s^2).'
                                % (args.acceleration, ', '.join('%g' % value for value in degrees)))
        acceleration = 0 if args.acceleration is None else int(np.argmin([abs(args.acceleration - value) for value in degrees]))
        # 3\ The stored validation interception of the run: its margins, read and never written.
        datasets = _Decision_Datasets(configuration)
        stored_parent = os.path.join(CONST_REPO_ROOT, interception.Output_Folder) if args.interceptions is None else args.interceptions
        stored_interception = os.path.join(_Run_Folder(configuration, stored_parent, training.Name), 'validation')
        stored_catch = os.path.join(_Run_Folder(configuration, os.path.join(CONST_REPO_ROOT, contact.Output_Folder) if args.catches is None
                                                else args.catches, training.Name), 'validation')
        try:
            stored = Read_Manifest(os.path.join(stored_interception, 'manifest.json'))
            margins = Read_Margins(os.path.join(stored_interception, 'margins.csv'))
        except (OSError, ValueError, Catch_Error_Cls) as error:
            raise Run_Error_Cls('view decides with the ladder\'s margins of the run\'s stored validation interception: %s cannot be read '
                                '(%s). Run scripts/catch.py catch --run %s first, or name the interceptions\' parent folder with --interceptions.'
                                % (stored_interception, error, args.run)) from error
        differing = [key for key, value in _Planner_Fields(configuration, config_bytes, datasets).items() if stored.get(key) != value]
        if differing:
            raise Run_Error_Cls('The stored interception %s differs from this call in %s; run scripts/catch.py catch again with this '
                                'configuration and run.'
                                % (stored_interception, ', '.join(differing)))
        missing = [n for n in interception.Stages if (seed, n) not in margins]
        if missing:
            raise Run_Error_Cls('The stored margins of %s hold no seed %r at N %s.' % (stored_interception, seed, missing))
        # 4\ The methods shown beside the run's, for the display only.
        flight_model, observation_model = _Flight_Model(configuration), _Observation_Model(configuration)
        overlays: tp.Dict[str, tp.Callable[..., tp.Any]] = {}
        for run in args.compare:
            other, _, _ = _Load_Run_Copy(run)
            name = other.Model_Name if other.Model_Name != training.Model_Name and other.Model_Name not in overlays else other.Name
            if name in overlays or name == training.Model_Name:
                raise Run_Error_Cls('--compare names the run %s twice, or the run that drives the robot.' % run)
            overlays[name] = Run_Predictor(other.Model_Name, os.path.abspath(run), other.Seeds[0] if other.Model_Name in CONST_LEARNED_MODEL_NAMES
                                           else None, Flight_Model_Str=flight_model, Observation_Model_Str=observation_model)
        # 5\ The scene, the area's map and the setup, as 'catch' builds them.
        model = Build_Scene(Areas_Str=configuration.Areas)
        context = Catch_Context(model, Scene_Robot(model), Criteria_Str=_Catch_Criteria(configuration))
        setup, _, _ = _Interception_Setup(configuration, training, args.run, context)
        setup.Margins = margins
        settings = Contact_Str(Step=contact.Step, Friction=list(contact.Friction), Settle_Time=contact.Settle_Time, Rest_Speed=contact.Rest_Speed,
                               Rest_Time=contact.Rest_Time, Levels=[(contact.Time_Constant, contact.Damping_Ratio)])
        # 6\ The throw, recorded.
        with tempfile.TemporaryDirectory() as temporary:
            if args.new is not None:
                label, partition = 'New_%d' % args.new, 'new, aimed %s' % ('into the catch area' if args.aim == 'catch' else 'anywhere')
                path = _New_Throw(configuration, args.dataset_config, args.new, args.aim, temporary)
            print('[INFO] Computing the throw %s %d (%s) with %s%s at %g deg/s^2.'
                  % (label, index, partition, training.Model_Name, ' seed %s' % seed if seed else '', degrees[acceleration]), file=sys.stderr)
            record = Record_Throw(setup, label, path, index, seed, acceleration, settings, overlays=overlays, every=args.every)
    except (Configuration_Error_Cls, Run_Error_Cls, Dataset_Error_Cls, Scene_Error_Cls, Robot_Error_Cls, Catch_Error_Cls, Motion_Error_Cls,
            Ball_Error_Cls, View_Error_Cls, Observation_Error_Cls, Recording_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    for level, message in configuration.Notes + training.Notes:
        print('[%s] %s' % (level, message), file=sys.stderr)

    # 7\ The report.
    go = record.Decision == 'go'
    print('Throw:      %s %d (%s)%s, %d observed samples, on the floor at %.3f s'
          % (label, index, partition, ', picked with --pick-seed %d' % pick_seed if args.random else '', len(record.Observed_Time),
             record.Truth_Time[-1]))
    print('Method:     %s%s, %g deg/s^2%s' % (record.Method, ' (seed %s)' % seed if seed else '', degrees[acceleration],
                                               '; shown beside it: %s' % ', '.join(record.Methods[1:]) if len(record.Methods) > 1 else ''))
    print('Decision:   %s%s; catchable: %s' % (record.Decision, ' at N %d (%.3f s), the catch time %.3f s'
                                               % (record.Go_N, record.Go_Time, record.Catch_Time) if go else '',
                                               'true' if record.Catchable else 'false'))
    print('Plans:      %d accepted, %d rejected' % (len(record.Plan_Time), record.Rejected))
    print('Class:      %s (%s)' % (record.Class, record.Outcome))
    print('Predictions shown: %d per method (%d asked by the decision and the planner, the others for the display only)'
          % (len(record.Sample_N), int(record.Used.sum())))
    # 8\ The stored result of a stored validation throw.
    equal: tp.Optional[bool] = None
    found = 'a new throw has no stored result' if args.new is not None else 'only the validation partitions have a stored result'
    if partition == 'validation':
        equal, found = _Stored_Check(record, stored_interception, stored_catch)
    if equal is False:
        print('[ERROR] Stored check: %s. The stored results and this code no longer agree; nothing is shown.' % found, file=sys.stderr)
        return 1
    print('[INFO] Stored check: %s.' % (found if equal else 'not made, %s' % found), file=sys.stderr)
    print('[INFO] Computed in %.0f s. The inference times shown are this run\'s wall times, not reported measurements. Simulated throws '
          'only (SPEC §2.14). %s' % (time.perf_counter() - started, CONST_VIEW_SAFETY), file=sys.stderr)
    # 9\ The frames, or the window.
    try:
        if args.frames is not None:
            count = _View_Frames(record, configuration, args, size)
            print('[INFO] Wrote %d frames of %d x %d px and frames.csv into %s (%s, %s%s).'
                  % (count, size[0], size[1], args.frames, args.camera, args.theme, ', the overlay burnt in' if args.overlay else ', clean'),
                  file=sys.stderr)
        else:
            print('[INFO] Keys: SPACE pause, UP / DOWN one control period forward / back, - / = half / twice the speed, ENTER again from the '
                  'release, 6 the next camera, 7 the zones, 8 the ghost pose, 9 the loop; the mouse moves the camera.', file=sys.stderr)
            _View_Window(record, configuration, args)
    except (Run_Error_Cls, View_Error_Cls, Scene_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    return 0


def _Git_State() -> tp.Tuple[str, bool]:
    """
    Description:
        The repository's commit and whether its tree has uncommitted changes, for the ledger.

    Returns:
        (1) parameter 1 [string]: The commit hash ('unknown' outside a git checkout).
        (2) parameter 2 [bool]: Whether the tree is dirty.
    """

    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=CONST_REPO_ROOT, capture_output=True, text=True, timeout=60)
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=CONST_REPO_ROOT, capture_output=True, text=True, timeout=60)
    except OSError:
        return 'unknown', True
    if commit.returncode != 0:
        return 'unknown', True
    return commit.stdout.strip(), status.stdout.strip() != ''


def _Flight_Model(configuration: Configuration_Str) -> Flight_Model_Str:
    """
    Description:
        The one flight model from the system configuration (SPEC §7.4): gravity along -z, F3's drag.

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.

    Returns:
        (1) parameter [Flight_Model_Str(object)]: The flight model.
    """

    flight = configuration.Flight
    return Flight_Model_Str(Gravity=np.array([0.0, 0.0, -flight.Gravity]), Drag=flight.Drag, Drag_Std=flight.Drag_Std)


def _Observation_Model(configuration: Configuration_Str) -> Observation_Model_Str:
    """
    Description:
        The observation error model M4 from the system configuration (SPEC §7.5, §8.1), its bounds included.

    Args:
        (1) configuration [Configuration_Str(object)]: The system configuration.

    Returns:
        (1) parameter [Observation_Model_Str(object)]: The error model.
    """

    o = configuration.Observation
    return Observation_Model_Str(White_Std=np.array(o.White_Std), White_Dof=o.White_Dof, Drift_Std=np.array(o.Drift_Std),
                                 Drift_Time=o.Drift_Time, Offset_Std=np.array(o.Offset_Std), Offset_Time=o.Offset_Time,
                                 Offset_Dof=o.Offset_Dof, Offset_Bound=o.Offset_Bound, White_Bound=o.White_Bound)


def _Generate_Into(folder: str, dataset_bytes: bytes, system_bytes: bytes, dataset_name: str,
                   stored_copy: bool = False) -> tp.Tuple[tp.Dict[str, tp.Any], float]:
    """
    Description:
        Generate a dataset from configuration bytes into an existing, empty folder: the two configuration
        copies (dataset.yaml, ball_catching.yaml), flights.npz and manifest.json (SPEC §9.5). The copies
        are what is read, so the check (scripts/Maintenance/derive.py dataset-check) regenerates exactly what was
        stored. A new dataset's system copy is loaded in full; a stored one (the check) only for the sections a dataset depends on
        (CONST_DATASET_SECTIONS).

    Args:
        (1) folder [string]: The folder.
        (2) dataset_bytes [bytes]: The dataset configuration's bytes.
        (3) system_bytes [bytes]: The system configuration's bytes.
        (4) dataset_name [string]: The dataset configuration's file name, for messages.
        (5) stored_copy [bool]: True when the bytes are a stored dataset's copies (the check).

    Returns:
        (1) parameter 1 [Dict<string, Any>]: The manifest.
        (2) parameter 2 [float]: The generation time [s].
    """

    for name, data in (('dataset.yaml', dataset_bytes), ('ball_catching.yaml', system_bytes)):
        with open(os.path.join(folder, name), 'wb') as file:
            file.write(data)
    dataset = Load_Dataset_Configuration(os.path.join(folder, 'dataset.yaml'))
    configuration = Load_Configuration(os.path.join(folder, 'ball_catching.yaml'), stored_copy=stored_copy)
    for level, message in dataset.Notes + configuration.Notes:
        print('[%s] %s' % (level, message.replace(folder, dataset_name)), file=sys.stderr)
    donors = Load_Donors(CONST_RECORDINGS_DIR)
    flight, observation = _Flight_Model(configuration), _Observation_Model(configuration)
    start = time.perf_counter()
    data = Generate_Dataset(dataset, configuration, donors, Flight_Model_Str=flight, Observation_Model_Str=observation,
                            progress=lambda done, total: print('[INFO] %d of %d flights' % (done, total), file=sys.stderr))
    Write_Dataset(os.path.join(folder, 'flights.npz'), data)
    elapsed = time.perf_counter() - start
    commit, dirty = _Git_State()
    sizes = Partition_Sizes(dataset.Flights, (dataset.Partitions.Train, dataset.Partitions.Validation, dataset.Partitions.Test))
    manifest = Build_Manifest('dataset', {'configuration': Sha256_Bytes(dataset_bytes), 'system_configuration': Sha256_Bytes(system_bytes),
                                          'data': Sha256_File(os.path.join(folder, 'flights.npz'))},
                              {'dataset_name': dataset.Name, 'seed': dataset.Seed, 'flights': dataset.Flights, 'partition_sizes': sizes,
                               'rates': dataset.Observation.Rates, 'format_version': 1, 'samples': int(data.Observed_Time.size),
                               'model_sha256': Model_Hash(Flight_Model_Str=flight, Observation_Model_Str=observation)}, commit, dirty)
    Write_Manifest(os.path.join(folder, 'manifest.json'), manifest)
    return manifest, elapsed


def _Dataset(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'dataset' (F6; SPEC §2.13.1-13.4, §9.5): generate Dataset.Flights flights from the
        dataset configuration (--config) and the system configuration it names, into
        <Output.Folder or --out>/<Dataset.Name>/, written to a temporary folder beside it and renamed at the
        end; an existing target is refused. A stored dataset is regenerated and compared byte for byte by
        scripts/Maintenance/derive.py dataset-check.

    Args:
        (1) args [argparse.Namespace]: --config, --out.

    Returns:
        (1) parameter [int]: 0 written; 1 refused (invalid configuration, a rate without donors, an existing target).
    """

    try:
        dataset = Load_Dataset_Configuration(args.config)
        system_path = os.path.join(CONST_REPO_ROOT, dataset.System_Configuration)
        with open(args.config, 'rb') as file:
            dataset_bytes = file.read()
        with open(system_path, 'rb') as file:
            system_bytes = file.read()
    except (Configuration_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    parent = os.path.join(CONST_REPO_ROOT, dataset.Output_Folder) if args.out is None else args.out
    target = os.path.join(parent, dataset.Name)
    if os.path.exists(target):
        print('[ERROR] The dataset folder %s exists; a dataset is never overwritten. Choose another Dataset.Name or remove it.' % target,
              file=sys.stderr)
        return 1
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix='.%s.' % dataset.Name, dir=parent)
    os.chmod(temporary, 0o755)
    try:
        manifest, elapsed = _Generate_Into(temporary, dataset_bytes, system_bytes, args.config)
        os.rename(temporary, target)
    except (Configuration_Error_Cls, Recording_Error_Cls, Dataset_Error_Cls, Observation_Error_Cls) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    size = os.path.getsize(os.path.join(target, 'flights.npz'))
    print('dataset %s: %d flights (train / validation / test %s), rates %s Hz, %d observed samples' % (
        manifest['dataset_name'], manifest['flights'], ' / '.join(str(n) for n in manifest['partition_sizes']),
        ', '.join('%g' % r for r in manifest['rates']), manifest['samples']))
    print('flights.npz %.1f MB, sha256 %s; generated in %.1f s' % (size / 1e6, manifest['sha256']['data'], elapsed))
    print('[INFO] Wrote %s. %s' % (target, CONST_SAFETY), file=sys.stderr)
    return 0


class Run_Error_Cls(ValueError):
    """
    Description:
        A run cannot be written or evaluated: a dataset whose hash differs from the run's or its manifest's,
        datasets with different models, a dirty run given to 'evaluate --partition test', an existing target folder. The message names
        it and what to do; it is printed verbatim as an [ERROR] line.
    """


def _Read_Bytes(path: str) -> bytes:
    """
    Description:
        A file's bytes.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [bytes]: Its content.
    """

    with open(path, 'rb') as file:
        return file.read()


def _Run_Datasets(training: tp.Any, trained_on: bool = False) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Description:
        The datasets of a run, checked (SPEC §8.3, §9.6): each dataset's flights.npz has the run's hash and its
        manifest's, its two stored copies their manifest's hashes, the models of its system copy the manifest's
        model hash, and all share one flight model and M4. The system copy is read only for
        the sections a dataset depends on (CONST_DATASET_SECTIONS). The data itself is not read here. A learned
        predictor's run names the datasets it is scored on (Run.Datasets) and those it is trained on (Training.Datasets).

    Args:
        (1) training [Training_Configuration_Str(object)]: The run's training configuration.
        (2) trained_on [bool]: True for the datasets trained on (Training.Datasets); False for Run.Datasets.

    Returns:
        (1) parameter [List<Dict<string, Any>>]: Per dataset: label, folder, hash, system configuration, dataset
                                                  configuration, flight and observation models, model hash.
    """

    out, problems = [], []
    for item in (training.Training_Datasets if trained_on else training.Datasets):
        folder = os.path.join(CONST_REPO_ROOT, item.Path)
        try:
            manifest = Read_Manifest(os.path.join(folder, 'manifest.json'))
            data_hash = Sha256_File(os.path.join(folder, 'flights.npz'))
        except (OSError, ValueError) as error:
            problems.append('the dataset %s cannot be read (%s); generate it with scripts/catch.py dataset (SPEC §9.5)' % (item.Path, error))
            continue
        if data_hash != item.Hash or manifest.get('sha256', {}).get('data') != item.Hash:
            problems.append('the dataset %s has flights.npz %s and manifest %s, not the run\'s %s; a run never mixes datasets (SPEC §8.3)'
                            % (item.Path, data_hash, manifest.get('sha256', {}).get('data'), item.Hash))
            continue
        copies = [(name, key, Sha256_File(os.path.join(folder, name)), manifest.get('sha256', {}).get(key))
                  for name, key in (('dataset.yaml', 'configuration'), ('ball_catching.yaml', 'system_configuration'))]
        edited = ['%s %s, manifest %s' % (name, copy, recorded) for name, _, copy, recorded in copies if copy != recorded]
        if edited:
            problems.append('the dataset %s has stored copies that differ from its manifest (%s); a stored copy is never edited, '
                            'generate a new dataset (SPEC §9.5)' % (item.Path, '; '.join(edited)))
            continue
        configuration = Load_Configuration(os.path.join(folder, 'ball_catching.yaml'), stored_copy=True)
        dataset = Load_Dataset_Configuration(os.path.join(folder, 'dataset.yaml'))
        flight, observation = _Flight_Model(configuration), _Observation_Model(configuration)
        model_sha256 = Model_Hash(Flight_Model_Str=flight, Observation_Model_Str=observation)
        if model_sha256 != manifest.get('model_sha256'):
            problems.append('the dataset %s\'s stored ball_catching.yaml gives the models %s, not its manifest\'s model_sha256 %s '
                            '(SPEC §9.5)' % (item.Path, model_sha256, manifest.get('model_sha256')))
            continue
        out.append({'label': os.path.basename(os.path.normpath(item.Path)), 'folder': folder, 'path': item.Path, 'hash': data_hash,
                    'configuration': configuration, 'dataset': dataset, 'flight': flight, 'observation': observation,
                    'model_sha256': model_sha256})
    if problems:
        raise Run_Error_Cls('The run\'s datasets cannot be used (%d problem(s)):\n  %s' % (len(problems), '\n  '.join(problems)))
    if len({d['model_sha256'] for d in out}) > 1:
        raise Run_Error_Cls('The run\'s datasets were generated with different flight or observation models (%s); one flight model drives '
                            'the datasets and the predictor (SPEC §2.5). Use datasets of one system configuration.'
                            % ', '.join('%s %s' % (d['path'], d['model_sha256'][:12]) for d in out))
    return out


def _Write_Folder(target: str, write: tp.Callable[[str], None]) -> None:
    """
    Description:
        Write a folder through a temporary folder beside it, renamed at the end; an existing target is refused and
        nothing is left behind on a failure (SPEC §9.5-9.7).

    Args:
        (1) target [string]: The folder.
        (2) write [Callable[[string], None]]: Writes the files into the folder it is given.
    """

    if os.path.exists(target):
        raise Run_Error_Cls('The folder %s exists; a run or an evaluation is never overwritten. Choose another Run.Name, --out, or remove '
                            'it.' % target)
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix='.%s.' % os.path.basename(target), dir=parent)
    os.chmod(temporary, 0o755)
    try:
        write(temporary)
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _Train_Learned(training: tp.Any, datasets: tp.List[tp.Dict[str, tp.Any]], folder: str) -> tp.Tuple[tp.Dict[str, tp.Any], tp.List[str]]:
    """
    Description:
        The training of a learned predictor (SPEC §2.13.7, §7.8, §9.6; F8): the first Training.Flights flights of
        the train partition of each dataset trained on are pooled, the validation partitions of the same datasets
        stop the training; the normalisation is fitted on the training flights only; one model per seed is written
        to <folder>/seed_<s>/ (weights.npz, normalisation.json, epochs.csv). Only the train and the validation
        partition are selected (Partition_Indices), and no recording is read. The training's runtime is imported
        here, so no other command needs it.

    Args:
        (1) training [Training_Configuration_Str(object)]: The run's configuration (a learned predictor's).
        (2) datasets [List<Dict<string, Any>>]: The datasets trained on (_Run_Datasets).
        (3) folder [string]: The run folder being written.

    Returns:
        (1) parameter 1 [Dict<string, Any>]: The manifest's training fields: the seeds, the device, the threads, the
                                             torch version, the datasets trained on, the training and validation
                                             flights, and per seed its epochs, its best epoch and the SHA-256 of
                                             its weights file.
        (2) parameter 2 [List<string>]: The report's lines, one per seed.
    """

    # PyTorch (the training only) [pip3 install torch]
    import torch
    #   ../Learning/Training - the training (imports PyTorch)
    from Learning.Training import Prepare_Flights, Merge_Flights, Fit_Normalisation, Train, Write_Epochs, CONST_THREADS

    samples = max(training.Observed_Samples)
    parts: tp.Dict[int, tp.List[tp.Any]] = {CONST_PARTITION_TRAIN: [], CONST_PARTITION_VALIDATION: []}
    for d in datasets:
        data = Read_Dataset(os.path.join(d['folder'], 'flights.npz'))
        for code in parts:
            indices = Partition_Indices(data, code)
            if code == CONST_PARTITION_TRAIN:
                if training.Training_Flights > len(indices):
                    raise Run_Error_Cls('Training.Flights is %d, and the train partition of %s holds %d flights. Lower it, or use a '
                                        'larger dataset (SPEC §8.3).' % (training.Training_Flights, d['path'], len(indices)))
                indices = indices[:training.Training_Flights]
            parts[code].append(Prepare_Flights(data, indices, samples, Flight_Model_Str=d['flight']))
    train, validation = Merge_Flights(parts[CONST_PARTITION_TRAIN]), Merge_Flights(parts[CONST_PARTITION_VALIDATION])
    flight = datasets[0]['flight']
    normalisation = Fit_Normalisation(train, training.Observed_Samples, Flight_Model_Str=flight)
    print('[INFO] %s: %d training and %d validation flights of %d dataset(s); %d seed(s) on %s.'
          % (training.Model_Name, len(train.Rate), len(validation.Rate), len(datasets), len(training.Seeds), training.Device), file=sys.stderr)
    lines, seeds = [], {}
    for seed in training.Seeds:
        start = time.perf_counter()
        weights, epochs = Train(training, train, validation, normalisation, seed,
                                progress=lambda epoch, a, b: print('[INFO]   seed %d epoch %d: train %.2f mm, validation %.2f mm'
                                                                   % (seed, epoch, 1e3 * a, 1e3 * b), file=sys.stderr) if epoch % 10 == 0 else None)
        target = os.path.join(folder, 'seed_%d' % seed)
        os.makedirs(target)
        Write_Weights(os.path.join(target, CONST_WEIGHTS_FILE), training.Model_Name, training.Hidden, weights)
        Write_Normalisation(os.path.join(target, CONST_NORMALISATION_FILE), normalisation)
        Write_Epochs(os.path.join(target, 'epochs.csv'), epochs)
        best = min(epochs, key=lambda e: e[2])
        seeds['%d' % seed] = {'epochs': len(epochs), 'best_epoch': best[0], 'weights_sha256': Sha256_File(os.path.join(target, CONST_WEIGHTS_FILE))}
        lines.append('seed %d: %d epochs, the best %d, validation loss %.3f mm (position %.3f mm)' % (seed, len(epochs), best[0], 1e3 * best[2],
                                                                                                  1e3 * best[3]))
        print('[INFO] seed %d trained in %.1f s (%.2f s per epoch).' % (seed, time.perf_counter() - start,
                                                                       (time.perf_counter() - start) / len(epochs)), file=sys.stderr)
    fields = {'seeds': list(training.Seeds), 'device': training.Device, 'threads': CONST_THREADS if training.Device == 'cpu' else None,
              'torch': torch.__version__, 'training_datasets': [{'path': d['path'], 'sha256': d['hash']} for d in datasets],
              'training_flights': int(len(train.Rate)), 'validation_flights': int(len(validation.Rate)), 'trained': seeds}
    return fields, lines


def _Train(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'train' (SPEC §2.13.7, §9.6). For a learned predictor (F8) it trains one model per seed of
        Run.Seeds on the train partitions of Training.Datasets, early-stopped on their validation partitions, and
        writes the run folder: a copy of the training configuration, manifest.json and seed_<s>/ per seed
        (_Train_Learned). For a numerical predictor (F7) it writes the configuration's copy and manifest.json only:
        nothing is fitted. Each dataset's hash is checked against the run's and its manifest's.

    Args:
        (1) args [argparse.Namespace]: --config, --out.

    Returns:
        (1) parameter [int]: 0 written; 1 refused (invalid configuration, a dataset mismatch, an existing target,
                             more training flights than a partition holds).
    """

    try:
        training = Load_Training_Configuration(args.config)
        config_bytes = _Read_Bytes(args.config)
        datasets = _Run_Datasets(training)
        learned = training.Model_Name in CONST_LEARNED_MODEL_NAMES
        trained_on = _Run_Datasets(training, True) if learned else []
        if len({d['model_sha256'] for d in datasets + trained_on}) > 1:
            raise Run_Error_Cls('The datasets trained on and the datasets scored were generated with different flight or observation '
                                'models; one flight model drives the datasets and the predictor (SPEC §2.5). Use datasets of one '
                                'system configuration.')
    except (Configuration_Error_Cls, Run_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    for level, message in training.Notes:
        print('[%s] %s' % (level, message), file=sys.stderr)
    target = os.path.join(CONST_REPO_ROOT, training.Output_Folder if args.out is None else args.out, training.Name)
    commit, dirty = _Git_State()
    lines: tp.List[str] = []

    def Write(folder: str) -> None:
        """
        Description:
            The run's files.

        Args:
            (1) folder [string]: The folder.
        """

        with open(os.path.join(folder, 'training.yaml'), 'wb') as file:
            file.write(config_bytes)
        fields = {'run_name': training.Name, 'model': training.Model_Name, 'observed_samples': training.Observed_Samples,
                  'datasets': [{'path': d['path'], 'sha256': d['hash']} for d in datasets], 'model_sha256': datasets[0]['model_sha256']}
        if learned:
            further, report = _Train_Learned(training, trained_on, folder)
            fields.update(further)
            lines.extend(report)
        Write_Manifest(os.path.join(folder, 'manifest.json'), Build_Manifest('run', {'configuration': Sha256_Bytes(config_bytes)}, fields,
                                                                             commit, dirty))

    try:
        _Write_Folder(target, Write)
    except (Run_Error_Cls, Dataset_Error_Cls, ValueError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    print('run %s: %s on %d dataset(s), N = %s; %s' % (training.Name, training.Model_Name, len(datasets),
                                                       ', '.join(str(n) for n in training.Observed_Samples),
                                                       '%d seed(s) trained on %d dataset(s)' % (len(training.Seeds), len(trained_on)) if learned
                                                       else 'nothing fitted (a numerical predictor)'))
    for line in lines:
        print(line)
    print('[INFO] Wrote %s%s. %s' % (target, ' (from a tree with uncommitted changes: evaluate --partition test will refuse it)' if dirty else '',
                                     CONST_SAFETY), file=sys.stderr)
    return 0


def _Load_Run(run: str) -> tp.Tuple[tp.Any, tp.Dict[str, tp.Any], bytes, tp.List[tp.Dict[str, tp.Any]]]:
    """
    Description:
        A run folder: its training configuration copy, its manifest and its checked datasets (_Run_Datasets).

    Args:
        (1) run [string]: The run folder.

    Returns:
        (1) parameter 1 [Training_Configuration_Str(object)]: The configuration.
        (2) parameter 2 [Dict<string, Any>]: The manifest.
        (3) parameter 3 [bytes]: The configuration copy's bytes.
        (4) parameter 4 [List<Dict<string, Any>>]: The datasets.
    """

    training, manifest, config_bytes = _Load_Run_Copy(run)
    return training, manifest, config_bytes, _Run_Datasets(training)


def _Load_Run_Copy(run: str) -> tp.Tuple[tp.Any, tp.Dict[str, tp.Any], bytes]:
    """
    Description:
        A run folder's training configuration copy and manifest, the copy checked against the manifest; its datasets are
        not read.

    Args:
        (1) run [string]: The run folder.

    Returns:
        (1) parameter 1 [Training_Configuration_Str(object)]: The configuration.
        (2) parameter 2 [Dict<string, Any>]: The manifest.
        (3) parameter 3 [bytes]: The configuration copy's bytes.
    """

    try:
        manifest = Read_Manifest(os.path.join(run, 'manifest.json'))
        config_bytes = _Read_Bytes(os.path.join(run, 'training.yaml'))
    except (OSError, ValueError) as error:
        raise Run_Error_Cls('%s is not a run folder (%s): it holds training.yaml and manifest.json written by train (SPEC §9.6).'
                            % (run, error)) from error
    training = Load_Training_Configuration(os.path.join(run, 'training.yaml'))
    if manifest.get('sha256', {}).get('configuration') != Sha256_Bytes(config_bytes):
        raise Run_Error_Cls('The run %s\'s training.yaml differs from its manifest; the run was changed after train. Write a new run.' % run)
    return training, manifest, config_bytes


def _Prediction_Records(results: tp.List[Flight_Result_Str], predictions: tp.List[tp.Any]) -> tp.List[Prediction_Record_Str]:
    """
    Description:
        The predictions.npz records of evaluated flights.

    Args:
        (1) results [List<Flight_Result_Str>]: The results.
        (2) predictions [List<Tuple>]: Per result (trajectory, query times, query states).

    Returns:
        (1) parameter [List<Prediction_Record_Str>]: The records (a refused prediction has none).
    """

    records = []
    for r, prediction in zip(results, predictions):
        if prediction is None:
            continue
        trajectory, t, states = prediction
        # A trajectory integrated under the flight model is rebuilt from its first state; a query-time model's has none.
        first = trajectory.Grid_State[0].copy() if isinstance(trajectory, Trajectory_Str) else np.full(6, np.nan)
        records.append(Prediction_Record_Str(Flight=r.Flight, Partition=r.Partition, Dataset=r.Dataset, Seed=r.Seed, N=r.N,
                                             Grid_Start=trajectory.Grid_Start, T_0=trajectory.T_0, T_End=trajectory.T_End, X_First=first,
                                             Query_Time=np.asarray(t), Query_State=states))
    return records


def _Report_Lines(results: tp.List[Flight_Result_Str]) -> tp.List[str]:
    """
    Description:
        The printed report of an evaluation: per partition, dataset, training seed and N the flights, the remainder
        RMS median / p90, the end mismatch, the inference time; for simulated flights the error at 0.5 s, the
        catch-point error, the crossing disagreement and the velocity's direction and speed errors at the catch
        plane; for real flights the horizon ranges with their flight counts.

    Args:
        (1) results [List<Flight_Result_Str>]: The results.

    Returns:
        (1) parameter [List<string>]: The lines.
    """

    lines = []
    rows = {(r['partition'], r['dataset'], r['seed'], r['n'], r['metric']): r for r in Summary_Rows(results) if r['group'] == 'all'}
    keys = sorted({(r.Partition, r.Dataset, r.Seed, r.N) for r in results})
    F = lambda value, form='%.1f': '-' if not np.isfinite(value) else form % value
    for partition, dataset, seed, n in keys:
        key = (partition, dataset, seed, n)
        get = lambda metric, statistic: rows[key + (metric,)][statistic] if key + (metric,) in rows else float('nan')
        line = ('%s %s%s N=%d: %d flights, %d refused; remainder RMS median %s / p90 %s mm; end mismatch |median| %s ms; '
                'inference median %s ms') % (
            partition, dataset or '(real)', ' seed %s' % seed if seed else '', n, rows[key + ('refused_share',)]['count'],
            rows[key + ('refused',)]['count'],
            F(get('remainder_rms_mm', 'median')), F(get('remainder_rms_mm', 'p90')), F(get('end_mismatch_abs_ms', 'median')),
            F(get('inference_ms', 'median'), '%.2f'))
        if partition != 'real':
            line += '; 0.5 s %s mm; catch point %s mm; disagreement %s %%; velocity at the catch plane: direction %s deg, |speed| %s m/s' % (
                F(get('fixed_0.5_mm', 'median')), F(get('catch_point_mm', 'median')), F(100.0 * get('crossing_disagreement', 'mean')),
                F(get('direction_catch_deg', 'median'), '%.2f'), F(get('speed_abs_catch_m_s', 'median'), '%.3f'))
        else:
            ranges = [(m, r) for (p, d, s, k, m), r in rows.items() if (p, d, s, k) == key and m.startswith('range_')
                      and not m.startswith('range_observed')]
            line += '; ranges ' + ', '.join('%s %s mm (%d)' % (m[6:-3].replace('_', '-'), F(r['median']), r['count']) for m, r in ranges)
        lines.append(line)
    return lines


def _Write_Evaluation(folder: str, results: tp.List[Flight_Result_Str], predictions: tp.List[tp.Any], manifest: tp.Dict[str, tp.Any]) -> None:
    """
    Description:
        An evaluation's files (SPEC §9.7) into a folder.

    Args:
        (1) folder [string]: The folder.
        (2) results [List<Flight_Result_Str>]: The results.
        (3) predictions [List<Tuple>]: Per result (trajectory, query times, query states).
        (4) manifest [Dict<string, Any>]: The manifest.
    """

    Write_Per_Flight(os.path.join(folder, 'per_flight.csv'), results)
    Write_Per_Horizon(os.path.join(folder, 'per_horizon.csv'), results)
    Write_Summary(os.path.join(folder, 'summary.csv'), Summary_Rows(results))
    Write_Predictions(os.path.join(folder, 'predictions.npz'), _Prediction_Records(results, predictions))
    files = dict(manifest['sha256'])
    for name in ('per_flight.csv', 'per_horizon.csv', 'summary.csv', 'predictions.npz'):
        files[name] = Sha256_File(os.path.join(folder, name))
    manifest['sha256'] = files
    Write_Manifest(os.path.join(folder, 'manifest.json'), manifest)


def _Predictors(training: tp.Any, dataset: tp.Dict[str, tp.Any],
                run: str) -> tp.List[tp.Tuple[str, tp.Callable[[np.ndarray, np.ndarray, float], tp.Any]]]:
    """
    Description:
        The predictors of a run on one dataset, as Evaluation takes them: a numerical predictor is one function of a
        prefix under the dataset's stored flight model and M4, without a training seed; a learned predictor is one
        function per training seed, from the seed's folder of the run (Learning.Predictor, NumPy).

    Args:
        (1) training [Training_Configuration_Str(object)]: The run's configuration.
        (2) dataset [Dict<string, Any>]: The dataset (_Run_Datasets): its flight and observation models.
        (3) run [string]: The run folder (a learned predictor's seeds).

    Returns:
        (1) parameter [List<Tuple<string, Callable>>]: Per predictor its seed's label and the function of a prefix.
    """

    if training.Model_Name in CONST_LEARNED_MODEL_NAMES:
        return [('%d' % seed, Run_Predictor(training.Model_Name, run, seed, Flight_Model_Str=dataset['flight'],
                                            Observation_Model_Str=dataset['observation'])) for seed in training.Seeds]
    return [('', Run_Predictor(training.Model_Name, run, None, Flight_Model_Str=dataset['flight'], Observation_Model_Str=dataset['observation']))]


def _Evaluate_Partition(training: tp.Any, datasets: tp.List[tp.Dict[str, tp.Any]], partition: str,
                        run: str) -> tp.Tuple[tp.List[Flight_Result_Str], tp.List[tp.Any]]:
    """
    Description:
        Every dataset's flights of one partition, predicted and scored; each dataset with its own stored models.

    Args:
        (1) training [Training_Configuration_Str(object)]: The run's configuration.
        (2) datasets [List<Dict<string, Any>>]: The run's datasets.
        (3) partition [string]: 'validation' or 'test'.
        (4) run [string]: The run folder.

    Returns:
        (1) parameter 1 [List<Flight_Result_Str>]: The results.
        (2) parameter 2 [List<Tuple>]: The predictions.
    """

    results, predictions = [], []
    for d in datasets:
        data = Read_Dataset(os.path.join(d['folder'], 'flights.npz'))
        indices = Partition_Indices(data, CONST_PARTITION_CODES[partition])
        print('[INFO] %s: %d %s flights with %s, N = %s.' % (d['label'], len(indices), partition, training.Model_Name,
                                                           ', '.join(str(n) for n in training.Observed_Samples)), file=sys.stderr)
        r, p = Evaluate_Simulated(data, indices, _Predictors(training, d, run), training.Observed_Samples, partition, d['label'],
                                  Catch_Height(d['configuration']), Flight_Model_Str=d['flight'],
                                  progress=lambda done, total: print('[INFO]   %d of %d flights' % (done, total), file=sys.stderr))
        results += r
        predictions += p
    return results, predictions


def _Evaluation_Manifest(training: tp.Any, config_bytes: bytes, run: str, datasets: tp.List[tp.Dict[str, tp.Any]], partition: str,
                         results: int) -> tp.Dict[str, tp.Any]:
    """
    Description:
        An evaluation's manifest (SPEC §2.13.10): the hashes of the run's configuration and manifest and of each
        dataset, the predictor, the partition, the number of results, the git state, the versions.

    Args:
        (1) training [Training_Configuration_Str(object)]: The run's configuration.
        (2) config_bytes [bytes]: Its copy's bytes.
        (3) run [string]: The run folder.
        (4) datasets [List<Dict<string, Any>>]: The run's datasets.
        (5) partition [string]: 'validation' or 'test'.
        (6) results [int]: The results written.

    Returns:
        (1) parameter [Dict<string, Any>]: The manifest.
    """

    commit, dirty = _Git_State()
    files = {'configuration': Sha256_Bytes(config_bytes), 'run_manifest': Sha256_File(os.path.join(run, 'manifest.json'))}
    return Build_Manifest('evaluation', files,
                          {'run_name': training.Name, 'model': training.Model_Name, 'partition': partition,
                           'datasets': [{'path': d['path'], 'sha256': d['hash']} for d in datasets], 'model_sha256': datasets[0]['model_sha256'],
                           'results': results, 'seeds': list(training.Seeds)}, commit, dirty)


def _Evaluate_Validation(args: argparse.Namespace) -> int:
    """
    Description:
        'evaluate' on the validation partition (SPEC §2.13.8, §9.7): predict and score the simulated validation partition of
        each of the run's datasets, write <Evaluations or --out>/<Run.Name>/validation/ and print the report. It
        is the only input to selection and tuning; it never reads the test partition or a recording.

    Args:
        (1) args [argparse.Namespace]: --run, --out.

    Returns:
        (1) parameter [int]: 0 written; 1 refused.
    """

    try:
        training, run_manifest, config_bytes, datasets = _Load_Run(args.run)
        target = os.path.join(CONST_EVALUATIONS_DIR if args.out is None else args.out, training.Name, 'validation')
        if os.path.exists(target):
            raise Run_Error_Cls('The folder %s exists; an evaluation is never overwritten. Use --out or remove it.' % target)
        results, predictions = _Evaluate_Partition(training, datasets, 'validation', args.run)
        manifest = _Evaluation_Manifest(training, config_bytes, args.run, datasets, 'validation', len(results))
        _Write_Folder(target, lambda folder: _Write_Evaluation(folder, results, predictions, manifest))
    except (Configuration_Error_Cls, Run_Error_Cls, Dataset_Error_Cls, Prediction_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    for line in _Report_Lines(results):
        print(line)
    print('[INFO] Wrote %s. The test partition and the recordings were not read. %s' % (target, CONST_SAFETY), file=sys.stderr)
    return 0


def _Real_Flights() -> tp.List[tp.Any]:
    """
    Description:
        The 45 real validation flights of the F2 split, F2's segments untrimmed, in the QTM frame. Read only by
        'evaluate --partition test' (SPEC §2.13.9, §2.14); a leakage test replaces it.

    Returns:
        (1) parameter [List<Flight_Str>]: The flights, in the order of the recordings.
    """

    return [flight for set_name, flight, _ in Segment_Flights(Load_Recordings(CONST_RECORDINGS_DIR))
            if flight is not None and set_name == CONST_SET_VALIDATION]


def _Evaluate_Test(args: argparse.Namespace) -> int:
    """
    Description:
        'evaluate --partition test', the final evaluation (SPEC §2.13.9, §9.7): refuse a run made from a tree with uncommitted changes, a
        dataset whose hash differs and an existing output folder; then append the evaluation to the test ledger
        and predict and score the simulated test partition of each dataset and the 45 real validation flights
        (their first 40 ms trimmed, in the robot frame by the assumed transform; the metrics of SPEC §2.14
        only), check their coverage of SPEC §2.13.6, and write <Evaluations or --out>/<Run.Name>/test/ with
        coverage.csv.

    Args:
        (1) args [argparse.Namespace]: --run, --out.

    Returns:
        (1) parameter [int]: 0 written; 1 refused.
    """

    try:
        training, run_manifest, config_bytes, datasets = _Load_Run(args.run)
        if run_manifest.get('git_dirty') is not False:
            raise Run_Error_Cls('The run %s was made from a tree with uncommitted changes (git_dirty in its manifest); evaluate --partition '
                                'test scores only a run of a committed tree (SPEC §2.13.9). Commit, then train a new run.' % args.run)
        if run_manifest.get('device') not in (None, 'cpu'):
            raise Run_Error_Cls('The run %s was trained on the device %s; every reported number comes from the CPU, so evaluate '
                                '--partition test scores only a run trained on cpu (SPEC §7.8). Train it with Run.Device cpu.'
                                % (args.run, run_manifest.get('device')))
        parent = CONST_EVALUATIONS_DIR if args.out is None else args.out
        target = os.path.join(parent, training.Name, 'test')
        if os.path.exists(target):
            raise Run_Error_Cls('The folder %s exists; an evaluation is never overwritten. Use --out or remove it.' % target)
        os.makedirs(parent, exist_ok=True)
        ledger = os.path.join(parent, CONST_TEST_LEDGER)
        commit, dirty = _Git_State()
        evaluation = Append_Test_Ledger(ledger, training.Name, training.Model_Name, Sha256_Bytes(config_bytes), [d['hash'] for d in datasets],
                                        commit, dirty)
        print('[INFO] Test evaluation %d recorded in %s (%s%s).' % (evaluation, ledger, commit[:12], ', dirty' if dirty else ''), file=sys.stderr)
        results, predictions = _Evaluate_Partition(training, datasets, 'test', args.run)
        first = datasets[0]
        configuration, flight_model = first['configuration'], first['flight']
        flights = _Real_Flights()
        real = []
        for flight in flights:
            trimmed = Trim_Flight(flight, CONST_IDENTIFICATION_TRIM)
            real.append((Flight_Name(flight), float(flight.Rate), trimmed.Time,
                         Transform_To_Robot(trimmed.Position, Transform_Str=configuration.Transform)))
        print('[INFO] %d real validation flights (first %g ms trimmed) with %s.' % (len(real), 1e3 * CONST_IDENTIFICATION_TRIM, training.Model_Name),
              file=sys.stderr)
        r, p = Evaluate_Real(real, _Predictors(training, first, args.run), training.Observed_Samples)
        results, predictions = results + r, predictions + p
        fitted = Validate_Flight_Model(flights, Flight_Model_Str=flight_model, trim=CONST_IDENTIFICATION_TRIM)
        coverage = Coverage([Flight_Name(f) for f in flights], Robot_States(fitted, configuration), configuration, first['dataset'],
                            Flight_Model_Str=flight_model)
        manifest = _Evaluation_Manifest(training, config_bytes, args.run, datasets, 'test', len(results))
        manifest['test_evaluation'], manifest['real_flights'] = evaluation, len(real)
        manifest['coverage_inside'] = sum(1 for _, reason in coverage if reason == '')

        def Write(folder: str) -> None:
            """
            Description:
                The evaluation's files and coverage.csv.

            Args:
                (1) folder [string]: The folder.
            """

            Write_Coverage(os.path.join(folder, 'coverage.csv'), coverage)
            manifest['sha256'] = dict(manifest['sha256'], **{'coverage.csv': Sha256_File(os.path.join(folder, 'coverage.csv'))})
            _Write_Evaluation(folder, results, predictions, manifest)

        _Write_Folder(target, Write)
    except (Configuration_Error_Cls, Run_Error_Cls, Dataset_Error_Cls, Prediction_Error_Cls, Recording_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    for line in _Report_Lines(results):
        print(line)
    print('coverage (SPEC §2.13.6): %d of %d real validation flights inside the support of the generated throws'
          % (manifest['coverage_inside'], len(coverage)))
    for name, reason in coverage:
        if reason:
            print('  outside: %s: %s' % (name, reason))
    for result in results:
        if result.Partition == 'real' and result.Crossing == 'refused':
            print('refused (information, not a bound): %s N=%d: %s' % (result.Flight, result.N, result.Refusal))
    print('[INFO] Wrote %s. Real flights rest on the assumed transform (SPEC §7.3); no catch metric on them (SPEC §2.14). %s'
          % (target, CONST_SAFETY), file=sys.stderr)
    return 0


def _Stored_Evaluation(folder: str) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Any, tp.Optional[tp.Dict[str, bool]]]:
    """
    Description:
        One stored evaluation for a comparison (SPEC §9.7): its manifest, its table and, for an evaluation of
        'test', its coverage. Only the files of the evaluation's folder are read (manifest.json, per_flight.csv,
        per_horizon.csv, coverage.csv), each checked against the SHA-256 its manifest holds; no recording, no
        dataset and no run is opened. An evaluation of a predictor that is not one of the program's two (a stored
        one of LS, QT_MLP or HYBRID) is refused before any of its files is read.

    Args:
        (1) folder [string]: The evaluation folder (<...>/<Run.Name>/validation or test).

    Returns:
        (1) parameter 1 [Dict<string, Any>]: The evaluation's manifest.
        (2) parameter 2 [Evaluation_Table_Str(object)]: Its table (Evaluation.Comparison.Build_Table).
        (3) parameter 3 [Dict<string, bool>]: Per real flight whether it is inside the support, or None.
    """

    try:
        manifest = Read_Manifest(os.path.join(folder, 'manifest.json'))
    except (OSError, ValueError) as error:
        raise Run_Error_Cls('%s is not an evaluation folder (%s): it holds manifest.json and per_flight.csv written by '
                            'evaluate (SPEC §9.7).' % (folder, error)) from error
    if manifest.get('kind') != 'evaluation' or not manifest.get('datasets'):
        raise Run_Error_Cls('%s holds the manifest of a %s, not of an evaluation; give the folder <Run.Name>/validation or '
                            '<Run.Name>/test (SPEC §9.7).' % (folder, manifest.get('kind')))
    if manifest.get('model') not in CONST_MODEL_NAMES + CONST_LEARNED_MODEL_NAMES:
        raise Run_Error_Cls('The evaluation %s is of the predictor %s; the program compares evaluations of %s only (LS, QT_MLP and '
                            'HYBRID were removed, owner, 2026-11-22; their recorded results are in SPEC §4).'
                            % (folder, manifest.get('model'), ' and '.join(CONST_MODEL_NAMES + CONST_LEARNED_MODEL_NAMES)))
    names = ['per_flight.csv', 'per_horizon.csv'] + (['coverage.csv'] if manifest.get('partition') == 'test' else [])
    for name in names:
        try:
            found = Sha256_File(os.path.join(folder, name))
        except OSError as error:
            raise Run_Error_Cls('The evaluation %s has no %s (%s). Evaluate the run again.' % (folder, name, error)) from error
        if found != manifest.get('sha256', {}).get(name):
            raise Run_Error_Cls('The file %s of the evaluation %s has the SHA-256 %s, its manifest %s; it was changed after the '
                                'evaluation. Evaluate the run again.' % (name, folder, found, manifest.get('sha256', {}).get(name)))
    rows = Read_Per_Flight(os.path.join(folder, 'per_flight.csv'))
    real = Read_Per_Horizon(os.path.join(folder, 'per_horizon.csv'), CONST_REAL) if 'coverage.csv' in names else []
    table = Build_Table(manifest['run_name'], os.path.basename(os.path.normpath(manifest['datasets'][0]['path'])), rows, real)
    return manifest, table, Read_Coverage(os.path.join(folder, 'coverage.csv')) if 'coverage.csv' in names else None


def _Compare(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'compare' (SPEC §9.7; F14): compare stored evaluations. It reads the
        evaluation folders given and nothing else, never a recording, a dataset or a run, so it is not a test
        evaluation and appends nothing to the test ledger. It writes <Comparisons or --out>/<--name>/ with
        summary.csv (per run, partition, dataset, support and N), paired.csv (each run minus each run of
        --against, with bootstrap intervals), sim_to_real.csv (a run's real-flight error against its own
        simulated one) and manifest.json, and prints the report. The real flights are also reported inside and
        outside the support of the generated throws, from the evaluations' coverage.csv, which must agree.

    Args:
        (1) args [argparse.Namespace]: --evaluations, --against, --name, --out.

    Returns:
        (1) parameter [int]: 0 written; 1 refused (a folder that is not an evaluation, a file changed after its
                             evaluation, a run given twice, an unknown --against, coverages that differ, an
                             existing target, a --name that is not a folder name).
    """

    try:
        if not CONST_NAME_PATTERN.match(args.name):
            raise Run_Error_Cls('--name is "%s"; a comparison\'s folder is named with letters, digits, _ and - only.' % args.name)
        tables, inputs, coverage, covered = [], {}, None, ''
        for folder in args.evaluations:
            manifest, table, own = _Stored_Evaluation(folder)
            if table.Name in inputs:
                raise Run_Error_Cls('The run %s is given twice (%s); a comparison holds each run once.' % (table.Name, folder))
            inputs[table.Name] = Sha256_File(os.path.join(folder, 'manifest.json'))
            tables.append(table)
            if own is not None and coverage is None:
                coverage, covered = own, table.Name
            elif own is not None and own != coverage:
                raise Run_Error_Cls('The coverage of the real flights differs between %s and %s; the support split needs one support '
                                    '(SPEC §2.13.6). Compare runs whose first datasets share their throws\' configuration.'
                                    % (covered, table.Name))
        unknown = [name for name in args.against if name not in inputs]
        if unknown:
            raise Run_Error_Cls('--against names %s, which is not among the evaluations given (%s).'
                                % (', '.join(unknown), ', '.join(sorted(inputs))))
        by_name = {table.Name: table for table in tables}
        summary = [row for table in tables for row in Comparison_Summary(table, coverage)]
        paired = [row for name in args.against for table in tables if table.Name != name
                  for row in Comparison_Paired(table, by_name[name], coverage)]
        sim_to_real = [row for table in tables for row in Comparison_Sim_To_Real(table, 'test')]
        commit, dirty = _Git_State()
        manifest = Build_Manifest('comparison', {'%s/manifest.json' % name: digest for name, digest in inputs.items()},
                                  {'name': args.name, 'runs': [table.Name for table in tables], 'against': list(args.against),
                                   'bootstrap_seed': CONST_BOOTSTRAP_SEED, 'bootstrap_resamples': CONST_BOOTSTRAP_RESAMPLES,
                                   'coverage_inside': None if coverage is None else sum(coverage.values()),
                                   'real_flights': None if coverage is None else len(coverage)}, commit, dirty)

        def Write(folder: str) -> None:
            """
            Description:
                The comparison's files.

            Args:
                (1) folder [string]: The folder.
            """

            files = dict(manifest['sha256'])
            for name, columns, rows in (('summary.csv', CONST_COMPARISON_SUMMARY_COLUMNS, summary),
                                        ('paired.csv', CONST_COMPARISON_PAIRED_COLUMNS, paired),
                                        ('sim_to_real.csv', CONST_COMPARISON_SIM_TO_REAL_COLUMNS, sim_to_real)):
                Write_Comparison_Table(os.path.join(folder, name), columns, rows)
                files[name] = Sha256_File(os.path.join(folder, name))
            manifest['sha256'] = files
            Write_Manifest(os.path.join(folder, 'manifest.json'), manifest)

        target = os.path.join(CONST_COMPARISONS_DIR if args.out is None else args.out, args.name)
        _Write_Folder(target, Write)
    except (Run_Error_Cls, Report_Error_Cls, Comparison_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    for line in Report_Lines(summary, paired, sim_to_real):
        print(line)
    if coverage is not None:
        print('coverage (SPEC §2.13.6): %d of %d real validation flights inside the support of the generated throws'
              % (sum(coverage.values()), len(coverage)))
        for flight, inside in coverage.items():
            if not inside:
                print('  outside: %s' % flight)
    print('[INFO] Wrote %s from %d stored evaluation(s); no recording, dataset or run was read. Real flights rest on the assumed '
          'transform (SPEC §7.3). %s' % (target, len(tables), CONST_SAFETY), file=sys.stderr)
    return 0


def _Evaluate(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'evaluate' (SPEC §2.13.8, §2.13.9, §9.7): predict and score a run.
        --partition validation (the default): the simulated validation partition of each of the run's datasets
        (_Evaluate_Validation); it never reads the test partition or a recording. --partition test: the final
        evaluation, the simulated test partition and the 45 real validation flights, counted in the test ledger
        (_Evaluate_Test).

    Args:
        (1) args [argparse.Namespace]: --run, --partition, --out.

    Returns:
        (1) parameter [int]: 0 written; 1 refused.
    """

    return _Evaluate_Test(args) if args.partition == 'test' else _Evaluate_Validation(args)


def main():
    """
    Description:
        The user's entry point. Sub-commands, in the order of use:

            .venv/bin/python scripts/catch.py dataset [--config Config/dataset.yaml] [--out <dir>]
            .venv/bin/python scripts/catch.py train [--config Config/training.yaml] [--out <dir>]
            .venv/bin/python scripts/catch.py evaluate --run <run dir> [--partition validation|test] [--out <dir>]
            .venv/bin/python scripts/catch.py compare --evaluations <evaluation dir> [<evaluation dir> ...] --name <name>
                                                      [--against <run> ...] [--out <dir>]
            .venv/bin/python scripts/catch.py catch --run <run dir> [--partition validation|test] [--workers <n>]
                                                    [--config Config/ball_catching.yaml] [--out <dir>]
            .venv/bin/mjpython scripts/catch.py view --run <run dir> (--index <i> | --random [--pick-seed <s>] | --new <seed>)
                                                     [--compare <run> ...] [--theme dark|light] [--camera side|overview|cup|thrower]
            .venv/bin/python scripts/catch.py view --run <run dir> ... --frames <dir> [--size <w>x<h>] [--fps <f>] [--speed <x>] [--overlay]

        'evaluate --partition test' and 'catch --partition test' are counted reads, each in its ledger. 'view' opens
        MuJoCo's viewer, which on macOS needs MuJoCo's launcher (.venv/bin/mjpython); with --frames it writes PNG frames
        and needs no launcher. The project's derivations (recordings, identify, observation, workspace, scene,
        dataset-check, sweep) are the sub-commands of scripts/Maintenance/derive.py. An option is given by its full
        name: an abbreviation is a usage error.

        Exit codes: 0 done; 1 refused (invalid input, nothing written); 2 a usage error (argparse).
        The report goes to stdout, the [INFO] / [WARNING] / [ERROR] lines to stderr.
    """

    parser = argparse.ArgumentParser(description='Robotic ball catching in simulation: generate data, train, evaluate, compare, catch, watch.',
                                     allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    dataset = commands.add_parser('dataset', allow_abbrev=False, help='generate a dataset of simulated flights')
    dataset.add_argument('--config', default=CONST_DATASET_CONFIG_FILE, help='the dataset configuration (default: Config/dataset.yaml)')
    dataset.add_argument('--out', default=None, help='the parent folder of the dataset instead of Output.Folder')
    train = commands.add_parser('train', allow_abbrev=False, help='train the learned predictor, one model per seed, into a run folder (the '
                                                                  'numerical predictor: the run folder only)')
    train.add_argument('--config', default=CONST_TRAINING_CONFIG_FILE, help='the training configuration (default: Config/training.yaml)')
    train.add_argument('--out', default=None, help='the parent folder of the run instead of Output.Folder')
    evaluate = commands.add_parser('evaluate', allow_abbrev=False, help='predict and score a run: the simulated validation partition, or '
                                                                        'with --partition test the final evaluation (COUNTED)')
    evaluate.add_argument('--run', required=True, metavar='DIR', help='the run folder (written by train)')
    evaluate.add_argument('--partition', default='validation', choices=('validation', 'test'),
                          help='validation (default); test: the simulated test partition and the 45 real validation flights, counted in '
                               'the test ledger, a run of a committed tree only')
    evaluate.add_argument('--out', default=None, help='the parent folder of the evaluations and the ledger instead of Data/Generated/Evaluations')
    compare = commands.add_parser('compare', allow_abbrev=False, help='compare stored evaluations: the summary per run, the paired '
                                                                      'differences with bootstrap intervals, sim-to-real')
    compare.add_argument('--evaluations', required=True, nargs='+', metavar='DIR',
                         help='the evaluation folders, <...>/<Run.Name>/validation or <...>/<Run.Name>/test')
    compare.add_argument('--name', required=True, help='the comparison\'s folder under Data/Generated/Comparisons')
    compare.add_argument('--against', default=[], nargs='+', metavar='RUN',
                         help='the reference run(s) every other run is paired with (default: none, no paired table)')
    compare.add_argument('--out', default=None, help='the parent folder of the comparison instead of Data/Generated/Comparisons')
    catch = commands.add_parser('catch', allow_abbrev=False, help='the whole chain on the simulated throws of Decision.Datasets: the '
                                                                  'decision, the interception and the physical catch of every go')
    catch.add_argument('--run', required=True, metavar='DIR', help='the run folder (written by train)')
    catch.add_argument('--partition', default='validation', choices=('validation', 'test'),
                       help='validation (default: calibrates the margins); test: the stored validation margins, a committed tree, COUNTED in '
                            'the catch ledger')
    catch.add_argument('--config', default=CONST_CONFIG_FILE, help='the system configuration (default: Config/ball_catching.yaml)')
    catch.add_argument('--out', default=None, help='one folder holding Decisions/, Interceptions/, Catches/ and the ledger instead of the '
                                                   'configured folders')
    catch.add_argument('--workers', default=1, type=int, help='the processes the throws run in (default: 1; the outputs are the same)')
    view = commands.add_parser('view', allow_abbrev=False, help='one simulated throw computed with the code of catch and played back in '
                                                                'MuJoCo\'s viewer (macOS: start it with .venv/bin/mjpython), or written '
                                                                'as PNG frames (--frames)')
    view.add_argument('--run', required=True, metavar='DIR', help='the run whose predictor drives the robot (written by train; its validation '
                                                                 'interception must be stored)')
    view.add_argument('--dataset', default=None, metavar='DIR', help='the stored dataset of the throw (default: the first of Decision.Datasets)')
    view.add_argument('--index', default=None, type=int, metavar='I', help='the stored throw\'s index (train or validation; a test throw is refused)')
    view.add_argument('--random', action='store_true', help='pick a stored train or validation throw (see --pick-seed)')
    view.add_argument('--pick-seed', default=None, type=int, metavar='S', help='the seed of --random (printed when not given, to repeat the pick)')
    view.add_argument('--new', default=None, type=int, metavar='SEED', help='a new throw generated from --dataset-config with this seed')
    view.add_argument('--aim', default='catch', choices=('catch', 'anywhere'), help='where a new throw is aimed: into the catch area '
                                                                                   '(default) or anywhere in the file\'s target area')
    view.add_argument('--dataset-config', default=CONST_DATASET_CONFIG_FILE, help='the dataset configuration of a new throw '
                                                                                  '(default: Config/dataset.yaml)')
    view.add_argument('--seed', default=None, type=int, help='the training seed of a learned predictor (default: the run\'s first)')
    view.add_argument('--acceleration', default=None, type=float, metavar='DEG_S2', help='one of Robot.Joint_Accelerations (default: the first)')
    view.add_argument('--compare', default=[], nargs='+', metavar='RUN', help='further runs whose predictions are shown beside the run\'s, '
                                                                             'for the display only')
    view.add_argument('--every', default=1, type=int, metavar='K', help='the display predicts at every K-th observed sample (default: 1)')
    view.add_argument('--theme', default='dark', help='dark (default) or light')
    view.add_argument('--camera', default='side', help='the camera preset: side (default), overview, cup or thrower')
    view.add_argument('--speed', default=1.0, type=float, help='the playback speed against real time (default: 1; 0.1 is ten times slower)')
    view.add_argument('--loop', action='store_true', help='the window: start again from the release at the end')
    view.add_argument('--ghost', action='store_true', help='show the goal of the plan being executed as a ghost pose')
    view.add_argument('--markers', action='store_true', help='draw the ball\'s marked mesh instead of its collision sphere (the markers '
                                                             'are visual only and show through the cup)')
    view.add_argument('--no-zones', action='store_true', help='do not draw the throw, flight and catch areas')
    view.add_argument('--no-labels', action='store_true', help='do not draw the labels of the zones and of the two markers')
    view.add_argument('--frames', default=None, metavar='DIR', help='write PNG frames and frames.csv into DIR instead of opening the window')
    view.add_argument('--size', default='1920x1080', metavar='WxH', help='the frames\' size in pixels, up to 3840x2160 (default: 1920x1080)')
    view.add_argument('--fps', default=60.0, type=float, help='the frames per second of the video the frames are for (default: 60)')
    view.add_argument('--overlay', action='store_true', help='burn the text overlay and the legend into the frames (default: clean frames)')
    view.add_argument('--from', dest='start', default=None, type=float, metavar='T', help='the first time shown [s] (default: the release)')
    view.add_argument('--to', dest='end', default=None, type=float, metavar='T', help='the last frame\'s time [s] (default: the record\'s end)')
    view.add_argument('--config', default=CONST_CONFIG_FILE, help='the system configuration (default: Config/ball_catching.yaml)')
    view.add_argument('--interceptions', default=None, help='the parent folder of the stored interception instead of Interception.Output_Folder')
    view.add_argument('--catches', default=None, help='the parent folder of the stored catch (the stored check) instead of Contact.Output_Folder')
    args = parser.parse_args()
    if args.command == 'dataset':
        return _Dataset(args)
    if args.command == 'train':
        return _Train(args)
    if args.command == 'evaluate':
        return _Evaluate(args)
    if args.command == 'compare':
        return _Compare(args)
    if args.command == 'catch':
        return _Catch(args)
    return _View(args)


if __name__ == '__main__':
    sys.exit(main())
