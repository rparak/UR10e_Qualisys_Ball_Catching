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
# File Name: ../scripts/Maintenance/derive.py
## =========================================================================== ##

# System (Default)
import sys
# OS (Operating system interfaces)
import os
# The repository root, built from this file's location, so the script runs from any working
# directory.
#   Unit [string]
CONST_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
#   Add access if it is not in the system path.
if os.path.join(CONST_REPO_ROOT, 'src') not in sys.path:
    sys.path.append(os.path.join(CONST_REPO_ROOT, 'src'))
# Typing (Support for type hints)
import typing as tp
# Argparse (Command-line parsing)
import argparse
# Functools (the sweep's injected predictor with its table and scale)
import functools
# Importlib (scripts/catch.py loaded as a module, for its helpers)
import importlib.util
# Shutil (High-level file operations; the temporary folder of dataset-check)
import shutil
# Tempfile (Temporary folders; a dataset is regenerated into one)
import tempfile
# Time (the times printed)
import time
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Lib.:
#   ../Configuration/Core - the only reader of the configuration
from Configuration.Core import Load_Configuration, Load_Dataset_Configuration, Configuration_Error_Cls
#   ../Recording/Core, Segmentation, Report - the recordings, their flights and the reports
from Recording.Core import Load_Recordings, Recording_Error_Cls, CONST_SET_VALIDATION
from Recording.Segmentation import Segment_Flights
from Recording.Report import (Write_Recording_Summary, Write_Recording_Steps, Summary_Lines, Write_Identification_Summary,
                              Write_Identification_Flights, Append_Ledger, Identification_Lines)
#   ../Recording/Identification - the flight-model identification (F3)
from Recording.Identification import Identify_Flight_Model, Validate_Flight_Model, Session_Name, CONST_IDENTIFICATION_TRIM
#   ../Dataset/Core, Manifest, Calibration - the identification flights, a dataset and its train partition, the manifests,
#   the calibration of the generation models (F6)
from Dataset.Core import Identification_Flights, Dataset_Error_Cls, Read_Dataset, Partition_Indices, CONST_PARTITION_TRAIN
from Dataset.Manifest import Build_Manifest, Read_Manifest, Sha256_Bytes, Sha256_File, Write_Manifest
from Observation.Core import Observation_Error_Cls
from Dataset.Calibration import Calibrate, Calibration_Error_Cls, Summary_Lines as Observation_Lines, Write_Observation_Summary
#   ../Evaluation/Report, Comparison - the stored per-horizon errors (the sweep) and the real flights' partition
from Evaluation.Report import Read_Per_Horizon, Report_Error_Cls
from Evaluation.Comparison import CONST_REAL


def _Load_Catch_Script() -> tp.Any:
    """
    Description:
        The module of scripts/catch.py, loaded from its file: the derivations share the user script's helpers (the
        models from the configuration, the git state, a run's copy, the decision's datasets, the interception's setup,
        the worker pool, the dataset's generation). One direction only: this script loads that one, never the reverse.
        A helper is called through the module, so a test's guard set on that module acts here.

    Returns:
        (1) parameter [module]: The module.
    """

    specification = importlib.util.spec_from_file_location('catch_script', os.path.join(CONST_REPO_ROOT, 'scripts', 'catch.py'))
    module = importlib.util.module_from_spec(specification)
    sys.modules['catch_script'] = module
    specification.loader.exec_module(module)
    return module


"""
Description:
    Initialization of constants.

    The project's derivations, kept for reproducibility (SPEC §6): what the shipped
    configuration and the catch area were derived from, and the checks of what is stored. One sub-command each:
    recordings, identify (COUNTED in its ledger), observation, workspace, scene, dataset-check, sweep. A user who
    generates data, trains, evaluates, compares, catches and watches needs none of them: scripts/catch.py.
"""
# The user script's module (_Load_Catch_Script) and the constants taken from it: one source each.
#   Unit [module], [string]
CONST_CATCH_SCRIPT = _Load_Catch_Script()
CONST_CONFIG_FILE = CONST_CATCH_SCRIPT.CONST_CONFIG_FILE
CONST_DATASET_CONFIG_FILE = CONST_CATCH_SCRIPT.CONST_DATASET_CONFIG_FILE
CONST_RECORDINGS_DIR = CONST_CATCH_SCRIPT.CONST_RECORDINGS_DIR
CONST_SAFETY = CONST_CATCH_SCRIPT.CONST_SAFETY


def _Scene(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'scene' (F4): build the MuJoCo scene (the stand, the vendored UR10e with the cup,
        the ball, the rail, the cameras and the areas of the configuration), print its summary on stdout
        and compare its fingerprint with the stored one (models/scene_fingerprint.json; --fingerprint FILE
        names another). Nothing is written. MuJoCo is imported here only, so that every other sub-command
        starts without it.

    Args:
        (1) args [argparse.Namespace]: --config, --fingerprint.

    Returns:
        (1) parameter [int]: 0 the fingerprint is identical; 1 it differs (each difference named); 2 could
                             not check (an invalid configuration, a scene that cannot be built, a
                             fingerprint that cannot be read).
    """

    #   ../Scene/Core - the scene's builder and its fingerprint
    from Scene.Core import (Build_Scene, Fingerprint, Compare_Fingerprints, Read_Fingerprint, Summary_Lines, Scene_Error_Cls,
                            CONST_MODELS_DIR, CONST_FINGERPRINT_FILE)
    #   ../Robot/Core - the robot's error
    from Robot.Core import Robot_Error_Cls

    try:
        configuration = Load_Configuration(args.config)
        started = time.perf_counter()
        model = Build_Scene(Areas_Str=configuration.Areas)
        seconds = time.perf_counter() - started
        lines = Summary_Lines(model)
        stored = Read_Fingerprint(args.fingerprint or os.path.join(CONST_MODELS_DIR, CONST_FINGERPRINT_FILE))
    except (Configuration_Error_Cls, Scene_Error_Cls, Robot_Error_Cls) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 2
    for level, message in configuration.Notes:
        print('[%s] %s' % (level, message), file=sys.stderr)
    differences = Compare_Fingerprints(stored, Fingerprint(model))
    for line in lines:
        print(line)
    print('Fingerprint: %s' % ('identical to the stored one' if not differences else 'DIFFERS from the stored one in %d fields'
                               % len(differences)))
    for difference in differences:
        print('    ' + difference)
    print('[INFO] The scene was built in %.2f s. %s' % (seconds, CONST_SAFETY), file=sys.stderr)
    if differences:
        print('[ERROR] The compiled scene is not the one that was pinned. A decided change of the scene is written with '
              'scripts/Maintenance/build_scene.py --write, in its own commit.', file=sys.stderr)
        return 1
    return 0


def _Workspace(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'workspace' (F4; SPEC §7.11, §9.11): derive where the robot may catch. Every point of
        the lattice (Catch.Workspace_Step) within the robot's reach is checked with the closed-form inverse
        kinematics for all 25 arrival directions of the configured cone. The ready pose and the catch area
        are chosen together (the rule B800; Search_Catch_Area): the largest
        cuboid of the qualifying points reached within CONST_REACH_TIME from the ready pose at the first of
        Robot.Joint_Accelerations, shrunk by one step on every face, the ready pose the one that makes it the
        largest. The throws of the train partition of Workspace.Dataset (no other partition is read) give the
        measured arrival cone and the reported shares: the throws through the area, and those catchable in it
        and anywhere, of all throws and of those through the area, at both Robot.Joint_Accelerations. Writes
        Workspace.Map_File, Workspace.Summary_File and manifest.json beside them (--out DIR moves the three)
        and prints the report. A configured Areas.Catch or Robot.Ready_Pose that differs from the derived one,
        or an arrival cone that does not hold the measured one, is a WARNING (a slow test pins the area). No
        recording is read.

    Args:
        (1) args [argparse.Namespace]: --config, --out.

    Returns:
        (1) parameter [int]: 0 written; 1 refused (an invalid configuration, a missing or unusable dataset,
                             no qualifying cuboid; nothing written).
    """

    # MuJoCo (Physics simulation; a data for the ready pose's forward kinematics) [pip3 install mujoco]
    import mujoco
    #   ../Scene/Core - the scene and its robot
    from Scene.Core import Build_Scene, Scene_Robot, Scene_Error_Cls
    #   ../Robot/Core - the forward kinematics, the robot's error
    from Robot.Core import Forward_Kinematics, Robot_Error_Cls
    #   ../Catch/Core, Workspace - the catch pose, the lattice, the map, the reach and the search
    from Catch.Core import (Catch_Context, Arrival_Axes, Lattice, Lattice_Domain, Largest_Cuboid, Catch_Error_Cls, CONST_CRITERIA,
                            CONST_JOINTS_USED)
    from Catch.Workspace import (Derive_Workspace, Configured_Cuboid, Flight_Samples, Cell_Visits, Through_Share, Box_Share, Reach_Times,
                                 Catchable_Share, Search_Catch_Area, Write_Workspace_Map, Write_Workspace_Summary, Joint_Wrap,
                                 CONST_REACH_TIME, CONST_DECISION_SAMPLE, CONST_ALLOWANCE)

    started = time.perf_counter()
    try:
        configuration = Load_Configuration(args.config)
        model = Build_Scene(Areas_Str=configuration.Areas)
        robot = Scene_Robot(model)
        context = Catch_Context(model, robot, Criteria_Str=CONST_CATCH_SCRIPT._Catch_Criteria(configuration))
        data_file = os.path.join(CONST_REPO_ROOT, configuration.Workspace.Dataset, 'flights.npz')
        data = Read_Dataset(data_file)
    except (Configuration_Error_Cls, Scene_Error_Cls, Robot_Error_Cls, Dataset_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    for level, message in configuration.Notes:
        print('[%s] %s' % (level, message), file=sys.stderr)
    catch, robot_limits, areas = configuration.Catch, configuration.Robot, configuration.Areas
    paths = _Output_Paths([configuration.Workspace.Map_File, configuration.Workspace.Summary_File], args.out)
    step = catch.Workspace_Step
    axes = Arrival_Axes(catch.Arrival_Elevation, catch.Arrival_Azimuth)
    centre, reach = Lattice_Domain(context)
    points, index = Lattice(centre, reach, step, 0.0)
    print('[INFO] Checking %d lattice points (step %.3f m, reach %.3f m about the shoulder) for %d arrival directions; about 8 '
          'minutes.' % (len(points), step, reach, len(axes)), file=sys.stderr)

    # 1\ The map; the largest cuboid of the qualifying points without the time, for comparison (F4 Q5 A's area).
    workspace = Derive_Workspace(points, index, step, axes, Catch_Context_Str=context,
                                 progress=lambda done, total, good: print('[INFO] %d of %d points, %d qualify (%.0f s).'
                                                                          % (done, total, good, time.perf_counter() - started),
                                                                          file=sys.stderr))
    derived_seconds = time.perf_counter() - started
    speed = np.array(robot_limits.Joint_Speed[:CONST_JOINTS_USED])
    wrap = Joint_Wrap(context)
    nominal = robot_limits.Joint_Accelerations[0]
    try:
        static_lower, static_upper = Largest_Cuboid(workspace.Index, workspace.Qualifies)
        # The ready pose and the area together, at the nominal acceleration (the rule B800).
        print('[INFO] The map took %.0f s; searching the ready pose and the catch area.' % derived_seconds, file=sys.stderr)
        ready, _, lower, upper, _ = Search_Catch_Area(workspace, speed, nominal, wrap, len(axes) // 2)
        area_centre, area_size = Configured_Cuboid(lower, upper, step)
    except Catch_Error_Cls as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1

    # 2\ The throws of the train partition only.
    train = Partition_Indices(data, CONST_PARTITION_TRAIN)
    decision = np.full(len(train), np.nan)
    for row, flight in enumerate(train):
        times = data.Observed_Time[data.Observed_Offset[flight]:data.Observed_Offset[flight + 1]]
        if len(times) >= CONST_DECISION_SAMPLE:
            decision[row] = times[CONST_DECISION_SAMPLE - 1]
    sample_times, states, valid = Flight_Samples(data.Release_State[train], data.Floor_Time[train],
                                                 Flight_Model_Str=CONST_CATCH_SCRIPT._Flight_Model(configuration))
    positions = states[..., :3]
    visits = Cell_Visits(positions, valid, workspace)
    flights = len(train)

    # 3\ The arrival cone of the throws inside the 2022 hemisphere, its 0.5 and 99.5 % quantiles.
    robot_area = areas.Robot
    inside = valid & (np.linalg.norm(positions - np.array(robot_area.Centre), axis=-1) <= robot_area.Radius) \
        & (positions[..., 2] >= robot_area.Centre[2])
    velocity = states[..., 3:][inside]
    elevation = np.degrees(np.arcsin(-velocity[:, 2] / np.linalg.norm(velocity, axis=1)))
    azimuth = np.degrees(np.arctan2(velocity[:, 0], velocity[:, 1]))
    measured = np.percentile(elevation, [0.5, 99.5]), np.percentile(azimuth, [0.5, 99.5])

    # 4\ The ready pose's cup, the area's cells; the reach at both accelerations.
    ready_full = np.concatenate([ready, np.zeros(len(robot.Joint_Id) - CONST_JOINTS_USED)])
    ready_tcp, _ = Forward_Kinematics(model, mujoco.MjData(model), ready_full, Robot_Str=robot)
    half = 0.5 * np.array(area_size)
    in_area = np.all(np.abs(workspace.Points - np.array(area_centre)) <= half + 1e-9, axis=1) & workspace.Qualifies
    through_area = Through_Share(visits[0], visits[1], in_area, flights)
    rows = [('lattice_points', '%d' % len(points), '-'), ('lattice_step', '%.3f' % step, 'm'), ('arrival_directions', '%d' % len(axes), '-'),
            ('qualifying_points', '%d' % workspace.Qualifies.sum(), '-'),
            ('qualifying_volume', '%.4f' % (workspace.Qualifies.sum() * step ** 3), 'm^3'),
            ('flights_through_qualifying', '%.4f' % Through_Share(visits[0], visits[1], workspace.Qualifies, flights), '-')]
    hemisphere = (np.linalg.norm(workspace.Points - np.array(robot_area.Centre), axis=1) <= robot_area.Radius) \
        & (workspace.Points[:, 2] >= robot_area.Centre[2])
    rows += [('hemisphere_points', '%d' % hemisphere.sum(), '-'),
             ('hemisphere_qualifying_share', '%.4f' % workspace.Qualifies[hemisphere].mean(), '-')]
    for code, name in enumerate(CONST_CRITERIA):
        rows.append(('hemisphere_failing_%s' % name, '%d' % np.sum(hemisphere & (workspace.Failed_Criterion == code)), '-'))
    rows += [('qualifying_cuboid_lower', ' '.join('%.3f' % v for v in static_lower * step), 'm'),
             ('qualifying_cuboid_upper', ' '.join('%.3f' % v for v in static_upper * step), 'm'),
             ('qualifying_cuboid_points', '%d' % np.prod(static_upper - static_lower + 1), '-'),
             ('reach_time', '%.3f' % CONST_REACH_TIME, 's'), ('search_acceleration', '%.0f' % np.degrees(nominal), 'deg/s^2'),
             ('largest_cuboid_lower', ' '.join('%.3f' % v for v in lower * step), 'm'),
             ('largest_cuboid_upper', ' '.join('%.3f' % v for v in upper * step), 'm'),
             ('largest_cuboid_points', '%d' % np.prod(upper - lower + 1), '-'),
             ('catch_area_centre', ' '.join('%.3f' % v for v in area_centre), 'm'),
             ('catch_area_size', ' '.join('%.3f' % v for v in area_size), 'm'),
             ('catch_area_volume', '%.4f' % float(np.prod(area_size)), 'm^3'),
             ('flights_through_catch_area', '%.4f' % Box_Share(positions, valid, area_centre, area_size), '-'),
             ('flights_through_catch_area_cells', '%.4f' % through_area, '-'),
             ('flights_through_hemisphere', '%.4f' % float(inside.any(axis=1).mean()), '-'),
             ('train_flights', '%d' % flights, '-'),
             ('measured_elevation', '%.2f %.2f' % tuple(measured[0]), 'deg'), ('measured_azimuth', '%.2f %.2f' % tuple(measured[1]), 'deg'),
             ('ready_pose', ' '.join('%.6f' % v for v in np.degrees(ready_full)), 'deg'),
             ('ready_pose_tcp', ' '.join('%.3f' % v for v in ready_tcp), 'm'),
             ('ready_pose_tcp_in_area', '%d' % bool(np.all(np.abs(ready_tcp - np.array(area_centre)) <= half + 1e-9)), '-')]
    for acceleration in robot_limits.Joint_Accelerations:
        reach_time = Reach_Times(workspace, ready, speed, acceleration, wrap)
        within = reach_time <= CONST_REACH_TIME
        label = '%.0f' % np.degrees(acceleration)
        in_area_share = Catchable_Share(*visits, sample_times, decision, reach_time, in_area, flights)
        rows += [('reach_%s_volume' % label, '%.4f' % (within.sum() * step ** 3), 'm^3'),
                 ('reach_%s_flights_through' % label, '%.4f' % Through_Share(visits[0], visits[1], within, flights), '-'),
                 ('reach_%s_catchable' % label,
                  '%.4f' % Catchable_Share(*visits, sample_times, decision, reach_time, workspace.Qualifies, flights), '-'),
                 ('reach_%s_catchable_in_area' % label, '%.4f' % in_area_share, '-'),
                 ('reach_%s_catchable_in_area_of_through' % label, '%.4f' % (in_area_share / through_area if through_area > 0.0 else np.nan), '-')]
    seconds = time.perf_counter() - started
    rows += [('map_seconds', '%.0f' % derived_seconds, 's'), ('seconds', '%.0f' % seconds, 's')]

    # 5\ The files: the map, the summary, the manifest.
    folder = os.path.dirname(paths[0]) or '.'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.dirname(paths[1]) or '.', exist_ok=True)
    Write_Workspace_Map(paths[0], workspace, {'largest_cuboid': np.stack([lower, upper]), 'catch_area': np.array([area_centre, area_size]),
                                              'ready_pose': ready_full})
    Write_Workspace_Summary(paths[1], rows)
    commit, dirty = CONST_CATCH_SCRIPT._Git_State()
    manifest = Build_Manifest('workspace', {'configuration': Sha256_File(args.config), 'dataset': Sha256_File(data_file),
                                            'map': Sha256_File(paths[0]), 'summary': Sha256_File(paths[1])},
                              {'partition': 'train', 'flights': int(flights), 'lattice_points': int(len(points))}, commit, dirty)
    Write_Manifest(os.path.join(folder, 'manifest.json'), manifest)
    for quantity, value, unit in rows:
        print('%-36s %s %s' % (quantity, value, unit))

    if _Differs(areas.Catch.Centre, area_centre) or _Differs(areas.Catch.Size, area_size):
        print('[WARNING] The configured Areas.Catch (centre %s, size %s) differs from the derived one (centre %s, size %s). Write it into %s '
              '(SPEC §8.1).' % (areas.Catch.Centre, areas.Catch.Size, area_centre, area_size, args.config), file=sys.stderr)
    if np.max(np.abs(np.degrees(np.array(robot_limits.Ready_Pose) - ready_full))) > 1e-6:
        print('[WARNING] The configured Robot.Ready_Pose %s deg differs from the searched one %s deg. Write it into %s (SPEC §8.1).'
              % (['%.6f' % v for v in np.degrees(robot_limits.Ready_Pose)], ['%.6f' % v for v in np.degrees(ready_full)], args.config),
              file=sys.stderr)
    configured_cone = np.degrees(catch.Arrival_Elevation), np.degrees(catch.Arrival_Azimuth)
    for name, cone, quantiles in (('Arrival_Elevation', configured_cone[0], measured[0]), ('Arrival_Azimuth', configured_cone[1], measured[1])):
        if quantiles[0] < cone[0] or quantiles[1] > cone[1]:
            print('[WARNING] The configured Catch.%s [%.1f, %.1f] deg does not hold the measured 0.5-99.5 %% quantiles [%.2f, %.2f] deg.'
                  % (name, cone[0], cone[1], quantiles[0], quantiles[1]), file=sys.stderr)
    print('[INFO] Wrote %s, %s and the manifest in %.0f s. Only the train partition of %s was read; no recording. %s'
          % (paths[0], paths[1], seconds, configuration.Workspace.Dataset, CONST_SAFETY), file=sys.stderr)
    return 0


def _Recordings(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'recordings' (F2): load the 90 recordings, segment their flights, write the
        summary and the steps (Recording.Summary_File, Recording.Steps_File; --out DIR moves both),
        and print the summary on stdout. Invalid input is refused before anything is written.

    Args:
        (1) args [argparse.Namespace]: --config, --out.

    Returns:
        (1) parameter [int]: 0 written; 1 refused (an invalid configuration or recording).
    """

    try:
        configuration = Load_Configuration(args.config)
        recordings = Load_Recordings(CONST_RECORDINGS_DIR)
    except (Configuration_Error_Cls, Recording_Error_Cls) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    for level, message in configuration.Notes:
        print('[%s] %s' % (level, message), file=sys.stderr)
    segmented = Segment_Flights(recordings)

    paths = _Output_Paths([configuration.Recording.Summary_File, configuration.Recording.Steps_File], args.out)
    Write_Recording_Summary(paths[0], recordings, segmented, Transform_Str=configuration.Transform)
    Write_Recording_Steps(paths[1], recordings)
    for line in Summary_Lines(recordings, segmented):
        print(line)
    for path in paths:
        print('[INFO] Wrote %s.' % path, file=sys.stderr)
    print('[INFO] Robot-frame points rest on the assumed transform (SPEC §7.3). %s' % CONST_SAFETY, file=sys.stderr)
    return 0


def _Output_Paths(configured: tp.List[str], out: tp.Optional[str]) -> tp.List[str]:
    """
    Description:
        The files a sub-command writes: the configured paths below the repository, or their names in
        --out DIR; their folders are created.

    Args:
        (1) configured [List<string>]: The configured, repository-relative paths.
        (2) out [string]: --out DIR, or None.

    Returns:
        (1) parameter [List<string>]: The paths.
    """

    paths = []
    for path in configured:
        path = os.path.join(CONST_REPO_ROOT, path) if out is None else os.path.join(out, os.path.basename(path))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        paths.append(path)
    return paths


def _Identify(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'identify' (F3): identify the flight model on the identification flights
        (Recording.Identification.Identify_Flight_Model), evaluate it on the validation flights with the
        identification's trim and untrimmed (no model parameter fitted on them), record the run in the
        ledger, write the summary and the flights (Identification.*; --out DIR moves all three) and print
        the report. Invalid input is refused before anything is written. A configured Flight.Drag that
        differs from the identified k is a WARNING (the slow test pins them equal).

    Args:
        (1) args [argparse.Namespace]: --config, --out.

    Returns:
        (1) parameter [int]: 0 written; 1 refused (an invalid configuration or recording).
    """

    try:
        configuration = Load_Configuration(args.config)
        recordings = Load_Recordings(CONST_RECORDINGS_DIR)
    except (Configuration_Error_Cls, Recording_Error_Cls) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    for level, message in configuration.Notes:
        print('[%s] %s' % (level, message), file=sys.stderr)
    output = configuration.Identification
    summary, flights, ledger = _Output_Paths([output.Summary_File, output.Flights_File, output.Ledger_File], args.out)

    print('[INFO] Identifying the flight model on the identification flights (about a minute).', file=sys.stderr)
    identification = Identify_Flight_Model(recordings, gravity=configuration.Flight.Gravity)
    model = identification.Flight_Model_Str
    sessions = {'%s/Experiment_%d.txt' % (recording.Folder, recording.Experiment): Session_Name(recording) for recording in recordings}
    validation_flights = [flight for set_name, flight, _ in Segment_Flights(recordings) if flight is not None and set_name == CONST_SET_VALIDATION]
    commit, dirty = CONST_CATCH_SCRIPT._Git_State()
    evaluation = Append_Ledger(ledger, commit, dirty, model)
    validation = [(trim, Validate_Flight_Model(validation_flights, Flight_Model_Str=model, trim=trim)) for trim in (CONST_IDENTIFICATION_TRIM, 0.0)]

    Write_Identification_Summary(summary, identification, validation, sessions, evaluation)
    Write_Identification_Flights(flights, [('identification', CONST_IDENTIFICATION_TRIM, identification.Variants[0].Fitted)]
                                 + [('validation', trim, fitted) for trim, fitted in validation], sessions)
    for line in Identification_Lines(identification, validation, sessions, evaluation):
        print(line)
    for path in (summary, flights, ledger):
        print('[INFO] Wrote %s.' % path, file=sys.stderr)
    print('[INFO] Validation evaluation %d recorded in %s (%s%s).' % (evaluation, ledger, commit[:12], ', dirty' if dirty else ''),
          file=sys.stderr)
    if abs(configuration.Flight.Drag - model.Drag) > 5e-9 or abs(configuration.Flight.Drag_Std - model.Drag_Std) > 5e-9:
        print('[WARNING] The configured Flight.Drag %.8f / Drag_Std %.8f 1/m differ from the identified %.8f / %.8f 1/m. Write the '
              'identified values into %s (SPEC §8.1).' % (configuration.Flight.Drag, configuration.Flight.Drag_Std, model.Drag,
                                                          model.Drag_Std, args.config), file=sys.stderr)
    print('[INFO] The flights are in the QTM frame; the model does not depend on the frame (SPEC §7.4). %s' % CONST_SAFETY, file=sys.stderr)
    return 0


def _Differs(configured: tp.Any, estimated: tp.Any) -> bool:
    """
    Description:
        Whether a configured value differs from its estimate at the 9 significant digits written into the
        configuration.

    Args:
        (1) configured [Any]: A number or a list of numbers.
        (2) estimated [Any]: A number or an array.

    Returns:
        (1) parameter [bool]: True when any element differs.
    """

    return ['%.9g' % v for v in np.ravel(configured)] != ['%.9g' % v for v in np.ravel(estimated)]


def _Observation(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'observation' (F6; SPEC §9.9): calibrate the generation models on the 45
        identification flights only (Dataset.Calibration.Calibrate): the error model M4's estimate and its
        spread, the error and dropout-pattern statistics and the throw shape against the identification
        flights' bootstrap bounds, the release distribution, and the coverage of SPEC §2.13.6; the largest
        values of the identification residuals and the error bounds derived from them. Writes
        Observation.Summary_File (--out DIR moves it) and prints the report. A configured Observation.* or
        Throw.Release_* / Apex_* value that differs from its estimate, or a bound that differs from the derived
        one, is a WARNING (a slow test pins them equal). The validation flights are never read.

    Args:
        (1) args [argparse.Namespace]: --config, --dataset-config, --out.

    Returns:
        (1) parameter [int]: 0 written; 1 refused (an invalid configuration or recording, nothing written).
    """

    try:
        configuration = Load_Configuration(args.config)
        dataset = Load_Dataset_Configuration(args.dataset_config)
        flights = Identification_Flights(CONST_RECORDINGS_DIR)
    except (Configuration_Error_Cls, Recording_Error_Cls, Dataset_Error_Cls) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    for level, message in configuration.Notes + dataset.Notes:
        print('[%s] %s' % (level, message), file=sys.stderr)
    path = _Output_Paths([configuration.Observation.Summary_File], args.out)[0]
    print('[INFO] Calibrating on the %d identification flights (about a minute).' % len(flights), file=sys.stderr)
    try:
        calibration = Calibrate(flights, configuration, dataset, Flight_Model_Str=CONST_CATCH_SCRIPT._Flight_Model(configuration),
                                Observation_Model_Str=CONST_CATCH_SCRIPT._Observation_Model(configuration),
                                progress=lambda step: print('[INFO] %s' % step, file=sys.stderr))
    except (Calibration_Error_Cls, Dataset_Error_Cls) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    Write_Observation_Summary(path, calibration)
    for line in Observation_Lines(calibration):
        print(line)
    estimate, o = calibration.Estimate, configuration.Observation
    for name in ('White_Std', 'White_Dof', 'Drift_Std', 'Drift_Time', 'Offset_Std', 'Offset_Time', 'Offset_Dof'):
        if _Differs(getattr(o, name), getattr(estimate, name)):
            print('[WARNING] The configured Observation.%s %s differs from the estimate %s. Write the estimate into %s (SPEC §8.1).'
                  % (name, getattr(o, name), ['%.9g' % v for v in np.ravel(getattr(estimate, name))], args.config), file=sys.stderr)
    for name in ('Offset_Bound', 'White_Bound'):
        if _Differs(getattr(o, name), getattr(calibration, name)):
            print('[WARNING] The configured Observation.%s %g m differs from the bound derived from the identification flights, %g m. '
                  'Write it into %s (SPEC §8.1).' % (name, getattr(o, name), getattr(calibration, name), args.config), file=sys.stderr)
    throw = dataset.Throw
    for name, configured, estimated in (('Release_Mean', throw.Release_Mean, calibration.Release_Mean),
                                        ('Release_Std', throw.Release_Std, calibration.Release_Std),
                                        ('Apex_Mean', throw.Apex_Mean, calibration.Apex_Mean), ('Apex_Std', throw.Apex_Std, calibration.Apex_Std)):
        if _Differs(configured, estimated):
            print('[WARNING] The configured Throw.%s %s differs from the measured %s. Write it into %s (SPEC §8.2).'
                  % (name, configured, ['%.9g' % v for v in np.ravel(estimated)], args.dataset_config), file=sys.stderr)
    print('[INFO] Wrote %s. The validation flights were not read; robot-frame values rest on the assumed transform (SPEC §7.3). %s'
          % (path, CONST_SAFETY), file=sys.stderr)
    return 0


def _Check_Dataset(folder: str) -> int:
    """
    Description:
        The sub-command 'dataset-check DIR' (SPEC §2.13.2): regenerate the dataset of DIR from its stored dataset.yaml and
        ball_catching.yaml into a temporary folder and compare flights.npz byte for byte (its SHA-256) with
        DIR/flights.npz and with the manifest's data hash. The system copy is read only for the sections a
        dataset depends on (CONST_DATASET_SECTIONS), so a copy written before a later REQUIRED section is
        checked; each copy's SHA-256 and the model hash must also equal the manifest's.
        Reproducible only with the same code, NumPy and platform (SPEC §9.5).

    Args:
        (1) folder [string]: The dataset folder.

    Returns:
        (1) parameter [int]: 0 identical; 1 differs (each difference named); 2 could not check.
    """

    try:
        manifest = Read_Manifest(os.path.join(folder, 'manifest.json'))
        with open(os.path.join(folder, 'dataset.yaml'), 'rb') as file:
            dataset_bytes = file.read()
        with open(os.path.join(folder, 'ball_catching.yaml'), 'rb') as file:
            system_bytes = file.read()
        stored = Sha256_File(os.path.join(folder, 'flights.npz'))
    except (OSError, ValueError) as error:
        print('[ERROR] %s cannot be checked: %s. A dataset folder holds manifest.json, dataset.yaml, ball_catching.yaml and '
              'flights.npz (SPEC §9.5).' % (folder, error), file=sys.stderr)
        return 2
    temporary = tempfile.mkdtemp(prefix='.check.')
    try:
        regenerated, _ = CONST_CATCH_SCRIPT._Generate_Into(temporary, dataset_bytes, system_bytes, os.path.join(folder, 'dataset.yaml'),
                                                           stored_copy=True)
    except (Configuration_Error_Cls, Recording_Error_Cls, Dataset_Error_Cls, Observation_Error_Cls) as error:
        print('[ERROR] %s cannot be regenerated: %s' % (folder, error), file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    differences = []
    if regenerated['sha256']['data'] != stored:
        differences.append('the regenerated flights.npz (%s) differs from the stored one (%s)' % (regenerated['sha256']['data'], stored))
    if manifest.get('sha256', {}).get('data') != stored:
        differences.append('the stored flights.npz (%s) differs from its manifest (%s)' % (stored, manifest.get('sha256', {}).get('data')))
    for key, name in (('configuration', 'dataset.yaml'), ('system_configuration', 'ball_catching.yaml')):
        if manifest.get('sha256', {}).get(key) != regenerated['sha256'][key]:
            differences.append('the stored %s (%s) differs from its manifest (%s); a stored copy is never edited'
                               % (name, regenerated['sha256'][key], manifest.get('sha256', {}).get(key)))
    if manifest.get('model_sha256') != regenerated['model_sha256']:
        differences.append('the models of the stored ball_catching.yaml (%s) differ from the manifest\'s model_sha256 (%s)'
                           % (regenerated['model_sha256'], manifest.get('model_sha256')))
    for key in ('numpy', 'python', 'platform'):
        if manifest.get(key) != regenerated[key]:
            differences.append('the manifest\'s %s %s differs from this one, %s (SPEC §9.5)' % (key, manifest.get(key), regenerated[key]))
    if differences:
        for difference in differences:
            print('[ERROR] %s.' % difference, file=sys.stderr)
        print('dataset %s: differs' % folder)
        return 1
    print('dataset %s: identical (sha256 %s)' % (folder, stored))
    return 0


def _Sweep_Errors(evaluations: str) -> tp.Dict[str, tp.Any]:
    """
    Description:
        The error tables of the robustness sweep, from a run's stored
        evaluations: 'real', the per-flight errors of the real validation flights that F14's final test evaluation measured
        (<evaluations>/test/per_horizon.csv, its rows of the partition 'real'), and 'simulated', those of the simulated
        validation partition of the evaluation's first dataset against the exact truth
        (<evaluations>/validation/per_horizon.csv), the construction's own check. No recording is read. GUARD: this is
        the only reader of the stored real-flight errors besides 'compare' of scripts/catch.py; only 'sweep' calls it, and what it
        returns goes to the injected prediction only, never to a predictor, the decision's margins or the planner.

    Args:
        (1) evaluations [string]: The run's stored evaluations, the folder holding test/ and validation/.

    Returns:
        (1) parameter [Dict<string, Error_Table_Str>]: The tables by source.
    """

    #   ../Evaluation/Core - the horizon ranges of a stored evaluation
    from Evaluation.Core import CONST_HORIZON_RANGES
    #   ../Catch/Robustness - the error table
    from Catch.Robustness import Error_Table

    real = Read_Per_Horizon(os.path.join(evaluations, 'test', 'per_horizon.csv'), CONST_REAL)
    simulated = Read_Per_Horizon(os.path.join(evaluations, 'validation', 'per_horizon.csv'), 'validation')
    first = simulated[0]['dataset'] if simulated else ''
    return {'real': Error_Table(real, CONST_HORIZON_RANGES),
            'simulated': Error_Table([row for row in simulated if row['dataset'] == first and row['truth'] == 'exact'], CONST_HORIZON_RANGES)}


def _Sweep(args: argparse.Namespace) -> int:
    """
    Description:
        The sub-command 'sweep' (SPEC §2.10, §9.14): the robustness sweep of
        a run on the first Contact.Sweep_Throws validation throws of each of Decision.Datasets, with the first seed's
        margins of the run's stored validation interception (written by scripts/catch.py catch; read, never
        written): the run's own predictor, then the true trajectory plus an injected error at each of
        Contact.Sweep_Scales of the stored real-flight errors and at the simulated level (_Sweep_Errors). Every throw is
        intercepted and every go simulated with the nominal contact and classified (Catch.Physical.Catch_Chunk). The
        validation partitions only; not counted. With a moving catch (Interception.Catch_Mode; F15) the stored
        interception and the sweep are those of <Run.Name>/moving/, and the rows carry its further columns. R2: every
        throw has exactly one class and the robot touches nothing but the ball. --workers N runs the throws in N
        processes; the outputs are the same for every N. Writes <Contact.Output_Folder or --out>/<Run.Name>/sweep/:
        per_throw.csv, summary.csv and manifest.json. No recording is read.

    Args:
        (1) args [argparse.Namespace]: --run, --evaluations, --config, --out, --interceptions, --workers.

    Returns:
        (1) parameter [int]: 0 written and R2 holds; 1 refused (nothing written), or written with R2 violated (named,
                             and in the manifest).
    """

    #   ../Scene/Core - the scene and its robot
    from Scene.Core import Build_Scene, Scene_Robot, Scene_Error_Cls
    #   ../Scene/Ball - the ball's error
    from Scene.Ball import Ball_Error_Cls
    #   ../Robot/Core, Motion - the robot's and the motion's errors
    from Robot.Core import Robot_Error_Cls
    from Robot.Motion import Motion_Error_Cls
    #   ../Catch/Core, Decision, Physical, Robustness - the catch pose, the stored margins, the physical catch, the
    #   sweep's predictor
    from Catch.Core import Catch_Context, Catch_Error_Cls
    from Catch.Decision import Read_Margins
    from Catch.Physical import Catch_Chunk, Contact_Str, Summary_Rows as Catch_Summary, Write_Per_Throw, Write_Summary, CONST_CLASSES, \
        CONST_PER_THROW_COLUMNS, CONST_SUMMARY_COLUMNS
    from Catch.Robustness import Injected_Predictor

    script = CONST_CATCH_SCRIPT
    started = time.perf_counter()
    try:
        configuration = Load_Configuration(args.config)
        if args.workers < 1:
            raise script.Run_Error_Cls('--workers must be an integer >= 1, found %d.' % args.workers)
        interception, contact = configuration.Interception, configuration.Contact
        training, run_manifest, config_bytes = script._Load_Run_Copy(args.run)
        if run_manifest.get('device') not in (None, 'cpu'):
            raise script.Run_Error_Cls('The run %s was trained on the device %s; every reported number comes from the CPU (SPEC §7.8). Train '
                                       'it with Run.Device cpu.' % (args.run, run_manifest.get('device')))
        datasets = script._Decision_Datasets(configuration)
        parent = os.path.join(CONST_REPO_ROOT, contact.Output_Folder) if args.out is None else args.out
        target = os.path.join(script._Run_Folder(configuration, parent, training.Name), 'sweep')
        moving = script._Moving(configuration)
        if os.path.exists(target):
            raise script.Run_Error_Cls('The folder %s exists; a sweep is never overwritten. Use --out or remove it.' % target)
        system_sha256, commit, dirty = Sha256_File(args.config), *script._Git_State()
        # 1\ The stored validation interception: its margins and, for the manifest, its rows.
        stored_parent = os.path.join(CONST_REPO_ROOT, interception.Output_Folder) if args.interceptions is None else args.interceptions
        source = os.path.join(script._Run_Folder(configuration, stored_parent, training.Name), 'validation')
        margins_file = os.path.join(source, 'margins.csv')
        planner = script._Planner_Fields(configuration, config_bytes, datasets)
        try:
            stored = Read_Manifest(os.path.join(source, 'manifest.json'))
            margins = Read_Margins(margins_file)
            margins_sha256 = Sha256_File(margins_file)
            stored_rows = script._Read_Bytes(os.path.join(source, 'per_throw.csv'))
        except (OSError, ValueError, Catch_Error_Cls) as error:
            raise script.Run_Error_Cls('The sweep runs with the margins of the stored validation interception: %s cannot be read (%s). Run '
                                       'scripts/catch.py catch on the validation partition first, or name the interceptions\' parent folder '
                                       'with --interceptions.' % (source, error)) from error
        differing = [key for key, value in planner.items() if stored.get(key) != value]
        if differing:
            raise script.Run_Error_Cls('The stored interception %s differs from this call in %s; run scripts/catch.py catch again with this '
                                       'configuration and run.' % (source, ', '.join(differing)))
        fields = dict(planner, system_configuration=system_sha256, step_s=contact.Step, time_constant_s=contact.Time_Constant,
                      damping_ratio=contact.Damping_Ratio, friction=list(contact.Friction), settle_time_s=contact.Settle_Time,
                      rest_speed_m_s=contact.Rest_Speed, rest_time_s=contact.Rest_Time)
        tables = _Sweep_Errors(args.evaluations)
        model = Build_Scene(Areas_Str=configuration.Areas)
        context = Catch_Context(model, Scene_Robot(model), Criteria_Str=script._Catch_Criteria(configuration))
        setup, seeds, accelerations = script._Interception_Setup(configuration, training, args.run, context)
    except (Configuration_Error_Cls, script.Run_Error_Cls, Dataset_Error_Cls, Scene_Error_Cls, Robot_Error_Cls, Catch_Error_Cls, Report_Error_Cls,
            OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    for level, message in configuration.Notes + training.Notes:
        print('[%s] %s' % (level, message), file=sys.stderr)
    missing = [(seed, n) for seed in seeds for n in interception.Stages if (seed, n) not in margins]
    if missing:
        print('[ERROR] The stored margins %s hold no seed and N %s.' % (margins_file, missing), file=sys.stderr)
        return 1
    setup.Margins = margins
    nominal = (contact.Time_Constant, contact.Damping_Ratio)
    settings = (lambda levels: Contact_Str(Step=contact.Step, Friction=list(contact.Friction), Settle_Time=contact.Settle_Time,
                                           Rest_Speed=contact.Rest_Speed, Rest_Time=contact.Rest_Time, Levels=levels))
    report = (lambda what: (lambda done, total: print('[INFO]   %s: %d of %d tasks (%.0f s)' % (what, done, total, time.perf_counter() - started),
                                                      file=sys.stderr)))
    # 2\ The tasks: per dataset, error source and scale, and chunk of throws, for the first seed.
    cases: tp.List[tp.Tuple[str, float, tp.Any]] = [('run', float('nan'), None)]
    cases += [('real', scale, functools.partial(Injected_Predictor, table=tables['real'], scale=scale)) for scale in contact.Sweep_Scales]
    cases.append(('simulated', 1.0, functools.partial(Injected_Predictor, table=tables['simulated'], scale=1.0)))
    tasks, throws = [], []
    owners: tp.List[int] = []
    try:
        for d in datasets:
            data = Read_Dataset(os.path.join(d['folder'], 'flights.npz'))
            indices = Partition_Indices(data, script.CONST_PARTITION_CODES['validation'])[:contact.Sweep_Throws]
            throws.append(int(len(indices)))
            for number, case in enumerate(cases):
                for seed in seeds[:1]:
                    for k in range(0, len(indices), script.CONST_CHUNK_THROWS):
                        tasks.append((setup, d['label'], os.path.join(d['folder'], 'flights.npz'), indices[k:k + script.CONST_CHUNK_THROWS], seed,
                                      settings([nominal]), case[2]))
                        owners.append(number)
        print('[INFO] Catching the validation partitions of %s (the robustness sweep, %d cases): %s throws, 1 seed(s), %d tasks, %d worker(s).'
              % (', '.join(d['label'] for d in datasets), len(cases), ' + '.join('%d' % n for n in throws), len(tasks), args.workers),
              file=sys.stderr)
        results = script._Map(Catch_Chunk, tasks, args.workers, report('catch'))
    except (Motion_Error_Cls, Catch_Error_Cls, Dataset_Error_Cls, Ball_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    prefix = (lambda number: (cases[number][0], cases[number][1]))
    per_throw = [prefix(owner) + tuple(row) for owner, (_, rows) in zip(owners, results) for row in rows]
    summary = []
    for number in range(len(cases)):
        rows = [row for owner, (_, chunk) in zip(owners, results) if owner == number for row in chunk]
        summary += [prefix(number) + tuple(row) for row in Catch_Summary(rows, 'validation')]

    # 3\ The reading fixed before the data. R2: every throw has exactly one class, and the robot touched nothing but the ball.
    column = {key: k + 2 for k, key in enumerate(CONST_PER_THROW_COLUMNS)}
    expected = sum(len(task[3]) * len(accelerations) * len(task[5].Levels) for task in tasks)
    unclassified = sum(1 for row in per_throw if row[column['class']] not in CONST_CLASSES)
    touching = [row for row in per_throw if row[column['robot_contact']]]
    r2 = len(per_throw) == expected and unclassified == 0 and not touching
    if not r2:
        print('[ERROR] R2 is violated: %d rows for %d throws, accelerations and contacts; %d rows without one of the classes; %d goes whose '
              'robot touched something but the ball (the first: %s).'
              % (len(per_throw), expected, unclassified, len(touching), touching[0][column['robot_contact']] if touching else '-'),
              file=sys.stderr)

    # 4\ The files.
    manifest = Build_Manifest('catch', {'run_manifest': Sha256_File(os.path.join(args.run, 'manifest.json')),
                                        'interception_margins': margins_sha256, 'interception_per_throw': Sha256_Bytes(stored_rows)},
                              dict(fields, run_name=training.Name, model=training.Model_Name, partition='validation', seeds=seeds[:1],
                                   dataset_paths=[d['path'] for d in datasets],
                                   accelerations_deg_s2=[float(np.degrees(a)) for a in accelerations], throws=throws, levels=[],
                                   r2_one_class_and_no_robot_contact=r2, sweep_scales=list(contact.Sweep_Scales),
                                   sweep_throws=contact.Sweep_Throws,
                                   sweep_evaluations={name: Sha256_File(os.path.join(args.evaluations, name, 'per_horizon.csv'))
                                                      for name in ('test', 'validation')}), commit, dirty)

    def Write(folder: str) -> None:
        """
        Description:
            The sweep's files and its manifest.

        Args:
            (1) folder [string]: The folder.
        """

        Write_Per_Throw(os.path.join(folder, 'per_throw.csv'), per_throw, True, moving)
        Write_Summary(os.path.join(folder, 'summary.csv'), summary, True, moving)
        manifest['sha256'] = dict(manifest['sha256'], **{name: Sha256_File(os.path.join(folder, name)) for name in ('per_throw.csv', 'summary.csv')})
        Write_Manifest(os.path.join(folder, 'manifest.json'), manifest)

    try:
        script._Write_Folder(target, Write)
        if Sha256_File(margins_file) != margins_sha256:
            raise script.Run_Error_Cls('The stored margins %s changed while the sweep ran; they are read, never written.' % margins_file)
    except (script.Run_Error_Cls, OSError) as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    key = {key: k + 2 for k, key in enumerate(CONST_SUMMARY_COLUMNS)}
    print('%-22s %-11s %-4s %4s %-14s %6s %6s %6s  %-28s %-28s %7s %s' % ('error', 'dataset', 'seed', 'acc', 'contact', 'throws',
                                                                          'go', 'f. go', 'retained of the goes [95 %]',
                                                                          'retained of catchable [95 %]', 'at rest', 'bounced / rim / missed / hit'))
    for row in summary:
        get = (lambda name: row[key[name]])
        print('%-22s %-11s %-4s %4s %-14s %6d %6d %6d  %-28s %-28s %7d %d / %d / %d / %d' % (
            '%s x %g' % (row[0], row[1]) if not np.isnan(row[1]) else row[0], get('dataset'), get('seed') or '-',
            get('acceleration_deg_s2'), '%g, %g' % (get('time_constant_s'), get('damping_ratio')), get('throws'), get('go'), get('false_go'),
            '%d %.1f%% [%.1f, %.1f]' % (get('retained'), 100 * get('retained_of_go'), 100 * get('retained_of_go_lower'),
                                       100 * get('retained_of_go_upper')),
            '%d %.1f%% [%.1f, %.1f]' % (get('retained_catchable'), 100 * get('retained_of_catchable'), 100 * get('retained_of_catchable_lower'),
                                       100 * get('retained_of_catchable_upper')),
            get('at_rest'), get('bounced_out'), get('rim'), get('missed'), get('hit_robot') + get('hit_stand')))
    print('[INFO] R2 (every throw has one class; the robot touches nothing but the ball): %s. R3: the retained shares are descriptive.'
          % ('holds' if r2 else 'VIOLATED'), file=sys.stderr)
    print('[INFO] Wrote %s in %.0f s (the robustness sweep, validation partition). The ball\'s contact is assumed (SPEC §4.14). Simulated '
          'throws only (SPEC §2.14). %s' % (target, time.perf_counter() - started, CONST_SAFETY), file=sys.stderr)
    return 0 if r2 else 1


def main():
    """
    Description:
        The entry point of the project's derivations. Sub-commands:

            .venv/bin/python scripts/Maintenance/derive.py recordings [--config Config/ball_catching.yaml] [--out <dir>]
            .venv/bin/python scripts/Maintenance/derive.py identify [--config Config/ball_catching.yaml] [--out <dir>]
            .venv/bin/python scripts/Maintenance/derive.py observation [--config Config/ball_catching.yaml]
                                                                        [--dataset-config Config/dataset.yaml] [--out <dir>]
            .venv/bin/python scripts/Maintenance/derive.py workspace [--config Config/ball_catching.yaml] [--out <dir>]
            .venv/bin/python scripts/Maintenance/derive.py scene [--config Config/ball_catching.yaml] [--fingerprint <file>]
            .venv/bin/python scripts/Maintenance/derive.py dataset-check <dataset dir>
            .venv/bin/python scripts/Maintenance/derive.py sweep --run <run dir> --evaluations <the run's evaluations> [--workers <n>]
                                                                 [--config Config/ball_catching.yaml] [--out <dir>] [--interceptions <dir>]

        'identify' is a counted read of the 45 real validation flights (its ledger). 'dataset-check' and 'scene' exit 0
        identical, 1 differs, 2 could not check. An option is given by its full name: an abbreviation is a usage error.

        Exit codes: 0 done; 1 refused (invalid input, nothing written); 2 a usage error (argparse).
        The report goes to stdout, the [INFO] / [WARNING] / [ERROR] lines to stderr.
    """

    parser = argparse.ArgumentParser(description='The derivations of the ball-catching simulation, kept for reproducibility.', allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    recordings = commands.add_parser('recordings', allow_abbrev=False, help='load the recordings, segment the flights, write the summary and '
                                                                            'the steps')
    recordings.add_argument('--config', default=CONST_CONFIG_FILE, help='the configuration (default: Config/ball_catching.yaml)')
    recordings.add_argument('--out', default=None, help='write both files into this directory instead (default: the configured paths)')
    identify = commands.add_parser('identify', allow_abbrev=False, help='identify the flight model on the identification flights, evaluate it '
                                                                        'on the validation flights (COUNTED in its ledger), write the reports')
    identify.add_argument('--config', default=CONST_CONFIG_FILE, help='the configuration (default: Config/ball_catching.yaml)')
    identify.add_argument('--out', default=None, help='write the three files into this directory instead (default: the configured paths)')
    observation = commands.add_parser('observation', allow_abbrev=False, help='calibrate the observation model and the throws on the '
                                                                              'identification flights, write the summary')
    observation.add_argument('--config', default=CONST_CONFIG_FILE, help='the system configuration (default: Config/ball_catching.yaml)')
    observation.add_argument('--dataset-config', default=CONST_DATASET_CONFIG_FILE,
                             help='the dataset configuration (default: Config/dataset.yaml)')
    observation.add_argument('--out', default=None, help='write the summary into this directory instead (default: the configured path)')
    workspace = commands.add_parser('workspace', allow_abbrev=False, help='derive where the robot may catch: the workspace map, the catch '
                                                                          'area, the ready pose (about 12 minutes)')
    workspace.add_argument('--config', default=CONST_CONFIG_FILE, help='the configuration (default: Config/ball_catching.yaml)')
    workspace.add_argument('--out', default=None, help='write the map, the summary and the manifest into this directory instead '
                                                       '(default: the configured paths)')
    scene = commands.add_parser('scene', allow_abbrev=False, help='build the MuJoCo scene, print its summary and compare its fingerprint with '
                                                                  'the stored one')
    scene.add_argument('--config', default=CONST_CONFIG_FILE, help='the configuration whose areas the scene shows '
                                                                   '(default: Config/ball_catching.yaml)')
    scene.add_argument('--fingerprint', default=None, metavar='FILE', help='the stored fingerprint (default: models/scene_fingerprint.json)')
    check = commands.add_parser('dataset-check', allow_abbrev=False, help='regenerate a stored dataset from its configuration copies and '
                                                                          'compare it byte for byte')
    check.add_argument('dataset', metavar='DIR', help='the dataset folder')
    sweep = commands.add_parser('sweep', allow_abbrev=False, help='the robustness sweep of a run on validation throws: the prediction error '
                                                                  'scaled from the stored real-flight errors (not counted)')
    sweep.add_argument('--run', required=True, metavar='DIR', help='the run folder (written by scripts/catch.py train)')
    sweep.add_argument('--evaluations', required=True, metavar='DIR', help='the run\'s stored evaluations: the folder holding test/ and '
                                                                           'validation/')
    sweep.add_argument('--config', default=CONST_CONFIG_FILE, help='the system configuration (default: Config/ball_catching.yaml)')
    sweep.add_argument('--out', default=None, help='the parent folder of the sweep instead of Contact.Output_Folder')
    sweep.add_argument('--interceptions', default=None,
                       help='the parent folder of the stored interception instead of Interception.Output_Folder')
    sweep.add_argument('--workers', default=1, type=int, help='the processes the throws run in (default: 1; the outputs are the same)')
    args = parser.parse_args()
    if args.command == 'recordings':
        return _Recordings(args)
    if args.command == 'identify':
        return _Identify(args)
    if args.command == 'observation':
        return _Observation(args)
    if args.command == 'workspace':
        return _Workspace(args)
    if args.command == 'scene':
        return _Scene(args)
    if args.command == 'dataset-check':
        return _Check_Dataset(args.dataset)
    return _Sweep(args)


if __name__ == '__main__':
    sys.exit(main())
