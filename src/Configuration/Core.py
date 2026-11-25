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
# File Name: ../Configuration/Core.py
## =========================================================================== ##

# System (Default)
import os
# Math (Mathematical functions)
import math
# Regular expressions (Default)
import re
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# PyYAML (YAML parser and emitter) [pip3 install pyyaml]
import yaml
# Typing (Support for type hints)
import typing as tp

"""
Description:
    Initialization of constants.

    The only reader of the project's configuration files (SPEC §7.1, §8): the system configuration
    (Load_Configuration), the dataset configuration (Load_Dataset_Configuration, F6) and the training
    configuration (Load_Training_Configuration, F7; the learning sections F8). It reads the
    YAML strictly (duplicate keys refused, only true / false as booleans), checks every field, collects
    every problem and fails once, and returns the values in canonical units (SPEC §9.1). It never
    prints: unknown keys come back as WARNING notes for the caller to print.
"""
# The shipped configurations, relative to the repository root (SPEC §6).
#   Unit [string]
CONST_CONFIG_FILE = os.path.join('Config', 'ball_catching.yaml')
CONST_DATASET_CONFIG_FILE = os.path.join('Config', 'dataset.yaml')
CONST_TRAINING_CONFIG_FILE = os.path.join('Config', 'training.yaml')
# The shapes of the robot's catch area (SPEC §7.2; the earlier code's Modular_Sphere types).
#   Unit [Tuple<string>]
CONST_ROBOT_AREA_SHAPES = ('sphere', 'hemisphere', 'quarter_sphere')
# The dropout patterns and the donor pools of the observation model (SPEC §7.5, §8.2);
# Observation.Core uses these.
#   Unit [Tuple<string>]
CONST_DROPOUT_PATTERNS = ('composed', 'recorded', 'none')
CONST_DONOR_POOLS = ('all', 'A', 'B')
# How the robot catches (SPEC §7.11, §8.1): at rest in the catch pose before the ball arrives
# (F10, F11), or on a stroke along the ball's path (F15); Catch.Interception uses these.
#   Unit [Tuple<string>]
CONST_CATCH_MODES = ('static', 'moving')
# The numerical predictor a training configuration may name in Model.Name (SPEC §7.7, §8.3):
# generalised least squares under M4's covariance; Prediction.Core uses it. The unweighted least squares
# 'LS' was removed from the program: a configuration or a stored run that names it is refused.
#   Unit [Tuple<string>]
CONST_MODEL_NAMES = ('GLS_M4',)
# The learned predictor Model.Name may name (SPEC §7.8, §8.3): the query-time model
# with the basis decoder; Learning.Core uses it. The query-time model with the MLP decoder ('QT_MLP') and the
# hybrid ('HYBRID') were removed from the program: a configuration, a stored run or a weights
# file that names one is refused.
#   Unit [Tuple<string>]
CONST_LEARNED_MODEL_NAMES = ('QT_BASIS',)
# The devices of Run.Device (SPEC §7.8, §8.3): every reported number comes from 'cpu';
# 'mps' serves tuning sweeps only, and 'test' refuses a run trained on it.
#   Unit [Tuple<string>]
CONST_DEVICES = ('cpu', 'mps')
# The fewest training seeds of a learned run (SPEC §2.13.7).
#   Unit [int]
CONST_MINIMUM_SEEDS = 5
# A dataset's name: its folder under Output.Folder (SPEC §8.2).
#   Unit [re.Pattern]
CONST_NAME_PATTERN = re.compile(r'^[A-Za-z0-9_-]+$')
# The partition ratios sum to 1 within this (SPEC §8.2).
#   Unit [float, -]
CONST_PARTITION_SUM_TOLERANCE = 1e-9
# The sections of the system file a dataset depends on, the only ones read from a dataset's stored copy (SPEC §8.1,
# §9.5): the areas (the throws, the catch area of a catch dataset, the robot's area of the catch
# height), the assumed transform, the flight model and the error model. A copy written before a later REQUIRED section
# keeps loading; a stored copy is a record and is never edited.
#   Unit [Tuple<string>]
CONST_DATASET_SECTIONS = ('Areas', 'Transform', 'Flight', 'Observation')
# Every field of the system file, by its full key path, with the check it must pass: 'vector' (3
# numbers), 'positive_vector' (3 numbers > 0), 'non_negative_vector' (3 numbers >= 0), 'positive' (a
# number > 0), 'non_negative' (a number >= 0), 'dof' (a number > 2: a Student-t with a variance),
# 'angle_vector' (3 numbers in [deg], converted to [rad]), 'shape' (one of CONST_ROBOT_AREA_SHAPES) or
# 'path' (a non-empty repository-relative path, never absolute: the shipped file holds no
# machine-specific value), 'joint_speeds' (6 numbers > 0, [deg/s]), 'joint_angles' (6 numbers in
# [-360, 360], [deg]), 'accelerations' (2 numbers > 0, the nominal first, [deg/s^2]), 'elevation_range'
# ([min, max] with 0 <= min < max <= 90, [deg]), 'azimuth_range' ([min, max] with -180 <= min < max <=
# 180, [deg]); for the decision (F9): 'datasets' (as CONST_TRAINING_FIELDS), 'calibration' (a non-empty list of
# distinct repository-relative paths, each among Decision.Datasets), 'quantile' (a number in (0, 1)), 'stages' (a
# non-empty list of increasing integers >= 2); for the interception (F10): 'stages' and 'calibration' as above (the
# calibration's paths each among Decision.Datasets, Decision.Stages each among Interception.Stages), 'positive' times,
# step and angle ([deg], converted to [rad]), 'mode' (one of CONST_CATCH_MODES) and 'stroke_scales' (a non-empty list of
# decreasing numbers in (0, 1]) for the moving catch (F15); for the physical catch (F11): 'levels' (a list of [time constant > 0,
# damping ratio > 0] pairs), 'scales' (a non-empty list of increasing numbers >= 0), 'count' (an integer >= 1), and no
# time constant below two steps. Every field is REQUIRED.
#   Unit [Dict<string, string>]
CONST_FIELDS = {'Areas.Throw.Centre': 'vector', 'Areas.Throw.Size': 'positive_vector',
                'Areas.Flight.Centre': 'vector', 'Areas.Flight.Size': 'positive_vector',
                'Areas.Catch.Centre': 'vector', 'Areas.Catch.Size': 'positive_vector',
                'Areas.Robot.Shape': 'shape', 'Areas.Robot.Centre': 'vector',
                'Areas.Robot.Orientation': 'angle_vector', 'Areas.Robot.Radius': 'positive',
                'Transform.Orientation': 'angle_vector', 'Transform.Position': 'vector',
                'Recording.Summary_File': 'path', 'Recording.Steps_File': 'path',
                'Flight.Gravity': 'positive', 'Flight.Drag': 'non_negative', 'Flight.Drag_Std': 'non_negative',
                'Identification.Summary_File': 'path', 'Identification.Flights_File': 'path',
                'Identification.Ledger_File': 'path',
                'Observation.White_Std': 'non_negative_vector', 'Observation.White_Dof': 'dof',
                'Observation.Drift_Std': 'non_negative_vector', 'Observation.Drift_Time': 'positive',
                'Observation.Offset_Std': 'non_negative_vector', 'Observation.Offset_Time': 'positive',
                'Observation.Offset_Dof': 'dof', 'Observation.Offset_Bound': 'positive', 'Observation.White_Bound': 'positive',
                'Observation.Summary_File': 'path',
                'Robot.Joint_Speed': 'joint_speeds', 'Robot.Joint_Accelerations': 'accelerations', 'Robot.Ready_Pose': 'joint_angles',
                'Catch.Joint_Margin': 'positive', 'Catch.Clearance': 'positive', 'Catch.Speed_Margin': 'positive',
                'Catch.Arrival_Elevation': 'elevation_range', 'Catch.Arrival_Azimuth': 'azimuth_range', 'Catch.Workspace_Step': 'positive',
                'Workspace.Dataset': 'path', 'Workspace.Map_File': 'path', 'Workspace.Summary_File': 'path',
                'Decision.Datasets': 'datasets', 'Decision.Calibration': 'calibration', 'Decision.Quantile': 'quantile',
                'Decision.Stages': 'stages', 'Decision.Output_Folder': 'path',
                'Interception.Stages': 'stages', 'Interception.Calibration': 'calibration', 'Interception.Latency': 'positive',
                'Interception.Compute_Budget': 'positive', 'Interception.Guard': 'positive', 'Interception.Path_Step': 'positive',
                'Interception.Angle_Tolerance': 'positive', 'Interception.Catch_Mode': 'mode', 'Interception.Stroke_Scales': 'stroke_scales',
                'Interception.Stroke_Compute_Budget': 'positive', 'Interception.Output_Folder': 'path',
                'Contact.Step': 'positive', 'Contact.Time_Constant': 'positive', 'Contact.Damping_Ratio': 'positive',
                'Contact.Friction': 'positive_vector', 'Contact.Settle_Time': 'positive', 'Contact.Rest_Speed': 'positive',
                'Contact.Rest_Time': 'positive', 'Contact.Levels': 'levels', 'Contact.Sweep_Scales': 'scales',
                'Contact.Sweep_Throws': 'count', 'Contact.Output_Folder': 'path', 'Contact.Ledger_File': 'path'}
# Every field of the dataset file (SPEC §8.2), as CONST_FIELDS, and: 'name' (CONST_NAME_PATTERN), 'seed'
# (an integer >= 0), 'count' (an integer >= 1), 'range' ([min, max] with 0 < min < max), 'rates' (a
# non-empty list of distinct numbers > 0; which rates have donors is checked by the dataset command),
# 'pattern' (one of CONST_DROPOUT_PATTERNS), 'pool' (one of CONST_DONOR_POOLS). Every field is REQUIRED.
#   Unit [Dict<string, string>]
CONST_DATASET_FIELDS = {'Dataset.Name': 'name', 'Dataset.Seed': 'seed', 'Dataset.Flights': 'count', 'Dataset.System_Configuration': 'path',
                        'Partitions.Train': 'positive', 'Partitions.Validation': 'positive', 'Partitions.Test': 'positive',
                        'Throw.Release_Mean': 'vector', 'Throw.Release_Std': 'positive_vector', 'Throw.Apex_Mean': 'positive',
                        'Throw.Apex_Std': 'positive', 'Throw.Flight_Time': 'range', 'Throw.Release_Speed': 'range',
                        'Observation.Rates': 'rates', 'Observation.Noise.Scale': 'non_negative', 'Observation.Dropout.Pattern': 'pattern',
                        'Observation.Dropout.Donors': 'pool', 'Output.Folder': 'path'}
# The two target areas of the dataset file, exactly one of which a file gives (SPEC §8.2): Throw.Target_Area, the
# quarter sphere of the prediction datasets, and Throw.Target_Catch_Area, Areas.Catch grown
# by a margin on every face, for the catch datasets. The datasets stored before F9 give the
# first, so they keep loading. Kinds as CONST_FIELDS.
#   Unit [Dict<string, string>]
CONST_QUARTER_SPHERE_TARGET_FIELDS = {'Throw.Target_Area.Centre': 'vector', 'Throw.Target_Area.Orientation': 'angle_vector',
                                      'Throw.Target_Area.Radius': 'positive'}
CONST_CATCH_TARGET_FIELDS = {'Throw.Target_Catch_Area.Margin': 'non_negative'}
# The target shapes a dataset configuration gives (Throw_Str.Target_Shape).
#   Unit [Tuple<string>]
CONST_TARGET_SHAPES = ('quarter_sphere', 'catch_area')
# Every field of the training file (SPEC §8.3; F7: without the learning sections, which join with F8), as
# CONST_DATASET_FIELDS, and: 'datasets' (a non-empty list of mappings of exactly Path, a repository-relative
# path, and Hash, a SHA-256 of 64 lower-case hex digits; the paths distinct),
# 'samples' (a non-empty list of distinct integers >= 2), 'model' (one of CONST_MODEL_NAMES or
# CONST_LEARNED_MODEL_NAMES). Every field is REQUIRED.
#   Unit [Dict<string, string>]
CONST_TRAINING_FIELDS = {'Run.Name': 'name', 'Run.Datasets': 'datasets', 'Input.Observed_Samples': 'samples', 'Model.Name': 'model',
                         'Output.Folder': 'path'}
# The learning fields of the training file (SPEC §8.3; F8), REQUIRED when Model.Name is a learned predictor
# and without effect for a numerical one (a present one is then reported in one WARNING): 'seeds' (at least
# CONST_MINIMUM_SEEDS distinct integers >= 0), 'device' (one of CONST_DEVICES); the others as above.
#   Unit [Dict<string, string>]
CONST_LEARNING_FIELDS = {'Run.Seeds': 'seeds', 'Run.Device': 'device', 'Training.Datasets': 'datasets', 'Training.Flights': 'count',
                         'Model.Hidden': 'count', 'Loss.Velocity_Weight': 'non_negative', 'Optimiser.Learning_Rate': 'positive',
                         'Optimiser.Batch_Size': 'count', 'Optimiser.Max_Epochs': 'count', 'Optimiser.Weight_Decay': 'non_negative',
                         'Early_Stopping.Patience': 'count', 'Early_Stopping.Min_Delta': 'non_negative'}
# A SHA-256 as hex text.
#   Unit [re.Pattern]
CONST_SHA256_PATTERN = re.compile(r'^[0-9a-f]{64}$')
# The YAML tag of a boolean.
#   Unit [string]
CONST_BOOL_TAG = 'tag:yaml.org,2002:bool'


class Configuration_Error_Cls(ValueError):
    """
    Description:
        The configuration cannot be used. The message lists every problem found, each with the file,
        the line and the full key, and says what to do; a script prints it verbatim as an [ERROR] line.
    """


@dataclass
class Cuboid_Area_Str:
    """
    Description:
        An axis-aligned cuboid area (SPEC §7.2).
    """

    # Centre of the cuboid.
    #   Unit [Vector<float> 1x3, m]
    Centre: tp.List[float] = field(default_factory=list)
    # Edge lengths along x, y, z.
    #   Unit [Vector<float> 1x3, m]
    Size: tp.List[float] = field(default_factory=list)


@dataclass
class Robot_Area_Str:
    """
    Description:
        The robot's catch area: a sphere, hemisphere or quarter sphere (SPEC §7.2).
    """

    # One of CONST_ROBOT_AREA_SHAPES.
    #   Unit [string]
    Shape: str = ''
    # Centre of the sphere.
    #   Unit [Vector<float> 1x3, m]
    Centre: tp.List[float] = field(default_factory=list)
    # Rotations about x, y, z, applied X then Y then Z.
    #   Unit [Vector<float> 1x3, rad]
    Orientation: tp.List[float] = field(default_factory=list)
    # Radius of the sphere.
    #   Unit [float, m]
    Radius: float = 0.0


@dataclass
class Areas_Str:
    """
    Description:
        The areas of the experiment (SPEC §8, section Areas).
    """

    # The cuboid the ball is thrown from.
    #   Unit [Cuboid_Area_Str]
    Throw: Cuboid_Area_Str = field(default_factory=Cuboid_Area_Str)
    # The cuboid the ball flies through.
    #   Unit [Cuboid_Area_Str]
    Flight: Cuboid_Area_Str = field(default_factory=Cuboid_Area_Str)
    # The cuboid around the robot where the ball is caught.
    #   Unit [Cuboid_Area_Str]
    Catch: Cuboid_Area_Str = field(default_factory=Cuboid_Area_Str)
    # The robot's catch area.
    #   Unit [Robot_Area_Str]
    Robot: Robot_Area_Str = field(default_factory=Robot_Area_Str)


@dataclass
class Transform_Str:
    """
    Description:
        The assumed transform from the recordings' frame (QTM) to the robot base frame (SPEC §7.3):
        p_robot = R p_QTM + Position, with R = R_x R_y R_z of Orientation, as for the areas (SPEC §7.2).
    """

    # Rotations about x, y, z.
    #   Unit [Vector<float> 1x3, rad]
    Orientation: tp.List[float] = field(default_factory=list)
    # Translation added after the rotation.
    #   Unit [Vector<float> 1x3, m]
    Position: tp.List[float] = field(default_factory=list)


@dataclass
class Recording_Output_Str:
    """
    Description:
        The files the recording loader writes (scripts/catch.py recordings; SPEC §8).
    """

    # One row per recording, repository-relative.
    #   Unit [string, path]
    Summary_File: str = ''
    # Every reported step with file and line, repository-relative.
    #   Unit [string, path]
    Steps_File: str = ''


@dataclass
class Flight_Parameters_Str:
    """
    Description:
        The flight model's parameters (SPEC §2.3, §8.1): gravity and the drag F3 identified, with its
        uncertainty. Flight.Core.Flight_Model_Str is built from them by the caller.
    """

    # The magnitude of gravity, along -z.
    #   Unit [float, m/s^2]
    Gravity: float = 0.0
    # The drag constant k of a = g - k |v| v, and its standard deviation.
    #   Unit [float, 1/m]
    Drag: float = 0.0
    Drag_Std: float = 0.0


@dataclass
class Identification_Output_Str:
    """
    Description:
        The files the identification writes (scripts/catch.py identify; SPEC §9.8).
    """

    # The identified values, the sensitivity rows and the validation rows, repository-relative.
    #   Unit [string, path]
    Summary_File: str = ''
    # One row per fitted flight, repository-relative.
    #   Unit [string, path]
    Flights_File: str = ''
    # One row per identify run that read the validation flights (the count, SPEC §2.13), repository-relative.
    #   Unit [string, path]
    Ledger_File: str = ''


@dataclass
class Observation_Parameters_Str:
    """
    Description:
        The measured parameters of the observation error model M4 (SPEC §7.5, §8.1) and the output of
        their estimation (scripts/catch.py observation). Observation.Core.Observation_Model_Str is built
        from them by the caller.
    """

    # The white noise: standard deviation per axis and degrees of freedom.
    #   Unit [Vector<float> 1x3, m], [float, -]
    White_Std: tp.List[float] = field(default_factory=list)
    White_Dof: float = 0.0
    # The drift: stationary standard deviation per axis and time constant.
    #   Unit [Vector<float> 1x3, m], [float, s]
    Drift_Std: tp.List[float] = field(default_factory=list)
    Drift_Time: float = 0.0
    # The offset: stationary standard deviation per axis, time constant, degrees of freedom.
    #   Unit [Vector<float> 1x3, m], [float, s], [float, -]
    Offset_Std: tp.List[float] = field(default_factory=list)
    Offset_Time: float = 0.0
    Offset_Dof: float = 0.0
    # The largest norm of the offset vector and of a sample's white-noise vector.
    #   Unit [float, m]
    Offset_Bound: float = 0.0
    White_Bound: float = 0.0
    # The estimation's summary, repository-relative.
    #   Unit [string, path]
    Summary_File: str = ''


@dataclass
class Robot_Parameters_Str:
    """
    Description:
        The robot's limits and its ready pose (SPEC §8.1; F4).
    """

    # The joints' speed limits (UR10e datasheet).
    #   Unit [Vector<float> 1x6, rad/s]
    Joint_Speed: tp.List[float] = field(default_factory=list)
    # The joints' acceleration limits, assumed: the nominal first, then the one every result is also reported at.
    #   Unit [Vector<float> 1x2, rad/s^2]
    Joint_Accelerations: tp.List[float] = field(default_factory=list)
    # The pose the robot waits in before a throw.
    #   Unit [Vector<float> 1x6, rad]
    Ready_Pose: tp.List[float] = field(default_factory=list)


@dataclass
class Catch_Parameters_Str:
    """
    Description:
        The catch pose's criteria and the arrival cone (SPEC §8.1; F4).
    """

    # The margin each joint keeps to its range.
    #   Unit [float, rad]
    Joint_Margin: float = 0.0
    # The smallest exact distance to the stand, the floor and the robot itself.
    #   Unit [float, m]
    Clearance: float = 0.0
    # The smallest speed of the tool centre point in every direction with the cup's axis held.
    #   Unit [float, m/s]
    Speed_Margin: float = 0.0
    # The arrival cone: the ball's velocity below the horizontal, and from +y towards +x.
    #   Unit [Vector<float> 1x2, rad]
    Arrival_Elevation: tp.List[float] = field(default_factory=list)
    Arrival_Azimuth: tp.List[float] = field(default_factory=list)
    # The step of the workspace's lattice.
    #   Unit [float, m]
    Workspace_Step: float = 0.0


@dataclass
class Workspace_Output_Str:
    """
    Description:
        The workspace command's dataset and outputs (scripts/catch.py workspace; SPEC §9.11).
    """

    # The dataset whose train partition gives the throws (the arrival cone's check, the ready pose).
    #   Unit [string]
    Dataset: str = ''
    # The workspace map and the summary.
    #   Unit [string]
    Map_File: str = ''
    Summary_File: str = ''


@dataclass
class Run_Dataset_Str:
    """
    Description:
        One dataset of a run (SPEC §8.3): its folder and the SHA-256 its flights.npz must have.
    """

    # The dataset folder, repository-relative.
    #   Unit [string, path]
    Path: str = ''
    # The expected SHA-256 of its flights.npz.
    #   Unit [string, hex]
    Hash: str = ''


@dataclass
class Decision_Parameters_Str:
    """
    Description:
        The catch decision's datasets, margin, stages and outputs (scripts/catch.py decide; SPEC §7.11, §8.1, §9.12; F9).
    """

    # The datasets whose partition decide evaluates, each with the SHA-256 its flights.npz must have.
    #   Unit [List<Run_Dataset_Str>]
    Datasets: tp.List[Run_Dataset_Str] = field(default_factory=list)
    # The datasets (among Datasets) whose validation partitions calibrate the margins.
    #   Unit [List<string>]
    Calibration: tp.List[str] = field(default_factory=list)
    # The quantile of the margins, in (0, 1).
    #   Unit [float, -]
    Quantile: float = 0.0
    # The prefixes' N at which the decision is made, in this order.
    #   Unit [List<int>]
    Stages: tp.List[int] = field(default_factory=list)
    # The parent folder of the decisions.
    #   Unit [string]
    Output_Folder: str = ''


@dataclass
class Interception_Parameters_Str:
    """
    Description:
        The interception planner's decision ladder, its calibration, its times, its path check and its outputs
        (scripts/catch.py intercept; SPEC §7.11, §8.1, §9.13; F10).
    """

    # The ladder of stages: a decision at N observed samples uses the margins of the largest stage <= N.
    #   Unit [List<int>]
    Stages: tp.List[int] = field(default_factory=list)
    # The datasets (among Decision.Datasets) whose train partitions calibrate the ladder's margins.
    #   Unit [List<string>]
    Calibration: tp.List[str] = field(default_factory=list)
    # The motion-capture latency, the compute budget of one plan, and the guard the robot rests before the catch time.
    #   Unit [float, s]
    Latency: float = 0.0
    Compute_Budget: float = 0.0
    Guard: float = 0.0
    # The largest motion of any robot point between two poses of a path check.
    #   Unit [float, m]
    Path_Step: float = 0.0
    # The largest angle between the cup's axis and the ball's reversed velocity at the crossing.
    #   Unit [float, rad]
    Angle_Tolerance: float = 0.0
    # How the robot catches (one of CONST_CATCH_MODES), and for a moving catch the stroke's speed scales in the order they
    # are tried and the compute budget of one plan (F15).
    #   Unit [string], [List<float>], [float, s]
    Catch_Mode: str = ''
    Stroke_Scales: tp.List[float] = field(default_factory=list)
    Stroke_Compute_Budget: float = 0.0
    # The parent folder of the interceptions.
    #   Unit [string]
    Output_Folder: str = ''


@dataclass
class Contact_Parameters_Str:
    """
    Description:
        The physical catch's step, the ball's contact, the catch's criteria, the sensitivity levels, the sweep and the
        outputs (scripts/catch.py catch; SPEC §7.11, §8.1, §9.14; F11).
    """

    # The simulation step of a catch.
    #   Unit [float, s]
    Step: float = 0.0
    # The ball's contact, MuJoCo's solref: the time constant and the damping ratio (the nominal level).
    #   Unit [float, s], [float, -]
    Time_Constant: float = 0.0
    Damping_Ratio: float = 0.0
    # The ball's friction: sliding, torsional, rolling.
    #   Unit [Vector<float> 1x3: -, m, m]
    Friction: tp.List[float] = field(default_factory=list)
    # The settle window after the catch time, at whose end a retained ball is caught.
    #   Unit [float, s]
    Settle_Time: float = 0.0
    # At rest: the ball's speed relative to the cup below Rest_Speed for Rest_Time.
    #   Unit [float, m/s], [float, s]
    Rest_Speed: float = 0.0
    Rest_Time: float = 0.0
    # The sensitivity levels: (time constant, damping ratio) each.
    #   Unit [List<Tuple<float, float>>: s, -]
    Levels: tp.List[tp.Tuple[float, float]] = field(default_factory=list)
    # The sweep's error scales and the first validation throws of each dataset it runs on.
    #   Unit [List<float>], [int]
    Sweep_Scales: tp.List[float] = field(default_factory=list)
    Sweep_Throws: int = 0
    # The parent folder of the catches and the ledger of the test-partition reads.
    #   Unit [string]
    Output_Folder: str = ''
    Ledger_File: str = ''


@dataclass
class Configuration_Str:
    """
    Description:
        The loaded configuration, in canonical units, and the loader's notes. Loaded from a dataset's stored copy
        (Load_Configuration with stored_copy=True), only the sections of CONST_DATASET_SECTIONS are given and every
        other section is None, so a reader of one fails instead of taking a default.
    """

    # The areas of the experiment.
    #   Unit [Areas_Str]
    Areas: Areas_Str = field(default_factory=Areas_Str)
    # The assumed transform from the recordings' frame to the robot base frame.
    #   Unit [Transform_Str]
    Transform: Transform_Str = field(default_factory=Transform_Str)
    # The recording loader's output files.
    #   Unit [Recording_Output_Str]
    Recording: Recording_Output_Str = field(default_factory=Recording_Output_Str)
    # The flight model's parameters.
    #   Unit [Flight_Parameters_Str]
    Flight: Flight_Parameters_Str = field(default_factory=Flight_Parameters_Str)
    # The identification's output files.
    #   Unit [Identification_Output_Str]
    Identification: Identification_Output_Str = field(default_factory=Identification_Output_Str)
    # The observation error model's parameters.
    #   Unit [Observation_Parameters_Str]
    Observation: Observation_Parameters_Str = field(default_factory=Observation_Parameters_Str)
    # The robot's limits and ready pose (F4).
    #   Unit [Robot_Parameters_Str]
    Robot: Robot_Parameters_Str = field(default_factory=Robot_Parameters_Str)
    # The catch pose's criteria and the arrival cone (F4).
    #   Unit [Catch_Parameters_Str]
    Catch: Catch_Parameters_Str = field(default_factory=Catch_Parameters_Str)
    # The workspace command's dataset and outputs (F4).
    #   Unit [Workspace_Output_Str]
    Workspace: Workspace_Output_Str = field(default_factory=Workspace_Output_Str)
    # The catch decision (F9).
    #   Unit [Decision_Parameters_Str]
    Decision: Decision_Parameters_Str = field(default_factory=Decision_Parameters_Str)
    # The interception planner (F10).
    #   Unit [Interception_Parameters_Str]
    Interception: Interception_Parameters_Str = field(default_factory=Interception_Parameters_Str)
    # The physical catch (F11).
    #   Unit [Contact_Parameters_Str]
    Contact: Contact_Parameters_Str = field(default_factory=Contact_Parameters_Str)
    # (level, message) notes for the caller to print: the WARNING for each unknown key.
    #   Unit [List<Tuple<string, string>>]
    Notes: tp.List[tp.Tuple[str, str]] = field(default_factory=list)


@dataclass
class Partitions_Str:
    """
    Description:
        The partition ratios of a dataset (SPEC §8.2), summing to 1.
    """

    # The shares of the train, validation and test partitions.
    #   Unit [float, -]
    Train: float = 0.0
    Validation: float = 0.0
    Test: float = 0.0


@dataclass
class Target_Area_Str:
    """
    Description:
        The area a dataset's throws are aimed into (SPEC §7.6, §8.2): the half, facing the
        thrower, of the upper hemisphere of a sphere (a quarter sphere). It is the targets' own area, not
        Areas.Robot, the area in which the robot catches.
    """

    # The centre of the sphere (robot frame).
    #   Unit [Vector<float> 1x3, m]
    Centre: tp.List[float] = field(default_factory=list)
    # The rotations about x, y, z (applied X then Y then Z), as Areas.Robot.Orientation.
    #   Unit [Vector<float> 1x3, rad]
    Orientation: tp.List[float] = field(default_factory=list)
    # The radius.
    #   Unit [float, m]
    Radius: float = 0.0


@dataclass
class Throw_Str:
    """
    Description:
        How a dataset's throws are drawn (SPEC §7.6, §8.2).
    """

    # The truncated Gaussian of the releases (robot frame): mean and standard deviation per axis.
    #   Unit [Vector<float> 1x3, m]
    Release_Mean: tp.List[float] = field(default_factory=list)
    Release_Std: tp.List[float] = field(default_factory=list)
    # The Gaussian of the apex height (robot frame z) from which the flight time is derived.
    #   Unit [float, m]
    Apex_Mean: float = 0.0
    Apex_Std: float = 0.0
    # [min, max] of the time from the release to the target; a throw outside is drawn again.
    #   Unit [Vector<float> 1x2, s]
    Flight_Time: tp.List[float] = field(default_factory=list)
    # [min, max] of the release speed; a throw outside is drawn again.
    #   Unit [Vector<float> 1x2, m/s]
    Release_Speed: tp.List[float] = field(default_factory=list)
    # The shape of the area the throws are aimed into (one of CONST_TARGET_SHAPES): 'quarter_sphere' (Target_Area)
    # or 'catch_area' (Areas.Catch of the system configuration grown by Target_Margin on every face).
    #   Unit [string]
    Target_Shape: str = 'quarter_sphere'
    # The quarter sphere; empty for the catch area.
    #   Unit [Target_Area_Str]
    Target_Area: Target_Area_Str = field(default_factory=Target_Area_Str)
    # The catch area's margin; 0 for the quarter sphere.
    #   Unit [float, m]
    Target_Margin: float = 0.0


@dataclass
class Observation_Choice_Str:
    """
    Description:
        A dataset's choices of the observation model (SPEC §8.2); its measured parameters are in the
        system configuration (Observation_Parameters_Str).
    """

    # The rates a flight draws one of.
    #   Unit [List<float>, Hz]
    Rates: tp.List[float] = field(default_factory=list)
    # The multiplier of the position error.
    #   Unit [float, -]
    Noise_Scale: float = 0.0
    # One of CONST_DROPOUT_PATTERNS and one of CONST_DONOR_POOLS.
    #   Unit [string]
    Pattern: str = ''
    Donors: str = ''


@dataclass
class Dataset_Configuration_Str:
    """
    Description:
        The loaded dataset configuration (SPEC §8.2) and the loader's notes.
    """

    # The dataset's folder name under Output_Folder, the root seed and the number of flights.
    #   Unit [string], [int], [int]
    Name: str = ''
    Seed: int = 0
    Flights: int = 0
    # The system configuration the dataset reads, repository-relative.
    #   Unit [string, path]
    System_Configuration: str = ''
    # The partition ratios.
    #   Unit [Partitions_Str]
    Partitions: Partitions_Str = field(default_factory=Partitions_Str)
    # How the throws are drawn.
    #   Unit [Throw_Str]
    Throw: Throw_Str = field(default_factory=Throw_Str)
    # The observation model's choices.
    #   Unit [Observation_Choice_Str]
    Observation: Observation_Choice_Str = field(default_factory=Observation_Choice_Str)
    # The parent folder of the datasets, repository-relative.
    #   Unit [string, path]
    Output_Folder: str = ''
    # (level, message) notes for the caller to print: the WARNING for each unknown key.
    #   Unit [List<Tuple<string, string>>]
    Notes: tp.List[tp.Tuple[str, str]] = field(default_factory=list)


@dataclass
class Training_Configuration_Str:
    """
    Description:
        The loaded training configuration (SPEC §8.3; F7) and the loader's notes.
    """

    # The run's folder name under Output_Folder.
    #   Unit [string]
    Name: str = ''
    # The datasets of the run: for a numerical predictor its seed axis.
    #   Unit [List<Run_Dataset_Str>]
    Datasets: tp.List[Run_Dataset_Str] = field(default_factory=list)
    # The values of N, the number of observed samples of a prefix.
    #   Unit [List<int>]
    Observed_Samples: tp.List[int] = field(default_factory=list)
    # The predictor, one of CONST_MODEL_NAMES or CONST_LEARNED_MODEL_NAMES.
    #   Unit [string]
    Model_Name: str = ''
    # The parent folder of the runs, repository-relative.
    #   Unit [string, path]
    Output_Folder: str = ''
    # The learning fields (SPEC §8.3; F8), read for a learned predictor only (empty or zero otherwise).
    # The training seeds, one model each, and the device ('cpu' or 'mps').
    #   Unit [List<int>], [string]
    Seeds: tp.List[int] = field(default_factory=list)
    Device: str = ''
    # The datasets trained on (their train partitions; their validation partitions stop the training) and
    # the number of training flights taken from each, the first of its train partition.
    #   Unit [List<Run_Dataset_Str>], [int]
    Training_Datasets: tp.List[Run_Dataset_Str] = field(default_factory=list)
    Training_Flights: int = 0
    # The size of the encoder's state.
    #   Unit [int]
    Hidden: int = 0
    # The weight of the velocity term of the loss (0: the position loss alone).
    #   Unit [float, s]
    Velocity_Weight: float = 0.0
    # The optimiser: the largest learning rate, the prefixes per step, the epochs at most, the weight decay.
    #   Unit [float, -], [int], [int], [float, -]
    Learning_Rate: float = 0.0
    Batch_Size: int = 0
    Max_Epochs: int = 0
    Weight_Decay: float = 0.0
    # Early stopping on the simulated validation partition: the epochs without an improvement of at least
    # Min_Delta after which the training stops.
    #   Unit [int], [float, m]
    Patience: int = 0
    Min_Delta: float = 0.0
    # (level, message) notes for the caller to print: the WARNING for each unknown key.
    #   Unit [List<Tuple<string, string>>]
    Notes: tp.List[tp.Tuple[str, str]] = field(default_factory=list)


class Strict_Loader_Cls(yaml.SafeLoader):
    """
    Description:
        A YAML loader for a hand-written file: only true / false are booleans (PyYAML would turn on,
        off, yes and no into booleans too); everything else that looks boolean stays text. A key
        repeated in one mapping is found by _Walk_Keys on the composed
        nodes.

    Initialization of the Class:
        Args:
            (1) stream [string]: The YAML text.

        Example:
            Initialization:
                # Initialization of the class.
                Cls = Strict_Loader_Cls(text)

            Features:
                # Functions of the class.
                Cls.get_single_node()
    """


# Only true / false resolve to a boolean: every other boolean resolver of SafeLoader is dropped.
Strict_Loader_Cls.yaml_implicit_resolvers = {first: [(tag, regexp) for tag, regexp in resolvers if tag != CONST_BOOL_TAG]
                                             for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()}
Strict_Loader_Cls.add_implicit_resolver(CONST_BOOL_TAG, re.compile(r'^(?:true|false)$'), list('tf'))


def _Walk_Keys(node: yaml.Node, prefix: str, lines: tp.Dict[str, int], problems: tp.List[str]) -> None:
    """
    Description:
        Record the line of every key path of a composed YAML node, and every key that appears twice
        in one mapping (PyYAML would keep the last silently).

    Args:
        (1) node [yaml.Node]: The node.
        (2) prefix [string]: The key path of the node ('' at the top).
        (3) lines [Dict<string, int>]: The key path -> 1-based line, filled in place.
        (4) problems [List<string>]: One sentence per repeated key, with both of its lines, appended.
    """

    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            path = '%s.%s' % (prefix, key_node.value) if prefix else str(key_node.value)
            line = key_node.start_mark.line + 1
            if path in lines:
                problems.append('line %d: the key %s appears again (first on line %d). Keep one of them.'
                                % (line, path, lines[path]))
            else:
                lines[path] = line
            _Walk_Keys(value_node, path, lines, problems)


def _Leaf_Paths(data: tp.Any, prefix: str) -> tp.List[str]:
    """
    Description:
        The key path of every leaf (non-mapping value) of the loaded document.

    Args:
        (1) data [Any]: The loaded document or one of its values.
        (2) prefix [string]: The key path of data ('' at the top).

    Returns:
        (1) parameter [List<string>]: The leaf key paths.
    """

    if not isinstance(data, dict):
        return [prefix]
    paths: tp.List[str] = []
    for key, value in data.items():
        paths.extend(_Leaf_Paths(value, '%s.%s' % (prefix, key) if prefix else str(key)))
    return paths


def _Is_Number(value: tp.Any) -> bool:
    """
    Description:
        True for an int or a finite float, not a boolean.

    Args:
        (1) value [Any]: The value.

    Returns:
        (1) parameter [bool]: Whether the value is a usable number.
    """

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _Check_Field(kind: str, value: tp.Any) -> tp.Optional[str]:
    """
    Description:
        Check one value against its kind (CONST_FIELDS).

    Args:
        (1) kind [string]: The kind of the field.
        (2) value [Any]: The loaded value.

    Returns:
        (1) parameter [string]: What is wrong, as the end of a sentence, or None when the value is valid.
    """

    if kind in ('vector', 'positive_vector', 'non_negative_vector', 'angle_vector'):
        if not (isinstance(value, list) and len(value) == 3 and all(_Is_Number(v) for v in value)):
            return 'must be a list of 3 numbers, found %r' % (value,)
        if kind == 'positive_vector' and not all(v > 0 for v in value):
            return 'must have every element > 0, found %r' % (value,)
        if kind == 'non_negative_vector' and not all(v >= 0 for v in value):
            return 'must have every element >= 0, found %r' % (value,)
        return None
    if kind == 'dof':
        if not _Is_Number(value) or value <= 2:
            return 'must be a number > 2 (a Student-t with a variance), found %r' % (value,)
        return None
    if kind == 'name':
        if not isinstance(value, str) or CONST_NAME_PATTERN.match(value) is None:
            return 'must be a name of letters, digits, _ and - only, found %r' % (value,)
        return None
    if kind in ('seed', 'count'):
        least = 0 if kind == 'seed' else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < least:
            return 'must be an integer >= %d, found %r' % (least, value)
        return None
    if kind == 'range':
        if not (isinstance(value, list) and len(value) == 2 and all(_Is_Number(v) for v in value)) or not 0 < value[0] < value[1]:
            return 'must be [min, max] with 0 < min < max, found %r' % (value,)
        return None
    if kind == 'rates':
        if not (isinstance(value, list) and value and all(_Is_Number(v) and v > 0 for v in value)) or len(set(value)) != len(value):
            return 'must be a non-empty list of distinct numbers > 0, found %r' % (value,)
        return None
    if kind in ('pattern', 'pool'):
        allowed = CONST_DROPOUT_PATTERNS if kind == 'pattern' else CONST_DONOR_POOLS
        if value not in allowed:
            return 'must be one of %s, found %r' % (', '.join(allowed), value)
        return None
    if kind == 'datasets':
        if not (isinstance(value, list) and value):
            return 'must be a non-empty list of datasets, each with Path and Hash, found %r' % (value,)
        for i, item in enumerate(value):
            if not isinstance(item, dict) or set(item) != {'Path', 'Hash'}:
                return 'item %d must have exactly the keys Path and Hash, found %r' % (i + 1, item)
            reason = _Check_Field('path', item['Path'])
            if reason is not None:
                return 'item %d: Path %s' % (i + 1, reason)
            if not isinstance(item['Hash'], str) or CONST_SHA256_PATTERN.match(item['Hash']) is None:
                return 'item %d: Hash must be a SHA-256 of 64 lower-case hex digits, found %r' % (i + 1, item['Hash'])
        if len({item['Path'] for item in value}) != len(value):
            return 'must name each dataset once, found %r' % ([item['Path'] for item in value],)
        return None
    if kind == 'calibration':
        if not (isinstance(value, list) and value) or len(set(map(str, value))) != len(value) \
                or any(_Check_Field('path', v) is not None for v in value):
            return 'must be a non-empty list of distinct repository-relative paths, found %r' % (value,)
        return None
    if kind == 'quantile':
        if not _Is_Number(value) or not 0.0 < value < 1.0:
            return 'must be a number in (0, 1), found %r' % (value,)
        return None
    if kind == 'stages':
        if not (isinstance(value, list) and value and all(isinstance(v, int) and not isinstance(v, bool) and v >= 2 for v in value)) \
                or any(b <= a for a, b in zip(value, value[1:])):
            return 'must be a non-empty list of increasing integers >= 2, found %r' % (value,)
        return None
    if kind == 'levels':
        if not isinstance(value, list) or not all(isinstance(v, list) and len(v) == 2 and all(_Is_Number(x) and x > 0 for x in v) for v in value):
            return 'must be a list of [time constant > 0, damping ratio > 0] pairs, found %r' % (value,)
        return None
    if kind == 'mode':
        if value not in CONST_CATCH_MODES:
            return 'must be one of %s, found %r' % (', '.join(CONST_CATCH_MODES), value)
        return None
    if kind == 'stroke_scales':
        if not (isinstance(value, list) and value and all(_Is_Number(v) and 0 < v <= 1 for v in value)) \
                or any(b >= a for a, b in zip(value, value[1:])):
            return 'must be a non-empty list of decreasing numbers in (0, 1], found %r' % (value,)
        return None
    if kind == 'scales':
        if not (isinstance(value, list) and value and all(_Is_Number(v) and v >= 0 for v in value)) or any(b <= a for a, b in zip(value, value[1:])):
            return 'must be a non-empty list of increasing numbers >= 0, found %r' % (value,)
        return None
    if kind == 'samples':
        if not (isinstance(value, list) and value and all(isinstance(v, int) and not isinstance(v, bool) and v >= 2 for v in value)) \
                or len(set(value)) != len(value):
            return 'must be a non-empty list of distinct integers >= 2, found %r' % (value,)
        return None
    if kind == 'model':
        if value not in CONST_MODEL_NAMES + CONST_LEARNED_MODEL_NAMES:
            return 'must be one of %s, found %r' % (', '.join(CONST_MODEL_NAMES + CONST_LEARNED_MODEL_NAMES), value)
        return None
    if kind == 'seeds':
        if not (isinstance(value, list) and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value)) \
                or len(set(value)) != len(value) or len(value) < CONST_MINIMUM_SEEDS:
            return 'must be a list of at least %d distinct integers >= 0, found %r' % (CONST_MINIMUM_SEEDS, value)
        return None
    if kind == 'device':
        if value not in CONST_DEVICES:
            return 'must be one of %s, found %r' % (', '.join(CONST_DEVICES), value)
        return None
    if kind == 'positive':
        if not _Is_Number(value) or value <= 0:
            return 'must be a number > 0, found %r' % (value,)
        return None
    if kind == 'non_negative':
        if not _Is_Number(value) or value < 0:
            return 'must be a number >= 0, found %r' % (value,)
        return None
    if kind in ('joint_speeds', 'joint_angles'):
        if not (isinstance(value, list) and len(value) == 6 and all(_Is_Number(v) for v in value)):
            return 'must be a list of 6 numbers, one per joint, found %r' % (value,)
        if kind == 'joint_speeds' and not all(v > 0 for v in value):
            return 'must have every element > 0, found %r' % (value,)
        if kind == 'joint_angles' and not all(-360.0 <= v <= 360.0 for v in value):
            return 'must have every element in [-360, 360] deg, found %r' % (value,)
        return None
    if kind == 'accelerations':
        if not (isinstance(value, list) and len(value) == 2 and all(_Is_Number(v) and v > 0 for v in value)):
            return 'must be a list of 2 numbers > 0 (the nominal, then the one every result is also reported at), found %r' % (value,)
        return None
    if kind in ('elevation_range', 'azimuth_range'):
        low, high = (0.0, 90.0) if kind == 'elevation_range' else (-180.0, 180.0)
        if not (isinstance(value, list) and len(value) == 2 and all(_Is_Number(v) for v in value)) \
                or not low <= value[0] < value[1] <= high:
            return 'must be [min, max] with %g <= min < max <= %g deg, found %r' % (low, high, value)
        return None
    if kind == 'exclusive_target':
        return ('is given together with Throw.Target_Catch_Area; a dataset has one target area: keep Throw.Target_Area (the '
                'quarter sphere) or Throw.Target_Catch_Area (the catch area grown by its margin), not both (SPEC §8.2)')
    if kind == 'path':
        if not isinstance(value, str) or value.strip() == '':
            return 'must be a non-empty repository-relative path, found %r' % (value,)
        if os.path.isabs(value):
            return 'must be repository-relative, not absolute, found %r' % (value,)
        return None
    if value not in CONST_ROBOT_AREA_SHAPES:
        return 'must be one of %s, found %r' % (', '.join(CONST_ROBOT_AREA_SHAPES), value)
    return None


def _Get(data: tp.Any, path: str) -> tp.Tuple[bool, tp.Any]:
    """
    Description:
        The value at a key path of the loaded document.

    Args:
        (1) data [Any]: The loaded document.
        (2) path [string]: The full key path, e.g. 'Areas.Throw.Size'.

    Returns:
        (1) parameter 1 [bool]: Whether the key is present.
        (2) parameter 2 [Any]: The value, or None.
    """

    for key in path.split('.'):
        if not isinstance(data, dict) or key not in data:
            return False, None
        data = data[key]
    return True, data


def _Cuboid(section: tp.Dict[str, tp.Any]) -> Cuboid_Area_Str:
    """
    Description:
        A cuboid area from its checked section.

    Args:
        (1) section [Dict<string, Any>]: The section with Centre and Size.

    Returns:
        (1) parameter [Cuboid_Area_Str(object)]: The area [m].
    """

    return Cuboid_Area_Str(Centre=[float(v) for v in section['Centre']], Size=[float(v) for v in section['Size']])


def _Load_Checked(path: str, fields: tp.Dict[str, str], shipped: str,
                  further: tp.Optional[tp.Callable[[tp.Dict[str, tp.Any]], tp.Dict[str, str]]] = None,
                  sections: tp.Optional[tp.Tuple[str, ...]] = None) -> tp.Tuple[tp.Dict[str, tp.Any], tp.List[tp.Tuple[str, str]]]:
    """
    Description:
        Read a configuration file strictly and check its fields. Every problem (unreadable file, invalid
        YAML, a repeated key, a missing or invalid field) is collected first; then the load fails once,
        listing all of them. An unknown key does not fail the load: it becomes a WARNING note. With
        `sections`, only those top-level sections are read: their fields checked, their unknown keys
        reported; the other sections are neither checked nor reported (the YAML and its repeated keys are
        still checked in the whole file).

    Args:
        (1) path [string]: The configuration file.
        (2) fields [Dict<string, string>]: Every field's full key path and kind (CONST_FIELDS, CONST_DATASET_FIELDS,
                                           CONST_TRAINING_FIELDS).
        (3) shipped [string]: The shipped file of this kind, named in a message.
        (4) further [Callable[[Dict], Dict<string, string>]]: The fields the loaded document requires beyond
                                                             `fields` (the learning fields of a learned
                                                             predictor), or None.
        (5) sections [Tuple<string>]: The top-level sections read (CONST_DATASET_SECTIONS), or None for all.

    Returns:
        (1) parameter 1 [Dict<string, Any>]: The loaded document (only `sections`, when given), every field valid.
        (2) parameter 2 [List<Tuple<string, string>>]: The notes.
    """

    try:
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
    except OSError as error:
        raise Configuration_Error_Cls('The configuration "%s" cannot be read (%s). Check the path given with --config.'
                                      % (path, error.strerror)) from error

    loader = Strict_Loader_Cls(text)
    try:
        node = loader.get_single_node()
        data = loader.construct_document(node) if node is not None else None
    except yaml.YAMLError as error:
        raise Configuration_Error_Cls('The configuration "%s" is not valid YAML: %s. Correct the file.'
                                      % (path, str(error).replace('\n', ' '))) from error
    finally:
        loader.dispose()

    duplicates: tp.List[str] = []
    lines: tp.Dict[str, int] = {}
    if node is not None:
        _Walk_Keys(node, '', lines, duplicates)
    problems = ['%s: %s' % (path, duplicate) for duplicate in duplicates]
    if not isinstance(data, dict):
        problems.append('%s: the file holds no mapping of sections. Start from the shipped %s.' % (path, shipped))
        data = {}
    if sections is not None:
        data = {key: value for key, value in data.items() if key in sections}
        fields = {key_path: kind for key_path, kind in fields.items() if key_path.split('.')[0] in sections}

    # 1\ Every required field: present and valid.
    if further is not None:
        fields = dict(fields, **further(data))
    for key_path, kind in fields.items():
        present, value = _Get(data, key_path)
        if not present:
            problems.append('%s: %s is missing. Add it (SPEC §8).' % (path, key_path))
            continue
        reason = _Check_Field(kind, value)
        if reason is not None:
            problems.append('%s: line %d: %s %s.' % (path, lines.get(key_path, 0), key_path, reason))
    if 'Partitions.Train' in fields and not problems:
        total = sum(_Get(data, 'Partitions.%s' % name)[1] for name in ('Train', 'Validation', 'Test'))
        if abs(total - 1.0) > CONST_PARTITION_SUM_TOLERANCE:
            problems.append('%s: line %d: Partitions.Train, .Validation and .Test must sum to 1 within %g, found %r.'
                            % (path, lines.get('Partitions.Train', 0), CONST_PARTITION_SUM_TOLERANCE, total))
    if problems:
        raise Configuration_Error_Cls('The configuration cannot be used (%d problem(s)):\n  %s'
                                      % (len(problems), '\n  '.join(problems)))

    # 2\ An unknown key is reported, not refused (a likely misspelling).
    notes = [('WARNING', '%s: line %d: the key %s is unknown and is ignored. Check its spelling (SPEC §8).'
              % (path, lines.get(leaf, 0), leaf))
             for leaf in _Leaf_Paths(data, '') if leaf not in fields]
    return data, notes


def Load_Configuration(path: str, *, stored_copy: bool = False) -> Configuration_Str:
    """
    Description:
        Read and check the system configuration file (SPEC §8.1). Every problem is collected first; then
        the load fails once, listing all of them. An unknown key does not fail the load: it becomes a
        WARNING note. A dataset's stored copy (stored_copy=True; SPEC §9.5) is read only for the sections a
        dataset depends on (CONST_DATASET_SECTIONS), with the same checks; its other sections are neither
        checked nor read and come back as None.

    Args:
        (1) path [string]: The configuration file.
        (2) stored_copy [bool]: True for a dataset's stored copy (dataset --check and the datasets of a run);
                                False for a configuration given with --config and a new dataset's copy.

    Returns:
        (1) parameter [Configuration_Str(object)]: The configuration in canonical units (angles in
                                                   [rad]) and its notes.
    """

    data, notes = _Load_Checked(path, CONST_FIELDS, CONST_CONFIG_FILE, sections=CONST_DATASET_SECTIONS if stored_copy else None)

    # 3\ Canonical units: angles from [deg] to [rad] (SPEC §9.1).
    robot = data['Areas']['Robot']
    areas = Areas_Str(Throw=_Cuboid(data['Areas']['Throw']), Flight=_Cuboid(data['Areas']['Flight']),
                      Catch=_Cuboid(data['Areas']['Catch']),
                      Robot=Robot_Area_Str(Shape=robot['Shape'], Centre=[float(v) for v in robot['Centre']],
                                           Orientation=[math.radians(v) for v in robot['Orientation']],
                                           Radius=float(robot['Radius'])))
    transform = Transform_Str(Orientation=[math.radians(v) for v in data['Transform']['Orientation']],
                              Position=[float(v) for v in data['Transform']['Position']])
    flight = Flight_Parameters_Str(Gravity=float(data['Flight']['Gravity']), Drag=float(data['Flight']['Drag']),
                                   Drag_Std=float(data['Flight']['Drag_Std']))
    section = data['Observation']
    observation = Observation_Parameters_Str(White_Std=[float(v) for v in section['White_Std']], White_Dof=float(section['White_Dof']),
                                             Drift_Std=[float(v) for v in section['Drift_Std']], Drift_Time=float(section['Drift_Time']),
                                             Offset_Std=[float(v) for v in section['Offset_Std']], Offset_Time=float(section['Offset_Time']),
                                             Offset_Dof=float(section['Offset_Dof']), Offset_Bound=float(section['Offset_Bound']),
                                             White_Bound=float(section['White_Bound']), Summary_File=section['Summary_File'])
    if stored_copy:
        # 4\ A dataset's stored copy: the sections it was not read for are None, never a default.
        return Configuration_Str(Areas=areas, Transform=transform, Recording=None, Flight=flight, Identification=None,
                                 Observation=observation, Robot=None, Catch=None, Workspace=None, Decision=None, Interception=None,
                                 Contact=None, Notes=notes)
    recording = Recording_Output_Str(Summary_File=data['Recording']['Summary_File'], Steps_File=data['Recording']['Steps_File'])
    section = data['Identification']
    identification = Identification_Output_Str(Summary_File=section['Summary_File'], Flights_File=section['Flights_File'],
                                               Ledger_File=section['Ledger_File'])
    section = data['Robot']
    robot_limits = Robot_Parameters_Str(Joint_Speed=[math.radians(v) for v in section['Joint_Speed']],
                                        Joint_Accelerations=[math.radians(v) for v in section['Joint_Accelerations']],
                                        Ready_Pose=[math.radians(v) for v in section['Ready_Pose']])
    section = data['Catch']
    catch = Catch_Parameters_Str(Joint_Margin=math.radians(section['Joint_Margin']), Clearance=float(section['Clearance']),
                                 Speed_Margin=float(section['Speed_Margin']),
                                 Arrival_Elevation=[math.radians(v) for v in section['Arrival_Elevation']],
                                 Arrival_Azimuth=[math.radians(v) for v in section['Arrival_Azimuth']],
                                 Workspace_Step=float(section['Workspace_Step']))
    section = data['Workspace']
    workspace = Workspace_Output_Str(Dataset=section['Dataset'], Map_File=section['Map_File'], Summary_File=section['Summary_File'])
    section = data['Decision']
    paths = [item['Path'] for item in section['Datasets']]
    unknown = [path for path in section['Calibration'] if path not in paths]
    if unknown:
        raise Configuration_Error_Cls('The configuration cannot be used (1 problem(s)):\n  %s: Decision.Calibration names %s, not among '
                                      'Decision.Datasets %s. Calibrate on datasets the decision evaluates (SPEC §8.1).' % (path, unknown, paths))
    decision = Decision_Parameters_Str(Datasets=[Run_Dataset_Str(Path=item['Path'], Hash=item['Hash']) for item in section['Datasets']],
                                       Calibration=list(section['Calibration']), Quantile=float(section['Quantile']),
                                       Stages=[int(v) for v in section['Stages']], Output_Folder=section['Output_Folder'])
    section = data['Interception']
    unknown = [item for item in section['Calibration'] if item not in paths]
    missing = [n for n in decision.Stages if n not in section['Stages']]
    problems = ['Interception.Calibration names %s, not among Decision.Datasets %s; the ladder is calibrated on the train partitions of '
                'datasets the decision evaluates (owner, 2026-11-21, F10 Q3)' % (unknown, paths)] if unknown else []
    problems += ['Interception.Stages %s does not hold Decision.Stages %s; F9\'s stages are the baseline, decided with the ladder\'s margins '
                 '(SPEC §8.1)' % (section['Stages'], missing)] if missing else []
    if problems:
        raise Configuration_Error_Cls('The configuration cannot be used (%d problem(s)):\n  %s' % (len(problems), '\n  '.join('%s: %s.' % (path, p)
                                                                                                                     for p in problems)))
    interception = Interception_Parameters_Str(Stages=[int(v) for v in section['Stages']], Calibration=list(section['Calibration']),
                                               Latency=float(section['Latency']), Compute_Budget=float(section['Compute_Budget']),
                                               Guard=float(section['Guard']), Path_Step=float(section['Path_Step']),
                                               Angle_Tolerance=math.radians(section['Angle_Tolerance']), Catch_Mode=section['Catch_Mode'],
                                               Stroke_Scales=[float(v) for v in section['Stroke_Scales']],
                                               Stroke_Compute_Budget=float(section['Stroke_Compute_Budget']),
                                               Output_Folder=section['Output_Folder'])
    section = data['Contact']
    pairs = [('Contact.Time_Constant', section['Time_Constant'])]
    pairs += [('Contact.Levels[%d]' % k, level[0]) for k, level in enumerate(section['Levels'])]
    problems = ['%s %r s is below two steps (Contact.Step %r s); MuJoCo raises such a time constant to two steps, so the contact would not '
                'be the configured one (SPEC §7.11)' % (name, value, section['Step']) for name, value in pairs if value < 2.0 * section['Step']]
    if problems:
        raise Configuration_Error_Cls('The configuration cannot be used (%d problem(s)):\n  %s' % (len(problems), '\n  '.join('%s: %s.' % (path, p)
                                                                                                                     for p in problems)))
    contact = Contact_Parameters_Str(Step=float(section['Step']), Time_Constant=float(section['Time_Constant']),
                                     Damping_Ratio=float(section['Damping_Ratio']), Friction=[float(v) for v in section['Friction']],
                                     Settle_Time=float(section['Settle_Time']), Rest_Speed=float(section['Rest_Speed']),
                                     Rest_Time=float(section['Rest_Time']), Levels=[(float(a), float(b)) for a, b in section['Levels']],
                                     Sweep_Scales=[float(v) for v in section['Sweep_Scales']], Sweep_Throws=int(section['Sweep_Throws']),
                                     Output_Folder=section['Output_Folder'], Ledger_File=section['Ledger_File'])
    return Configuration_Str(Areas=areas, Transform=transform, Recording=recording, Flight=flight,
                             Identification=identification, Observation=observation, Robot=robot_limits, Catch=catch,
                             Workspace=workspace, Decision=decision, Interception=interception, Contact=contact, Notes=notes)


def Load_Dataset_Configuration(path: str) -> Dataset_Configuration_Str:
    """
    Description:
        Read and check a dataset configuration file (SPEC §8.2; F6), with the same strict rules as the
        system file; the partition ratios must also sum to 1 within CONST_PARTITION_SUM_TOLERANCE. The file gives
        exactly one target area: Throw.Target_Area (the quarter sphere) or Throw.Target_Catch_Area (Areas.Catch grown
        by its margin).

    Args:
        (1) path [string]: The dataset configuration file.

    Returns:
        (1) parameter [Dataset_Configuration_Str(object)]: The configuration and its notes.
    """

    def Targets(data: tp.Dict[str, tp.Any]) -> tp.Dict[str, str]:
        """
        Description:
            The fields of the target area the document gives: the catch area's when it gives Throw.Target_Catch_Area,
            else the quarter sphere's; both given is a problem, listed with the others.

        Args:
            (1) data [Dict<string, Any>]: The loaded document.

        Returns:
            (1) parameter [Dict<string, string>]: The target area's fields.
        """

        catch, sphere = _Get(data, 'Throw.Target_Catch_Area')[0], _Get(data, 'Throw.Target_Area')[0]
        if catch and sphere:
            return dict(CONST_CATCH_TARGET_FIELDS, **{'Throw.Target_Area': 'exclusive_target'})
        return CONST_CATCH_TARGET_FIELDS if catch else CONST_QUARTER_SPHERE_TARGET_FIELDS

    data, notes = _Load_Checked(path, CONST_DATASET_FIELDS, CONST_DATASET_CONFIG_FILE, Targets)
    dataset, partitions, throw, observation = data['Dataset'], data['Partitions'], data['Throw'], data['Observation']
    if 'Target_Catch_Area' in throw:
        shape, target, margin = 'catch_area', Target_Area_Str(), float(throw['Target_Catch_Area']['Margin'])
    else:
        shape, margin = 'quarter_sphere', 0.0
        target = Target_Area_Str(Centre=[float(v) for v in throw['Target_Area']['Centre']],
                                 Orientation=[math.radians(v) for v in throw['Target_Area']['Orientation']],
                                 Radius=float(throw['Target_Area']['Radius']))
    return Dataset_Configuration_Str(
        Name=dataset['Name'], Seed=int(dataset['Seed']), Flights=int(dataset['Flights']), System_Configuration=dataset['System_Configuration'],
        Partitions=Partitions_Str(Train=float(partitions['Train']), Validation=float(partitions['Validation']), Test=float(partitions['Test'])),
        Throw=Throw_Str(Release_Mean=[float(v) for v in throw['Release_Mean']], Release_Std=[float(v) for v in throw['Release_Std']],
                        Apex_Mean=float(throw['Apex_Mean']), Apex_Std=float(throw['Apex_Std']),
                        Flight_Time=[float(v) for v in throw['Flight_Time']], Release_Speed=[float(v) for v in throw['Release_Speed']],
                        Target_Shape=shape, Target_Area=target, Target_Margin=margin),
        Observation=Observation_Choice_Str(Rates=[float(v) for v in observation['Rates']], Noise_Scale=float(observation['Noise']['Scale']),
                                           Pattern=observation['Dropout']['Pattern'], Donors=observation['Dropout']['Donors']),
        Output_Folder=data['Output']['Folder'], Notes=notes)


def Load_Training_Configuration(path: str) -> Training_Configuration_Str:
    """
    Description:
        Read and check a training configuration file (SPEC §8.3; F7, F8), with the same strict rules as the system
        file. The learning fields (CONST_LEARNING_FIELDS: Run.Seeds, Run.Device, Training, Model.Hidden, Loss,
        Optimiser, Early_Stopping) are REQUIRED for a learned predictor (CONST_LEARNED_MODEL_NAMES); for a
        numerical one they cannot take effect, and a present one is named in one WARNING note.

    Args:
        (1) path [string]: The training configuration file.

    Returns:
        (1) parameter [Training_Configuration_Str(object)]: The configuration and its notes.
    """

    def Learning(data: tp.Dict[str, tp.Any]) -> tp.Dict[str, str]:
        """
        Description:
            The learning fields, when the document names a learned predictor.

        Args:
            (1) data [Dict<string, Any>]: The loaded document.

        Returns:
            (1) parameter [Dict<string, string>]: CONST_LEARNING_FIELDS, or no field.
        """

        return CONST_LEARNING_FIELDS if _Get(data, 'Model.Name')[1] in CONST_LEARNED_MODEL_NAMES else {}

    data, notes = _Load_Checked(path, CONST_TRAINING_FIELDS, CONST_TRAINING_CONFIG_FILE, Learning)
    configuration = Training_Configuration_Str(Name=data['Run']['Name'], Datasets=[Run_Dataset_Str(Path=item['Path'], Hash=item['Hash'])
                                                                                   for item in data['Run']['Datasets']],
                                               Observed_Samples=[int(v) for v in data['Input']['Observed_Samples']],
                                               Model_Name=data['Model']['Name'], Output_Folder=data['Output']['Folder'])
    if configuration.Model_Name not in CONST_LEARNED_MODEL_NAMES:
        # A learning field of a numerical predictor cannot take effect: one WARNING names them all.
        present = [key for key in CONST_LEARNING_FIELDS if _Get(data, key)[0]]
        notes = [note for note in notes if not any('the key %s is unknown' % key in note[1] for key in present)]
        if present:
            notes.append(('WARNING', '%s: %s have no effect for the numerical predictor %s and are ignored (SPEC §8.3).'
                          % (path, ', '.join(present), configuration.Model_Name)))
        configuration.Notes = notes
        return configuration
    optimiser, stopping = data['Optimiser'], data['Early_Stopping']
    configuration.Seeds, configuration.Device = [int(v) for v in data['Run']['Seeds']], data['Run']['Device']
    configuration.Training_Datasets = [Run_Dataset_Str(Path=item['Path'], Hash=item['Hash']) for item in data['Training']['Datasets']]
    configuration.Training_Flights, configuration.Hidden = int(data['Training']['Flights']), int(data['Model']['Hidden'])
    configuration.Velocity_Weight = float(data['Loss']['Velocity_Weight'])
    configuration.Learning_Rate, configuration.Batch_Size = float(optimiser['Learning_Rate']), int(optimiser['Batch_Size'])
    configuration.Max_Epochs, configuration.Weight_Decay = int(optimiser['Max_Epochs']), float(optimiser['Weight_Decay'])
    configuration.Patience, configuration.Min_Delta = int(stopping['Patience']), float(stopping['Min_Delta'])
    configuration.Notes = notes
    return configuration
