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
# File Name: ../View/Core.py
## =========================================================================== ##

# Dataclasses (Data classes)
from dataclasses import dataclass, field
# Struct (Binary packing; the PNG's chunks)
import struct
# Time (The wall time of a prediction shown in the overlay)
import time
# Zlib (Compression and checksums; the PNG's image data)
import zlib
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation; here its abstract scene and its renderer) [pip3 install mujoco]
import mujoco
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the areas' structures
from Configuration.Core import Areas_Str
#   ../Flight/Core - the true flight
from Flight.Core import Simulate
#   ../Prediction/Core - a refused prediction
from Prediction.Core import Prediction_Error_Cls
#   ../Dataset/Core - a dataset, its reader and the held-out partition's code
from Dataset.Core import Read_Dataset, CONST_PARTITION_HELD_OUT
#   ../Learning/Predictor - the run's predictor
from Learning.Predictor import Run_Predictor
#   ../Scene/Core - the ball's body
from Scene.Core import CONST_BALL_BODY
#   ../Catch/Interception - the interception of a throw, its setup and its row
from Catch.Interception import (Intercept_Chunk, Interception_Setup_Str, Plan_Str, CONST_MOVING, CONST_MOVING_COLUMNS,
                                CONST_PER_THROW_COLUMNS as CONST_INTERCEPTION_COLUMNS)
#   ../Catch/Physical - the catch of a go, its scene, its row
from Catch.Physical import (Catch_Scene, Set_Contact, Simulate_Catch, Catch_Row, Ball_In_Cup, Contact_Str, Catch_Scene_Str,
                            CONST_PER_THROW_COLUMNS as CONST_CATCH_COLUMNS)

"""
Description:
    Initialization of constants.

    The view of one throw (SPEC §2.11, §7.13): a throw is
    computed with the code of scripts/catch.py catch (Catch.Interception.Intercept_Chunk with its two hooks,
    Catch.Physical.Simulate_Catch with its per-step callback) and recorded; what is shown at a time is derived from the
    record alone, so the window and the frames only play back. This module holds the record, the frame at a time, the
    overlay's text, the decorations (the zones, the trajectory by phase, the markers), the camera presets, the
    offscreen renderer and the frames' two files (PNG, frames.csv). The look of the scene's model is View.Theme's.
    Nothing is printed; no window is opened here.

    The visual language is the owner's diagram (CONST_DIAGRAM): the zones as coloured volumes, the observed samples
    red until the first decision sample ("Throw Detection": the sample of the ladder's first stage, N = 20; no
    detector runs, a simulated throw starts at its release, Q2), the remaining true path dotted, each method's newest
    prediction in its own colour, ending at the predicted catch point after a go ("Catch Prediction").
"""
# The owner's diagram the colours were sampled from (file name and SHA-256, 1462 x 526 px; Part A, 2026-11-22; the file
# is kept outside the repository, Q5). The zones' fills there have alpha 0.10 over white.
#   Unit [Tuple<string, string>]
CONST_DIAGRAM = ('1795373317963_image.png', 'd44d0f66d5ce2fbcff732a177d846a47894911b49305f86052736732c084f3d3')
# The zones of the diagram: the area's name in Areas (SPEC §8.1), its label and its colour (sRGB, 0-255, sampled).
#   Unit [Tuple<Tuple<string, string, Tuple<int, int, int>>>]
CONST_ZONES = (('Throw', 'Throw Area', (255, 0, 0)), ('Flight', 'Flight Area', (69, 118, 196)), ('Catch', 'Catch Area', (108, 177, 69)))
# The diagram's markers: the red of the detection phase and the green of the catch prediction (sampled).
#   Unit [Tuple<int, int, int>]
CONST_DETECTION_COLOUR = (255, 0, 0)
CONST_PREDICTION_COLOUR = (112, 173, 71)
# One colour per method in every view (the driving method included): QT_BASIS takes the diagram's green; the others
# are shades that stay apart from it, from the zones and from the ball's orange on both themes (looked at, Part A).
# A method without an entry takes CONST_OTHER_COLOURS in turn.
#   Unit [Dict<string, Tuple<int, int, int>>], [Tuple<Tuple<int, int, int>>]
CONST_METHOD_COLOURS = {'QT_BASIS': CONST_PREDICTION_COLOUR, 'GLS_M4': (46, 139, 160)}
CONST_OTHER_COLOURS = ((120, 120, 200), (200, 120, 120), (120, 200, 200))
# The labels of the two markers (the diagram's words).
#   Unit [string]
CONST_DETECTION_LABEL = 'Throw Detection'
CONST_PREDICTION_LABEL = 'Catch Prediction'
# What the overlay says of a go's ball between the catch time and the class: while its centre is
# inside the cup, and once it has been inside and is not.
#   Unit [string]
CONST_IN_CUP = 'in cup (settling)'
CONST_OUT_OF_CUP = 'out of cup'
# The legend's names of the observed samples drawn after the detection phase and of the true path's dots.
#   Unit [string]
CONST_SAMPLES_NAME = 'observed samples'
CONST_TRUTH_NAME = 'true path'
# The points a predicted path is drawn through, and the spacing of the true path's dots along it.
#   Unit [int], [float, m]
CONST_PATH_POINTS = 40
CONST_DOT_SPACING = 0.05
# The step of the stored true path (the scene's control period).
#   Unit [float, s]
CONST_TRUTH_STEP = 0.002
# The sizes of what is drawn: an observed sample of the detection phase and a later one, the two markers and a
# comparison's marker, the true path's dot, a path's radius for the driving method and for a comparison.
#   Unit [float, m]
CONST_SAMPLE_RADIUS = 0.011
CONST_LATER_RADIUS = 0.006
CONST_MARKER_RADIUS = 0.030
CONST_COMPARISON_RADIUS = 0.018
CONST_DOT_RADIUS = 0.0045
CONST_PATH_RADIUS = 0.006
CONST_COMPARISON_PATH_RADIUS = 0.004
# The free camera's vertical field of view in a perspective preset (narrower than MuJoCo's 45 degrees: less
# distortion of the zones' boxes; looked at, Part A).
#   Unit [float, deg]
CONST_FOVY = 32.0
# A ghost pose's opacity against the robot's own.
#   Unit [float]
CONST_GHOST_ALPHA = 0.28
# The most decoration geoms a scene takes beside the model's (a 300 Hz throw with three methods draws under 1 500).
#   Unit [int]
CONST_MAX_GEOMS = 20000
# The frame height from which the offscreen text takes MuJoCo's largest font scale (a 4K frame).
#   Unit [int, px]
CONST_LARGE_FONT_HEIGHT = 1200
# The PNG's compression level (zlib; 6 is its default: a 3840 x 2160 frame of the scene in about 0.2 s).
#   Unit [int]
CONST_PNG_LEVEL = 6
# The columns of frames.csv (SPEC §9.15).
#   Unit [Tuple<string>]
CONST_FRAME_COLUMNS = ('frame', 'file', 'time_s', 'phase', 'samples', 'method', 'prediction_n', 'inference_ms', 'plan', 'plans', 'outcome')
# The project's sentence, shown in every overlay (SPEC §1).
#   Unit [string]
CONST_SAFETY = 'Simulation only: nothing is run on, or validated on, the real UR10e.'


class View_Error_Cls(ValueError):
    """
    Description:
        A throw cannot be shown as asked: a throw of a test partition, an index outside a dataset, a frame size the
        model's offscreen buffer does not hold, an image that is not rows of RGB bytes. The message names the value
        and what to do; a script prints it verbatim as an [ERROR] line.
    """


@dataclass
class Theme_Str:
    """
    Description:
        A theme of the view: the colours of what surrounds the robot and of the decorations (sRGB, 0-1). The model's
        side of it is applied by View.Theme.Themed_Scene, the decorations' by Decorations.
    """

    # The theme's name.
    #   Unit [string]
    Name: str = ''
    # The background's gradient: at the top and at the horizon.
    #   Unit [Vector<float> 1x3]
    Sky_Top: tp.Sequence[float] = (0.0, 0.0, 0.0)
    Sky_Horizon: tp.Sequence[float] = (0.0, 0.0, 0.0)
    # The floor's checkerboard: its two tones and the line between two tiles.
    #   Unit [Vector<float> 1x3]
    Floor: tp.Sequence[float] = (0.0, 0.0, 0.0)
    Floor_Second: tp.Sequence[float] = (0.0, 0.0, 0.0)
    Grid: tp.Sequence[float] = (0.0, 0.0, 0.0)
    # The stand, and the camera rail with its cameras.
    #   Unit [Vector<float> 1x3]
    Stand: tp.Sequence[float] = (0.0, 0.0, 0.0)
    Rail: tp.Sequence[float] = (0.0, 0.0, 0.0)
    # The opacity of a zone's faces and of its edges.
    #   Unit [float]
    Zone_Alpha: float = 0.0
    Edge_Alpha: float = 0.0
    # The true path's dots (a mid grey) and the observed samples after the detection phase (the strongest tone on the
    # background), so that the legend's two swatches differ.
    #   Unit [Vector<float> 1x3]
    Dots: tp.Sequence[float] = (0.0, 0.0, 0.0)
    Samples: tp.Sequence[float] = (0.0, 0.0, 0.0)


# The themes (the dark one is the default for the viewer and the
# frames, the light one stays selectable; the floor a checkerboard in MuJoCo's style, two grey tones matched to the
# theme, the background a plain gradient). The values were chosen by looking at rendered frames and are the owner's to
# approve, not measured.
#   Unit [Dict<string, Theme_Str>]
CONST_THEMES = {
    'dark': Theme_Str(Name='dark', Sky_Top=(0.07, 0.08, 0.10), Sky_Horizon=(0.14, 0.15, 0.18), Floor=(0.21, 0.22, 0.24),
                      Floor_Second=(0.16, 0.17, 0.19), Grid=(0.25, 0.26, 0.29), Stand=(0.42, 0.43, 0.45), Rail=(0.30, 0.31, 0.33),
                      Zone_Alpha=0.13, Edge_Alpha=0.55, Dots=(0.60, 0.62, 0.66), Samples=(0.95, 0.95, 0.95)),
    'light': Theme_Str(Name='light', Sky_Top=(0.93, 0.94, 0.95), Sky_Horizon=(0.80, 0.81, 0.83), Floor=(0.76, 0.77, 0.79),
                       Floor_Second=(0.70, 0.71, 0.73), Grid=(0.66, 0.67, 0.69), Stand=(0.55, 0.56, 0.58), Rail=(0.62, 0.63, 0.65),
                       Zone_Alpha=0.10, Edge_Alpha=0.55, Dots=(0.48, 0.49, 0.51), Samples=(0.15, 0.15, 0.15))}
CONST_DEFAULT_THEME = 'dark'


@dataclass
class Camera_Str:
    """
    Description:
        A camera preset: MuJoCo's free camera (the point looked at, the distance to it, the azimuth and the elevation
        of the viewing direction) and what the preset shows.
    """

    # The point looked at.
    #   Unit [Vector<float> 1x3, m]
    Lookat: tp.Sequence[float] = (0.0, 0.0, 0.0)
    # The camera's distance from it.
    #   Unit [float, m]
    Distance: float = 0.0
    # The viewing direction: its azimuth about z from +x and its elevation (negative: looking down).
    #   Unit [float, deg], [float, deg]
    Azimuth: float = 0.0
    Elevation: float = 0.0
    # The height of an orthographic view; 0 for a perspective view (CONST_FOVY).
    #   Unit [float, m]
    Extent: float = 0.0
    # Whether the camera rail and its cameras are drawn.
    #   Unit [bool]
    Rail: bool = False
    # The zones (their names in Areas) whose faces are not drawn in this preset, their edges and labels staying; and
    # the zones not drawn at all.
    #   Unit [Tuple<string>], [Tuple<string>]
    Open_Zones: tp.Sequence[str] = ()
    Hidden_Zones: tp.Sequence[str] = ()


# The camera presets. 'side' is the diagram's view (Q4): orthographic, from -x, perpendicular to
# the throw, the robot on the left and the throwing side on the right, 4.9 m high from 0.25 m below the floor.
# 'overview' looks from beyond the throwing side over the whole cell. 'cup' looks into the cup from its open side, from
# the thrower's side above the catch area (the mean arrival direction is (0.10, 0.56, -0.82), SPEC §4.10). 'thrower' is
# the eye behind the release, 1.77 m above the floor, inside the throw zone: that zone is not drawn there and the flight
# zone's faces are open, so that nothing tints the view.
#   Unit [Dict<string, Camera_Str>]
CONST_CAMERAS = {
    'side': Camera_Str(Lookat=(0.0, -2.65, 2.2), Distance=8.6, Azimuth=0.0, Elevation=0.0, Extent=4.9),
    'overview': Camera_Str(Lookat=(0.1, -1.9, 1.0), Distance=8.2, Azimuth=38.0, Elevation=-18.0, Rail=True),
    'cup': Camera_Str(Lookat=(0.7, -0.175, 0.95), Distance=2.3, Azimuth=58.0, Elevation=-34.0),
    'thrower': Camera_Str(Lookat=(0.45, -0.6, 1.25), Distance=5.0, Azimuth=83.0, Elevation=-6.0, Rail=True, Open_Zones=('Flight',),
                          Hidden_Zones=('Throw',))}
CONST_DEFAULT_CAMERA = 'side'


@dataclass
class Throw_Record_Str:
    """
    Description:
        One throw as the catch computed it (Record_Throw): the truth, the observed samples, the predictions shown, the
        decision, the plans, the class and the scene's positions at every step. Times from the release.
    """

    # The dataset's label, the throw's index in it, the training seed's label ('' for a numerical predictor), the
    # acceleration and the driving method (the run's Model.Name).
    #   Unit [string], [int], [string], [float, rad/s^2], [string]
    Dataset: str = ''
    Flight: int = 0
    Seed: str = ''
    Acceleration: float = 0.0
    Method: str = ''
    # The true path every CONST_TRUTH_STEP from the release to the floor.
    #   Unit [Vector<float> 1xT, s], [Matrix<float> Tx3, m]
    Truth_Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Truth_Position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # The observed samples, as observed.
    #   Unit [Vector<float> 1xM, s], [Matrix<float> Mx3, m]
    Observed_Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Observed_Position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # The first decision sample's N (the ladder's first stage): the end of the detection phase.
    #   Unit [int]
    First_N: int = 0
    # The methods shown, the driving one first.
    #   Unit [List<string>]
    Methods: tp.List[str] = field(default_factory=list)
    # The predictions shown: the N of each, and whether the pipeline asked the driving method for it (a decision or a
    # plan) or it was computed for the display only.
    #   Unit [Vector<int> 1xK], [Vector<bool> 1xK]
    Sample_N: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    Used: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Per method the predicted path of each prediction (NaN when the prediction was refused), whether it ends at the
    # catch time (then its last point is the predicted catch point) and the wall time the prediction took here.
    #   Unit [Dict<string, Array<float> KxPx3, m>], [Dict<string, Vector<bool> 1xK>], [Dict<string, Vector<float> 1xK, s>]
    Paths: tp.Dict[str, np.ndarray] = field(default_factory=dict)
    Catch_Point: tp.Dict[str, np.ndarray] = field(default_factory=dict)
    Inference: tp.Dict[str, np.ndarray] = field(default_factory=dict)
    # The decision ('go' or 'no-go'), the go's N, time and catch time (0 and NaN without one), the largest N the
    # pipeline decided at, and the truth that judges the decision.
    #   Unit [string], [int], [float, s], [float, s], [int], [bool]
    Decision: str = ''
    Go_N: int = 0
    Go_Time: float = float('nan')
    Catch_Time: float = float('nan')
    Decided_N: int = 0
    Catchable: bool = False
    # The accepted plans: each one's start time and the catch pose it aims the cup at (the ghost; a static plan's goal,
    # a moving plan's pose at the catch time, not its stroke's start or end), and the goals rejected.
    #   Unit [Vector<float> 1xQ, s], [Matrix<float> Qx6, rad], [int]
    Plan_Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Plan_Goal: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    Rejected: int = 0
    # A moving catch (F15): whether the catch is computed in moving mode, whether the go's last
    # plan is a stroke, its scale and start, and at the crossing of the moving mouth plane the cup's speed along the ball
    # and the ball's speed relative to the cup (NaN without a stroke or a crossing).
    #   Unit [bool], [bool], [float], [float, s], [float, m/s], [float, m/s]
    Moving: bool = False
    Stroke: bool = False
    Stroke_Scale: float = 0.0
    Stroke_Time: float = float('nan')
    Cup_Speed: float = float('nan')
    Relative_Speed: float = float('nan')
    # The catch's class (Catch.Physical.CONST_CLASSES) and SPEC §2.10's class.
    #   Unit [string], [string]
    Class: str = ''
    Outcome: str = ''
    # The scene's positions (qpos: the robot's joints and the ball's free joint) at the start and after every step
    # of a go's simulation; for a no-go, which the catch does not simulate, the robot at the ready pose and the ball
    # on its true path every CONST_TRUTH_STEP.
    #   Unit [Vector<float> 1xS, s], [Matrix<float> SxQ]
    State_Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    State_Position: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # Per state of a go, whether the ball's centre is inside the cup (Catch.Physical.Ball_In_Cup); empty for a no-go.
    #   Unit [Vector<bool> 1xS]
    State_In_Cup: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Where the robot's six joints and the ball's position sit in the scene's positions.
    #   Unit [Vector<int> 1x6], [int]
    Robot_Address: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    Ball_Address: int = 0
    # The interception's and the catch's row of this throw and acceleration, as scripts/catch.py intercept and catch
    # write them (the stored result is compared with them).
    #   Unit [Tuple<Any>], [Tuple<Any>]
    Interception_Row: tp.Tuple[tp.Any, ...] = ()
    Catch_Row: tp.Tuple[tp.Any, ...] = ()


@dataclass
class Frame_Str:
    """
    Description:
        What a record shows at one time (Frame_At).
    """

    # The time.
    #   Unit [float, s]
    Time: float = 0.0
    # The index of the scene's positions shown, the observed samples seen and the index of the newest prediction
    # (-1 before the first).
    #   Unit [int], [int], [int]
    State: int = 0
    Seen: int = 0
    Prediction: int = -1
    # The decision's phase: 'waiting' (before the first decision sample), 'deciding', 'GO' or 'NO-GO'.
    #   Unit [string]
    Phase: str = ''
    # The index of the plan being executed (-1 before the first), and the class once it is known ('' before).
    #   Unit [int], [string]
    Plan: int = -1
    Outcome: str = ''
    # A go's ball after the catch time, until the class is known: CONST_IN_CUP while its centre is inside the cup,
    # CONST_OUT_OF_CUP once it has been inside and is not, '' otherwise.
    #   Unit [string]
    Settling: str = ''


@dataclass
class Decoration_Str:
    """
    Description:
        One thing drawn beside the model (Decorations): a sphere, a line between two points, a box or a box's edges.
    """

    # 'sphere', 'line', 'box' or 'frame'.
    #   Unit [string]
    Kind: str = ''
    # The centre (a line: its start).
    #   Unit [Vector<float> 1x3, m]
    Position: tp.Sequence[float] = (0.0, 0.0, 0.0)
    # A line's end; a box's or a frame's half sizes.
    #   Unit [Vector<float> 1x3, m]
    Extent: tp.Sequence[float] = (0.0, 0.0, 0.0)
    # A sphere's or a line's radius.
    #   Unit [float, m]
    Radius: float = 0.0
    # The colour and its opacity (0-1).
    #   Unit [Vector<float> 1x4]
    Colour: tp.Sequence[float] = (0.0, 0.0, 0.0, 1.0)
    # The text drawn at it ('' for none).
    #   Unit [string]
    Label: str = ''


def Partition_Code(index: int, sizes: tp.Sequence[int]) -> int:
    """
    Description:
        The partition of a flight from the dataset's partition sizes alone (its manifest's partition_sizes; the
        partitions are index blocks: train, then validation, then test, SPEC §7.6), so that a throw of a test
        partition is refused before the dataset's file is opened.

    Args:
        (1) index [int]: The flight's index.
        (2) sizes [Sequence<int>]: The sizes of the train, the validation and the test partition.

    Returns:
        (1) parameter [int]: 0 train, 1 validation, 2 test.
    """

    if len(sizes) != 3 or any(int(size) < 0 for size in sizes):
        raise View_Error_Cls('The dataset\'s partition sizes %r are not three counts (train, validation, test); its manifest was not '
                             'written by scripts/catch.py dataset (SPEC §9.5).' % (list(sizes),))
    total = int(sum(sizes))
    if not 0 <= index < total:
        raise View_Error_Cls('The throw %d is not in the dataset: it holds the flights 0 to %d.' % (index, total - 1))
    return 0 if index < sizes[0] else (1 if index < sizes[0] + sizes[1] else 2)


def Pick_Throw(sizes: tp.Sequence[int], *, index: tp.Optional[int] = None, pick_seed: tp.Optional[int] = None) -> int:
    """
    Description:
        The throw to show: the index given, or one drawn from the train and the validation partition with the seed
        given (np.random.default_rng). A throw of the test partition is refused: the simulated test partitions are
        read by the counted commands only (SPEC §2.14).

    Args:
        (1) sizes [Sequence<int>]: The dataset's partition sizes (train, validation, test).
        (2) index [int or None]: Keyword-only: the throw's index.
        (3) pick_seed [int or None]: Keyword-only: the seed of the draw, when no index is given.

    Returns:
        (1) parameter [int]: The throw's index.
    """

    if index is None:
        if pick_seed is None:
            raise View_Error_Cls('No throw is named: give an index, or a seed to pick one with.')
        Partition_Code(0, sizes)
        allowed = int(sizes[0]) + int(sizes[1])
        if allowed < 1:
            raise View_Error_Cls('The dataset holds no train or validation throw to pick from (partition sizes %r).' % (list(sizes),))
        index = int(np.random.default_rng(pick_seed).integers(allowed))
    if Partition_Code(index, sizes) == CONST_PARTITION_HELD_OUT:
        raise View_Error_Cls('The throw %d is in the dataset\'s test partition (the flights %d to %d): the simulated test partitions are '
                             'read by the counted commands only. Choose a train or validation throw (0 to %d).'
                             % (index, sizes[0] + sizes[1], sum(sizes) - 1, sizes[0] + sizes[1] - 1))
    return int(index)


def Method_Colour(name: str, methods: tp.Sequence[str]) -> tp.Tuple[int, int, int]:
    """
    Description:
        A method's colour: its own (CONST_METHOD_COLOURS), or for a name without one the next of CONST_OTHER_COLOURS by
        its place among the methods without one.

    Args:
        (1) name [string]: The method.
        (2) methods [Sequence<string>]: The methods shown.

    Returns:
        (1) parameter [Tuple<int, int, int>]: The colour (sRGB, 0-255).
    """

    if name in CONST_METHOD_COLOURS:
        return CONST_METHOD_COLOURS[name]
    others = [method for method in methods if method not in CONST_METHOD_COLOURS]
    return CONST_OTHER_COLOURS[(others.index(name) if name in others else 0) % len(CONST_OTHER_COLOURS)]


def Record_Throw(setup: Interception_Setup_Str, label: str, path: str, index: int, seed: str, acceleration: int, contact: Contact_Str, *,
                 overlays: tp.Optional[tp.Dict[str, tp.Callable[..., tp.Any]]] = None, every: int = 1,
                 scene: tp.Optional[Catch_Scene_Str] = None) -> Throw_Record_Str:
    """
    Description:
        Compute one throw with the catch's own code and record it. The interception is Catch.Interception.Intercept_Chunk
        on this one throw, the run's predictor passed through a wrapper that keeps every prediction it is asked for and
        each go's plan kept (its two hooks); a go is simulated by Catch.Physical.Simulate_Catch with the nominal contact
        (contact.Levels' first), every step's positions kept (its callback), and its row is Catch.Physical.Catch_Row's.
        A no-go is not simulated, as in the catch: its ball follows the flight model and the robot holds the ready pose.
        Beside the predictions the pipeline asked for, the driving method and every overlay method predict from every
        `every`-th observed sample from the first decision sample on, for the display only: nothing of them reaches
        the decision or a plan. A throw of a test partition is refused.

    Args:
        (1) setup [Interception_Setup_Str(object)]: The interception's setup, with its margins.
        (2) label [string]: The dataset's label.
        (3) path [string]: Its flights.npz.
        (4) index [int]: The throw's index.
        (5) seed [string]: The training seed's label ('' for a numerical predictor).
        (6) acceleration [int]: The index of the acceleration shown, in setup.Accelerations.
        (7) contact [Contact_Str(object)]: The catch's settings; its first level is the contact simulated.
        (8) overlays [Dict<string, Callable> or None]: Keyword-only: further methods by name, each a function of a
                                                       prefix (times, positions, rate) giving a trajectory.
        (9) every [int]: Keyword-only: the display's predictions at every `every`-th sample.
        (10) scene [Catch_Scene_Str(object) or None]: Keyword-only: the catch's scene, built here when None.

    Returns:
        (1) parameter [Throw_Record_Str(object)]: The record.
    """

    if every < 1:
        raise View_Error_Cls('The display predicts at every K-th sample with K >= 1, found %d.' % every)
    if not 0 <= acceleration < len(setup.Accelerations):
        raise View_Error_Cls('The acceleration %d is not one of the %d of Robot.Joint_Accelerations.' % (acceleration, len(setup.Accelerations)))
    data = Read_Dataset(path)
    if not 0 <= index < data.Flights:
        raise View_Error_Cls('The throw %d is not in the dataset %s: it holds the flights 0 to %d.' % (index, label, data.Flights - 1))
    if int(data.Partition[index]) == CONST_PARTITION_HELD_OUT:
        raise View_Error_Cls('The throw %d of %s is in its test partition: the simulated test partitions are read by the counted commands '
                             'only. Choose a train or validation throw.' % (index, label))
    t_obs, x_obs = data.Observations(index)
    rate = float(data.Rate[index])
    base = Run_Predictor(setup.Model_Name, setup.Run, int(seed) if seed else None, Flight_Model_Str=setup.Flight,
                         Observation_Model_Str=setup.Observation)
    asked: tp.Dict[int, tp.Tuple[tp.Any, float]] = {}

    def Timed(predict: tp.Callable[..., tp.Any], n: int) -> tp.Tuple[tp.Any, float]:
        """
        Description:
            A method's prediction of the first n samples with the wall time it took; None when it is refused.

        Args:
            (1) predict [Callable]: The method.
            (2) n [int]: The samples.

        Returns:
            (1) parameter [trajectory or None]: The trajectory.
            (2) parameter [float]: The wall time [s].
        """

        start = time.perf_counter()
        try:
            trajectory = predict(t_obs[:n], x_obs[:n], rate)
        except Prediction_Error_Cls:
            trajectory = None
        return trajectory, time.perf_counter() - start

    def Driving(time_prefix: np.ndarray, position_prefix: np.ndarray, prefix_rate: float) -> tp.Any:
        """
        Description:
            The run's predictor as the interception calls it, each prediction kept; a refusal is raised on.

        Args:
            (1 - 3) time_prefix, position_prefix, prefix_rate: The prefix [s], [m], [Hz].

        Returns:
            (1) parameter [trajectory]: The trajectory.
        """

        start = time.perf_counter()
        try:
            trajectory = base(time_prefix, position_prefix, prefix_rate)
        except Prediction_Error_Cls:
            asked[len(time_prefix)] = (None, time.perf_counter() - start)
            raise
        asked[len(time_prefix)] = (trajectory, time.perf_counter() - start)
        return trajectory

    # 1\ The interception, as scripts/catch.py catch computes it, with the predictions and the plan kept.
    plans: tp.Dict[int, Plan_Str] = {}
    rows = Intercept_Chunk((setup, label, path, np.array([index]), seed), predictor=lambda _: Driving,
                           on_plan=lambda flight, a, plan: plans.__setitem__(a, plan))
    row = rows[acceleration]
    moving = setup.Catch_Mode == CONST_MOVING
    column = {name: k for k, name in enumerate(CONST_INTERCEPTION_COLUMNS + (CONST_MOVING_COLUMNS if moving else ()))}
    go = row[column['decision']] == 'go'
    plan = plans.get(acceleration, Plan_Str())
    record = Throw_Record_Str(Dataset=label, Flight=int(index), Seed=seed, Acceleration=float(setup.Accelerations[acceleration]),
                              Method=setup.Model_Name, Observed_Time=np.array(t_obs), Observed_Position=np.array(x_obs),
                              First_N=int(setup.Stages[0]), Decision=row[column['decision']], Catchable=row[column['catchable']] == 'true',
                              Decided_N=max(asked) if asked else 0, Interception_Row=tuple(row), Rejected=plan.Rejected, Moving=moving)
    floor_time = float(data.Floor_Time[index])
    record.Truth_Time = np.append(np.arange(0.0, floor_time, CONST_TRUTH_STEP), floor_time)
    record.Truth_Position = Simulate(data.Release_State[index], record.Truth_Time, Flight_Model_Str=setup.Flight)[:, :3]
    if go:
        record.Go_N, record.Go_Time, record.Catch_Time = int(row[column['go_n']]), float(row[column['go_time_s']]), float(row[column['catch_time_s']])
        record.Plan_Time = np.array([profile.Start_Time for profile in plan.Profiles])
        record.Plan_Goal = np.array(plan.Poses if moving else [profile.Goal for profile in plan.Profiles]).reshape(-1, len(setup.Ready))
        if plan.Stroke:
            number = (lambda name: float(row[column[name]]) if row[column[name]] != '' else float('nan'))
            record.Stroke, record.Stroke_Scale, record.Stroke_Time = True, plan.Scale, plan.Profiles[-1].Start_Time
            record.Cup_Speed, record.Relative_Speed = number('cup_speed_m_s'), number('relative_speed_m_s')
    # 2\ The catch: a go simulated with the nominal contact, every step kept; a no-go by its truth.
    scene = Catch_Scene(setup.Areas, contact, Flight_Model_Str=setup.Flight) if scene is None else scene
    record.Robot_Address, record.Ball_Address = np.array(scene.Robot.Position_Address, dtype=np.int64), int(scene.Ball.Position_Address)
    time_constant, damping_ratio = contact.Levels[0]
    if go:
        Set_Contact(scene, time_constant, damping_ratio)
        times, states, inside = [], [], []
        result = Simulate_Catch(scene, data.Release_State[index], plan, setup.Ready, record.Catch_Time, Contact_Str=contact,
                                on_step=lambda t, q: (times.append(t), states.append(q), inside.append(Ball_In_Cup(scene)[0])))
        record.Catch_Row = Catch_Row(row, time_constant, damping_ratio, bool(plan.Profiles), result, moving=moving)
        record.State_Time, record.State_Position, record.State_In_Cup = np.array(times), np.array(states), np.array(inside, dtype=bool)
    else:
        record.Catch_Row = Catch_Row(row, time_constant, damping_ratio, False, None, moving=moving)
        record.State_Time = record.Truth_Time
        record.State_Position = np.zeros((len(record.Truth_Time), scene.Model.nq))
        record.State_Position[:, record.Robot_Address] = setup.Ready
        record.State_Position[:, record.Ball_Address:record.Ball_Address + 3] = record.Truth_Position
        record.State_Position[:, record.Ball_Address + 3] = 1.0
    catch_column = {name: k for k, name in enumerate(CONST_CATCH_COLUMNS)}
    record.Class, record.Outcome = record.Catch_Row[catch_column['class']], record.Catch_Row[catch_column['outcome']]
    # 3\ The predictions shown: what the pipeline asked for, and every `every`-th sample from the first decision
    #    sample, before the catch time of a go.
    last = len(t_obs) if not go else int(np.searchsorted(t_obs, record.Catch_Time, side='left'))
    shown = sorted(set(range(record.First_N, last + 1, every)) | {n for n in asked if n <= last})
    methods = {setup.Model_Name: base}
    for name, predict in (overlays or {}).items():
        if name in methods:
            raise View_Error_Cls('The method %s is given twice (the run\'s and an overlay); an overlay is another method.' % name)
        methods[name] = predict
    record.Methods = list(methods)
    record.Sample_N = np.array(shown, dtype=np.int64)
    record.Used = np.array([n in asked for n in shown], dtype=bool)
    for name, predict in methods.items():
        paths = np.full((len(shown), CONST_PATH_POINTS, 3), np.nan)
        marked, seconds = np.zeros(len(shown), dtype=bool), np.full(len(shown), np.nan)
        for k, n in enumerate(shown):
            trajectory, seconds[k] = asked[n] if name == setup.Model_Name and n in asked else Timed(predict, n)
            if trajectory is None:
                continue
            marked[k] = bool(go and n >= record.Go_N and trajectory.T_0 < record.Catch_Time <= trajectory.T_End)
            end = record.Catch_Time if marked[k] else trajectory.T_End
            paths[k] = trajectory.State(np.linspace(trajectory.T_0, end, CONST_PATH_POINTS))[:, :3]
        record.Paths[name], record.Catch_Point[name], record.Inference[name] = paths, marked, seconds
    return record


def Frame_At(record: Throw_Record_Str, t: float) -> Frame_Str:
    """
    Description:
        What a record shows at a time: the scene's positions at or before it, the samples captured by then, the newest
        prediction among them, the decision's phase and the class once the record has ended (a no-go: once the
        decisions have ended); for a go after its catch time and before its end, whether the ball is in the cup.

    Args:
        (1) record [Throw_Record_Str(object)]: The record.
        (2) t [float]: The time from the release [s].

    Returns:
        (1) parameter [Frame_Str(object)]: The frame.
    """

    state = int(np.clip(np.searchsorted(record.State_Time, t + 1e-12, side='right') - 1, 0, len(record.State_Time) - 1))
    seen = int(np.searchsorted(record.Observed_Time, t + 1e-12, side='right'))
    prediction = int(np.searchsorted(record.Sample_N, seen, side='right')) - 1
    go = record.Decision == 'go'
    if seen < record.First_N:
        phase = 'waiting'
    elif go:
        phase = 'GO' if seen >= record.Go_N else 'deciding'
    else:
        phase = 'NO-GO' if seen >= max(record.Decided_N, record.First_N) or seen >= len(record.Observed_Time) else 'deciding'
    ended = t >= record.State_Time[-1] - 1e-9 if go else phase == 'NO-GO'
    plan = int(np.searchsorted(record.Plan_Time, t + 1e-12, side='right')) - 1 if len(record.Plan_Time) else -1
    settling = ''
    if go and not ended and t > record.Catch_Time and len(record.State_In_Cup):
        settling = CONST_IN_CUP if record.State_In_Cup[state] else (CONST_OUT_OF_CUP if record.State_In_Cup[:state].any() else '')
    return Frame_Str(Time=float(t), State=state, Seen=seen, Prediction=prediction, Phase=phase, Plan=plan, Outcome=record.Class if ended else '',
                     Settling=settling)


def Frame_Times(record: Throw_Record_Str, fps: float, speed: float, start: tp.Optional[float] = None,
                end: tp.Optional[float] = None) -> np.ndarray:
    """
    Description:
        The times of a record's frames: from `start` (the record's first time) to `end` (its last) every speed / fps,
        so that the frames played at `fps` show the throw at `speed` times its real speed (0.1: ten times slower).

    Args:
        (1) record [Throw_Record_Str(object)]: The record.
        (2) fps [float]: The frames per second of the video [1/s].
        (3) speed [float]: The playback speed against real time [-].
        (4 - 5) start, end [float or None]: The first and the last time [s].

    Returns:
        (1) parameter [Vector<float> 1xF]: The times [s].
    """

    if not (fps > 0.0 and speed > 0.0):
        raise View_Error_Cls('The frames need a rate and a speed above 0, found %r frames per second and a speed of %r.' % (fps, speed))
    low = float(record.State_Time[0]) if start is None else float(start)
    high = float(record.State_Time[-1]) if end is None else float(end)
    if not low <= high:
        raise View_Error_Cls('The frames run from %r s to %r s: the first time must not be after the last.' % (low, high))
    return low + (speed / fps) * np.arange(int(np.floor((high - low) * fps / speed + 1e-9)) + 1)


def Overlay_Lines(record: Throw_Record_Str, frame: Frame_Str) -> tp.Tuple[str, str]:
    """
    Description:
        The overlay's two text columns at a frame: the method, the throw, the samples seen, the wall time of the newest
        prediction (measured in this run, on this machine as it was: not a reported inference time, SPEC §7.12), the
        decision, the plan being executed, the class and the time; until a go's class is known the outcome's row says
        after the catch time whether the ball is in the cup (settling) or out of it. A moving catch's record (F15) names
        the stroke as the plan while it runs and has three more rows: the catch's mode with
        the stroke's scale (or the static plan it fell back to), and from the stroke's start on the cup's speed along
        the ball and the ball's speed relative to the cup at the crossing of the mouth plane.

    Args:
        (1) record [Throw_Record_Str(object)]: The record.
        (2) frame [Frame_Str(object)]: The frame.

    Returns:
        (1) parameter [string]: The names, one per line.
        (2) parameter [string]: The values, one per line.
    """

    inference = '-'
    if frame.Prediction >= 0 and np.isfinite(record.Inference[record.Method][frame.Prediction]):
        inference = '%.1f ms (this run)' % (1e3 * record.Inference[record.Method][frame.Prediction])
    decision = {'waiting': 'waiting', 'deciding': 'deciding', 'GO': 'GO at N %d' % record.Go_N, 'NO-GO': 'NO-GO'}[frame.Phase]
    plan = '%d of %d' % (frame.Plan + 1, len(record.Plan_Time)) if frame.Plan >= 0 else '-'
    if record.Stroke and frame.Plan == len(record.Plan_Time) - 1:
        plan = 'the stroke'
    rows = [('method', record.Method + (' (seed %s)' % record.Seed if record.Seed else '')),
            ('throw', '%s %d, %.0f deg/s2' % (record.Dataset, record.Flight, np.degrees(record.Acceleration))),
            ('samples N', '%d' % frame.Seen), ('inference', inference), ('decision', decision), ('plan', plan),
            ('outcome', frame.Outcome.replace('_', ' ') if frame.Outcome else (frame.Settling or '-')),
            ('time', '%.3f s' % frame.Time)]
    if record.Moving:
        shown = record.Stroke and frame.Time >= record.Stroke_Time
        speed = (lambda value: '%.2f m/s' % value if shown and np.isfinite(value) else '-')
        mode = 'moving' if record.Decision != 'go' or not len(record.Plan_Time) else \
            ('moving, stroke x %.2f' % record.Stroke_Scale if record.Stroke else 'moving, a static plan')
        rows[-2:-2] = [('catch', mode), ('cup speed', speed(record.Cup_Speed)), ('relative speed', speed(record.Relative_Speed))]
    return '\n'.join(name for name, _ in rows), '\n'.join(value for _, value in rows)


def Legend(record: Throw_Record_Str, *, Theme_Str: Theme_Str) -> tp.List[tp.Tuple[str, tp.Tuple[int, int, int]]]:
    """
    Description:
        The legend's entries: each method shown with its colour, the driving one first; then what the dots are: the
        observed samples drawn after the detection phase, and the true path's dots ahead of the ball,
        each in the theme's colour. A gap in the samples is a dropout of the observation: every sample is drawn.

    Args:
        (1) record [Throw_Record_Str(object)]: The record.
        (2) Theme_Str [Theme_Str(object)]: REQUIRED, keyword-only: the theme.

    Returns:
        (1) parameter [List<Tuple<string, Tuple<int, int, int>>>]: The names and their colours (sRGB, 0-255).
    """

    tone = (lambda colour: tuple(int(round(255.0 * value)) for value in colour))
    return [(name, Method_Colour(name, record.Methods)) for name in record.Methods] + \
        [(CONST_SAMPLES_NAME, tone(Theme_Str.Samples)), (CONST_TRUTH_NAME, tone(Theme_Str.Dots))]


def _Rgba(colour: tp.Sequence[float], alpha: float = 1.0, scale: float = 255.0) -> tp.Tuple[float, float, float, float]:
    """
    Description:
        A colour with an opacity, 0-1.

    Args:
        (1) colour [Vector<float> 1x3]: The colour.
        (2) alpha [float]: The opacity.
        (3) scale [float]: The colour's full scale (255 for sRGB bytes, 1 for 0-1).

    Returns:
        (1) parameter [Tuple<float, float, float, float>]: Red, green, blue, opacity.
    """

    return (colour[0] / scale, colour[1] / scale, colour[2] / scale, alpha)


def Zone_Box(name: str, areas: Areas_Str) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        A zone's box as the view draws it: the area of the configuration, except that the flight zone begins at the
        throw area's face towards the robot where the two overlap along the throw (y), as in the owner's diagram
        (the configured Areas.Flight, the 2022 lab's, reaches 0.96 m into Areas.Throw, the measured
        releases' support; nothing is computed from Areas.Flight or from the overlap). For the display only: the
        configuration's areas are not changed.

    Args:
        (1) name [string]: The area's name in Areas: 'Throw', 'Flight' or 'Catch'.
        (2) areas [Areas_Str(object)]: The areas (SPEC §8.1).

    Returns:
        (1) parameter [Vector<float> 1x3]: The box's centre [m].
        (2) parameter [Vector<float> 1x3]: Its half sizes [m].
    """

    area = getattr(areas, name)
    low = np.asarray(area.Centre, dtype=np.float64) - 0.5 * np.asarray(area.Size, dtype=np.float64)
    high = np.asarray(area.Centre, dtype=np.float64) + 0.5 * np.asarray(area.Size, dtype=np.float64)
    if name == 'Flight':
        near = areas.Throw.Centre[1] + 0.5 * areas.Throw.Size[1]
        if low[1] < near < high[1]:
            low[1] = near
    return 0.5 * (low + high), 0.5 * (high - low)


def Decorations(record: Throw_Record_Str, frame: Frame_Str, *, Theme_Str: Theme_Str, Areas_Str: Areas_Str, Camera_Str: Camera_Str,
                zones: bool = True, labels: bool = True) -> tp.List[Decoration_Str]:
    """
    Description:
        What is drawn beside the model at a frame, in the diagram's language. The zones: each area's box (Zone_Box:
        the flight zone from the throw area's near face) in its colour (the faces at the theme's opacity, left out
        for a zone the camera opens; the edges), labelled; a zone the camera hides is left out. The observed
        samples seen so far: red and joined by a red line up to the first decision sample, which carries the red
        marker "Throw Detection" once it is seen; the later ones small, in the theme's colour for them, every one,
        so that a gap is a dropout of the observation. The true path still ahead of the ball as dots every
        CONST_DOT_SPACING. Each method's newest prediction as
        a line in its colour, ending in a marker when it ends at the catch time (the driving method's labelled "Catch
        Prediction"); after a go's catch time no prediction is drawn: the ball is at the cup.

    Args:
        (1) record [Throw_Record_Str(object)]: The record.
        (2) frame [Frame_Str(object)]: The frame.
        (3) Theme_Str [Theme_Str(object)]: REQUIRED, keyword-only: the theme.
        (4) Areas_Str [Areas_Str(object)]: REQUIRED, keyword-only: the areas (SPEC §8.1).
        (5) Camera_Str [Camera_Str(object)]: REQUIRED, keyword-only: the camera preset (the zones it opens).
        (6) zones [bool]: Keyword-only: draw the zones.
        (7) labels [bool]: Keyword-only: draw the labels.

    Returns:
        (1) parameter [List<Decoration_Str>]: The decorations.
    """

    out: tp.List[Decoration_Str] = []
    # 1\ The zones.
    if zones:
        for name, label, colour in CONST_ZONES:
            if name in Camera_Str.Hidden_Zones:
                continue
            centre, half = Zone_Box(name, Areas_Str)
            if name not in Camera_Str.Open_Zones:
                out.append(Decoration_Str(Kind='box', Position=tuple(centre), Extent=tuple(half), Colour=_Rgba(colour, Theme_Str.Zone_Alpha)))
            out.append(Decoration_Str(Kind='frame', Position=tuple(centre), Extent=tuple(half), Colour=_Rgba(colour, Theme_Str.Edge_Alpha)))
            if labels:
                top = (centre[0], centre[1], centre[2] + half[2])
                out.append(Decoration_Str(Kind='sphere', Position=top, Radius=0.001, Colour=_Rgba(colour, 0.0), Label=label))
    # 2\ The observed samples: the detection phase in red, the later ones neutral.
    x, first = record.Observed_Position, min(frame.Seen, record.First_N)
    red = _Rgba(CONST_DETECTION_COLOUR)
    for k in range(first):
        out.append(Decoration_Str(Kind='sphere', Position=tuple(x[k]), Radius=CONST_SAMPLE_RADIUS, Colour=red))
    for k in range(first - 1):
        out.append(Decoration_Str(Kind='line', Position=tuple(x[k]), Extent=tuple(x[k + 1]), Radius=0.5 * CONST_SAMPLE_RADIUS, Colour=red))
    if frame.Seen >= record.First_N and record.First_N >= 1:
        out.append(Decoration_Str(Kind='sphere', Position=tuple(x[record.First_N - 1]), Radius=CONST_MARKER_RADIUS, Colour=red,
                                  Label=CONST_DETECTION_LABEL if labels else ''))
    for k in range(record.First_N, frame.Seen):
        out.append(Decoration_Str(Kind='sphere', Position=tuple(x[k]), Radius=CONST_LATER_RADIUS, Colour=_Rgba(Theme_Str.Samples, 0.9, 1.0)))
    # 3\ The true path ahead of the ball, dotted.
    ahead = record.Truth_Position[record.Truth_Time >= frame.Time]
    if len(ahead) > 1:
        length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ahead, axis=0), axis=1))])
        for at in np.searchsorted(length, np.arange(CONST_DOT_SPACING, length[-1], CONST_DOT_SPACING)):
            out.append(Decoration_Str(Kind='sphere', Position=tuple(ahead[at]), Radius=CONST_DOT_RADIUS,
                                      Colour=_Rgba(Theme_Str.Dots, 0.9, 1.0)))
    # 4\ Each method's newest prediction, until a go's catch time; the comparisons first, so that the driving method is
    #    drawn over them.
    if frame.Prediction >= 0 and not (record.Decision == 'go' and frame.Time > record.Catch_Time):
        for name in reversed(record.Methods):
            path = record.Paths[name][frame.Prediction]
            if not np.all(np.isfinite(path)):
                continue
            driving = name == record.Method
            colour = _Rgba(Method_Colour(name, record.Methods))
            for a, b in zip(path[:-1], path[1:]):
                out.append(Decoration_Str(Kind='line', Position=tuple(a), Extent=tuple(b),
                                          Radius=CONST_PATH_RADIUS if driving else CONST_COMPARISON_PATH_RADIUS, Colour=colour))
            if record.Catch_Point[name][frame.Prediction]:
                out.append(Decoration_Str(Kind='sphere', Position=tuple(path[-1]), Colour=colour,
                                          Radius=CONST_MARKER_RADIUS if driving else CONST_COMPARISON_RADIUS,
                                          Label=CONST_PREDICTION_LABEL if labels and driving else ''))
    return out


def Draw(scene: mujoco.MjvScene, decorations: tp.Sequence[Decoration_Str]) -> None:
    """
    Description:
        Add decorations to MuJoCo's abstract scene (the offscreen renderer's, or the viewer's user scene) as geoms of the
        decoration category, which cast no shadow. A scene without room for them is refused.

    Args:
        (1) scene [mujoco.MjvScene]: The scene, extended in place.
        (2) decorations [Sequence<Decoration_Str>]: The decorations.
    """

    if scene.ngeom + len(decorations) > scene.maxgeom:
        raise View_Error_Cls('The scene has room for %d geoms and holds %d; %d decorations do not fit. Show fewer methods or predict at '
                             'every K-th sample (--every).' % (scene.maxgeom, scene.ngeom, len(decorations)))
    kinds = {'sphere': mujoco.mjtGeom.mjGEOM_SPHERE, 'box': mujoco.mjtGeom.mjGEOM_BOX, 'frame': mujoco.mjtGeom.mjGEOM_LINEBOX,
             'line': mujoco.mjtGeom.mjGEOM_CAPSULE}
    identity = np.eye(3).ravel()
    for decoration in decorations:
        geom = scene.geoms[scene.ngeom]
        size = np.full(3, decoration.Radius) if decoration.Kind in ('sphere', 'line') else np.asarray(decoration.Extent, dtype=np.float64)
        mujoco.mjv_initGeom(geom, kinds[decoration.Kind], size, np.asarray(decoration.Position, dtype=np.float64), identity,
                            np.asarray(decoration.Colour, dtype=np.float32))
        if decoration.Kind == 'line':
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, decoration.Radius, np.asarray(decoration.Position, dtype=np.float64),
                                 np.asarray(decoration.Extent, dtype=np.float64))
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.label = decoration.Label
        # A zone's face takes the scene's light; everything else glows a little, so that it reads in a shadow.
        geom.emission, geom.specular = (0.0 if decoration.Kind == 'box' else 0.6), 0.0
        scene.ngeom += 1


def Add_Ghost(model: mujoco.MjModel, positions: np.ndarray, option: mujoco.MjvOption, scene: mujoco.MjvScene) -> None:
    """
    Description:
        Add the robot at other joint positions to MuJoCo's abstract scene as a ghost: the moving bodies' geoms at those
        positions, at CONST_GHOST_ALPHA of their opacity, as decorations; the ball's geoms and the sites are invisible.

    Args:
        (1) model [mujoco.MjModel]: The model shown.
        (2) positions [Vector<float> 1xQ]: The scene's positions (qpos) with the robot's joints at the ghost's pose.
        (3) option [mujoco.MjvOption]: The view's options (the geom groups shown).
        (4) scene [mujoco.MjvScene]: The scene, extended in place.
    """

    ghost = mujoco.MjData(model)
    ghost.qpos[:] = positions
    mujoco.mj_kinematics(model, ghost)
    first = scene.ngeom
    mujoco.mjv_addGeoms(model, ghost, option, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONST_BALL_BODY)
    for k in range(first, scene.ngeom):
        geom = scene.geoms[k]
        of_robot = geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and model.geom_bodyid[geom.objid] != ball
        geom.rgba[3] = CONST_GHOST_ALPHA * geom.rgba[3] if of_robot else 0.0
        geom.transparent, geom.category = 1, mujoco.mjtCatBit.mjCAT_DECOR


def Ghost_Positions(record: Throw_Record_Str, frame: Frame_Str) -> tp.Optional[np.ndarray]:
    """
    Description:
        The scene's positions of the ghost at a frame: the robot at the goal of the plan being executed, the rest as
        shown; None before the first plan.

    Args:
        (1) record [Throw_Record_Str(object)]: The record.
        (2) frame [Frame_Str(object)]: The frame.

    Returns:
        (1) parameter [Vector<float> 1xQ or None]: The positions.
    """

    if frame.Plan < 0:
        return None
    positions = record.State_Position[frame.State].copy()
    positions[record.Robot_Address] = record.Plan_Goal[frame.Plan]
    return positions


def Set_Camera(model: mujoco.MjModel, camera: mujoco.MjvCamera, option: mujoco.MjvOption, *, Camera_Str: Camera_Str) -> None:
    """
    Description:
        Put MuJoCo's free camera at a preset and show what the preset shows: the floor and the visual geoms, never the
        collision shapes (groups 3 and 4), the model's own area boxes (group 1; the zones are decorations) or a site,
        and the camera rail (group 5, View.Theme) when the preset has it. An orthographic preset sets the model's free-camera
        mode and its height; a perspective one CONST_FOVY.

    Args:
        (1) model [mujoco.MjModel]: The themed model; its free-camera mode and field of view are set.
        (2) camera [mujoco.MjvCamera]: The camera, set in place.
        (3) option [mujoco.MjvOption]: The view's options, set in place.
        (4) Camera_Str [Camera_Str(object)]: REQUIRED, keyword-only: the preset.
    """

    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = Camera_Str.Lookat
    camera.distance, camera.azimuth, camera.elevation = Camera_Str.Distance, Camera_Str.Azimuth, Camera_Str.Elevation
    orthographic = Camera_Str.Extent > 0.0
    camera.orthographic = int(orthographic)
    model.vis.global_.orthographic = int(orthographic)
    model.vis.global_.fovy = Camera_Str.Extent if orthographic else CONST_FOVY
    option.geomgroup[:] = (1, 0, 1, 0, 0, int(Camera_Str.Rail))
    option.sitegroup[:] = 0


def Legend_Figure(entries: tp.Sequence[tp.Tuple[str, tp.Tuple[int, int, int]]]) -> mujoco.MjvFigure:
    """
    Description:
        The legend as one of MuJoCo's figures, for the viewer's window (its text overlay has one colour): a figure
        without data whose legend names each entry in the entry's colour.

    Args:
        (1) entries [Sequence<Tuple<string, Tuple<int, int, int>>>]: The names and their colours (sRGB, 0-255).

    Returns:
        (1) parameter [mujoco.MjvFigure]: The figure.
    """

    figure = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(figure)
    figure.flg_legend, figure.flg_ticklabel[0], figure.flg_ticklabel[1], figure.flg_extend = 1, 0, 0, 0
    figure.figurergba[:], figure.panergba[:], figure.legendrgba[:] = (0.1, 0.1, 0.1, 0.55), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
    figure.gridsize[:], figure.gridrgb[:] = (2, 2), (0.1, 0.1, 0.1)
    # A line needs two points to be listed; they lie outside the range shown, so no line is drawn.
    figure.range[0][:], figure.range[1][:] = (0.0, 1.0), (0.0, 1.0)
    for k, (name, colour) in enumerate(entries):
        figure.linename[k] = name
        figure.linergb[k] = [value / 255.0 for value in colour]
        figure.linepnt[k] = 2
        figure.linedata[k][:4] = (0.0, -1.0, 1.0, -1.0)
    return figure


class Offscreen_Cls(object):
    """
    Description:
        MuJoCo's renderer on an offscreen buffer, through its public calls (a GL context, a rendering context, an
        abstract scene): one frame of a themed model with the decorations, a ghost pose, and optionally the text
        overlay and the legend with colour swatches drawn over it.

    Initialization of the Class:
        Args:
            (1) model [mujoco.MjModel]: The themed model (View.Theme.Themed_Scene); its offscreen buffer must hold
                                        the frame.
            (2) width [int]: The frame's width [px].
            (3) height [int]: The frame's height [px].

        Example:
            Initialization:
                Cls = Offscreen_Cls(model, 1920, 1080)

            Features:
                image = Cls.Render(data, CONST_CAMERAS['side'], decorations)
                Cls.Close()
    """

    def __init__(self, model: mujoco.MjModel, width: int, height: int) -> None:
        if not (0 < width <= model.vis.global_.offwidth and 0 < height <= model.vis.global_.offheight):
            raise View_Error_Cls('A frame of %d x %d px does not fit the model\'s offscreen buffer of %d x %d px; build the themed scene '
                                 'for the frame\'s size (View.Theme.Themed_Scene).'
                                 % (width, height, model.vis.global_.offwidth, model.vis.global_.offheight))
        # << PRIVATE >> #
        self.__model, self.__width, self.__height = model, int(width), int(height)
        self.__gl = mujoco.GLContext(self.__width, self.__height)
        self.__gl.make_current()
        self.__font = mujoco.mjtFontScale.mjFONTSCALE_300 if height >= CONST_LARGE_FONT_HEIGHT else mujoco.mjtFontScale.mjFONTSCALE_150
        self.__context = mujoco.MjrContext(model, self.__font)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.__context)
        self.__scene = mujoco.MjvScene(model, maxgeom=CONST_MAX_GEOMS)
        self.__option, self.__camera = mujoco.MjvOption(), mujoco.MjvCamera()
        self.__viewport = mujoco.MjrRect(0, 0, self.__width, self.__height)

    def Render(self, data: mujoco.MjData, camera: Camera_Str, decorations: tp.Sequence[Decoration_Str], *,
               ghost: tp.Optional[np.ndarray] = None, texts: tp.Optional[tp.Tuple[str, str]] = None,
               legend: tp.Sequence[tp.Tuple[str, tp.Tuple[int, int, int]]] = ()) -> np.ndarray:
        """
        Description:
            One frame: the model at the data's positions from a camera preset, with the decorations and a ghost pose;
            with `texts` the overlay (the two columns at the top left, the project's sentence at the bottom left) and
            the legend (a colour swatch and a name per entry, at the top right) are drawn over it.

        Args:
            (1) data [mujoco.MjData]: The model's data, its kinematics computed.
            (2) camera [Camera_Str(object)]: The camera preset.
            (3) decorations [Sequence<Decoration_Str>]: The decorations.
            (4) ghost [Vector<float> 1xQ or None]: Keyword-only: the positions of a ghost pose.
            (5) texts [Tuple<string, string> or None]: Keyword-only: the overlay's two columns (Overlay_Lines).
            (6) legend [Sequence<Tuple<string, Tuple<int, int, int>>>]: Keyword-only: the legend's entries.

        Returns:
            (1) parameter [Array<uint8> HxWx3]: The image, the top row first.
        """

        model, scene, context, viewport = self.__model, self.__scene, self.__context, self.__viewport
        self.__gl.make_current()
        Set_Camera(model, self.__camera, self.__option, Camera_Str=camera)
        mujoco.mjv_updateScene(model, data, self.__option, None, self.__camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        if ghost is not None:
            Add_Ghost(model, ghost, self.__option, scene)
        Draw(scene, decorations)
        mujoco.mjr_render(viewport, scene, context)
        if texts is not None:
            mujoco.mjr_overlay(self.__font, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, texts[0], texts[1], context)
            mujoco.mjr_overlay(self.__font, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, CONST_SAFETY, '', context)
            self.__Legend(legend)
        image = np.zeros((self.__height, self.__width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(image, None, viewport, context)
        return np.flipud(image)

    def __Legend(self, entries: tp.Sequence[tp.Tuple[str, tp.Tuple[int, int, int]]]) -> None:
        """
        Description:
            The legend at the top right: per entry a colour swatch and its name, on a dark panel. The sizes follow the
            frame's height (a row is 3 % of it).

        Args:
            (1) entries [Sequence<Tuple<string, Tuple<int, int, int>>>]: The names and their colours.
        """

        if not entries:
            return
        row = max(int(round(0.03 * self.__height)), 12)
        width = int(round(10.5 * row))
        left, top = self.__width - width - row // 2, self.__height - row // 2
        mujoco.mjr_rectangle(mujoco.MjrRect(left, top - row * len(entries) - row // 2, width, row * len(entries) + row // 2), 0.1, 0.1, 0.1, 0.6)
        for k, (name, colour) in enumerate(entries):
            base = top - row * (k + 1) - row // 4
            mujoco.mjr_rectangle(mujoco.MjrRect(left + row // 2, base + row // 4, int(1.5 * row), row // 2),
                                 colour[0] / 255.0, colour[1] / 255.0, colour[2] / 255.0, 1.0)
            # The name, centred in the rest of the row on a transparent ground (mjr_text would take its position
            # relative to the rectangle drawn last, the swatch).
            mujoco.mjr_label(mujoco.MjrRect(left + int(2.25 * row), base, width - int(2.5 * row), row), mujoco.mjtFont.mjFONT_NORMAL, name,
                             0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, self.__context)

    def Close(self) -> None:
        """
        Description:
            Free the rendering context and the GL context.
        """

        self.__context.free()
        self.__gl.free()


def Write_Png(path: str, image: np.ndarray) -> None:
    """
    Description:
        Write an image as a PNG file (8-bit RGB, no interlace, every row unfiltered, one zlib stream), with the standard
        library only: the signature, IHDR, IDAT and IEND, each chunk with its CRC-32 (PNG specification, 3rd edition).

    Args:
        (1) path [string]: The file.
        (2) image [Array<uint8> HxWx3]: The image, the top row first.
    """

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or image.shape[0] < 1 or image.shape[1] < 1:
        raise View_Error_Cls('A PNG frame is rows of RGB bytes (height x width x 3, uint8), found the shape %r of %s.'
                             % (tuple(image.shape), image.dtype))
    height, width = image.shape[:2]
    rows = np.zeros((height, 1 + 3 * width), dtype=np.uint8)
    rows[:, 1:] = image.reshape(height, 3 * width)

    def Chunk(kind: bytes, payload: bytes) -> bytes:
        """
        Description:
            A PNG chunk: its length, its type, its data and the CRC-32 of the type and the data.

        Args:
            (1) kind [bytes]: The type.
            (2) payload [bytes]: The data.

        Returns:
            (1) parameter [bytes]: The chunk.
        """

        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff)

    with open(path, 'wb') as file:
        file.write(b'\x89PNG\r\n\x1a\n' + Chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                   + Chunk(b'IDAT', zlib.compress(rows.tobytes(), CONST_PNG_LEVEL)) + Chunk(b'IEND', b''))


def Frame_Row(record: Throw_Record_Str, frame: Frame_Str, number: int, name: str) -> tp.Tuple[tp.Any, ...]:
    """
    Description:
        A frame's row of frames.csv (CONST_FRAME_COLUMNS): what the frame shows, so that a video's titles can be set
        from the file instead of burnt-in text.

    Args:
        (1) record [Throw_Record_Str(object)]: The record.
        (2) frame [Frame_Str(object)]: The frame.
        (3) number [int]: The frame's number, from 0.
        (4) name [string]: The frame's file name.

    Returns:
        (1) parameter [Tuple<Any>]: The row.
    """

    shown = frame.Prediction >= 0
    seconds = record.Inference[record.Method][frame.Prediction] if shown else float('nan')
    return (number, name, '%.6f' % frame.Time, frame.Phase, frame.Seen, record.Method, int(record.Sample_N[frame.Prediction]) if shown else '',
            '%.3f' % (1e3 * seconds) if np.isfinite(seconds) else '', frame.Plan + 1 if frame.Plan >= 0 else '', len(record.Plan_Time),
            frame.Outcome)


def Write_Frames(path: str, rows: tp.Iterable[tp.Sequence[tp.Any]]) -> None:
    """
    Description:
        Write frames.csv (SPEC §9.15): comma-separated, a header line, LF, CONST_FRAME_COLUMNS' order.

    Args:
        (1) path [string]: The file.
        (2) rows [Iterable<Sequence<Any>>]: The rows (Frame_Row).
    """

    with open(path, 'w', encoding='ascii', newline='') as file:
        file.write(','.join(CONST_FRAME_COLUMNS) + '\n')
        for row in rows:
            file.write(','.join('%s' % value for value in row) + '\n')
