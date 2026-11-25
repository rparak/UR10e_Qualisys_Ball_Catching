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
# File Name: ../test/Legacy/Reference.py
## =========================================================================== ##

# System (Default)
import os
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.: the areas and the Euclidean norm
#   ../Geometry/Core
from Geometry.Core import Cuboid_Cls, Euclidean_Norm, Spherical_Area_Cls
# Custom Lib.: the flight model and the integrators
#   ../Flight/Core
from Flight.Core import Flight_Model_Str, Integrate, Integrator

"""
Description:
    Initialization of constants.

    The 2022 prediction benchmark of commit 0e59db2 (Prediction/Helpers/benchmark_prediction.py
    with Ball/Simulation.py, Prediction/Numerical_Analysis.py and Prediction/Utils.py), rewritten
    on src/Geometry and src/Flight. It is a frozen test reference and the regression guard of those
    two domains, not a project predictor: it is fixed-step and assumes the 2022 frame (the thrower
    at +y, the ball approaching the robot along -y; SPEC §2.1). Nothing in src/ imports it.

    Every float32 rounding of the 2022 code is kept, and the three places where 2022 mixed a
    float32 scalar with a Python float are cast to float64 explicitly: NumPy 1.x promoted them,
    NumPy 2 (NEP 50) does not.
"""
# The repository root.
#   Unit [string]
CONST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# The 2022 inputs and the per-throw baseline (SPEC §9.3).
#   Unit [string]
CONST_LEGACY_INPUT_DIR = os.path.join(CONST_ROOT, 'Data', 'Legacy')
CONST_LEGACY_BASELINE_DIR = os.path.join(CONST_ROOT, 'Data', 'Legacy', 'Baseline')
# The motion-capture rate of the 2022 code (Ball/Parameters.CONST_QTM_FREQUENCY) and its step;
# 1 / 370 is the very float the 2022 code used.
#   Unit [Hz], [s]
CONST_LEGACY_RATE = 370
CONST_LEGACY_STEP = 1 / CONST_LEGACY_RATE
# Gravity of the 2022 code (Ball/Parameters.CONST_GRAVITY_ACCELERATION) and its vacuum model.
#   Unit [m/s^2]
CONST_LEGACY_GRAVITY = 9.81
CONST_LEGACY_FLIGHT_MODEL = Flight_Model_Str(Gravity=np.array([0.0, 0.0, (-1) * CONST_LEGACY_GRAVITY]))
# The number of input states of the throw detection and of the approximate catch (the benchmark's
# NUM_OF_ATTEMPTS), and the points of the linear segment of Mean_Opt (Bezier.Linear(..., 20)).
#   Unit [int]
CONST_LEGACY_N_STATES = 20
CONST_LEGACY_SEGMENT_POINTS = 20
# The flight cuboid in which a throw is detected (the benchmark's Area_Flight).
#   Unit [Cuboid_Cls(object)]
CONST_LEGACY_FLIGHT_AREA = Cuboid_Cls([0.0, 3.0, 2.25], [4.0, 4.0, 3.5])
# The robot's catch hemisphere widened by 0.01 m for the prediction error (the benchmark's
# Area_Robot_Modified, radius 1.01 m; the configuration's Areas.Robot.Radius is 1.0 m).
#   Unit [Spherical_Area_Cls(object)]
CONST_LEGACY_ROBOT_AREA = Spherical_Area_Cls([0.0, 0.0, 0.1807 + 0.18], np.array([0.0, 0.0, -90.0]) * (np.pi / 180.0),
                                             1.01, 'hemisphere')
# The distance to the true end point at which the final prediction stops (the benchmark's 1e-5).
#   Unit [m]
CONST_LEGACY_TOLERANCE = 1e-5
# The integrators by the names the 2022 input files carry.
#   Unit [Dict<string, string>]
CONST_LEGACY_METHODS = {'Midpoint': 'Midpoint', 'Heun': 'Heun', 'Second-Runge-Kutta': 'RK2', 'Fourth-Runge-Kutta': 'RK4'}
# The key states (Prediction/Utils.Get_Key_State).
#   Unit [Tuple<string>]
CONST_LEGACY_KEY_STATES = ('Last', 'Mean', 'Mean_Opt')
# The header and the number of throws of each input file (SPEC §9.3).
#   Unit [string], [int]
CONST_LEGACY_HEADER = 'r_0_X,r_0_Y,r_0_Z,r_n_X,r_n_Y,r_n_Z,t_f'
CONST_LEGACY_THROWS = 1000


class Legacy_Error_Cls(ValueError):
    """
    Description:
        A 2022 input or baseline file is not as SPEC §9.3 states, or a throw cannot be reproduced
        (not detected, no approximate catch, a tie the 2022 code could not resolve). The message
        names the file, the row or throw, and what is wrong.
    """


@dataclass
class Legacy_Inputs_Str:
    """
    Description:
        One 2022 input file: the start and end point of each throw and the value passed as its
        flight time, in float32 as the 2022 benchmark held them.
    """

    # The file read.
    #   Unit [string]
    File: str = ''
    # The start point r_0 of each throw.
    #   Unit [Matrix<float32> Nx3, m]
    R_0: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    # The end point r_n of each throw.
    #   Unit [Matrix<float32> Nx3, m]
    R_N: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    # The t_f column, passed by the 2022 benchmark as the flight time; it holds r_n_Z, the catch
    # height in [m], not a time (save_data.py wrote the wrong column; KNOWN_ISSUES).
    #   Unit [Vector<float32> 1xN, used as s]
    Flight_Time_Input: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


@dataclass
class Legacy_Throw_Result_Str:
    """
    Description:
        What the 2022 pipeline computes for one throw, the intermediate values included.
    """

    # The smallest distance between a predicted state and the true end point (the baseline value).
    #   Unit [float, m]
    Error: float = 0.0
    # The sample at which the throw is detected (0 = the first sample of the trajectory).
    #   Unit [int]
    Detection_Index: int = -1
    # The key state of the detected throw and of the approximate catch.
    #   Unit [Vector<float32> 1x6, m and m/s]
    Key_State_Throw: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    Key_State_Approx: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    # The number of states the final prediction returns.
    #   Unit [int]
    N_Final_States: int = 0


def Read_Legacy_Inputs(path: str) -> Legacy_Inputs_Str:
    """
    Description:
        The only reader of Data/Legacy/Experiment_1_<method>_DRAG_False.txt (SPEC §9.3). It checks
        the header, 1000 rows of 7 numbers, and that t_f equals r_n_Z character for character (the
        known defect); every problem is collected, then the read fails once.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [Legacy_Inputs_Str(object)]: The throws, in float32.
    """

    with open(path, newline='') as handle:
        lines = handle.read().split('\n')
    if lines and lines[-1] == '':
        lines = lines[:-1]
    problems = []
    if not lines or lines[0] != CONST_LEGACY_HEADER:
        problems.append('line 1: the header is %r, not %r' % (lines[0] if lines else '', CONST_LEGACY_HEADER))
    values = []
    for number, line in enumerate(lines[1:], start=2):
        fields = line.split(',')
        try:
            row = [float(text) for text in fields]
        except ValueError:
            row = []
        if len(row) != 7:
            problems.append('line %d: not 7 numbers: %r' % (number, line))
            continue
        if fields[6] != fields[5]:
            problems.append('line %d: t_f %s is not r_n_Z %s (SPEC §9.3)' % (number, fields[6], fields[5]))
        values.append(row)
    if len(lines) - 1 != CONST_LEGACY_THROWS:
        problems.append('%d rows, not %d' % (len(lines) - 1, CONST_LEGACY_THROWS))
    if problems:
        raise Legacy_Error_Cls('The 2022 input %s is not as SPEC §9.3 states: %s.' % (path, '; '.join(problems)))
    data = np.array(values, dtype=np.float64).astype(np.float32)
    return Legacy_Inputs_Str(File=path, R_0=data[:, 0:3], R_N=data[:, 3:6], Flight_Time_Input=data[:, 6])


def Read_Baseline_File(path: str) -> np.ndarray:
    """
    Description:
        The only reader of Data/Legacy/Baseline/<method>_<key state>.txt (SPEC §9.3): one error per
        throw, 1000 lines in %.10e.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [Vector<float> 1x1000]: The per-throw errors [m], parsed from their text.
    """

    with open(path) as handle:
        lines = handle.read().splitlines()
    if len(lines) != CONST_LEGACY_THROWS:
        raise Legacy_Error_Cls('The baseline %s has %d lines, not %d (SPEC §9.3).' % (path, len(lines), CONST_LEGACY_THROWS))
    return np.array([float(line) for line in lines])


def Generate_Legacy_Trajectory(r_0: np.ndarray, r_n: np.ndarray, t_f: np.float32, method: str) -> np.ndarray:
    """
    Description:
        The 2022 throw (Ball/Simulation.Generate_Object_Trajectory): the vacuum shot from r_0 to r_n
        in t_f rounded to whole frames of 370 Hz, integrated with the given method, in float32.

        Note:
            The initial velocity solves r_n = r_0 + v t + g t^2 / 2 for the vacuum only, which is
            why this generator stays in the test reference (SPEC §2.1).

    Args:
        (1 - 2) r_0, r_n [Vector<float32> 1x3]: Start and end point [m].
        (3) t_f [float32]: The flight time input [s].
        (4) method [string]: A key of Flight.Core.CONST_INTEGRATORS.

    Returns:
        (1) parameter [Matrix<float32> nx6]: The states at 370 Hz [m, m/s].
    """

    r = np.array([r_0, r_n])
    d_r = r[-1] - r[0]
    # NEP 50: 2022 divided a float32 scalar by a Python float and NumPy 1.x promoted it to float64.
    number_of_frames = int(round((np.float64(t_f) / CONST_LEGACY_STEP), 0))
    t_flight = (number_of_frames - 1) * CONST_LEGACY_STEP
    # NEP 50, as above.
    v = [np.float64(d_r[0]) / t_flight, np.float64(d_r[1]) / t_flight,
         (np.float64(d_r[2]) + CONST_LEGACY_GRAVITY / 2 * t_flight ** 2) / t_flight]
    return np.float32(Integrate(np.append(r[0], v), number_of_frames, CONST_LEGACY_STEP, method,
                                Flight_Model_Str=CONST_LEGACY_FLIGHT_MODEL))


class Throw_Detector_Cls(object):
    """
    Description:
        The 2022 throw detection (Prediction/Utils.Throw_Detection): it buffers the last n samples
        and fires when the oldest buffered sample is inside the area, or when every one of the last
        n steps rose in y and z and the newest sample is inside.

        Note:
            The rising-in-y test expects a thrower at -y, while the 2022 areas put the thrower at
            +y; on the legacy throws v_y < 0 throughout, so that branch never fires (KNOWN_ISSUES).
            It is kept as it is.

    Initialization of the Class:
        Args:
            (1) n_states [int]: The number of buffered samples.
            (2) area [Cuboid_Cls(object)]: The flight area.

        Example:
            Initialization:
                # Initialization of the class.
                Cls = Throw_Detector_Cls(20, CONST_LEGACY_FLIGHT_AREA)

            Features:
                # Functions of the class.
                Cls.Update([x, y, z], t)
    """

    def __init__(self, n_states: int, area: Cuboid_Cls) -> None:
        # << PRIVATE >> #
        self.__n_states = n_states
        self.__area = area
        self.__samples = []
        self.__times = []
        self.__rising = [False] * n_states

    def __Get_States(self) -> np.ndarray:
        """
        Description:
            The buffered samples with finite-difference velocities. The first state's velocity is
            taken against the last buffered sample (index -1), as in 2022: the mean over the window.

        Returns:
            (1) parameter [Matrix<float> nx6]: The states [m, m/s].
        """

        if len(self.__samples) != self.__n_states:
            raise Legacy_Error_Cls('The throw was detected on %d samples, not %d.' % (len(self.__samples), self.__n_states))
        x_r = np.transpose(self.__samples)
        x_v = np.zeros((self.__n_states, 3))
        for i in range(0, self.__n_states):
            dt = float(self.__times[i] - self.__times[i - 1])
            # NEP 50: a float32 difference divided by a Python float, promoted to float64 as in NumPy 1.x.
            x_v[i] = [np.float64(x_r[0][i] - x_r[0][i - 1]) / dt,
                      np.float64(x_r[1][i] - x_r[1][i - 1]) / dt,
                      np.float64(x_r[2][i] - x_r[2][i - 1]) / dt]
        return np.array([np.hstack((r, v)) for r, v in zip(np.transpose(x_r), x_v)])

    def Update(self, p: tp.List[np.float32], t: np.float64) -> tp.Tuple[bool, tp.Optional[np.ndarray]]:
        """
        Description:
            Adds one sample and says whether the throw is detected.

        Args:
            (1) p [Vector<float32> 1x3]: The sample [m].
            (2) t [float]: Its time [s].

        Returns:
            (1) parameter 1 [bool]: True when the throw is detected.
            (2) parameter 2 [Matrix<float> nx6]: The buffered states when detected, else None.
        """

        self.__samples.append(p)
        self.__times.append(t)
        if len(self.__samples) > 1:
            self.__rising.pop(0)
            self.__rising.append(bool(self.__samples[-2][1] < self.__samples[-1][1] and self.__samples[-2][2] < self.__samples[-1][2]))
        if len(self.__samples) > self.__n_states:
            self.__samples.pop(0)
            self.__times.pop(0)
        if (all(self.__rising) and self.__area.Is_Point_Inside(self.__samples[-1])) or self.__area.Is_Point_Inside(self.__samples[0]):
            self.__rising = [False] * self.__n_states
            return (True, self.__Get_States())
        return (False, None)


class Approximate_Catch_Detector_Cls(object):
    """
    Description:
        The 2022 approximate catch detection (Prediction/Utils.Approximate_Catch_Detection): it
        buffers the last n predicted states and fires when every one of the last n steps fell in y
        and z and the newest state is inside the area.

    Initialization of the Class:
        Args:
            (1) n_states [int]: The number of buffered states.
            (2) area [Spherical_Area_Cls(object)]: The robot's catch area.

        Example:
            Initialization:
                # Initialization of the class.
                Cls = Approximate_Catch_Detector_Cls(20, CONST_LEGACY_ROBOT_AREA)

            Features:
                # Functions of the class.
                Cls.Update(x)
    """

    def __init__(self, n_states: int, area: Spherical_Area_Cls) -> None:
        # << PRIVATE >> #
        self.__n_states = n_states
        self.__area = area
        self.__states = []
        self.__falling = [False] * n_states

    def Update(self, x: np.ndarray) -> tp.Tuple[bool, tp.Optional[tp.List[np.ndarray]]]:
        """
        Description:
            Adds one predicted state and says whether the approximate catch is detected.

        Args:
            (1) x [Vector<float> 1x6]: The state [m, m/s].

        Returns:
            (1) parameter 1 [bool]: True when detected.
            (2) parameter 2 [List<Vector<float> 1x6>]: The buffered states when detected, else None.
        """

        self.__states.append(x)
        if len(self.__states) > 1:
            self.__falling.pop(0)
            self.__falling.append(bool(self.__states[-2][1] > self.__states[-1][1] and self.__states[-2][2] > self.__states[-1][2]))
        if len(self.__states) > self.__n_states:
            self.__states.pop(0)
        if all(self.__falling) and self.__area.Is_Point_Inside(self.__states[-1][0:3]):
            self.__falling = [False] * self.__n_states
            return (True, list(self.__states))
        return (False, None)


def _Unique_Index(values: np.ndarray, value: float, what: str) -> int:
    """
    Description:
        The index of the one entry equal to value; 2022 (np.asscalar) failed on a tie, so does this.
    """

    index = np.where(values == value)[0]
    if index.size != 1:
        raise Legacy_Error_Cls('%s: %d entries equal %r, the 2022 code needs exactly one.' % (what, index.size, value))
    return int(index[0])


def _Linear_Segment_Points(p_0: np.ndarray, p_1: np.ndarray, n: int) -> np.ndarray:
    """
    Description:
        n points of the segment (1 - t) p_0 + t p_1, t evenly in [0, 1] (Utilities/Bezier.Linear).
    """

    t = np.linspace(0, 1, n)
    return np.transpose([(1 - t) * p[0] + t * p[1] for p in np.transpose([p_0, p_1])])


def Key_State(x_in: np.ndarray, kind: str) -> np.ndarray:
    """
    Description:
        The 2022 key state of n states (Prediction/Utils.Get_Key_State): 'Last' the last state,
        'Mean' the mean state, 'Mean_Opt' the point nearest the mean position on the segment
        between the two states nearest the mean state, with the mean velocity.

    Args:
        (1) x_in [Matrix<float> nx6]: The states [m, m/s].
        (2) kind [string]: One of CONST_LEGACY_KEY_STATES.

    Returns:
        (1) parameter [Vector<float32> 1x6]: The key state [m, m/s].
    """

    if kind == 'Last':
        return np.float32(x_in[-1])
    if kind == 'Mean':
        return np.float32(x_in.mean(axis=0))
    if kind != 'Mean_Opt':
        raise Legacy_Error_Cls('The key state "%s" is not one of %s.' % (kind, ', '.join(CONST_LEGACY_KEY_STATES)))
    x_in_mean = x_in.mean(axis=0)
    e_d = np.array([Euclidean_Norm(x_in_i - x_in_mean) for x_in_i in x_in])
    e_d_sort = np.sort(e_d)
    i_0, i_1 = _Unique_Index(e_d, e_d_sort[0], 'Mean_Opt'), _Unique_Index(e_d, e_d_sort[1], 'Mean_Opt')
    points = _Linear_Segment_Points(x_in[i_0][0:3], x_in[i_1][0:3], CONST_LEGACY_SEGMENT_POINTS)
    e_d = np.array([Euclidean_Norm(b_t - x_in_mean[0:3]) for b_t in points])
    i_min = _Unique_Index(e_d, np.min(e_d), 'Mean_Opt segment')
    return np.float32(np.hstack((points[i_min], x_in_mean[3::])))


def Predict_Approximate_Catch(x_in: np.ndarray, area: Spherical_Area_Cls, n_states: int, n_steps: int,
                              method: str) -> tp.Tuple[bool, tp.Optional[np.ndarray]]:
    """
    Description:
        The 2022 approximate catch prediction (Calculate_Approximate_Catch_Prediction): steps from
        the key state and feeds the detector only while y < radius + 1 m (the 2022 frame).

    Args:
        (1) x_in [Vector<float32> 1x6]: The key state [m, m/s].
        (2) area [Spherical_Area_Cls(object)]: The robot's catch area.
        (3 - 4) n_states, n_steps [int]: The detector's states; the steps tried.
        (5) method [string]: A key of Flight.Core.CONST_INTEGRATORS.

    Returns:
        (1) parameter 1 [bool]: True when the approximate catch is detected.
        (2) parameter 2 [Matrix<float32> n_statesx6]: The detector's states, else None.
    """

    step = Integrator(method)
    detector = Approximate_Catch_Detector_Cls(n_states, area)
    offset = area.Radius + 1.0
    x = step(x_in, CONST_LEGACY_STEP, Flight_Model_Str=CONST_LEGACY_FLIGHT_MODEL)
    is_close, x_out = False, None
    for _ in range(n_steps):
        if x[1] < offset:
            (is_close, x_out) = detector.Update(x)
            if is_close:
                break
        x = step(x, CONST_LEGACY_STEP, Flight_Model_Str=CONST_LEGACY_FLIGHT_MODEL)
    return (is_close, np.float32(x_out) if is_close else None)


def Predict_Final_Catch(x_in: np.ndarray, target: np.ndarray, tolerance: float, n_steps: int,
                        method: str) -> tp.Tuple[float, np.ndarray]:
    """
    Description:
        The 2022 final prediction to a known end point (Calculate_Prediction_Method_Auto): steps
        from the key state until a state is within the tolerance of the target, and returns the
        smallest distance before that and the states up to it.

    Args:
        (1) x_in [Vector<float32> 1x6]: The key state [m, m/s].
        (2) target [Vector<float32> 1x3]: The true end point [m].
        (3) tolerance [float]: The stopping distance [m].
        (4) n_steps [int]: The steps tried.
        (5) method [string]: A key of Flight.Core.CONST_INTEGRATORS.

    Returns:
        (1) parameter 1 [float]: The smallest distance [m].
        (2) parameter 2 [Matrix<float32> kx6]: The states up to the nearest one [m, m/s].
    """

    step = Integrator(method)
    x = step(x_in, CONST_LEGACY_STEP, Flight_Model_Str=CONST_LEGACY_FLIGHT_MODEL)
    e_d, x_out = [], []
    for _ in range(n_steps):
        e_d_i = Euclidean_Norm(x[0:3] - target)
        if e_d_i < tolerance:
            break
        x_out.append(x)
        e_d.append(e_d_i)
        x = step(x, CONST_LEGACY_STEP, Flight_Model_Str=CONST_LEGACY_FLIGHT_MODEL)
    if not e_d:
        raise Legacy_Error_Cls('The first predicted state is already within %g m of the target.' % tolerance)
    e_d = np.array(e_d)
    e_d_min = np.min(e_d)
    i_min = _Unique_Index(e_d, e_d_min, 'final prediction')
    return (e_d_min, np.float32(x_out[0:i_min + 1]))


def Run_Legacy_Throw(inputs: Legacy_Inputs_Str, throw: int, method_name: str, kind: str) -> Legacy_Throw_Result_Str:
    """
    Description:
        One throw of the 2022 benchmark: the trajectory, its detection in the flight area, the key
        state, the approximate catch in the robot's area, its key state and the final prediction.

    Args:
        (1) inputs [Legacy_Inputs_Str(object)]: The input file.
        (2) throw [int]: The throw (0 = the first data row).
        (3) method_name [string]: A key of CONST_LEGACY_METHODS.
        (4) kind [string]: One of CONST_LEGACY_KEY_STATES.

    Returns:
        (1) parameter [Legacy_Throw_Result_Str(object)]: The error and the intermediate values.
    """

    method = CONST_LEGACY_METHODS[method_name]
    trajectory = Generate_Legacy_Trajectory(inputs.R_0[throw], inputs.R_N[throw], inputs.Flight_Time_Input[throw], method)
    n_steps = trajectory[:, 0].size
    detector = Throw_Detector_Cls(CONST_LEGACY_N_STATES, CONST_LEGACY_FLIGHT_AREA)
    t = np.zeros(n_steps)
    i, is_thrown, x_throw = 0, False, None
    while n_steps > i:
        t[i] = t[i - 1] + CONST_LEGACY_STEP
        (is_thrown, x_throw) = detector.Update([trajectory[:, 0][i], trajectory[:, 1][i], trajectory[:, 2][i]], t[i])
        if is_thrown:
            break
        i += 1
    if not is_thrown:
        raise Legacy_Error_Cls('Throw %d of %s is not detected.' % (throw, inputs.File))
    x_ks_throw = Key_State(x_throw, kind)
    (is_close, x_approx) = Predict_Approximate_Catch(x_ks_throw, CONST_LEGACY_ROBOT_AREA, CONST_LEGACY_N_STATES, n_steps, method)
    if not is_close:
        raise Legacy_Error_Cls('Throw %d of %s has no approximate catch.' % (throw, inputs.File))
    x_ks_approx = Key_State(np.array(x_approx), kind)
    (e_d, x_final) = Predict_Final_Catch(x_ks_approx, inputs.R_N[throw].copy(), CONST_LEGACY_TOLERANCE, n_steps, method)
    return Legacy_Throw_Result_Str(Error=float(e_d), Detection_Index=i, Key_State_Throw=x_ks_throw,
                                   Key_State_Approx=x_ks_approx, N_Final_States=len(x_final))
