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
# File Name: ../Prediction/Core.py
## =========================================================================== ##

# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the numerical predictor's name of the training configuration's Model.Name
from Configuration.Core import CONST_MODEL_NAMES
#   ../Flight/Core - the one flight model, Simulate, the fit of an initial state
from Flight.Core import Acceleration, Fit_Initial_States, Flight_Model_Str, Simulate, CONST_SIMULATION_STEP
#   ../Observation/Core - M4's error covariance (the GLS_M4 weights)
from Observation.Core import Error_Covariance, Observation_Model_Str
#   ../Dataset/Core - the floor height a trajectory ends at
from Dataset.Core import CONST_FLOOR_HEIGHT

"""
Description:
    Initialization of constants.

    The numerical predictor and the trajectory output contract (SPEC §2.13.13, §7.7; F7).
    A predictor receives the observed prefix only (its own time stamps, never resampled)
    and returns a Trajectory_Str fixed before any query: its last observed time t_0, its own predicted end
    t_end (the ball centre at the floor height) and the state at any times in [t_0, t_end]. The numerical
    predictor 'GLS_M4' fits the state at the first sample under F3's one flight model by Gauss-Newton
    (Flight.Core.Fit_Initial_States), weighted by M4's error covariance between
    the samples (Observation.Core.Error_Covariance; with no process noise this is the estimate of a Kalman
    smoother whose state carries M4's drift and offset, F7 Part A). The state is integrated by Simulate's RK4
    on its 1/600 s grid to the floor, and a query is answered by cubic Hermite interpolation of the states
    and their derivatives on the grid interval holding it (4e-14 m from Simulate, F7 Part A), so a time's
    answer depends on that time only. Pure computation: no file, no configuration, no printing.
"""
# The longest a predicted flight may last from its first sample to the floor; a trajectory that has not
# reached the floor by then is refused. The recorded flights last 1.02-1.42 s (SPEC §4.2). *Assumed*.
#   Unit [float, s]
CONST_LONGEST_FLIGHT = 10.0
# The grid integrated first, as a share of the vacuum time to the floor plus a margin; drag lengthens the
# descent by a few per cent (F3), and the grid is doubled until it reaches the floor.
#   Unit [float, -], [float, s]
CONST_GRID_FACTOR = 1.2
CONST_GRID_MARGIN = 0.1
# The bisection steps of the floor time on its grid interval: 1/600 s / 2^60 is below 1e-20 s.
#   Unit [int]
CONST_FLOOR_BISECTIONS = 60


class Prediction_Error_Cls(ValueError):
    """
    Description:
        A prediction cannot be made or queried as asked: an unknown predictor, too few or unordered samples,
        a query outside the trajectory's range, a trajectory that does not reach the floor. The message names
        the value and what is allowed; a script prints it verbatim as an [ERROR] line.
    """


@dataclass
class Trajectory_Str:
    """
    Description:
        A predicted trajectory (SPEC §2.13.13): fixed before any query, answered in [T_0, T_End] only, in the
        time base of the samples it was predicted from and the frame of their positions.
    """

    # The last observed time and the predicted end, the ball centre at CONST_FLOOR_HEIGHT.
    #   Unit [float, s]
    T_0: float = 0.0
    T_End: float = 0.0
    # The time of the grid's first state (the first observed sample) and the grid step.
    #   Unit [float, s]
    Grid_Start: float = 0.0
    Grid_Step: float = CONST_SIMULATION_STEP
    # The states on the grid from Grid_Start to past T_End, and their time derivatives (v, a).
    #   Unit [Matrix<float> Kx6, m, m/s], [Matrix<float> Kx6, m/s, m/s^2]
    Grid_State: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    Grid_Derivative: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))

    def State(self, t: np.ndarray) -> np.ndarray:
        """
        Description:
            The state at each time, by cubic Hermite interpolation of the grid states and their derivatives on
            the grid interval holding it; each answer depends on its own time only (query independence, SPEC
            §2.13.13). A time outside [T_0, T_End] is refused.

        Args:
            (1) t [Vector<float> 1xM]: The times [s], in any order.

        Returns:
            (1) parameter [Matrix<float> Mx6]: The states [m, m/s].
        """

        t = np.asarray(t, dtype=np.float64).reshape(-1)
        outside = t[~np.isfinite(t) | (t < self.T_0) | (t > self.T_End)]
        if outside.size:
            raise Prediction_Error_Cls('The time %r s is outside the trajectory\'s range [%r, %r] s: a trajectory is answered from its '
                                       'last observed sample to its predicted end only (SPEC §2.13.13).'
                                       % (float(outside[0]), self.T_0, self.T_End))
        h = self.Grid_Step
        k = np.minimum(np.floor((t - self.Grid_Start) / h).astype(np.int64), len(self.Grid_State) - 2)
        s = ((t - self.Grid_Start) - k * h) / h
        x_0, x_1 = self.Grid_State[k], self.Grid_State[k + 1]
        d_0, d_1 = self.Grid_Derivative[k], self.Grid_Derivative[k + 1]
        h_00, h_10 = (2.0 * s ** 3 - 3.0 * s ** 2 + 1.0)[:, None], (s ** 3 - 2.0 * s ** 2 + s)[:, None]
        h_01, h_11 = (-2.0 * s ** 3 + 3.0 * s ** 2)[:, None], (s ** 3 - s ** 2)[:, None]
        return h_00 * x_0 + h_10 * h * d_0 + h_01 * x_1 + h_11 * h * d_1

    def Scan_Times(self, low: float) -> np.ndarray:
        """
        Description:
            The times from low to T_End at which a crossing of a surface is looked for (Evaluation): low, every
            grid time between, and T_End.

        Args:
            (1) low [float]: The first time, in [T_0, T_End] [s].

        Returns:
            (1) parameter [Vector<float> 1xK]: The times, increasing, from low to T_End [s].
        """

        grid = self.Grid_Start + np.arange(len(self.Grid_State)) * self.Grid_Step
        return np.concatenate(([low], grid[(grid > low) & (grid < self.T_End)], [self.T_End]))


def Trajectory_From_State(x: np.ndarray, t_start: float, t_0: float, *, Flight_Model_Str: Flight_Model_Str) -> Trajectory_Str:
    """
    Description:
        The trajectory of a state under the flight model: Simulate's grid from t_start to past the floor, the
        floor time by bisection of the interpolated height on its grid interval (the first descent through
        CONST_FLOOR_HEIGHT after t_0), answered from t_0. The truth of a simulated flight is built the same way
        from its release state (Evaluation).

    Args:
        (1) x [Vector<float> 1x6]: The state at t_start [m, m/s].
        (2) t_start [float]: The time of x [s].
        (3) t_0 [float]: The first time answered (the last observed sample), >= t_start [s].
        (4) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Trajectory_Str(object)]: The trajectory.
    """

    h = CONST_SIMULATION_STEP
    g = -Flight_Model_Str.Gravity[2]
    z, v_z = x[2] - CONST_FLOOR_HEIGHT, x[5]
    duration = CONST_GRID_FACTOR * (v_z + np.sqrt(max(v_z ** 2 + 2.0 * g * z, 0.0))) / g + CONST_GRID_MARGIN
    duration = max(duration, t_0 - t_start + CONST_GRID_MARGIN)
    while True:
        n = int(np.ceil(min(duration, CONST_LONGEST_FLIGHT) / h)) + 2
        grid = Simulate(x, np.arange(n) * h, Flight_Model_Str=Flight_Model_Str)
        k_0 = int(np.floor((t_0 - t_start) / h))
        if grid[k_0, 2] < CONST_FLOOR_HEIGHT:
            raise Prediction_Error_Cls('The predicted trajectory is below the floor height %.3f m at its last observed time %r s: the '
                                       'prefix gives a flight that has already landed. Check the samples.' % (CONST_FLOOR_HEIGHT, float(t_0)))
        below = np.where((grid[k_0:-1, 2] >= CONST_FLOOR_HEIGHT) & (grid[k_0 + 1:, 2] < CONST_FLOOR_HEIGHT))[0]
        if below.size:
            break
        if duration >= CONST_LONGEST_FLIGHT:
            raise Prediction_Error_Cls('The predicted trajectory does not descend to the floor height %.3f m within %.1f s of its first '
                                       'sample (CONST_LONGEST_FLIGHT); the prefix gives no flight that lands. Check the samples.'
                                       % (CONST_FLOOR_HEIGHT, CONST_LONGEST_FLIGHT))
        duration *= 2.0
    k = k_0 + int(below[0])
    last = k + 2
    trajectory = Trajectory_Str(T_0=float(t_0), T_End=float(t_start + (k + 1) * h), Grid_Start=float(t_start), Grid_Step=h,
                                Grid_State=grid[:last], Grid_Derivative=np.hstack((grid[:last, 3:], Acceleration(grid[:last, 3:],
                                                                                                                Flight_Model_Str=Flight_Model_Str))))
    low, high = max(t_start + k * h, t_0), t_start + (k + 1) * h
    if trajectory.State([low])[0, 2] < CONST_FLOOR_HEIGHT:
        raise Prediction_Error_Cls('The predicted trajectory is below the floor height %.3f m at its last observed time %r s: the prefix '
                                   'gives a flight that has already landed. Check the samples.' % (CONST_FLOOR_HEIGHT, float(t_0)))
    for _ in range(CONST_FLOOR_BISECTIONS):
        middle = 0.5 * (low + high)
        trajectory.T_End = high
        if trajectory.State([middle])[0, 2] >= CONST_FLOOR_HEIGHT:
            low = middle
        else:
            high = middle
    trajectory.T_End = float(low)
    return trajectory


def Check_Prefix(time: np.ndarray, position: np.ndarray, rate: float) -> None:
    """
    Description:
        Refuse a prefix no predictor can use: fewer than 2 samples (6 unknowns), times not finite or not
        strictly increasing, positions not Mx3 or not finite, a rate not > 0.

    Args:
        (1) time [Vector<float> 1xM]: The sample times [s].
        (2) position [Matrix<float> Mx3]: The positions [m].
        (3) rate [float]: The rate [Hz].
    """

    if time.ndim != 1 or position.shape != (len(time), 3):
        raise Prediction_Error_Cls('A prefix is M times and Mx3 positions; %s times and %s positions were given.'
                                   % (time.shape, position.shape))
    if len(time) < 2:
        raise Prediction_Error_Cls('A prefix of %d sample(s) cannot fix a state of 6 values; give at least 2 samples.' % len(time))
    if not (np.all(np.isfinite(time)) and np.all(np.isfinite(position))):
        raise Prediction_Error_Cls('A prefix holds a time or position that is not finite; give finite samples.')
    if np.any(np.diff(time) <= 0.0):
        raise Prediction_Error_Cls('The prefix times are not strictly increasing (first at index %d); give the samples in time order.'
                                   % int(np.where(np.diff(time) <= 0.0)[0][0] + 1))
    if not rate > 0.0:
        raise Prediction_Error_Cls('The rate %r Hz is not > 0.' % (rate,))


def Predict(name: str, time: np.ndarray, position: np.ndarray, rate: float, *, Flight_Model_Str: Flight_Model_Str,
            Observation_Model_Str: Observation_Model_Str) -> Trajectory_Str:
    """
    Description:
        The predicted trajectory of a prefix (SPEC §7.7): the state at the first sample fitted under the flight
        model by Gauss-Newton, weighted by M4's covariance between the samples ('GLS_M4'; the rate gives the
        lost time of each dropout), then integrated to the floor. Only the prefix is read.

    Args:
        (1) name [string]: One of CONST_MODEL_NAMES.
        (2) time [Vector<float> 1xM]: The sample times [s], strictly increasing, as observed (never resampled).
        (3) position [Matrix<float> Mx3]: The positions [m].
        (4) rate [float]: The rate of the samples' frames [Hz].
        (5) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the one flight model.
        (6) Observation_Model_Str [Observation_Model_Str(object)]: REQUIRED, keyword-only: M4 (the weights).

    Returns:
        (1) parameter [Trajectory_Str(object)]: The trajectory from the last sample to the floor.
    """

    if name not in CONST_MODEL_NAMES:
        raise Prediction_Error_Cls('The predictor "%s" is not one of %s.' % (name, ', '.join(CONST_MODEL_NAMES)))
    time = np.asarray(time, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    Check_Prefix(time, position, float(rate))
    tau = time - time[0]
    covariance = Error_Covariance(tau, float(rate), Observation_Model_Str=Observation_Model_Str)
    whitening = [np.stack([np.linalg.inv(np.linalg.cholesky(covariance[a])) for a in range(3)])]
    x, _ = Fit_Initial_States([tau], [position], Flight_Model_Str=Flight_Model_Str, whitening=whitening)
    return Trajectory_From_State(x[0], float(time[0]), float(time[-1]), Flight_Model_Str=Flight_Model_Str)
