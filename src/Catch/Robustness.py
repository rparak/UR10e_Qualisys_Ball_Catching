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
# File Name: ../Catch/Robustness.py
## =========================================================================== ##


# Dataclasses (Data classes)
from dataclasses import dataclass, field
# Zlib (A dataset's number from its path, stable across processes)
import zlib
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Flight/Core - the true trajectory
from Flight.Core import Flight_Model_Str, Simulate
#   ../Prediction/Core - a query outside a trajectory's range
from Prediction.Core import Prediction_Error_Cls
#   ../Catch/Core - the catch's error
from Catch.Core import Catch_Error_Cls

"""
Description:
    Initialization of constants.

    The robustness sweep's prediction (SPEC §7.11): the true trajectory
    plus an injected error whose size per horizon range comes from the per-flight errors of a stored evaluation (the
    real validation flights' of F14's final test, or the simulated validation's as the construction's own check),
    times a scale. Per throw one quantile is drawn and used for every range and every prefix, and one direction: the
    error of a throw keeps its rank and its direction through the re-plans, which is harsher than a real predictor's
    error, which shrinks with more samples. The stored errors are read by the caller (scripts/catch.py, the sweep
    only); this module takes their rows and never reads a file. **Guard**: the stored real-flight errors are used here
    and nowhere else; they are never fed back into the predictor, the decision's margins or the planner.
"""
# The seed of the throws' draws (their quantile and direction), with the dataset's number and the throw's index.
#   Unit [int]
CONST_SWEEP_SEED = 20261122
# The fewest flights with an error in a horizon range for the range to have its own distribution; a range with fewer
# takes the distribution of the range before it.
#   Unit [int]
CONST_MINIMUM_FLIGHTS = 10
# The stored evaluations' prefixes whose errors are used: the shorter one's for a prediction from fewer samples than
# the longer, the longer one's from there on (the stored real errors exist at N 10, 20 and 40 only).
#   Unit [int]
CONST_SHORT_N = 20
CONST_LONG_N = 40


@dataclass
class Error_Table_Str:
    """
    Description:
        The error sizes of a stored evaluation (Error_Table): per horizon range the sorted per-flight position errors,
        for the shorter and the longer prefix.
    """

    # The horizon ranges' mid-points.
    #   Unit [Vector<float> 1xR, s]
    Mid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Per range the sorted errors of the flights, for a prefix below CONST_LONG_N samples and from it on.
    #   Unit [List<Vector<float>>, m]
    Short: tp.List[np.ndarray] = field(default_factory=list)
    Long: tp.List[np.ndarray] = field(default_factory=list)


def Range_Labels(ranges: tp.Sequence[tp.Tuple[float, float]]) -> tp.List[str]:
    """
    Description:
        The horizon ranges as a stored evaluation's per_horizon.csv names them (SPEC §9.7): '[low, high)', the last
        '[low, high]'.

    Args:
        (1) ranges [Sequence<Tuple<float, float>>]: The ranges [s] (Evaluation.Core.CONST_HORIZON_RANGES).

    Returns:
        (1) parameter [List<string>]: The labels.
    """

    return ['[%g, %g%s' % (low, high, ']' if k == len(ranges) - 1 else ')') for k, (low, high) in enumerate(ranges)]


def Error_Table(rows: tp.Sequence[tp.Dict[str, str]], ranges: tp.Sequence[tp.Tuple[float, float]]) -> Error_Table_Str:
    """
    Description:
        The error table of a stored evaluation's rows: per prefix (CONST_SHORT_N, CONST_LONG_N) and horizon range, each
        flight's position error (the mean over the run's seeds), sorted. A range with fewer than CONST_MINIMUM_FLIGHTS
        flights takes the range before it; rows without the first range for a prefix are refused.

    Args:
        (1) rows [Sequence<Dict<string, string>>]: Rows of per_horizon.csv (the columns 'flight', 'n', 'horizon',
                                                   'position_error_mm'), already selected by the caller.
        (2) ranges [Sequence<Tuple<float, float>>]: The horizon ranges [s].

    Returns:
        (1) parameter [Error_Table_Str(object)]: The table.
    """

    labels = Range_Labels(ranges)
    table = Error_Table_Str(Mid=np.array([0.5 * (low + high) for low, high in ranges]))
    for n, target in ((CONST_SHORT_N, table.Short), (CONST_LONG_N, table.Long)):
        for label in labels:
            flights: tp.Dict[str, tp.List[float]] = {}
            for row in rows:
                if row['n'] == '%d' % n and row['horizon'] == label and row['position_error_mm'] != '':
                    flights.setdefault(row['flight'], []).append(float(row['position_error_mm']))
            errors = np.sort(np.array([np.mean(values) for values in flights.values()])) * 1e-3
            if len(errors) >= CONST_MINIMUM_FLIGHTS:
                target.append(errors)
            elif target:
                target.append(target[-1])
            else:
                raise Catch_Error_Cls('The stored evaluation holds %d flights with a position error at N = %d in the horizon range %s; the '
                                      'sweep needs at least %d there (SPEC §7.11).' % (len(errors), n, label, CONST_MINIMUM_FLIGHTS))
    return table


@dataclass
class Injected_Trajectory_Str:
    """
    Description:
        A prediction of the sweep: the true trajectory plus an error of a fixed direction whose size depends on the
        horizon, linear between the horizon ranges' mid-points and constant beyond them. It answers as a predicted
        trajectory does (Prediction.Core.Trajectory_Str): in [T_0, T_End] only.
    """

    # The last observed time and the true end of the flight.
    #   Unit [float, s]
    T_0: float = 0.0
    T_End: float = 0.0
    # The true states at given times.
    #   Unit [Callable[[Vector<float> 1xM], Matrix<float> Mx6]]
    Truth: tp.Callable[[np.ndarray], np.ndarray] = None
    # The horizons of the error's nodes, the error's size at them and its direction.
    #   Unit [Vector<float> 1xR, s], [Vector<float> 1xR, m], [Vector<float> 1x3]
    Mid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Size: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Direction: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def State(self, t: np.ndarray) -> np.ndarray:
        """
        Description:
            The predicted state at each time: the true one with the error added to the position, and the error's
            rate of change to the velocity. A time outside [T_0, T_End] is refused.

        Args:
            (1) t [Vector<float> 1xM]: The times [s], in any order.

        Returns:
            (1) parameter [Matrix<float> Mx6]: The states [m, m/s].
        """

        t = np.asarray(t, dtype=np.float64).reshape(-1)
        outside = t[~np.isfinite(t) | (t < self.T_0) | (t > self.T_End)]
        if outside.size:
            raise Prediction_Error_Cls('The time %r s is outside the trajectory\'s range [%r, %r] s: a trajectory is answered from its '
                                       'last observed sample to its end only (SPEC §2.13.13).' % (float(outside[0]), self.T_0, self.T_End))
        state = np.array(self.Truth(t), dtype=np.float64)
        horizon = t - self.T_0
        size = np.interp(horizon, self.Mid, self.Size)
        node = np.clip(np.searchsorted(self.Mid, horizon) - 1, 0, len(self.Mid) - 2)
        slope = (self.Size[node + 1] - self.Size[node]) / (self.Mid[node + 1] - self.Mid[node])
        slope = np.where((horizon > self.Mid[0]) & (horizon < self.Mid[-1]), slope, 0.0)
        state[:, :3] += size[:, None] * self.Direction
        state[:, 3:] += slope[:, None] * self.Direction
        return state


def Injected_Predictor(data: tp.Any, flight: Flight_Model_Str, *, table: Error_Table_Str, scale: float
                       ) -> tp.Callable[[int], tp.Callable[..., Injected_Trajectory_Str]]:
    """
    Description:
        The sweep's predictor for the throws of a dataset (Catch.Interception.Intercept_Chunk's `predictor`): for a
        throw, a function of a prefix that returns the true trajectory plus the injected error. The throw's quantile u
        and direction are drawn once from (CONST_SWEEP_SEED, the dataset's number, the throw's index); the error's size
        in a horizon range is `scale` times the u-quantile of the table's errors there, the shorter prefix's below
        CONST_LONG_N samples and the longer one's from there on. The true states are computed once per time and throw
        (Flight.Core.Simulate answers each time on its own).

    Args:
        (1) data [Dataset_Str(object)]: The dataset (its release states, floor times and seed).
        (2) flight [Flight_Model_Str(object)]: The flight model.
        (3) table [Error_Table_Str(object)]: REQUIRED, keyword-only: the error table.
        (4) scale [float]: REQUIRED, keyword-only: the error's scale, >= 0 (1: the stored level).

    Returns:
        (1) parameter [Callable[[int], Callable]]: The predictor of a throw's prefixes, by the throw's index.
    """

    number = zlib.crc32(np.asarray(data.Release_State[:8], dtype=np.float64).tobytes())

    def For_Throw(index: int) -> tp.Callable[..., Injected_Trajectory_Str]:
        """
        Description:
            The predictor of one throw's prefixes.

        Args:
            (1) index [int]: The throw's index in the dataset.

        Returns:
            (1) parameter [Callable]: (time, position, rate) -> the injected trajectory.
        """

        release, end = np.asarray(data.Release_State[index], dtype=np.float64), float(data.Floor_Time[index])
        rng = np.random.default_rng(np.random.SeedSequence([CONST_SWEEP_SEED, number, int(index)]))
        u = float(rng.uniform())
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        sizes = {name: scale * np.array([np.quantile(errors, u) for errors in ranges])
                 for name, ranges in (('short', table.Short), ('long', table.Long))}
        known: tp.Dict[float, np.ndarray] = {}

        def Truth(t: np.ndarray) -> np.ndarray:
            """
            Description:
                The true states at the times, each computed once.

            Args:
                (1) t [Vector<float> 1xM]: The times [s].

            Returns:
                (1) parameter [Matrix<float> Mx6]: The states [m, m/s].
            """

            times = [float(value) for value in t]
            missing = sorted({value for value in times if value not in known})
            if missing:
                known.update(zip(missing, Simulate(release, np.array(missing), Flight_Model_Str=flight)))
            return np.array([known[value] for value in times]).reshape(-1, 6)

        def Predict(time: np.ndarray, position: np.ndarray, rate: float) -> Injected_Trajectory_Str:
            """
            Description:
                The injected prediction of a prefix; only the prefix's length and last time are used.

            Args:
                (1) time [Vector<float> 1xN]: The observed times [s].
                (2) position [Matrix<float> Nx3]: The observed positions [m] (not used: the truth is known).
                (3) rate [float]: The sample rate [Hz] (not used).

            Returns:
                (1) parameter [Injected_Trajectory_Str(object)]: The trajectory.
            """

            return Injected_Trajectory_Str(T_0=float(time[-1]), T_End=end, Truth=Truth, Mid=table.Mid,
                                           Size=sizes['short' if len(time) < CONST_LONG_N else 'long'], Direction=direction)

        return Predict

    return For_Throw
