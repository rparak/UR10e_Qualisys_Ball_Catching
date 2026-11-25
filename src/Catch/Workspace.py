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
# File Name: ../Catch/Workspace.py
## =========================================================================== ##

# CSV (The summary's file)
import csv
# Dataclasses (Data classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Flight/Core - the throws' trajectories
from Flight.Core import Simulate, Flight_Model_Str
#   ../Robot/Core - the move time
from Robot.Core import Move_Time, Wrapping_Joints
#   ../Catch/Core - the catch pose, the lattice and the largest cuboid
from Catch.Core import Catch_Context_Str, Check_Point, Largest_Cuboid, Catch_Error_Cls, CONST_CRITERIA, CONST_JOINTS_USED

"""
Description:
    Initialization of constants.

    The workspace (SPEC §7.11, F4): the map of the lattice points where a catch pose exists for every
    arrival direction, the times in which the robot reaches them from a ready pose, the throws that pass
    through them, the joint search of the ready pose and the catch area, and the workspace's two files
    (SPEC §9.11). No file but its own two is read or written here, and nothing is printed.
"""
# The version of the workspace map's file; a reader refuses another.
#   Unit [int]
CONST_MAP_VERSION = 1
# The columns of the summary's file.
#   Unit [Tuple<string>]
CONST_SUMMARY_COLUMNS = ('quantity', 'value', 'unit')
# The sampling of a throw's trajectory for the cells it passes through: every 10 ms, as in F4 Part A
# (the ball moves 6-8 m/s, so 60-80 mm per sample: a 0.05 m cell may be skipped on a sample, a 0.1 m one
# hardly; the reach sets are compared on the same samples).
#   Unit [s]
CONST_SAMPLE_STEP = 0.01
# The lowest centre of the ball that counts as in flight: its radius above the floor (SPEC §14.6).
#   Unit [m]
CONST_FLOOR = 0.030
# The reference time of the catch area (the rule B800): the area holds only
# points the robot reaches from its ready pose within this time. Its source is F4 Part A, the train
# partition of the dataset Default: the ball enters the 1.0 m sphere about the robot 0.76 s (5 % quantile)
# / 0.99 s (median) after its 20th observed sample, so 0.7 s lies below the time 95 % of the throws leave.
# Whether a throw leaves the time is still tested per throw (F9, F10). The decision made at this observed
# sample, and a catch needs the robot there this long before the ball (the reported catchable shares).
#   Unit [s], [int], [s]
CONST_REACH_TIME = 0.7
CONST_DECISION_SAMPLE = 20
CONST_ALLOWANCE = 0.10
# The joint search of the ready pose and the catch area (the rule B800; heuristic, not
# exhaustive): the first stage's candidates stand on every CONST_SEARCH_COARSE-th lattice point along each
# axis; its CONST_SEARCH_KEEP best are kept with every candidate on the scoring sub-lattice within
# CONST_SEARCH_REFINE lattice steps of them; both stages score on the sub-lattice of every
# CONST_SEARCH_SCORE-th point (0.1 m at a 0.05 m step), and the CONST_SEARCH_FINAL best of the second stage
# are scored again on the whole lattice.
#   Unit [int]
CONST_SEARCH_COARSE = 4
CONST_SEARCH_KEEP = 3
CONST_SEARCH_REFINE = 2
CONST_SEARCH_SCORE = 2
CONST_SEARCH_FINAL = 6


@dataclass
class Workspace_Str:
    """
    Description:
        The workspace map (Derive_Workspace): every lattice point, whether it qualifies, why not, and the
        joint vectors of every qualifying catch pose of the points that qualify.
    """

    # The lattice's step and its points with their whole multiples of the step.
    #   Unit [float, m], [Matrix<float> Nx3, m], [Matrix<int> Nx3]
    Step: float = 0.0
    Points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    Index: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    # The cup's axes of the arrival directions.
    #   Unit [Matrix<float> Dx3]
    Axes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # Whether each point qualifies; for one that does not, the axis without a qualifying pose and the code
    # (index into CONST_CRITERIA) of the criterion its best pose failed; -1 for a point that qualifies.
    #   Unit [Vector<bool> 1xN], [Vector<int> 1xN], [Vector<int> 1xN]
    Qualifies: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    Failed_Axis: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    Failed_Criterion: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    # The rows of the qualifying points, and for each its qualifying poses (joints 1 to CONST_JOINTS_USED),
    # NaN for a branch that does not qualify.
    #   Unit [Vector<int> 1xM], [Array<float32> MxDx8xJ, rad]
    Rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    Poses: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 8, CONST_JOINTS_USED), dtype=np.float32))


def Derive_Workspace(points: np.ndarray, index: np.ndarray, step: float, axes: np.ndarray,
                     progress: tp.Optional[tp.Callable[[int, int, int], None]] = None, *,
                     Catch_Context_Str: Catch_Context_Str) -> Workspace_Str:
    """
    Description:
        Check every lattice point (Catch.Core.Check_Point); a point that qualifies is checked again with
        every pose of every axis, so that the map holds all its qualifying poses.

    Args:
        (1) points [Matrix<float> Nx3]: The lattice's points [m] (Catch.Core.Lattice).
        (2) index [Matrix<int> Nx3]: Their whole multiples of the step.
        (3) step [float]: The lattice's step [m].
        (4) axes [Matrix<float> Dx3]: The cup's axes (Catch.Core.Arrival_Axes).
        (5) progress [Callable[[int, int, int], None]]: Called every 5 000 points with the points done, all
                                                       points and the qualifying ones so far, or None.
        (6) Catch_Context_Str [Catch_Context_Str(object)]: REQUIRED, keyword-only: the context.

    Returns:
        (1) parameter [Workspace_Str(object)]: The map.
    """

    count = len(points)
    qualifies = np.zeros(count, dtype=bool)
    failed_axis = np.full(count, -1, dtype=np.int8)
    failed_criterion = np.full(count, -1, dtype=np.int8)
    poses = []
    for row in range(count):
        good, _, axis, criterion = Check_Point(points[row], axes, Catch_Context_Str=Catch_Context_Str)
        if good:
            _, kept, _, _ = Check_Point(points[row], axes, every=True, Catch_Context_Str=Catch_Context_Str)
            qualifies[row] = True
            poses.append(kept[..., :CONST_JOINTS_USED].astype(np.float32))
        else:
            failed_axis[row], failed_criterion[row] = axis, CONST_CRITERIA.index(criterion)
        if progress is not None and (row + 1) % 5000 == 0:
            progress(row + 1, count, int(qualifies.sum()))
    rows = np.flatnonzero(qualifies).astype(np.int64)
    stacked = np.stack(poses) if poses else np.zeros((0, len(axes), 8, CONST_JOINTS_USED), dtype=np.float32)
    return Workspace_Str(Step=float(step), Points=np.asarray(points, dtype=np.float64), Index=np.asarray(index, dtype=np.int64),
                         Axes=np.asarray(axes, dtype=np.float64), Qualifies=qualifies, Failed_Axis=failed_axis,
                         Failed_Criterion=failed_criterion, Rows=rows, Poses=stacked)


def Configured_Cuboid(lower: np.ndarray, upper: np.ndarray, step: float) -> tp.Tuple[tp.List[float], tp.List[float]]:
    """
    Description:
        The catch area from the largest cuboid of qualifying lattice points: shrunk by one lattice step on
        every face, because only the lattice points were checked and the space between them was not.
        Refused when a side would hold fewer than one step.

    Args:
        (1) lower [Vector<int> 1x3]: The cuboid's lower corner in multiples of the step (Catch.Core.Largest_Cuboid).
        (2) upper [Vector<int> 1x3]: Its upper corner, inclusive.
        (3) step [float]: The lattice's step [m].

    Returns:
        (1) parameter [List<float>]: The area's centre [m], rounded to 1e-9 m.
        (2) parameter [List<float>]: Its edge lengths [m], rounded to 1e-9 m.
    """

    low, high = (np.asarray(lower) + 1) * step, (np.asarray(upper) - 1) * step
    if np.any(high - low < step - 1e-12):
        raise Catch_Error_Cls('The largest cuboid %s to %s (multiples of %.3f m) is too small to shrink by one step on every face.'
                              % (np.asarray(lower).tolist(), np.asarray(upper).tolist(), step))
    return [round(float(v), 9) + 0.0 for v in 0.5 * (low + high)], [round(float(v), 9) for v in high - low]


def Flight_Samples(release_state: np.ndarray, floor_time: np.ndarray, *,
                   Flight_Model_Str: Flight_Model_Str) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        The throws' trajectories every CONST_SAMPLE_STEP from their release (Flight.Core.Simulate), and which
        samples count: descending, the ball's centre at or above CONST_FLOOR, before its floor time.

    Args:
        (1) release_state [Matrix<float> Fx6]: The throws' states at the release [m, m/s].
        (2) floor_time [Vector<float> 1xF]: Their floor times [s].
        (3) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Vector<float> 1xT]: The sample times [s].
        (2) parameter [Tensor<float> FxTx6]: The states [m, m/s].
        (3) parameter [Matrix<bool> FxT]: Which samples count.
    """

    times = np.arange(0.0, float(np.max(floor_time)) + CONST_SAMPLE_STEP, CONST_SAMPLE_STEP)
    states = Simulate(np.asarray(release_state, dtype=np.float64), times, Flight_Model_Str=Flight_Model_Str)
    valid = (states[..., 5] < 0.0) & (states[..., 2] >= CONST_FLOOR) & (times[None, :] <= np.asarray(floor_time)[:, None])
    return times, states, valid


def Cell_Visits(positions: np.ndarray, valid: np.ndarray, Workspace_Str: Workspace_Str,
                stride: int = 1) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        The lattice cells the throws pass through: each counting sample is taken to the nearest point of the
        sub-lattice of every `stride`-th lattice point; a sample whose nearest point is not on the lattice
        (out of the robot's reach) is dropped.

    Args:
        (1) positions [Tensor<float> FxTx3]: The samples' positions [m].
        (2) valid [Matrix<bool> FxT]: Which samples count (Flight_Samples).
        (3) Workspace_Str [Workspace_Str(object)]: The map.
        (4) stride [int]: The sub-lattice (1: every point).

    Returns:
        (1) parameter [Vector<int> 1xV]: Each visit's flight.
        (2) parameter [Vector<int> 1xV]: Its lattice row.
        (3) parameter [Vector<int> 1xV]: Its sample.
    """

    w = Workspace_Str
    flights, samples = np.nonzero(valid)
    index = stride * np.round(positions[flights, samples] / (stride * w.Step)).astype(np.int64)
    low = w.Index.min(axis=0)
    shape = w.Index.max(axis=0) - low + 1
    lookup = np.full(shape, -1, dtype=np.int64)
    lookup[tuple((w.Index - low).T)] = np.arange(len(w.Index))
    inside = np.all((index >= low) & (index < low + shape), axis=1)
    rows = np.full(len(index), -1, dtype=np.int64)
    rows[inside] = lookup[tuple((index[inside] - low).T)]
    keep = rows >= 0
    return flights[keep], rows[keep], samples[keep]


def Through_Share(visit_flight: np.ndarray, visit_row: np.ndarray, cells: np.ndarray, flights: int) -> float:
    """
    Description:
        The share of the throws that pass through at least one of the given cells.

    Args:
        (1 - 2) visit_flight, visit_row [Vector<int> 1xV]: The visits (Cell_Visits).
        (3) cells [Vector<bool> 1xN]: The cells, by lattice row.
        (4) flights [int]: The number of throws.

    Returns:
        (1) parameter [float]: The share, 0 to 1.
    """

    return float(len(np.unique(visit_flight[cells[visit_row]]))) / float(flights)


def Box_Share(positions: np.ndarray, valid: np.ndarray, centre: tp.Sequence[float], size: tp.Sequence[float]) -> float:
    """
    Description:
        The share of the throws with a counting sample inside an axis-aligned box (its faces included).

    Args:
        (1) positions [Tensor<float> FxTx3]: The samples' positions [m].
        (2) valid [Matrix<bool> FxT]: Which samples count.
        (3 - 4) centre, size [Vector<float> 1x3]: The box [m].

    Returns:
        (1) parameter [float]: The share, 0 to 1.
    """

    half = 0.5 * np.asarray(size, dtype=np.float64)
    inside = np.all(np.abs(positions - np.asarray(centre, dtype=np.float64)) <= half + 1e-12, axis=-1) & valid
    return float(inside.any(axis=1).mean())


def Reach_Times(Workspace_Str: Workspace_Str, start: np.ndarray, speed: np.ndarray, acceleration: float, wrap: np.ndarray,
                rows: tp.Optional[np.ndarray] = None) -> np.ndarray:
    """
    Description:
        For each qualifying point, the time in which the robot, at rest in `start`, can stand at rest in a
        catch pose there for whichever arrival direction comes: per axis the fastest of its qualifying poses
        (Robot.Core.Move_Time), and the slowest axis decides. Points that do not qualify take infinity.

    Args:
        (1) Workspace_Str [Workspace_Str(object)]: The map.
        (2) start [Vector<float> 1xJ]: The ready pose, joints 1 to CONST_JOINTS_USED [rad].
        (3) speed [Vector<float> 1xJ]: Their speed limits [rad/s].
        (4) acceleration [float]: Their acceleration limit [rad/s^2].
        (5) wrap [Vector<bool> 1xJ]: Which of them wrap.
        (6) rows [Vector<int> 1xK]: Only these positions of Workspace_Str.Rows (None: all).

    Returns:
        (1) parameter [Vector<float> 1xN]: The time per lattice row [s].
    """

    w = Workspace_Str
    chosen = np.arange(len(w.Rows)) if rows is None else np.asarray(rows)
    times = Move_Time(np.asarray(start, dtype=np.float32), w.Poses[chosen], np.asarray(speed, dtype=np.float32), acceleration, wrap)
    times = np.where(np.isnan(times), np.inf, times)
    result = np.full(len(w.Points), np.inf)
    result[w.Rows[chosen]] = times.min(axis=2).max(axis=1)
    return result


def Catchable_Share(visit_flight: np.ndarray, visit_row: np.ndarray, visit_sample: np.ndarray, sample_times: np.ndarray,
                    decision: np.ndarray, reach: np.ndarray, allowed: np.ndarray, flights: int) -> float:
    """
    Description:
        The share of the throws with a catchable sample: a visit of an allowed cell whose reach time is at most
        the time left, the sample's time minus the decision's (the throw's CONST_DECISION_SAMPLE-th observed
        sample) minus CONST_ALLOWANCE. A throw without its decision sample is not catchable.

    Args:
        (1 - 3) visit_flight, visit_row, visit_sample [Vector<int> 1xV]: The visits (Cell_Visits).
        (4) sample_times [Vector<float> 1xT]: The samples' times [s].
        (5) decision [Vector<float> 1xF]: Each throw's decision time [s], NaN without one.
        (6) reach [Vector<float> 1xN]: Each cell's reach time [s] (Reach_Times).
        (7) allowed [Vector<bool> 1xN]: The cells a catch may use.
        (8) flights [int]: The number of throws.

    Returns:
        (1) parameter [float]: The share, 0 to 1.
    """

    left = sample_times[visit_sample] - decision[visit_flight] - CONST_ALLOWANCE
    with np.errstate(invalid='ignore'):
        good = allowed[visit_row] & (reach[visit_row] <= left)
    return float(len(np.unique(visit_flight[good]))) / float(flights)


def Reached_Cuboid(Workspace_Str: Workspace_Str, start: np.ndarray, speed: np.ndarray, acceleration: float, wrap: np.ndarray,
                   stride: int = 1) -> tp.Tuple[float, tp.Optional[np.ndarray], tp.Optional[np.ndarray]]:
    """
    Description:
        The catch area's cuboid from one ready pose (the rule B800): on the sub-lattice of every
        `stride`-th lattice point, the largest cuboid (Catch.Core.Largest_Cuboid, the most points) of the
        qualifying points reached within CONST_REACH_TIME (Reach_Times), and its volume after the shrink by
        one lattice step on every face (Configured_Cuboid); a cuboid too thin to shrink has the volume 0. With
        a stride above 1 the volume estimates the whole lattice's (the search's scoring).

    Args:
        (1) Workspace_Str [Workspace_Str(object)]: The map.
        (2 - 5) start, speed, acceleration, wrap: As Reach_Times.
        (6) stride [int]: The sub-lattice (1: every point).

    Returns:
        (1) parameter [float]: The shrunk cuboid's volume [m^3].
        (2) parameter [Vector<int> 1x3]: The cuboid's lower corner in multiples of the step, None when no point is reached.
        (3) parameter [Vector<int> 1x3]: Its upper corner, inclusive, None when no point is reached.
    """

    w = Workspace_Str
    rows = np.flatnonzero(np.all(w.Index[w.Rows] % stride == 0, axis=1))
    within = Reach_Times(w, start, speed, acceleration, wrap, rows=rows) <= CONST_REACH_TIME
    if not within.any():
        return 0.0, None, None
    lower, upper = Largest_Cuboid(w.Index[within] // stride, np.ones(int(within.sum()), dtype=bool))
    lower, upper = lower * stride, upper * stride
    return float(np.prod(np.maximum(upper - lower - 2, 0))) * w.Step ** 3, lower, upper


def Search_Catch_Area(Workspace_Str: Workspace_Str, speed: np.ndarray, acceleration: float, wrap: np.ndarray,
                      axis: int) -> tp.Tuple[np.ndarray, int, np.ndarray, np.ndarray, float]:
    """
    Description:
        The ready pose and the catch area chosen together (the rule B800):
        among the qualifying poses of one arrival axis at the qualifying points, the ready pose whose
        Reached_Cuboid has the largest volume. A heuristic search, not an exhaustive one, in three stages:
        the candidates on every CONST_SEARCH_COARSE-th lattice point, scored on the sub-lattice of every
        CONST_SEARCH_SCORE-th point; the CONST_SEARCH_KEEP best with every candidate on that sub-lattice within
        CONST_SEARCH_REFINE lattice steps of them, scored the same way; the CONST_SEARCH_FINAL best of these
        scored on the whole lattice. Each stage ranks by the score, then by the position in the map's rows
        and the branch, the later first; the last stage keeps the first strictly best. The result depends on
        the map and the limits only.

    Args:
        (1) Workspace_Str [Workspace_Str(object)]: The map.
        (2 - 4) speed, acceleration, wrap: As Reach_Times.
        (5) axis [int]: The arrival axis whose poses are the candidates (the cone's centre).

    Returns:
        (1) parameter [Vector<float> 1xJ]: The ready pose, joints 1 to CONST_JOINTS_USED [rad].
        (2) parameter [int]: Its lattice row.
        (3) parameter [Vector<int> 1x3]: The cuboid's lower corner in multiples of the step.
        (4) parameter [Vector<int> 1x3]: Its upper corner, inclusive.
        (5) parameter [float]: The shrunk cuboid's volume [m^3].
    """

    w = Workspace_Str
    limits = (speed, acceleration, wrap)
    coarse = _Candidates(w, np.flatnonzero(np.all(w.Index[w.Rows] % CONST_SEARCH_COARSE == 0, axis=1)), axis)
    if not coarse:
        raise Catch_Error_Cls('No qualifying pose of the arrival axis %d stands on the search\'s coarse lattice.' % axis)
    kept = _Ranked(w, coarse, axis, limits)[:CONST_SEARCH_KEEP]
    scored = np.all(w.Index[w.Rows] % CONST_SEARCH_SCORE == 0, axis=1)
    pool = {(p, s) for _, p, s in kept}
    for _, p, _ in kept:
        near = np.max(np.abs(w.Index[w.Rows] - w.Index[w.Rows[p]]), axis=1) <= CONST_SEARCH_REFINE
        pool.update(_Candidates(w, np.flatnonzero(near & scored), axis))
    best: tp.Tuple[float, int, int, tp.Any, tp.Any] = (-1.0, -1, -1, None, None)
    for _, p, s in _Ranked(w, sorted(pool), axis, limits)[:CONST_SEARCH_FINAL]:
        volume, lower, upper = Reached_Cuboid(w, w.Poses[p, axis, s], *limits)
        if volume > best[0]:
            best = (volume, p, s, lower, upper)
    volume, p, s, lower, upper = best
    if lower is None:
        raise Catch_Error_Cls('From none of the search\'s best ready poses is a qualifying point reached within %.2f s.' % CONST_REACH_TIME)
    return w.Poses[p, axis, s].astype(np.float64), int(w.Rows[p]), lower, upper, volume


def _Candidates(w: Workspace_Str, positions: np.ndarray, axis: int) -> tp.List[tp.Tuple[int, int]]:
    """
    Description:
        The ready-pose candidates at the given positions of the map's rows: each qualifying pose (branch) of
        the arrival axis, in the order of the positions, then of the branches.

    Args:
        (1) w [Workspace_Str(object)]: The map.
        (2) positions [Vector<int> 1xK]: The positions in w.Rows.
        (3) axis [int]: The arrival axis.

    Returns:
        (1) parameter [List<Tuple<int, int>>]: The positions and branches.
    """

    return [(int(p), s) for p in positions for s in range(w.Poses.shape[2]) if np.all(np.isfinite(w.Poses[p, axis, s]))]


def _Ranked(w: Workspace_Str, candidates: tp.Iterable[tp.Tuple[int, int]], axis: int,
            limits: tp.Tuple[np.ndarray, float, np.ndarray]) -> tp.List[tp.Tuple[float, int, int]]:
    """
    Description:
        The candidates scored by Reached_Cuboid on the sub-lattice of every CONST_SEARCH_SCORE-th point and
        ranked by the score, then by the position and the branch, the larger first.

    Args:
        (1) w [Workspace_Str(object)]: The map.
        (2) candidates [Iterable<Tuple<int, int>>]: The positions in w.Rows and the branches.
        (3) axis [int]: The arrival axis.
        (4) limits [Tuple]: The joints' speed limits [rad/s], the acceleration limit [rad/s^2], the wrap.

    Returns:
        (1) parameter [List<Tuple<float, int, int>>]: The scores [m^3], positions and branches.
    """

    return sorted(((Reached_Cuboid(w, w.Poses[p, axis, s], *limits, stride=CONST_SEARCH_SCORE)[0], p, s) for p, s in candidates),
                  reverse=True)


def Write_Workspace_Map(path: str, Workspace_Str: Workspace_Str, extra: tp.Dict[str, np.ndarray]) -> None:
    """
    Description:
        Write the workspace map (SPEC §9.11): np.savez, uncompressed, its format version, the map's arrays
        and the derived values given (the cuboid, the ready pose). The bytes depend on the arrays only.

    Args:
        (1) path [string]: The file.
        (2) Workspace_Str [Workspace_Str(object)]: The map.
        (3) extra [Dict<string, np.ndarray>]: Further named arrays.
    """

    w = Workspace_Str
    np.savez(path, format_version=np.array(CONST_MAP_VERSION), step=np.array(w.Step), points=w.Points, index=w.Index, axes=w.Axes,
             qualifies=w.Qualifies, failed_axis=w.Failed_Axis, failed_criterion=w.Failed_Criterion, criteria=np.array(CONST_CRITERIA),
             rows=w.Rows, poses=w.Poses, **extra)


def Write_Workspace_Summary(path: str, rows: tp.List[tp.Tuple[str, str, str]]) -> None:
    """
    Description:
        Write the workspace summary (SPEC §9.11): a CSV with the columns quantity, value, unit, one row per
        reported quantity, in the order given.

    Args:
        (1) path [string]: The file.
        (2) rows [List<Tuple<string>>]: The rows.
    """

    with open(path, 'w', newline='', encoding='ascii') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(CONST_SUMMARY_COLUMNS)
        writer.writerows(rows)


def Joint_Wrap(Catch_Context_Str: Catch_Context_Str) -> np.ndarray:
    """
    Description:
        Which of the joints a catch pose uses wrap (Robot.Core.Wrapping_Joints, joints 1 to CONST_JOINTS_USED).

    Args:
        (1) Catch_Context_Str [Catch_Context_Str(object)]: The context.

    Returns:
        (1) parameter [Vector<bool> 1xJ]: True for a wrapping joint.
    """

    return Wrapping_Joints(Robot_Str=Catch_Context_Str.Robot)[:CONST_JOINTS_USED]
