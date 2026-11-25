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
# File Name: ../Catch/Decision.py
## =========================================================================== ##

# CSV (The decision's files)
import csv
# Math (The binomial tail of the intervals)
import math
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Flight/Core - the throws' trajectories
from Flight.Core import Flight_Model_Str
#   ../Robot/Core - the move time
from Robot.Core import Move_Time
#   ../Catch/Core - the catch pose
from Catch.Core import Catch_Context_Str, Catch_Poses, Check_Pose, Catch_Error_Cls, Read_Table, CONST_JOINTS_USED
#   ../Catch/Workspace - the map, the reach, the throws' samples and the time a catch needs before the ball
from Catch.Workspace import Workspace_Str, Flight_Samples, Reach_Times, CONST_SAMPLE_STEP, CONST_FLOOR, CONST_ALLOWANCE

"""
Description:
    Initialization of constants.

    The catch decision (SPEC §2.8, §7.11; F9): the truth, whether a simulated throw is catchable
    from a decision made at its N-th observed sample, and the decision, go or no-go from the predicted trajectory of
    the prefix, conservative by margins calibrated on simulated validation flights only. The truth: a 10 ms sample
    of the true trajectory inside the truth area (Truth_Area: Areas.Catch grown by one lattice step, the unshrunk
    qualifying cuboid of B800; its faces included), descending, with a catch pose for that sample's own arrival
    direction that meets the catch criteria and is reached from the ready pose, rest to rest, within the sample's
    time minus the decision time minus CONST_ALLOWANCE (F9 Q1, amended 2026-11-21). The decision: go when a predicted 10 ms
    sample after the decision time lies inside Areas.Catch shrunk per axis by the cross-track margin, and the largest
    reach time of the area's lattice cells within that margin of it is at most the time left minus
    CONST_ALLOWANCE minus the time margin (F9 Q6). The rates with Clopper-Pearson intervals, and the decision's
    three files (SPEC §9.12). No file but its own is read or written here, and nothing is printed.
"""
# The horizon bins of the margins: the time from the decision to a predicted sample, in steps of
# CONST_HORIZON_BIN, the last bin holding every longer horizon (40 bins: 2.0 s, beyond the longest simulated
# flight, 1.52 s). A bin with fewer than CONST_MINIMUM_SAMPLES errors has no margin: no go is made there
# (when in doubt, no move).
#   Unit [s], [int], [int]
CONST_HORIZON_BIN = 0.05
CONST_HORIZON_BINS = 40
CONST_MINIMUM_SAMPLES = 50
# The intervals' confidence and the bisections that find their bounds (2^-60 of the unit interval).
#   Unit [-], [int]
CONST_CONFIDENCE = 0.95
CONST_INTERVAL_BISECTIONS = 60
# The columns of the decision's files (SPEC §9.12).
#   Unit [Tuple<string>]
CONST_MARGIN_COLUMNS = ('seed', 'n', 'bin', 'horizon_low_s', 'samples', 'cross_x_m', 'cross_y_m', 'cross_z_m', 'time_s')
CONST_PER_THROW_COLUMNS = ('dataset', 'flight', 'seed', 'acceleration_deg_s2', 'catchable', 'decision', 'go_stage', 'outcome', 'refused')
CONST_SUMMARY_COLUMNS = ('dataset', 'partition', 'seed', 'acceleration_deg_s2', 'throws', 'catchable', 'not_catchable', 'go', 'false_go',
                         'false_go_rate', 'false_go_lower', 'false_go_upper', 'false_no_go', 'false_no_go_rate', 'false_no_go_lower',
                         'false_no_go_upper', 'false_go_of_go', 'refused', 'go_by_stage')


def Area_Lattice(centre: tp.Sequence[float], size: tp.Sequence[float], step: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        The lattice points (every multiple of the step) inside an axis-aligned box, its faces included within 1e-9 m.

    Args:
        (1) centre [Vector<float> 1x3]: The box's centre [m].
        (2) size [Vector<float> 1x3]: Its edge lengths [m].
        (3) step [float]: The lattice's step [m].

    Returns:
        (1) parameter [Matrix<float> Nx3]: The points [m].
        (2) parameter [Matrix<int> Nx3]: Their whole multiples of the step.
    """

    low = np.ceil((np.asarray(centre) - 0.5 * np.asarray(size) - 1e-9) / step).astype(np.int64)
    high = np.floor((np.asarray(centre) + 0.5 * np.asarray(size) + 1e-9) / step).astype(np.int64)
    if np.any(high < low):
        raise Catch_Error_Cls('The box at %s m of size %s m holds no lattice point of step %.3f m.' % (list(centre), list(size), step))
    axes = [np.arange(a, b + 1) for a, b in zip(low, high)]
    index = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    return index * step, index


def Reach_Grid(Workspace_Str: Workspace_Str, start: np.ndarray, speed: np.ndarray, acceleration: float,
               wrap: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        The reach times of a map's lattice points (Catch.Workspace.Reach_Times: the slowest arrival axis of its fastest
        qualifying pose, rest to rest from the start) on a grid over the map's index box; a point that does not qualify
        takes infinity.

    Args:
        (1) Workspace_Str [Workspace_Str(object)]: The map (of the catch area's lattice points).
        (2 - 5) start, speed, acceleration, wrap: As Catch.Workspace.Reach_Times.

    Returns:
        (1) parameter [Array<float> XxYxZ]: The reach time per cell [s].
        (2) parameter [Vector<int> 1x3]: The index of the grid's first cell.
    """

    w = Workspace_Str
    low = w.Index.min(axis=0)
    grid = np.full(w.Index.max(axis=0) - low + 1, np.inf)
    grid[tuple((w.Index - low).T)] = Reach_Times(w, start, speed, acceleration, wrap)
    return grid, low


def Reach_Bound(grid: np.ndarray, steps: tp.Sequence[int]) -> np.ndarray:
    """
    Description:
        The conservative reach: per cell the largest reach time of the cells within `steps` cells along each axis, a
        cell beyond the grid counting as infinity (outside the catch area no catch is allowed).

    Args:
        (1) grid [Array<float> XxYxZ]: The reach times (Reach_Grid) [s].
        (2) steps [Vector<int> 1x3]: The cells per axis, each >= 0.

    Returns:
        (1) parameter [Array<float> XxYxZ]: The bound [s].
    """

    bound = np.asarray(grid, dtype=np.float64).copy()
    for axis, count in enumerate(steps):
        base = bound.copy()
        for d in range(1, int(count) + 1):
            for sign in (1, -1):
                shifted = np.full_like(base, np.inf)
                source, target = [slice(None)] * 3, [slice(None)] * 3
                if d < base.shape[axis]:
                    source[axis], target[axis] = (slice(d, None), slice(None, -d)) if sign > 0 else (slice(None, -d), slice(d, None))
                    shifted[tuple(target)] = base[tuple(source)]
                bound = np.maximum(bound, shifted)
    return bound


def Truth_Area(centre: tp.Sequence[float], size: tp.Sequence[float], step: float) -> tp.Tuple[tp.List[float], tp.List[float]]:
    """
    Description:
        The area the truth credits a catch in (amended on validation evidence): Areas.Catch grown
        by one lattice step on every face, the unshrunk qualifying cuboid of B800 (Catch.Workspace.Configured_Cuboid
        shrinks it by that step). A sample in the shell between the two is checked exactly like one inside the area
        (Exact_Reach); the decision's region stays Areas.Catch shrunk by the margins, a guard band inside the truth.

    Args:
        (1) centre [Vector<float> 1x3]: Areas.Catch's centre [m].
        (2) size [Vector<float> 1x3]: Its edge lengths [m].
        (3) step [float]: The lattice's step, Catch.Workspace_Step [m].

    Returns:
        (1) parameter [List<float>]: The truth area's centre [m].
        (2) parameter [List<float>]: Its edge lengths [m].
    """

    return [float(v) for v in centre], [float(v) + 2.0 * step for v in size]


def Truth_Samples(release_state: np.ndarray, floor_time: np.ndarray, centre: tp.Sequence[float], size: tp.Sequence[float], *,
                  Flight_Model_Str: Flight_Model_Str) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        The true trajectories' counting samples inside a box (the truth area, Truth_Area): every CONST_SAMPLE_STEP from the release
        (Catch.Workspace.Flight_Samples: descending, at or above the floor, before the floor time), inside the box with
        its faces (within 1e-12 m).

    Args:
        (1) release_state [Matrix<float> Fx6]: The throws' release states [m, m/s].
        (2) floor_time [Vector<float> 1xF]: Their floor times [s].
        (3 - 4) centre, size [Vector<float> 1x3]: The box (Truth_Area) [m].
        (5) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Vector<int> 1xS]: Each sample's throw (row of release_state).
        (2) parameter [Vector<float> 1xS]: Its time from the release [s].
        (3) parameter [Matrix<float> Sx3]: Its position [m].
        (4) parameter [Matrix<float> Sx3]: Its velocity [m/s].
    """

    times, states, valid = Flight_Samples(release_state, floor_time, Flight_Model_Str=Flight_Model_Str)
    half = 0.5 * np.asarray(size, dtype=np.float64)
    inside = valid & np.all(np.abs(states[..., :3] - np.asarray(centre, dtype=np.float64)) <= half + 1e-12, axis=-1)
    flight, sample = np.nonzero(inside)
    return flight, times[sample], states[flight, sample, :3], states[flight, sample, 3:]


def Exact_Reach(points: np.ndarray, velocities: np.ndarray, start: np.ndarray, speed: np.ndarray, accelerations: tp.Sequence[float],
                wrap: np.ndarray, *, Catch_Context_Str: Catch_Context_Str) -> np.ndarray:
    """
    Description:
        Per sample, the time in which the robot, at rest in `start`, stands at rest in the fastest catch pose for the
        sample's own arrival direction (the cup's axis against its velocity): every branch of the closed form checked
        against the criteria (Catch.Core.Check_Pose), the fastest qualifying one (Robot.Core.Move_Time); infinity when no
        branch qualifies.

    Args:
        (1) points [Matrix<float> Sx3]: The samples' positions [m].
        (2) velocities [Matrix<float> Sx3]: Their velocities [m/s].
        (3) start [Vector<float> 1xJ]: The ready pose, joints 1 to CONST_JOINTS_USED [rad].
        (4) speed [Vector<float> 1xJ]: Their speed limits [rad/s].
        (5) accelerations [Vector<float> 1xA]: The acceleration limits [rad/s^2].
        (6) wrap [Vector<bool> 1xJ]: Which joints wrap.
        (7) Catch_Context_Str [Catch_Context_Str(object)]: REQUIRED, keyword-only: the context.

    Returns:
        (1) parameter [Matrix<float> SxA]: The reach time per sample and acceleration [s].
    """

    reach = np.full((len(points), len(accelerations)), np.inf)
    for row, (point, velocity) in enumerate(zip(np.asarray(points, dtype=np.float64), np.asarray(velocities, dtype=np.float64))):
        axis = -velocity / np.linalg.norm(velocity)
        poses = Catch_Poses(point, axis[None], Catch_Context_Str=Catch_Context_Str)[0]
        good = [q[:CONST_JOINTS_USED] for q in poses if np.all(np.isfinite(q)) and Check_Pose(q, Catch_Context_Str=Catch_Context_Str).Failed == '']
        if good:
            for column, acceleration in enumerate(accelerations):
                reach[row, column] = float(np.min(Move_Time(start, np.array(good), speed, acceleration, wrap)))
    return reach


def Catchable(sample_flight: np.ndarray, sample_time: np.ndarray, reach: np.ndarray, decision_time: np.ndarray) -> np.ndarray:
    """
    Description:
        The truth per throw: catchable when one of its samples inside the truth area is reached in
        the time left, reach <= the sample's time - the decision time - CONST_ALLOWANCE. A throw without its decision
        time (fewer observed samples than the stage's N, NaN) is not catchable at that stage.

    Args:
        (1) sample_flight [Vector<int> 1xS]: Each sample's throw (Truth_Samples).
        (2) sample_time [Vector<float> 1xS]: Its time [s].
        (3) reach [Vector<float> 1xS]: Its reach time at one acceleration (Exact_Reach) [s].
        (4) decision_time [Vector<float> 1xF]: Each throw's decision time [s], NaN without one.

    Returns:
        (1) parameter [Vector<bool> 1xF]: Whether each throw is catchable.
    """

    out = np.zeros(len(decision_time), dtype=bool)
    with np.errstate(invalid='ignore'):
        good = np.asarray(reach) <= np.asarray(sample_time) - np.asarray(decision_time)[sample_flight] - CONST_ALLOWANCE
    out[np.unique(np.asarray(sample_flight)[good])] = True
    return out


def Horizon_Bins(h: np.ndarray) -> np.ndarray:
    """
    Description:
        The horizon bin of each time after the decision: floor(h / CONST_HORIZON_BIN + 1e-9) (a horizon on a bin's edge,
        such as 0.3 s, in the bin it starts), the last bin holding every longer horizon.

    Args:
        (1) h [Vector<float> 1xK]: The horizons, each > 0 [s].

    Returns:
        (1) parameter [Vector<int> 1xK]: The bins.
    """

    return np.minimum(np.floor(np.asarray(h) / CONST_HORIZON_BIN + 1e-9).astype(np.int64), CONST_HORIZON_BINS - 1)


def Margin_Errors(predicted: np.ndarray, truth: np.ndarray, h: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        The errors the margins are calibrated on, at the times where the truth is in flight
        and descending: the error e = predicted - true position split along the true velocity's direction u; the
        cross-track error e - (e . u) u per axis (absolute) and the time the ball is early at the predicted point,
        max(0, -(e . u) / |v|).

    Args:
        (1) predicted [Matrix<float> Kx6]: The predicted states [m, m/s].
        (2) truth [Matrix<float> Kx6]: The true states at the same times [m, m/s].
        (3) h [Vector<float> 1xK]: The times after the decision [s].

    Returns:
        (1) parameter [Vector<int> 1xM]: The horizon bins of the times used.
        (2) parameter [Matrix<float> Mx3]: The absolute cross-track errors [m].
        (3) parameter [Vector<float> 1xM]: The early times [s].
    """

    truth = np.asarray(truth, dtype=np.float64)
    keep = (np.asarray(h) > 0.0) & (truth[:, 5] < 0.0) & (truth[:, 2] >= CONST_FLOOR) & np.all(np.isfinite(predicted), axis=1)
    e = np.asarray(predicted, dtype=np.float64)[keep, :3] - truth[keep, :3]
    v = truth[keep, 3:]
    speed = np.linalg.norm(v, axis=1)
    u = v / speed[:, None]
    along = np.sum(e * u, axis=1)
    return Horizon_Bins(np.asarray(h)[keep]), np.abs(e - along[:, None] * u), np.maximum(0.0, -along / speed)


def Margins(bins: np.ndarray, cross: np.ndarray, early: np.ndarray, quantile: float) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        Per horizon bin the quantile of the cross-track error per axis and of the early time; a bin with fewer than
        CONST_MINIMUM_SAMPLES errors takes infinity (no go there).

    Args:
        (1) bins [Vector<int> 1xM]: The errors' bins (Margin_Errors).
        (2) cross [Matrix<float> Mx3]: The absolute cross-track errors [m].
        (3) early [Vector<float> 1xM]: The early times [s].
        (4) quantile [float]: The quantile, in (0, 1).

    Returns:
        (1) parameter [Matrix<float> Bx3]: The cross-track margin per bin and axis [m].
        (2) parameter [Vector<float> 1xB]: The time margin per bin [s].
        (3) parameter [Vector<int> 1xB]: The errors per bin.
    """

    cross_margin, time_margin = np.full((CONST_HORIZON_BINS, 3), np.inf), np.full(CONST_HORIZON_BINS, np.inf)
    counts = np.bincount(np.asarray(bins, dtype=np.int64), minlength=CONST_HORIZON_BINS)
    for b in np.flatnonzero(counts >= CONST_MINIMUM_SAMPLES):
        select = bins == b
        cross_margin[b] = np.quantile(cross[select], quantile, axis=0)
        time_margin[b] = np.quantile(early[select], quantile)
    return cross_margin, time_margin, counts


def Candidates(times: np.ndarray, states: np.ndarray, t_0: float, centre: tp.Sequence[float],
               size: tp.Sequence[float]) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        The predicted samples that may make a go: after the decision time, descending, at or above the floor, inside the
        unshrunk area with its faces.

    Args:
        (1) times [Vector<float> 1xK]: The predicted samples' times [s].
        (2) states [Matrix<float> Kx6]: Their predicted states [m, m/s].
        (3) t_0 [float]: The decision time [s].
        (4 - 5) centre, size [Vector<float> 1x3]: Areas.Catch [m].

    Returns:
        (1) parameter [Vector<float> 1xC]: Their horizons, the time after t_0 [s].
        (2) parameter [Matrix<float> Cx3]: Their positions [m].
    """

    states = np.asarray(states, dtype=np.float64)
    h = np.asarray(times, dtype=np.float64) - t_0
    with np.errstate(invalid='ignore'):
        keep = (h > 0.0) & (states[:, 5] < 0.0) & (states[:, 2] >= CONST_FLOOR) \
            & np.all(np.abs(states[:, :3] - np.asarray(centre)) <= 0.5 * np.asarray(size) + 1e-12, axis=1)
    return h[keep], states[keep, :3]


def Decide(rows: np.ndarray, h: np.ndarray, position: np.ndarray, flights: int, cross_margin: np.ndarray, time_margin: np.ndarray,
           grid: np.ndarray, low: np.ndarray, step: float, centre: tp.Sequence[float], size: tp.Sequence[float],
           bounds: tp.Optional[tp.Dict[tp.Tuple[int, int, int], np.ndarray]] = None) -> np.ndarray:
    """
    Description:
        The decision of one stage: go for a throw when one of its candidates (Candidates)
        lies inside Areas.Catch shrunk on each axis by the cross-track margin of its horizon bin, and the conservative
        reach (Reach_Bound over the cells within that margin, rounded up to whole cells) at its nearest cell is at most
        its horizon - CONST_ALLOWANCE - the bin's time margin. A throw without a candidate (a refused prediction, too few
        samples) is no-go.

    Args:
        (1) rows [Vector<int> 1xC]: Each candidate's throw.
        (2) h [Vector<float> 1xC]: Its horizon [s].
        (3) position [Matrix<float> Cx3]: Its predicted position [m].
        (4) flights [int]: The number of throws.
        (5) cross_margin [Matrix<float> Bx3]: The cross-track margins (Margins) [m].
        (6) time_margin [Vector<float> 1xB]: The time margins [s].
        (7) grid [Array<float> XxYxZ]: The reach times (Reach_Grid) [s].
        (8) low [Vector<int> 1x3]: The index of the grid's first cell.
        (9) step [float]: The lattice's step [m].
        (10 - 11) centre, size [Vector<float> 1x3]: Areas.Catch [m].
        (12) bounds [Dict]: A cache of Reach_Bound by the steps (filled here), or None.

    Returns:
        (1) parameter [Vector<bool> 1xF]: Go per throw.
    """

    go = np.zeros(flights, dtype=bool)
    if len(rows) == 0:
        return go
    cache = {} if bounds is None else bounds
    b = Horizon_Bins(h)
    m, tm = cross_margin[b], time_margin[b]
    with np.errstate(invalid='ignore'):
        inside = np.all(np.abs(np.asarray(position) - np.asarray(centre)) <= 0.5 * np.asarray(size) - m, axis=1)
    cell = np.round(np.asarray(position) / step).astype(np.int64) - low
    inside &= np.all((cell >= 0) & (cell < np.array(grid.shape)), axis=1)
    good = np.zeros(len(rows), dtype=bool)
    candidates = np.flatnonzero(inside)
    steps = np.ceil(m[candidates] / step - 1e-9).astype(np.int64)
    for key in {tuple(int(v) for v in s) for s in steps}:
        select = candidates[np.all(steps == key, axis=1)]
        if key not in cache:
            cache[key] = Reach_Bound(grid, key)
        good[select] = cache[key][tuple(cell[select].T)] <= h[select] - CONST_ALLOWANCE - tm[select]
    go[np.unique(np.asarray(rows)[good])] = True
    return go


def Stage_Rule(go: tp.List[np.ndarray], catchable: tp.List[np.ndarray]) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        The stages in their order: a throw goes at the first stage that says go; the truth
        that judges it is its catchability at that stage, and for a no-go its catchability at the first stage (the
        most time left).

    Args:
        (1) go [List<Vector<bool> 1xF>]: Go per stage (Decide).
        (2) catchable [List<Vector<bool> 1xF>]: The truth per stage (Catchable).

    Returns:
        (1) parameter [Vector<bool> 1xF]: The decision, go or not.
        (2) parameter [Vector<int> 1xF]: The stage of the go (its index), -1 for a no-go.
        (3) parameter [Vector<bool> 1xF]: The truth that judges each decision.
    """

    stage = np.full(len(go[0]), -1, dtype=np.int64)
    for k in range(len(go) - 1, -1, -1):
        stage[go[k]] = k
    truth = np.array(catchable[0], dtype=bool)
    for k in range(1, len(go)):
        truth[stage == k] = catchable[k][stage == k]
    return stage >= 0, stage, truth


def Clopper_Pearson(k: int, n: int, confidence: float = CONST_CONFIDENCE) -> tp.Tuple[float, float]:
    """
    Description:
        The Clopper-Pearson interval of a binomial share k / n: the lower bound p with
        P(X >= k | p) = (1 - confidence) / 2 (0 for k = 0), the upper with P(X <= k | p) = (1 - confidence) / 2 (1 for
        k = n), each by bisection on the exact binomial tail.

    Args:
        (1) k [int]: The events, 0 <= k <= n.
        (2) n [int]: The trials, >= 1.
        (3) confidence [float]: The interval's confidence.

    Returns:
        (1) parameter [float]: The lower bound.
        (2) parameter [float]: The upper bound.
    """

    if n < 1 or not 0 <= k <= n:
        raise Catch_Error_Cls('A share needs 0 <= k <= n and n >= 1; found k %d, n %d.' % (k, n))
    alpha = 0.5 * (1.0 - confidence)

    def Cdf(j: int, p: float) -> float:
        """
        Description:
            P(X <= j) for X ~ Binomial(n, p), 0 < p < 1.

        Args:
            (1) j [int]: The count.
            (2) p [float]: The probability.

        Returns:
            (1) parameter [float]: The probability.
        """

        terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)
                 for i in range(j + 1)]
        top = max(terms)
        return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))

    def Solve(below: tp.Callable[[float], bool]) -> float:
        """
        Description:
            The point of (0, 1) where `below` turns from True to False.

        Args:
            (1) below [Callable[[float], bool]]: True below the root.

        Returns:
            (1) parameter [float]: The root.
        """

        lo, hi = 0.0, 1.0
        for _ in range(CONST_INTERVAL_BISECTIONS):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if below(mid) else (lo, mid)
        return 0.5 * (lo + hi)

    lower = 0.0 if k == 0 else Solve(lambda p: 1.0 - Cdf(k - 1, p) < alpha)
    upper = 1.0 if k == n else Solve(lambda p: Cdf(k, p) > alpha)
    return lower, upper


def Rates(go: np.ndarray, truth: np.ndarray) -> tp.Dict[str, tp.Any]:
    """
    Description:
        The decision's rates: false-go = go among the throws not catchable, false-no-go =
        no-go among the catchable ones, each with its Clopper-Pearson interval (NaN without a throw of its kind), and the
        false-go share among the go decisions.

    Args:
        (1) go [Vector<bool> 1xF]: The decisions.
        (2) truth [Vector<bool> 1xF]: The truth that judges them (Stage_Rule).

    Returns:
        (1) parameter [Dict<string, Any>]: The counts, the rates and the bounds.
    """

    go, truth = np.asarray(go, dtype=bool), np.asarray(truth, dtype=bool)
    out: tp.Dict[str, tp.Any] = {'throws': int(len(go)), 'catchable': int(truth.sum()), 'not_catchable': int((~truth).sum()), 'go': int(go.sum()),
                                 'false_go': int(np.sum(go & ~truth)), 'false_no_go': int(np.sum(~go & truth))}
    for name, total in (('false_go', out['not_catchable']), ('false_no_go', out['catchable'])):
        lower, upper = Clopper_Pearson(out[name], total) if total else (float('nan'), float('nan'))
        out[name + '_rate'] = out[name] / total if total else float('nan')
        out[name + '_lower'], out[name + '_upper'] = lower, upper
    out['false_go_of_go'] = out['false_go'] / out['go'] if out['go'] else float('nan')
    return out


def _Write_Rows(path: str, columns: tp.Sequence[str], rows: tp.Iterable[tp.Sequence[tp.Any]]) -> None:
    """
    Description:
        Write a CSV with its header, floats by repr (17 significant digits), one row per sequence.

    Args:
        (1) path [string]: The file.
        (2) columns [Sequence<string>]: The header.
        (3) rows [Iterable<Sequence<Any>>]: The rows.
    """

    with open(path, 'w', newline='', encoding='ascii') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(columns)
        for row in rows:
            writer.writerow([repr(float(v)) if isinstance(v, (float, np.floating)) else v for v in row])


def Write_Margins(path: str, rows: tp.Iterable[tp.Sequence[tp.Any]]) -> None:
    """
    Description:
        Write margins.csv (SPEC §9.12): per training seed (empty for a numerical predictor), stage N and horizon bin, the
        bin's lower horizon, its errors, the cross-track margin per axis and the time margin ('inf' without a margin).

    Args:
        (1) path [string]: The file.
        (2) rows [Iterable<Sequence<Any>>]: The rows, in the order of CONST_MARGIN_COLUMNS.
    """

    _Write_Rows(path, CONST_MARGIN_COLUMNS, rows)


def Read_Margins(path: str) -> tp.Dict[tp.Tuple[str, int], tp.Tuple[np.ndarray, np.ndarray]]:
    """
    Description:
        Read margins.csv back (the test partition decides with the margins of the validation partition).

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [Dict<Tuple<string, int>, Tuple>]: Per (seed, N) the cross-track margins Bx3 [m] and the time
                                                         margins 1xB [s].
    """

    out: tp.Dict[tp.Tuple[str, int], tp.Tuple[np.ndarray, np.ndarray]] = {}
    with open(path, 'r', encoding='ascii', newline='') as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != CONST_MARGIN_COLUMNS:
            raise Catch_Error_Cls('%s does not hold the columns %s; it was not written by decide (SPEC §9.12).'
                                  % (path, ', '.join(CONST_MARGIN_COLUMNS)))
        for row in reader:
            key = (row['seed'], int(row['n']))
            cross, time = out.setdefault(key, (np.full((CONST_HORIZON_BINS, 3), np.inf), np.full(CONST_HORIZON_BINS, np.inf)))
            b = int(row['bin'])
            cross[b] = [float(row['cross_%s_m' % axis]) for axis in 'xyz']
            time[b] = float(row['time_s'])
    return out


def Write_Per_Throw(path: str, rows: tp.Iterable[tp.Sequence[tp.Any]]) -> None:
    """
    Description:
        Write per_throw.csv (SPEC §9.12): per dataset, throw, seed and acceleration the truth that judges the decision,
        the decision, the stage of a go, the outcome (true_go, false_go, true_no_go, false_no_go) and the stages whose
        prediction was refused.

    Args:
        (1) path [string]: The file.
        (2) rows [Iterable<Sequence<Any>>]: The rows, in the order of CONST_PER_THROW_COLUMNS.
    """

    _Write_Rows(path, CONST_PER_THROW_COLUMNS, rows)


def Write_Summary(path: str, rows: tp.Iterable[tp.Sequence[tp.Any]]) -> None:
    """
    Description:
        Write summary.csv (SPEC §9.12): per dataset, partition, seed and acceleration the counts, the false-go and
        false-no-go rates with their Clopper-Pearson bounds, the false-go share among the go decisions, the refused
        predictions and the go decisions per stage.

    Args:
        (1) path [string]: The file.
        (2) rows [Iterable<Sequence<Any>>]: The rows, in the order of CONST_SUMMARY_COLUMNS.
    """

    _Write_Rows(path, CONST_SUMMARY_COLUMNS, rows)


def Read_Summary(path: str) -> tp.List[tp.Dict[str, str]]:
    """
    Description:
        A decision's summary.csv (SPEC §9.12; Write_Summary) read back for the figures, as text by column.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [List<Dict<string, string>>]: The rows.
    """

    return Read_Table(path, [CONST_SUMMARY_COLUMNS], '§9.12')
