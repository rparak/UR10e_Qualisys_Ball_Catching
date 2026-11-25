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
# File Name: ../Dataset/Core.py
## =========================================================================== ##

# OS (Operating system interfaces)
import os
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the configurations' structures
from Configuration.Core import Configuration_Str, Dataset_Configuration_Str
#   ../Geometry/Core - the throw cuboid and the target area
from Geometry.Core import Cuboid_Cls, Spherical_Area_Cls
#   ../Flight/Core - the one flight model and its state function
from Flight.Core import Flight_Model_Str, Simulate, Simulate_At, _RK4_Batch, CONST_SIMULATION_STEP
#   ../Recording/Core, Segmentation, Identification - the identification flights, read only
from Recording.Core import Load_Recording, Assign_Sets, CONST_SET_IDENTIFICATION
from Recording.Segmentation import Flight_Str, Segment_Flight
from Recording.Identification import Trim_Flight, CONST_IDENTIFICATION_TRIM
#   ../Observation/Core - the donors, the patterns and the error model
from Observation.Core import Assign_Pools, Donor_Pool, Donor_Str, Draw_Pattern, Frame_Count, Observation_Error, Observation_Model_Str

"""
Description:
    Initialization of constants.

    The dataset generator (SPEC §7.6, §9.5; F6): random throws from one seed, flight i from stream i of
    np.random.SeedSequence(seed).spawn, so a flight depends on the seed and its index only (§2.13.3).
    A release from a Gaussian truncated to Areas.Throw, a target uniform in the dataset's target area (Throw.Target_Area,
    the half of a sphere's upper hemisphere facing the thrower; or Throw.Target_Catch_Area, Areas.Catch grown by a
    margin, for the catch datasets of F9), a flight time from an apex height drawn from a Gaussian, the release velocity by
    shooting under the one flight model; the truth by Simulate at every frame to the floor; the
    observations by the observation model from the identification flights. One file format,
    flights.npz, written and read here only.
"""
# The version of flights.npz (SPEC §9.5); a reader refuses another.
#   Unit [int]
CONST_FORMAT_VERSION = 1
# The partition codes of flights.npz (SPEC §9.5): train, validation and the held-out partition.
#   Unit [int]
CONST_PARTITION_TRAIN = 0
CONST_PARTITION_VALIDATION = 1
CONST_PARTITION_HELD_OUT = 2
# The floor: a flight ends where the ball centre reaches this height (SPEC §14.6; the ball's radius).
#   Unit [float, m]
CONST_FLOOR_HEIGHT = 0.030
# The flights generated at once (the peak memory below 2 GB).
#   Unit [int]
CONST_CHUNK = 1000
# The candidates a flight may draw before the generation stops (a configuration whose throws can hardly
# be drawn is refused, never generated with a silently different distribution).
#   Unit [int]
CONST_MAX_DRAWS = 1000
# The shooting: Newton on the release velocity with a forward-difference Jacobian.
#   Unit [float, m], [int], [float, m/s]
CONST_SHOOT_TOLERANCE = 1e-9
CONST_SHOOT_ITERATIONS = 12
CONST_SHOOT_STEP = 1e-6
# The floor time: Newton on z(t) - CONST_FLOOR_HEIGHT until a step is below this.
#   Unit [float, s], [int]
CONST_FLOOR_TOLERANCE = 1e-12
CONST_FLOOR_ITERATIONS = 12
# The shape of the targets' area: of the sphere of Throw.Target_Area, the upper
# hemisphere's half on the thrower's side.
#   Unit [string]
CONST_TARGET_SHAPE = 'quarter_sphere'
# The arrays of flights.npz (SPEC §9.5), in this order.
#   Unit [Tuple<string>]
CONST_ARRAYS = ('format_version', 'partition', 'rate', 'release_state', 'target', 'apex_drawn', 'flight_time', 'floor_time', 'draws',
                'truth_offset', 'truth_position', 'observed_offset', 'observed_frame', 'observed_time', 'observed_position',
                'donor_offset', 'donor_name')


class Dataset_Error_Cls(ValueError):
    """
    Description:
        A dataset cannot be generated or read: a rate without donors, throws that cannot be drawn, a
        target area that does not face the thrower, a file of another format version. The message names
        the value and what to do.
    """


@dataclass
class Dataset_Str:
    """
    Description:
        A generated dataset, as flights.npz holds it (SPEC §9.5): per flight its partition and parameters;
        ragged arrays concatenated with per-flight offsets (flight i is offset[i]:offset[i + 1]).
    """

    # The partition code of each flight (0 train, 1 validation, 2 the held-out partition; SPEC §9.5) and its rate.
    #   Unit [Vector<int8> 1xF], [Vector<float> 1xF, Hz]
    Partition: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    Rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # The release state (robot frame), the target, the apex drawn, the time to the target, the floor time,
    # and the number of candidates drawn.
    #   Unit [Matrix<float> Fx6, m, m/s], [Matrix<float> Fx3, m], [Vector<float> 1xF, m], [s], [s], [int]
    Release_State: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    Target: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    Apex_Drawn: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Flight_Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Floor_Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Draws: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # The truth positions at every frame of the flight's rate from the release (frame 0) to the floor.
    #   Unit [Vector<int64> 1x(F+1)], [Matrix<float> Tx3, m]
    Truth_Offset: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    Truth_Position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # The observations: frame, time (frame / rate) and position of every sample.
    #   Unit [Vector<int64> 1x(F+1)], [Vector<int64> 1xS], [Vector<float> 1xS, s], [Matrix<float> Sx3, m]
    Observed_Offset: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    Observed_Frame: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    Observed_Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Observed_Position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # The donors replayed by each flight, in the order of its segments.
    #   Unit [Vector<int64> 1x(F+1)], [Vector<string> 1xD]
    Donor_Offset: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    Donor_Name: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype='<U1'))

    @property
    def Flights(self) -> int:
        """
        Description:
            The number of flights.

        Returns:
            (1) parameter [int]: [-].
        """

        return int(self.Rate.size)

    def Observations(self, i: int) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Description:
            The observed samples of flight i, as observed: its own time stamps, never resampled.

        Args:
            (1) i [int]: The flight index.

        Returns:
            (1) parameter 1 [Vector<float> 1xM]: The times from the release [s].
            (2) parameter 2 [Matrix<float> Mx3]: The positions (robot frame) [m].
        """

        a, b = self.Observed_Offset[i], self.Observed_Offset[i + 1]
        return self.Observed_Time[a:b], self.Observed_Position[a:b]

    def Truth(self, i: int) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Description:
            The truth of flight i at every frame of its rate from the release to the floor.

        Args:
            (1) i [int]: The flight index.

        Returns:
            (1) parameter 1 [Vector<float> 1xN]: The frame times [s].
            (2) parameter 2 [Matrix<float> Nx3]: The true positions (robot frame) [m].
        """

        a, b = self.Truth_Offset[i], self.Truth_Offset[i + 1]
        return np.arange(b - a) / self.Rate[i], self.Truth_Position[a:b]


def Partition_Sizes(flights: int, ratios: tp.Sequence[float]) -> tp.List[int]:
    """
    Description:
        The partition sizes by the largest-remainder rule (SPEC §2.13.1): each partition gets the floor of
        its quota flights * ratio, and the flights left over go one each to the largest remainders, ties
        to the earlier partition (train, validation, test).

    Args:
        (1) flights [int]: The number of flights.
        (2) ratios [Sequence<float>]: The train, validation and test ratios, summing to 1.

    Returns:
        (1) parameter [List<int>]: The sizes, summing to flights.
    """

    quotas = [flights * ratio for ratio in ratios]
    sizes = [int(np.floor(quota)) for quota in quotas]
    order = sorted(range(len(ratios)), key=lambda i: (-(quotas[i] - sizes[i]), i))
    for i in order[:flights - sum(sizes)]:
        sizes[i] += 1
    return sizes


def Partition_Codes(flights: int, ratios: tp.Sequence[float]) -> np.ndarray:
    """
    Description:
        The partition of each flight, as index blocks: the first flights are
        train, then validation, then test. The flights are independent draws, so the blocks are as random
        as a permutation; a flight's partition depends on the dataset size, and a run refuses a dataset
        whose hash differs from its Run.Dataset_Hash (SPEC §8.3), which keeps datasets from being mixed.

    Args:
        (1) flights [int]: The number of flights.
        (2) ratios [Sequence<float>]: The train, validation and test ratios.

    Returns:
        (1) parameter [Vector<int8> 1xF]: The partition code of each flight.
    """

    return np.repeat(np.arange(len(ratios), dtype=np.int8), Partition_Sizes(flights, ratios))


def Partition_Indices(data: Dataset_Str, code: int) -> np.ndarray:
    """
    Description:
        The flights of one partition by its code (SPEC §9.5: CONST_PARTITION_TRAIN, CONST_PARTITION_VALIDATION,
        CONST_PARTITION_HELD_OUT): the one selector of a partition, so a leakage test can refuse the held-out
        partition here (SPEC §2.13.7-13.8).

    Args:
        (1) data [Dataset_Str(object)]: The dataset.
        (2) code [int]: The partition code, 0, 1 or 2.

    Returns:
        (1) parameter [Vector<int64> 1xP]: The flight indices, increasing.
    """

    if code not in (CONST_PARTITION_TRAIN, CONST_PARTITION_VALIDATION, CONST_PARTITION_HELD_OUT):
        raise Dataset_Error_Cls('The partition code %r is not one of 0 (train), 1 (validation), 2 (held out; SPEC §9.5).' % (code,))
    return np.where(data.Partition == code)[0].astype(np.int64)


def Identification_Flights(folder: str) -> tp.List[Flight_Str]:
    """
    Description:
        The identification flights, F2's segments with the first CONST_IDENTIFICATION_TRIM dropped, in the QTM frame.
        The sets come from the file names (Recording.Core.Assign_Sets);
        only the identification files are opened: the validation flights are never read (SPEC §2.14).

    Args:
        (1) folder [string]: Data/Recordings.

    Returns:
        (1) parameter [List<Flight_Str>]: The 45 trimmed flights, ordered by folder and experiment.
    """

    names = [(sub, name) for sub in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, sub))
             for name in sorted(os.listdir(os.path.join(folder, sub))) if name.endswith('.txt')]
    sets = Assign_Sets(names)
    flights = []
    for sub, name in names:
        if sets[(sub, name)] != CONST_SET_IDENTIFICATION:
            continue
        flight, reason = Segment_Flight(Load_Recording(os.path.join(folder, sub, name)), CONST_SET_IDENTIFICATION)
        if flight is None:
            raise Dataset_Error_Cls('The identification flight %s/%s has no segmented flight (%s); its dropout pattern cannot serve '
                                    'the observation model. Check the recording (SPEC §7.3).' % (sub, name, reason))
        flights.append(Trim_Flight(flight, CONST_IDENTIFICATION_TRIM))
    return sorted(flights, key=lambda flight: (flight.Folder, flight.Experiment))


def Load_Donors(folder: str) -> tp.List[Donor_Str]:
    """
    Description:
        The donors of the observation model: the identification flights' sample times from the first
        sample at or after the trim (SPEC §7.5), their pools A and B assigned.

    Args:
        (1) folder [string]: Data/Recordings.

    Returns:
        (1) parameter [List<Donor_Str>]: The 45 donors.
    """

    return Assign_Pools([Donor_Str(Name='%s/Experiment_%d.txt' % (flight.Folder, flight.Experiment), Rate=flight.Rate, Time=flight.Time)
                         for flight in Identification_Flights(folder)])


def Target_Area(dataset: Dataset_Configuration_Str, configuration: Configuration_Str) -> tp.Union[Spherical_Area_Cls, Cuboid_Cls]:
    """
    Description:
        The targets' area. Throw.Target_Area: the half, facing the thrower, of the upper
        hemisphere of a sphere, i.e. a quarter sphere of the configured centre, orientation and radius; it is the
        targets' own area, not Areas.Robot (in which the robot catches), and its facing is checked: a point at the
        centre + (0, -r/2, r/4) (towards the thrower at -y) must be inside and its mirror at +y outside.
        Throw.Target_Catch_Area: Areas.Catch of the system configuration grown by the
        margin on every face, so that the catch datasets follow the area where the robot may catch.

    Args:
        (1) dataset [Dataset_Configuration_Str(object)]: The dataset configuration.
        (2) configuration [Configuration_Str(object)]: The system configuration (Areas.Catch).

    Returns:
        (1) parameter [Spherical_Area_Cls(object) or Cuboid_Cls(object)]: The target area.
    """

    if dataset.Throw.Target_Shape == 'catch_area':
        catch = configuration.Areas.Catch
        return Cuboid_Cls(catch.Centre, [v + 2.0 * dataset.Throw.Target_Margin for v in catch.Size])
    target = dataset.Throw.Target_Area
    area = Spherical_Area_Cls(target.Centre, target.Orientation, target.Radius, CONST_TARGET_SHAPE)
    probe = np.array([0.0, 0.5 * target.Radius, 0.25 * target.Radius])
    if not area.Is_Point_Inside(np.array(target.Centre) - probe * [1, 1, -1]) or area.Is_Point_Inside(np.array(target.Centre) + probe):
        raise Dataset_Error_Cls('Throw.Target_Area does not face the thrower at -y with the orientation %s rad. Use '
                                'Throw.Target_Area.Orientation [0, 0, -90] deg (SPEC §8.2).' % (list(target.Orientation),))
    return area


def _Draw_Candidate(rng: np.random.Generator, dataset: Dataset_Configuration_Str, throw: Cuboid_Cls,
                    target: tp.Union[Spherical_Area_Cls, Cuboid_Cls],
                    gravity: float) -> tp.Tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Description:
        One throw candidate from the flight's generator: a release from the Gaussian until it lies in
        Areas.Throw, a target uniform in the target area, an apex height h from its Gaussian; the flight
        time is the vacuum time at which a throw of apex h from the release descends to the target's
        height, t = (v_z + sqrt(v_z^2 - 2 g (z_t - z_0))) / g with v_z = sqrt(2 g (h - z_0)). A candidate
        whose apex is not above both the release and the target, or whose time is outside
        Throw.Flight_Time, is drawn again.

    Args:
        (1) rng [np.random.Generator]: The flight's generator.
        (2) dataset [Dataset_Configuration_Str(object)]: The dataset configuration.
        (3) throw [Cuboid_Cls(object)]: Areas.Throw.
        (4) target [Spherical_Area_Cls(object) or Cuboid_Cls(object)]: The target area (Target_Area).
        (5) gravity [float]: The magnitude of gravity [m/s^2].

    Returns:
        (1) parameter 1 [Vector<float> 1x3]: The release position [m].
        (2) parameter 2 [Vector<float> 1x3]: The target [m].
        (3) parameter 3 [float]: The apex drawn [m].
        (4) parameter 4 [float]: The flight time to the target [s].
        (5) parameter 5 [int]: The candidates drawn, this one included.
    """

    spec = dataset.Throw
    for draws in range(1, CONST_MAX_DRAWS + 1):
        while True:
            release = rng.normal(spec.Release_Mean, spec.Release_Std)
            if throw.Is_Point_Inside(release):
                break
        aim = target.Random_Point_Inside(rng)
        apex = float(rng.normal(spec.Apex_Mean, spec.Apex_Std))
        if apex <= max(release[2], aim[2]):
            continue
        v_z = np.sqrt(2.0 * gravity * (apex - release[2]))
        time = float((v_z + np.sqrt(v_z ** 2 - 2.0 * gravity * (aim[2] - release[2]))) / gravity)
        if spec.Flight_Time[0] <= time <= spec.Flight_Time[1]:
            return release, aim, apex, time, draws
    raise Dataset_Error_Cls('No throw candidate met Throw.Flight_Time %s s in %d draws. Check Throw.Apex_Mean / Apex_Std against the '
                            'flight time (SPEC §8.2).' % (spec.Flight_Time, CONST_MAX_DRAWS))


def Shoot(release: np.ndarray, aim: np.ndarray, time: np.ndarray, *, Flight_Model_Str: Flight_Model_Str) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        The release velocities that reach each target at its time under the flight model (SPEC §7.6):
        Newton from the vacuum solution with a forward-difference Jacobian (CONST_SHOOT_STEP), every row on
        its own (Simulate_At), until the miss is below CONST_SHOOT_TOLERANCE; a row is not updated once it
        has converged.

    Args:
        (1) release [Matrix<float> Bx3]: The release positions [m].
        (2) aim [Matrix<float> Bx3]: The targets [m].
        (3) time [Vector<float> 1xB]: The flight times [s].
        (4) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter 1 [Matrix<float> Bx3]: The release velocities [m/s].
        (2) parameter 2 [Vector<bool> 1xB]: Whether each row converged.
    """

    B = len(release)
    v = (aim - release - 0.5 * Flight_Model_Str.Gravity * time[:, None] ** 2) / time[:, None]
    done = np.zeros(B, dtype=bool)
    for _ in range(CONST_SHOOT_ITERATIONS):
        todo = np.flatnonzero(~done)
        if not todo.size:
            break
        x = np.concatenate([np.hstack((release[todo], v[todo] + CONST_SHOOT_STEP * np.eye(3)[j] * (j >= 0))) for j in (-1, 0, 1, 2)])
        p = Simulate_At(x, np.tile(time[todo], 4), Flight_Model_Str=Flight_Model_Str)[:, :3].reshape(4, todo.size, 3)
        miss = p[0] - aim[todo]
        converged = np.linalg.norm(miss, axis=1) < CONST_SHOOT_TOLERANCE
        done[todo[converged]] = True
        J = np.stack([(p[j + 1] - p[0]) / CONST_SHOOT_STEP for j in range(3)], axis=2)
        step = np.linalg.solve(J[~converged], miss[~converged][..., None])[..., 0]
        v[todo[~converged]] -= step
    return v, done


def _Grid_Simulate_At(state: np.ndarray, *, Flight_Model_Str: Flight_Model_Str) -> tp.Callable[[np.ndarray], np.ndarray]:
    """
    Description:
        Flight.Core.Simulate_At for one set of rows at many successive times, with the grid stepped once:
        Simulate_At(state, t) is the grid state at n = floor(t / CONST_SIMULATION_STEP), stepped from the
        initial state by RK4, plus one partial RK4 step; the grid states do not depend on t, so they are
        kept and extended as the times grow. Each call returns what Simulate_At returns, bit for bit.

    Args:
        (1) state [Matrix<float> Bx6]: The initial states [m, m/s].
        (2) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Callable]: t [Vector<float> 1xB, s, each >= 0] -> the states [Matrix<float> Bx6].
    """

    h = CONST_SIMULATION_STEP
    grid = [np.asarray(state, dtype=np.float64).copy()]
    rows = np.arange(len(grid[0]))

    def At(t: np.ndarray) -> np.ndarray:
        """
        Description:
            The state of each row at its own time.

        Args:
            (1) t [Vector<float> 1xB]: The times [s], each >= 0.

        Returns:
            (1) parameter [Matrix<float> Bx6]: The states [m, m/s].
        """

        n = np.floor(t / h).astype(np.int64)
        while len(grid) <= n.max():
            grid.append(_RK4_Batch(grid[-1], h, Flight_Model_Str))
        start = np.empty_like(grid[0])
        for k in np.unique(n):
            at = rows[n == k]
            start[at] = grid[k][at]
        return _RK4_Batch(start, (t - n * h)[:, None], Flight_Model_Str)

    return At


def Floor_Time(state: np.ndarray, start: np.ndarray, *, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        The time at which each flight's ball centre descends to CONST_FLOOR_HEIGHT (SPEC §14.6): Newton on
        z(t) - CONST_FLOOR_HEIGHT with dz/dt = v_z from the state function, from the vacuum drop after
        `start` (a time on the descent above the floor), every row on its own; the Newton steps go on
        for all rows until every row's step is below CONST_FLOOR_TOLERANCE. The state function is
        Simulate_At's, with its grid stepped once per CONST_CHUNK rows and reused over the Newton steps
        (_Grid_Simulate_At; bit-identical, faster).

    Args:
        (1) state [Matrix<float> Bx6]: The release states [m, m/s].
        (2) start [Vector<float> 1xB]: A time of each flight on its descent, above the floor [s].
        (3) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Vector<float> 1xB]: The floor times [s].
    """

    g = -Flight_Model_Str.Gravity[2]
    chunks = [slice(first, first + CONST_CHUNK) for first in range(0, len(state), CONST_CHUNK)]
    functions = [_Grid_Simulate_At(state[rows], Flight_Model_Str=Flight_Model_Str) for rows in chunks]
    t = np.empty(len(state))
    for rows, At in zip(chunks, functions):
        x = At(start[rows])
        z, v_z = x[:, 2] - CONST_FLOOR_HEIGHT, x[:, 5]
        t[rows] = start[rows] + (v_z + np.sqrt(v_z ** 2 + 2.0 * g * z)) / g
    for _ in range(CONST_FLOOR_ITERATIONS):
        step = np.empty(len(state))
        for rows, At in zip(chunks, functions):
            x = At(t[rows])
            step[rows] = (x[:, 2] - CONST_FLOOR_HEIGHT) / x[:, 5]
        t = t - step
        if np.all(np.abs(step) < CONST_FLOOR_TOLERANCE):
            return t
    raise Dataset_Error_Cls('The floor time did not converge in %d Newton steps (largest step %.3e s). Report this with the dataset '
                            'configuration.' % (CONST_FLOOR_ITERATIONS, float(np.max(np.abs(step)))))


def Draw_Throws(rngs: tp.List[np.random.Generator], dataset: Dataset_Configuration_Str, configuration: Configuration_Str, *,
                Flight_Model_Str: Flight_Model_Str) -> tp.Dict[str, np.ndarray]:
    """
    Description:
        The throws of a set of flights, each from its own generator (SPEC §7.6): candidates
        (_Draw_Candidate) shot in rounds; a candidate whose shooting does not converge or whose release
        speed is outside Throw.Release_Speed is replaced by the flight's next candidate. A flight's
        throw depends on its generator only.

    Args:
        (1) rngs [List<np.random.Generator>]: The flights' generators, the rate already drawn.
        (2) dataset [Dataset_Configuration_Str(object)]: The dataset configuration.
        (3) configuration [Configuration_Str(object)]: The system configuration.
        (4) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Dict<string, np.ndarray>]: 'release_state' Fx6, 'target' Fx3, 'apex_drawn', 'flight_time',
                                                  'floor_time', 'draws'.
    """

    F = len(rngs)
    throw = Cuboid_Cls(configuration.Areas.Throw.Centre, configuration.Areas.Throw.Size)
    target = Target_Area(dataset, configuration)
    gravity = -Flight_Model_Str.Gravity[2]
    release, aim, velocity = np.zeros((F, 3)), np.zeros((F, 3)), np.zeros((F, 3))
    apex, time, draws = np.zeros(F), np.zeros(F), np.zeros(F, dtype=np.int64)
    todo = np.arange(F)
    while todo.size:
        for i in todo:
            release[i], aim[i], apex[i], time[i], n = _Draw_Candidate(rngs[i], dataset, throw, target, gravity)
            draws[i] += n
            if draws[i] > CONST_MAX_DRAWS:
                raise Dataset_Error_Cls('Flight %d drew %d candidates without a release speed inside Throw.Release_Speed %s m/s. Check '
                                        'Throw.Release_Speed (SPEC §8.2).' % (i, draws[i], dataset.Throw.Release_Speed))
        v, converged = Shoot(release[todo], aim[todo], time[todo], Flight_Model_Str=Flight_Model_Str)
        velocity[todo] = v
        speed = np.linalg.norm(v, axis=1)
        accepted = converged & (speed >= dataset.Throw.Release_Speed[0]) & (speed <= dataset.Throw.Release_Speed[1])
        todo = todo[~accepted]
    state = np.hstack((release, velocity))
    return {'release_state': state, 'target': aim, 'apex_drawn': apex, 'flight_time': time,
            'floor_time': Floor_Time(state, time, Flight_Model_Str=Flight_Model_Str), 'draws': draws}


def Check_Rates(dataset: Dataset_Configuration_Str, donors: tp.List[Donor_Str]) -> None:
    """
    Description:
        Every configured rate has donors in the configured pool (Observation.Core.Donor_Pool); checked
        before anything is generated, every rate without donors named at once.

    Args:
        (1) dataset [Dataset_Configuration_Str(object)]: The dataset configuration.
        (2) donors [List<Donor_Str>]: The donors.
    """

    problems = []
    for rate in dataset.Observation.Rates:
        try:
            Donor_Pool(donors, rate, dataset.Observation.Donors)
        except ValueError as error:
            problems.append(str(error))
    if problems:
        raise Dataset_Error_Cls('Observation.Rates cannot be used: %s' % ' '.join(problems))


def Generate_Dataset(dataset: Dataset_Configuration_Str, configuration: Configuration_Str, donors: tp.List[Donor_Str], *,
                     Flight_Model_Str: Flight_Model_Str, Observation_Model_Str: Observation_Model_Str,
                     progress: tp.Optional[tp.Callable[[int, int], None]] = None) -> Dataset_Str:
    """
    Description:
        Generate a dataset (SPEC §2.13.1-13.4, §7.6), CONST_CHUNK flights at a time. Per flight, from its
        own stream: the rate (one of Observation.Rates), the throw (Draw_Throws), the dropout pattern and
        the observation error (Observation.Core). The truth is Simulate at every frame of the rate from
        the release to the floor; an observation is the truth at a present frame plus the error, at the
        frame's time. The partitions are index blocks (Partition_Codes).

    Args:
        (1) dataset [Dataset_Configuration_Str(object)]: The dataset configuration.
        (2) configuration [Configuration_Str(object)]: The system configuration.
        (3) donors [List<Donor_Str>]: The identification donors (Load_Donors).
        (4) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the one flight model.
        (5) Observation_Model_Str [Observation_Model_Str(object)]: REQUIRED, keyword-only: the error model.
        (6) progress [Callable[[int, int], None]]: Called with (done, total) after each chunk, or None.

    Returns:
        (1) parameter [Dataset_Str(object)]: The dataset.
    """

    Check_Rates(dataset, donors)
    Target_Area(dataset, configuration)
    F = dataset.Flights
    choice = dataset.Observation
    sequences = np.random.SeedSequence(dataset.Seed).spawn(F)
    parts: tp.Dict[str, tp.List[np.ndarray]] = {key: [] for key in ('rate', 'release_state', 'target', 'apex_drawn', 'flight_time',
                                                                     'floor_time', 'draws')}
    truth, frames, times, positions, donor_names, donor_counts = [], [], [], [], [], []
    for start in range(0, F, CONST_CHUNK):
        rngs = [np.random.default_rng(sequence) for sequence in sequences[start:start + CONST_CHUNK]]
        rate = np.array([choice.Rates[int(rng.integers(len(choice.Rates)))] for rng in rngs])
        throws = Draw_Throws(rngs, dataset, configuration, Flight_Model_Str=Flight_Model_Str)
        chunk_truth: tp.List[tp.Optional[np.ndarray]] = [None] * len(rngs)
        for value in np.unique(rate):
            rows = np.flatnonzero(rate == value)
            counts = [Frame_Count(throws['floor_time'][k], value) for k in rows]
            state = Simulate(throws['release_state'][rows], np.arange(max(counts)) / value, Flight_Model_Str=Flight_Model_Str)
            for j, k in enumerate(rows):
                chunk_truth[k] = state[j, :counts[j], :3]
        for k, rng in enumerate(rngs):
            present, names = Draw_Pattern(donors, rate[k], throws['floor_time'][k], choice.Pattern, choice.Donors, rng)
            frame = np.flatnonzero(present)
            time = frame / rate[k]
            error = Observation_Error(time, rate[k], Observation_Model_Str=Observation_Model_Str, scale=choice.Noise_Scale, rng=rng)
            truth.append(chunk_truth[k])
            frames.append(frame)
            times.append(time)
            positions.append(chunk_truth[k][frame] + error)
            donor_names.extend(names)
            donor_counts.append(len(names))
        parts['rate'].append(rate)
        for key in ('release_state', 'target', 'apex_drawn', 'flight_time', 'floor_time', 'draws'):
            parts[key].append(throws[key])
        if progress is not None:
            progress(min(start + CONST_CHUNK, F), F)
    ratios = (dataset.Partitions.Train, dataset.Partitions.Validation, dataset.Partitions.Test)
    return Dataset_Str(Partition=Partition_Codes(F, ratios), Rate=np.concatenate(parts['rate']).astype(np.float64),
                       Release_State=np.concatenate(parts['release_state']), Target=np.concatenate(parts['target']),
                       Apex_Drawn=np.concatenate(parts['apex_drawn']), Flight_Time=np.concatenate(parts['flight_time']),
                       Floor_Time=np.concatenate(parts['floor_time']), Draws=np.concatenate(parts['draws']),
                       Truth_Offset=np.cumsum([0] + [len(a) for a in truth]).astype(np.int64), Truth_Position=np.concatenate(truth),
                       Observed_Offset=np.cumsum([0] + [len(a) for a in frames]).astype(np.int64),
                       Observed_Frame=np.concatenate(frames).astype(np.int64), Observed_Time=np.concatenate(times),
                       Observed_Position=np.concatenate(positions), Donor_Offset=np.cumsum([0] + donor_counts).astype(np.int64),
                       Donor_Name=np.array(donor_names, dtype=str) if donor_names else np.zeros(0, dtype='<U1'))


def Write_Dataset(path: str, data: Dataset_Str) -> None:
    """
    Description:
        Write flights.npz (SPEC §9.5): np.savez, uncompressed, float64, the
        arrays of CONST_ARRAYS in their order with the format version. Its bytes depend on the arrays only
        (measured, SPEC §4.5).

    Args:
        (1) path [string]: The file.
        (2) data [Dataset_Str(object)]: The dataset.
    """

    arrays = {'format_version': np.array(CONST_FORMAT_VERSION, dtype=np.int64), 'partition': data.Partition, 'rate': data.Rate,
              'release_state': data.Release_State, 'target': data.Target, 'apex_drawn': data.Apex_Drawn, 'flight_time': data.Flight_Time,
              'floor_time': data.Floor_Time, 'draws': data.Draws, 'truth_offset': data.Truth_Offset, 'truth_position': data.Truth_Position,
              'observed_offset': data.Observed_Offset, 'observed_frame': data.Observed_Frame, 'observed_time': data.Observed_Time,
              'observed_position': data.Observed_Position, 'donor_offset': data.Donor_Offset, 'donor_name': data.Donor_Name}
    with open(path, 'wb') as file:
        np.savez(file, **{key: arrays[key] for key in CONST_ARRAYS})


def Read_Dataset(path: str) -> Dataset_Str:
    """
    Description:
        Read flights.npz (SPEC §9.5); a file of another format version, or without one of CONST_ARRAYS,
        is refused.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [Dataset_Str(object)]: The dataset.
    """

    with np.load(path, allow_pickle=False) as file:
        missing = [key for key in CONST_ARRAYS if key not in file.files]
        if missing:
            raise Dataset_Error_Cls('%s is not a dataset of this project: it lacks %s. Generate it with scripts/catch.py dataset.'
                                    % (path, ', '.join(missing)))
        version = int(file['format_version'])
        if version != CONST_FORMAT_VERSION:
            raise Dataset_Error_Cls('%s has format version %d; this code reads version %d. Regenerate it with scripts/catch.py dataset.'
                                    % (path, version, CONST_FORMAT_VERSION))
        a = {key: file[key] for key in CONST_ARRAYS}
    return Dataset_Str(Partition=a['partition'], Rate=a['rate'], Release_State=a['release_state'], Target=a['target'],
                       Apex_Drawn=a['apex_drawn'], Flight_Time=a['flight_time'], Floor_Time=a['floor_time'], Draws=a['draws'],
                       Truth_Offset=a['truth_offset'], Truth_Position=a['truth_position'], Observed_Offset=a['observed_offset'],
                       Observed_Frame=a['observed_frame'], Observed_Time=a['observed_time'], Observed_Position=a['observed_position'],
                       Donor_Offset=a['donor_offset'], Donor_Name=a['donor_name'])
