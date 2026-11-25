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
# File Name: ../Catch/Core.py
## =========================================================================== ##

# CSV (the catch's files, read back for the figures)
import csv
# Dataclasses (Data classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation; the poses' exact distances) [pip3 install mujoco]
import mujoco
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Robot/Core - the robot's kinematics, limits, clearance and speed margin
from Robot.Core import (Robot_Str, Robot_Geometry_Str, Collision_Pairs_Str, Robot_Geometry, Collision_Pairs, Inverse_Kinematics,
                        Forward_Kinematics, Within_Margin, Clearance, Speed_Margin)

"""
Description:
    Initialization of constants.

    The catch pose and where it exists (SPEC §7.11, F4): a point qualifies when, for every arrival direction
    of the configured cone, a joint vector puts the centre of the cup's mouth on the point with the cup's
    axis against the ball's velocity, within the joints' ranges by a margin, clear of the stand, the floor
    and the robot itself by the clearance, and with the speed margin. The lattice of points, the arrival
    cone, and the largest cuboid of qualifying lattice points. Pure computation on a compiled model: no
    file, no configuration, no printing.
"""
# The arrival directions: CONST_ARRIVAL_STEPS elevations times CONST_ARRIVAL_STEPS azimuths, evenly over
# the configured cone, its edges included (5 x 5 = 25).
#   Unit [int]
CONST_ARRIVAL_STEPS = 5
# The joints a catch pose uses: the cup is symmetric about joint 6's axis, so joint 6 moves neither the
# point nor the cup's axis and is left at 0 (F4 Part A; test/Scene/test_core.py).
#   Unit [int]
CONST_JOINTS_USED = 5
# The criteria of a catch pose, in the order they are checked (the cheapest first), and the code of each
# as the workspace map stores the first one a pose fails; 'ik' is a direction without any solution.
#   Unit [Tuple<string>]
CONST_CRITERIA = ('ik', 'limits', 'environment', 'self', 'exempt', 'speed')


class Catch_Error_Cls(ValueError):
    """
    Description:
        A catch pose or an area cannot be evaluated. The message names the value and what it must be; a
        script prints it verbatim as an [ERROR] line.
    """


@dataclass
class Criteria_Str:
    """
    Description:
        The thresholds of a catch pose (all assumed) and the joints' speed limits.
    """

    # The margin each joint keeps to its range.
    #   Unit [float, rad]
    Joint_Margin: float = 0.0
    # The smallest exact distance to the stand and the floor, and between the robot's own parts (except the
    # exempt pair, which is held apart only).
    #   Unit [float, m]
    Clearance: float = 0.0
    # The smallest speed the tool centre point must reach in every direction with the cup's axis held.
    #   Unit [float, m/s]
    Speed_Margin: float = 0.0
    # The joints' speed limits.
    #   Unit [Vector<float> 1x6, rad/s]
    Joint_Speed: np.ndarray = field(default_factory=lambda: np.zeros(6))


@dataclass
class Catch_Context_Str:
    """
    Description:
        What every catch-pose check of one compiled scene needs (Catch_Context): the model, its data, the
        robot, its geometry, its collision pairs and the criteria.
    """

    # The compiled scene and a data for its checks.
    #   Unit [mujoco.MjModel], [mujoco.MjData]
    Model: tp.Any = None
    Data: tp.Any = None
    # The robot, its UR geometry and its collision pairs in the model.
    #   Unit [Robot_Str], [Robot_Geometry_Str], [Collision_Pairs_Str]
    Robot: Robot_Str = field(default_factory=Robot_Str)
    Geometry: Robot_Geometry_Str = field(default_factory=Robot_Geometry_Str)
    Pairs: Collision_Pairs_Str = field(default_factory=Collision_Pairs_Str)
    # The thresholds.
    #   Unit [Criteria_Str]
    Criteria: Criteria_Str = field(default_factory=Criteria_Str)


@dataclass
class Pose_Check_Str:
    """
    Description:
        One joint vector checked against the criteria (Check_Pose): every measured value, and the first
        criterion it fails ('' when it qualifies). A value not measured because an earlier criterion failed
        is NaN.
    """

    # The smallest distance of each joint to its range's limits.
    #   Unit [float, rad]
    Limit_Distance: float = float('nan')
    # The clearance to the environment and to the robot itself, and the exempt pair's distance, each cut at
    # Criteria.Clearance (a value equal to it means "at least").
    #   Unit [float, m]
    Environment: float = float('nan')
    Self: float = float('nan')
    Exempt: float = float('nan')
    # The speed margin.
    #   Unit [float, m/s]
    Speed: float = float('nan')
    # The first criterion of CONST_CRITERIA the pose fails, '' when it qualifies.
    #   Unit [string]
    Failed: str = ''


def Catch_Context(model: mujoco.MjModel, robot: Robot_Str, *, Criteria_Str: Criteria_Str) -> Catch_Context_Str:
    """
    Description:
        The context of the catch-pose checks in a compiled scene: its UR geometry and collision pairs are
        read from the model once.

    Args:
        (1) model [mujoco.MjModel]: The compiled scene (Scene.Core.Build_Scene).
        (2) robot [Robot_Str(object)]: The robot in it (Scene.Core.Scene_Robot).
        (3) Criteria_Str [Criteria_Str(object)]: REQUIRED, keyword-only: the thresholds.

    Returns:
        (1) parameter [Catch_Context_Str(object)]: The context.
    """

    return Catch_Context_Str(Model=model, Data=mujoco.MjData(model), Robot=robot, Geometry=Robot_Geometry(model, Robot_Str=robot),
                             Pairs=Collision_Pairs(model, Robot_Str=robot), Criteria=Criteria_Str)


def Arrival_Axes(elevation: tp.Sequence[float], azimuth: tp.Sequence[float]) -> np.ndarray:
    """
    Description:
        The cup's axes against the arrival directions of the cone: the ball's velocity descends at the
        elevation e below the horizontal and points at the azimuth z from +y towards +x, v = (cos e sin z,
        cos e cos z, -sin e); the cup's axis is -v. CONST_ARRIVAL_STEPS values of each, evenly from its min
        to its max, elevation major.

    Args:
        (1) elevation [Vector<float> 1x2]: [min, max] below the horizontal [rad].
        (2) azimuth [Vector<float> 1x2]: [min, max] from +y towards +x [rad].

    Returns:
        (1) parameter [Matrix<float> Dx3]: The cup's axes (unit vectors), D = CONST_ARRIVAL_STEPS^2.
    """

    e, z = np.meshgrid(np.linspace(elevation[0], elevation[1], CONST_ARRIVAL_STEPS), np.linspace(azimuth[0], azimuth[1], CONST_ARRIVAL_STEPS),
                       indexing='ij')
    velocity = np.stack([np.cos(e) * np.sin(z), np.cos(e) * np.cos(z), -np.sin(e)], axis=-1).reshape(-1, 3)
    return -velocity


def Check_Pose(q: np.ndarray, complete: bool = False, *, Catch_Context_Str: Catch_Context_Str) -> Pose_Check_Str:
    """
    Description:
        Check one joint vector against the criteria, the cheapest first: the joints' ranges by the margin,
        the clearance to the environment, the clearance to the robot itself, the exempt pair apart (> 0),
        the speed margin. The check stops at the first criterion that fails unless `complete`.

    Args:
        (1) q [Vector<float> 1x6]: The joint vector [rad].
        (2) complete [bool]: Measure every value even after a failure.
        (3) Catch_Context_Str [Catch_Context_Str(object)]: REQUIRED, keyword-only: the context.

    Returns:
        (1) parameter [Pose_Check_Str(object)]: The values and the first criterion failed.
    """

    c = Catch_Context_Str
    criteria, robot = c.Criteria, c.Robot
    check = Pose_Check_Str()
    q = np.asarray(q, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        check.Failed = 'ik'
        return check
    check.Limit_Distance = float(np.min(np.minimum(q - robot.Lower, robot.Upper - q)))
    if not Within_Margin(q, criteria.Joint_Margin, Robot_Str=robot):
        check.Failed = 'limits'
        if not complete:
            return check
    if not Within_Margin(q, 0.0, Robot_Str=robot):
        return check
    Forward_Kinematics(c.Model, c.Data, q, Robot_Str=robot)
    for name, pairs, bound in (('environment', c.Pairs.Environment_Set, criteria.Clearance), ('self', c.Pairs.Self_Set, criteria.Clearance),
                               ('exempt', c.Pairs.Exempt_Set, 0.0)):
        distance, _ = Clearance(c.Model, c.Data, pairs, criteria.Clearance)
        setattr(check, name.capitalize(), distance)
        if not distance >= bound or (name == 'exempt' and not distance > 0.0):
            check.Failed = check.Failed or name
            if not complete:
                return check
    check.Speed = Speed_Margin(c.Model, c.Data, q, criteria.Joint_Speed, CONST_JOINTS_USED, Robot_Str=robot)
    if not check.Speed >= criteria.Speed_Margin:
        check.Failed = check.Failed or 'speed'
    return check


def Catch_Poses(point: np.ndarray, axes: np.ndarray, *, Catch_Context_Str: Catch_Context_Str) -> np.ndarray:
    """
    Description:
        Every joint vector of the closed form for the point and each cup axis (Robot.Core.Inverse_Kinematics:
        up to 8 per axis, joint 6 at 0).

    Args:
        (1) point [Vector<float> 1x3]: The centre of the cup's mouth [m].
        (2) axes [Matrix<float> Dx3]: The cup's axes.
        (3) Catch_Context_Str [Catch_Context_Str(object)]: REQUIRED, keyword-only: the context.

    Returns:
        (1) parameter [Array<float> Dx8x6]: The joint vectors [rad], NaN where a branch has none.
    """

    axes = np.atleast_2d(axes)
    return Inverse_Kinematics(np.broadcast_to(np.asarray(point, dtype=np.float64), axes.shape), axes,
                              Robot_Geometry_Str=Catch_Context_Str.Geometry)


def Check_Point(point: np.ndarray, axes: np.ndarray, every: bool = False, *,
                Catch_Context_Str: Catch_Context_Str) -> tp.Tuple[bool, np.ndarray, int, str]:
    """
    Description:
        Whether a point qualifies: for every cup axis at least one of its joint vectors passes Check_Pose.
        The axes are checked in order and the check stops at the first axis without a qualifying vector;
        within an axis it stops at the first qualifying vector, unless `every`, which checks every vector of
        every axis (the workspace map keeps them all, for the move times).

    Args:
        (1) point [Vector<float> 1x3]: The centre of the cup's mouth [m].
        (2) axes [Matrix<float> Dx3]: The cup's axes (Arrival_Axes).
        (3) every [bool]: Check every joint vector of a point that qualifies.
        (4) Catch_Context_Str [Catch_Context_Str(object)]: REQUIRED, keyword-only: the context.

    Returns:
        (1) parameter [bool]: True when the point qualifies.
        (2) parameter [Array<float> Dx8x6]: The joint vectors; NaN for a branch without one, for one that
                                            failed and for one not checked.
        (3) parameter [int]: The axis without a qualifying vector, -1 when the point qualifies.
        (4) parameter [string]: For that axis, the criterion its best vector failed (the one that got the
                                furthest through CONST_CRITERIA), '' when the point qualifies.
    """

    poses = Catch_Poses(point, axes, Catch_Context_Str=Catch_Context_Str)
    kept = np.full(poses.shape, np.nan)
    for d in range(len(axes)):
        found, furthest = False, 'ik'
        for s in range(poses.shape[1]):
            if not np.all(np.isfinite(poses[d, s])):
                continue
            check = Check_Pose(poses[d, s], Catch_Context_Str=Catch_Context_Str)
            if check.Failed == '':
                kept[d, s], found = poses[d, s], True
                if not every:
                    break
            elif CONST_CRITERIA.index(check.Failed) > CONST_CRITERIA.index(furthest):
                furthest = check.Failed
        if not found:
            return False, kept, d, furthest
    return True, kept, -1, ''


def Lattice_Domain(Catch_Context_Str: Catch_Context_Str) -> tp.Tuple[np.ndarray, float]:
    """
    Description:
        Where a catch pose can exist at all: the centre is the point of joint 1's axis level with joint 2's
        axis (the shoulder), and the reach the sum of the lengths from it to the tool centre point along the
        chain (to joint 2's axis, to joint 3's, to the wrist point P, to W, to the tool centre point), which no
        pose exceeds. A lattice over this ball holds every point that can qualify.

    Args:
        (1) Catch_Context_Str [Catch_Context_Str(object)]: The context.

    Returns:
        (1) parameter [Vector<float> 1x3]: The centre [m].
        (2) parameter [float]: The reach [m].
    """

    g = Catch_Context_Str.Geometry
    centre = g.Base_Point + ((g.Shoulder_Point - g.Base_Point) @ g.Base_Axis) * g.Base_Axis
    reach = (np.linalg.norm(g.Shoulder_Point - centre) + np.linalg.norm(g.Elbow_Point - g.Shoulder_Point)
             + np.linalg.norm(g.Wrist_Point - g.Elbow_Point) + abs(g.Wrist_Offset) + abs(g.Tool_Offset))
    return centre, float(reach)


def Lattice(centre: np.ndarray, reach: float, step: float, floor: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        The lattice of candidate points: every point whose coordinates are whole multiples of the step,
        within `reach` of the centre and at or above the floor, ordered by x, then y, then z.

    Args:
        (1) centre [Vector<float> 1x3]: The centre of the reach [m].
        (2) reach [float]: The largest distance from the centre [m].
        (3) step [float]: The lattice's step [m].
        (4) floor [float]: The lowest z [m].

    Returns:
        (1) parameter [Matrix<float> Nx3]: The points [m].
        (2) parameter [Matrix<int> Nx3]: Their whole multiples of the step.
    """

    if not step > 0.0:
        raise Catch_Error_Cls('The lattice\'s step must be > 0, found %r.' % (step,))
    centre = np.asarray(centre, dtype=np.float64)
    low, high = np.floor((centre - reach) / step).astype(np.int64), np.ceil((centre + reach) / step).astype(np.int64)
    axes = [np.arange(low[i], high[i] + 1) for i in range(3)]
    index = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    points = index * step
    keep = (np.linalg.norm(points - centre, axis=1) <= reach) & (points[:, 2] >= floor - 1e-12)
    return points[keep], index[keep]


def Largest_Cuboid(index: np.ndarray, qualifies: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        The axis-aligned cuboid of lattice points with the most points, all of which qualify: for each
        range of x layers the points qualifying in every layer form a mask over (y, z), whose largest
        all-qualifying rectangle is found with the histogram method. The first largest in the order
        (x start, x end, y, z) is returned, so the result does not depend on anything but the lattice.

    Args:
        (1) index [Matrix<int> Nx3]: The lattice points' whole multiples of the step (Lattice).
        (2) qualifies [Vector<bool> 1xN]: Which points qualify.

    Returns:
        (1) parameter [Vector<int> 1x3]: The lower corner's multiples of the step.
        (2) parameter [Vector<int> 1x3]: The upper corner's multiples of the step (inclusive).
    """

    qualifies = np.asarray(qualifies, dtype=bool)
    if not qualifies.any():
        raise Catch_Error_Cls('No lattice point qualifies: there is no cuboid.')
    low = index[qualifies].min(axis=0)
    shape = index[qualifies].max(axis=0) - low + 1
    grid = np.zeros(shape, dtype=bool)
    inside = np.all((index >= low) & (index < low + shape), axis=1) & qualifies
    grid[tuple((index[inside] - low).T)] = True
    best, corner = 0, (None, None)
    for x0 in range(shape[0]):
        mask = np.ones(shape[1:], dtype=bool)
        for x1 in range(x0, shape[0]):
            mask &= grid[x1]
            if not mask.any():
                break
            area, (y0, y1, z0, z1) = _Largest_Rectangle(mask)
            if area * (x1 - x0 + 1) > best:
                best = area * (x1 - x0 + 1)
                corner = (np.array([x0, y0, z0]), np.array([x1, y1, z1]))
    return corner[0] + low, corner[1] + low


def _Largest_Rectangle(mask: np.ndarray) -> tp.Tuple[int, tp.Tuple[int, int, int, int]]:
    """
    Description:
        The largest all-True rectangle of a 2D mask (the histogram method: per row, the heights of the True
        runs ending in it, and the largest rectangle under that histogram with a stack). The first largest
        in the order of its last row, then its first column.

    Args:
        (1) mask [Matrix<bool> RxC]: The mask.

    Returns:
        (1) parameter [int]: Its number of cells (0 for a mask without a True).
        (2) parameter [Tuple<int>]: Its rows and columns, (row 0, row 1, column 0, column 1), inclusive.
    """

    rows, columns = mask.shape
    heights = np.zeros(columns, dtype=np.int64)
    best, where = 0, (0, -1, 0, -1)
    for row in range(rows):
        heights = np.where(mask[row], heights + 1, 0)
        stack: tp.List[int] = []
        for column in range(columns + 1):
            height = int(heights[column]) if column < columns else 0
            while stack and int(heights[stack[-1]]) >= height:
                top = stack.pop()
                top_height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                area = top_height * (column - left)
                if area > best:
                    best, where = area, (row - top_height + 1, row, left, column - 1)
            stack.append(column)
    return best, where


def Check_Area(points: np.ndarray, axes: np.ndarray, failures: int = 1, *,
               Catch_Context_Str: Catch_Context_Str) -> tp.Tuple[int, tp.List[tp.Tuple[np.ndarray, int, str]]]:
    """
    Description:
        Check that every point given (the lattice points of an area) qualifies, stopping after `failures`
        points that do not.

    Args:
        (1) points [Matrix<float> Nx3]: The points [m].
        (2) axes [Matrix<float> Dx3]: The cup's axes (Arrival_Axes).
        (3) failures [int]: The number of failing points after which the check stops.
        (4) Catch_Context_Str [Catch_Context_Str(object)]: REQUIRED, keyword-only: the context.

    Returns:
        (1) parameter [int]: The number of points checked.
        (2) parameter [List<Tuple>]: Each failing point with its axis without a qualifying vector and the
                                     criterion its best vector failed.
    """

    failed = []
    for checked, point in enumerate(np.atleast_2d(points), start=1):
        qualifies, _, axis, criterion = Check_Point(point, axes, Catch_Context_Str=Catch_Context_Str)
        if not qualifies:
            failed.append((np.array(point), axis, criterion))
            if len(failed) >= failures:
                return checked, failed
    return len(np.atleast_2d(points)), failed


def Read_Table(path: str, headers: tp.Sequence[tp.Sequence[str]], section: str) -> tp.List[tp.Dict[str, str]]:
    """
    Description:
        The rows of one of the catch's CSV files (a decision's, an interception's or a catch's), as text by column, in
        the file's order; read back for the figures (scripts/plots.py). A file whose header is none of the given ones is
        refused, naming the section of the SPEC that states it.

    Args:
        (1) path [string]: The file.
        (2) headers [Sequence<Sequence<string>>]: The headers its writer writes (a static, a moving, a sweep's file).
        (3) section [string]: The SPEC's section of the format, for the message.

    Returns:
        (1) parameter [List<Dict<string, string>>]: The rows.
    """

    with open(path, 'r', encoding='ascii', newline='') as file:
        reader = csv.reader(file)
        header = tuple(next(reader, None) or ())
        if header not in {tuple(h) for h in headers}:
            raise Catch_Error_Cls('%s does not have the columns its writer writes (SPEC %s); it was written by another version or is not '
                                  'that file.' % (path, section))
        return [dict(zip(header, row)) for row in reader]
