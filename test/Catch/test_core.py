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
# File Name: ../test/Catch/test_core.py
## =========================================================================== ##


# System (Default)
import itertools
# Dataclasses (A context with other pairs or criteria)
import dataclasses
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Robot/Core - the prepared pair sets of a context with other pairs
from Robot.Core import Pair_Set
#   ../Catch/Core
from Catch.Core import (Catch_Context, Criteria_Str, Arrival_Axes, Check_Pose, Check_Point, Catch_Poses, Lattice, Lattice_Domain,
                        Largest_Cuboid, Check_Area, Catch_Error_Cls, CONST_CRITERIA)
#   ../Scene/Core - the robot of the shipped scene
from Scene.Core import Scene_Robot

"""
Description:
    Initialization of constants.

    The catch pose and the lattice (src/Catch/Core.py). The thresholds are the chosen ones (F4),
    written out here; a point known to qualify comes from F4 Part A (the middle of its largest cuboid).
"""
# The criteria and the UR10e's joint speeds (SPEC §4.3).
#   Unit [rad], [m], [m/s], [Vector<float> 1x6, rad/s]
CONST_MARGIN = np.radians(10.0)
CONST_CLEARANCE = 0.05
CONST_SPEED = 0.5
CONST_JOINT_SPEED = np.radians([120.0, 120.0, 180.0, 180.0, 180.0, 180.0])
# The arrival cone of the shipped configuration (F4 Part A, rounded outwards).
#   Unit [deg]
CONST_ELEVATION = (44.0, 68.0)
CONST_AZIMUTH = (-25.0, 27.0)
# A point inside F4 Part A's largest cuboid (x +-0.7, y -0.9...-0.6, z 0.5...1.2 m), and one inside the stand's
# controller box behind the robot.
#   Unit [Vector<float> 1x3, m]
CONST_GOOD_POINT = (0.0, -0.75, 0.85)
CONST_STAND_POINT = (0.0, 1.0, 0.45)
# The catch area's reference time (the rule B800; SPEC §7.11), written out here.
#   Unit [s]
CONST_REACH_TIME = 0.7


@pytest.fixture(scope='module')
def context(scene):
    return Catch_Context(scene, Scene_Robot(scene), Criteria_Str=Criteria_Str(Joint_Margin=CONST_MARGIN, Clearance=CONST_CLEARANCE,
                                                                              Speed_Margin=CONST_SPEED, Joint_Speed=CONST_JOINT_SPEED))


@pytest.fixture(scope='module')
def axes():
    return Arrival_Axes(np.radians(CONST_ELEVATION), np.radians(CONST_AZIMUTH))


def _Good_Pose(context, axes):
    poses = Catch_Poses(CONST_GOOD_POINT, axes[12:13], Catch_Context_Str=context)[0]
    for q in poses:
        if Check_Pose(q, Catch_Context_Str=context).Failed == '':
            return q
    raise AssertionError('no qualifying pose at the good point')


def _With(context, **changes):
    criteria = dataclasses.replace(context.Criteria, **{k: v for k, v in changes.items() if hasattr(context.Criteria, k)})
    pair_changes = {k: v for k, v in changes.items() if hasattr(context.Pairs, k)}
    # A replaced pair array replaces its set prepared for Clearance too (Robot.Core.Pair_Set; F10 step 1).
    pair_changes.update({k + '_Set': Pair_Set(context.Model, v) for k, v in list(pair_changes.items()) if k in ('Environment', 'Self', 'Exempt')})
    pairs = dataclasses.replace(context.Pairs, **pair_changes)
    return dataclasses.replace(context, Criteria=criteria, Pairs=pairs)


def test_the_arrival_axes_are_the_cones_5_by_5_directions_against_the_velocity(axes):
    """
    Description:
        25 unit axes, the cone's corners among them: an axis is minus the velocity that descends e below the
        horizontal at the azimuth z from +y towards +x, so every axis points up, and the first is the corner
        (44 deg, -25 deg), the last (68 deg, 27 deg), the middle one (56 deg, 1 deg).
    """

    assert axes.shape == (25, 3) and np.allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-12)
    for row, (e, z) in ((0, (44.0, -25.0)), (24, (68.0, 27.0)), (12, (56.0, 1.0)), (4, (44.0, 27.0))):
        e, z = np.radians(e), np.radians(z)
        assert np.allclose(axes[row], [-np.cos(e) * np.sin(z), -np.cos(e) * np.cos(z), np.sin(e)], atol=1e-12)
    assert np.all(axes[:, 2] > 0.0)


def test_a_qualifying_pose_passes_every_criterion_and_reports_its_values(context, axes):
    """
    Description:
        A catch pose at a point inside Part A's cuboid passes (Failed ''), with every value measured: the
        limit distance above the margin, both clearances at the cut (at least 0.05 m), the exempt pair apart,
        the speed margin above 0.5 m/s; a complete check measures the same values.
    """

    q = _Good_Pose(context, axes)
    check = Check_Pose(q, Catch_Context_Str=context)
    assert check.Failed == '' and check.Limit_Distance >= CONST_MARGIN
    assert check.Environment == CONST_CLEARANCE and check.Self == CONST_CLEARANCE and 0.0 < check.Exempt <= CONST_CLEARANCE
    assert check.Speed >= CONST_SPEED
    assert Check_Pose(q, complete=True, Catch_Context_Str=context) == check


def test_each_criterion_fails_alone_on_one_side_of_its_threshold(context, axes):
    """
    Description:
        At the qualifying pose each criterion is moved across its measured value, with the others left as
        they are: the joint margin 1e-6 rad above the pose's limit distance fails 'limits' and 1e-6 below
        passes; the clearance just above the environment's distance (self pairs left out) fails
        'environment', just below passes; the same for 'self' (environment pairs left out); an exempt pair
        that touches (the cup against wrist 2, -17 mm in every pose) fails 'exempt'; the speed margin just
        above the pose's fails 'speed', just below passes. A pose without a solution fails 'ik'.
    """

    q = _Good_Pose(context, axes)
    full = Check_Pose(q, complete=True, Catch_Context_Str=_With(context, Clearance=1.0))
    empty = np.zeros((0, 2), dtype=np.int64)
    cases = [('limits', {'Joint_Margin': full.Limit_Distance}),
             ('environment', {'Clearance': full.Environment, 'Self': empty}),
             ('self', {'Clearance': full.Self, 'Environment': empty}),
             ('speed', {'Speed_Margin': full.Speed})]
    for name, changes in cases:
        key = [k for k in changes if k in ('Joint_Margin', 'Clearance', 'Speed_Margin')][0]
        above = dict(changes, **{key: changes[key] + 1e-6})
        below = dict(changes, **{key: changes[key] - 1e-6})
        if name in ('environment', 'self'):
            below['Exempt'] = empty
            above['Exempt'] = empty
        assert Check_Pose(q, Catch_Context_Str=_With(context, **above)).Failed == name, name
        assert Check_Pose(q, Catch_Context_Str=_With(context, **below)).Failed == '', name
    touching = context.Pairs.Constant[context.Pairs.Constant_Distance < 0.0]
    assert len(touching) == 1
    assert Check_Pose(q, Catch_Context_Str=_With(context, Exempt=touching)).Failed == 'exempt'
    assert Check_Pose(np.full(6, np.nan), Catch_Context_Str=context).Failed == 'ik'
    assert CONST_CRITERIA == ('ik', 'limits', 'environment', 'self', 'exempt', 'speed')


def test_a_point_qualifies_only_when_every_axis_has_a_qualifying_pose(context, axes):
    """
    Description:
        The good point qualifies; every pose kept with `every` passes the check, and `every` keeps at least as
        many as the first-per-axis check, one per axis at least. A point beyond the reach fails at the first
        axis for 'ik'; a point inside the stand's controller box fails for a clearance; an axis list whose
        last direction cannot be met there (the cup open towards the robot, +y, at (0, -0.75, 0.6) m: the arm
        folds into itself) makes the point fail at that axis for 'self'.
    """

    good, first, axis, criterion = Check_Point(CONST_GOOD_POINT, axes, Catch_Context_Str=context)
    assert good and axis == -1 and criterion == ''
    _, every, _, _ = Check_Point(CONST_GOOD_POINT, axes, every=True, Catch_Context_Str=context)
    assert np.all(np.isfinite(first[..., 0]).sum(axis=1) == 1)
    assert np.all(np.isfinite(every[..., 0]).sum(axis=1) >= 1) and np.isfinite(every[..., 0]).sum() > 25
    for q in every.reshape(-1, 6):
        if np.all(np.isfinite(q)):
            assert Check_Pose(q, Catch_Context_Str=context).Failed == ''
    assert Check_Point((2.5, 0.0, 0.5), axes, Catch_Context_Str=context)[2:] == (0, 'ik')
    stand = Check_Point(CONST_STAND_POINT, axes, Catch_Context_Str=context)
    assert not stand[0] and stand[3] in ('ik', 'environment', 'self')
    bad_axes = np.vstack([axes, [0.0, 1.0, 0.0]])
    result = Check_Point((0.0, -0.75, 0.6), bad_axes, Catch_Context_Str=context)
    assert not result[0] and result[2:] == (25, 'self')


def test_the_lattice_is_every_multiple_of_the_step_in_the_ball_above_the_floor():
    """
    Description:
        Around (0, 0, 0.1) with a reach of 0.12 m and a step of 0.05 m: every point whose coordinates are
        whole multiples of the step, within the reach, z >= 0; counted here by brute force over the cube,
        none missing and none extra; a step of 0 is refused.
    """

    points, index = Lattice((0.0, 0.0, 0.1), 0.12, 0.05, 0.0)
    expected = {(i, j, k) for i, j, k in itertools.product(range(-5, 6), repeat=3)
                if np.linalg.norm(np.array([i, j, k]) * 0.05 - [0.0, 0.0, 0.1]) <= 0.12 and k >= 0}
    assert {tuple(r) for r in index.tolist()} == expected and len(index) == len(expected)
    assert np.allclose(points, index * 0.05)
    with pytest.raises(Catch_Error_Cls, match='step'):
        Lattice((0.0, 0.0, 0.0), 1.0, 0.0, 0.0)


def test_the_lattice_domain_is_the_shoulder_and_the_chains_length(context):
    """
    Description:
        The centre is the shoulder, z 0.180 + 0.181 = 0.361 m on joint 1's axis; the reach is the chain's
        length from it, from the vendored file: 0.176 + sqrt(0.613^2 + 0.137^2) + sqrt(0.571^2 + 0.135^2) +
        0.12 + 0.29 m.
    """

    centre, reach = Lattice_Domain(context)
    assert np.allclose(centre, [0.0, 0.0, 0.361], atol=1e-12)
    assert reach == pytest.approx(0.176 + np.hypot(0.613, 0.137) + np.hypot(0.571, 0.135) + 0.12 + 0.29, abs=1e-12)


def _Brute_Cuboid(grid):
    best, corner = 0, None
    nx, ny, nz = grid.shape
    for x0, x1 in itertools.combinations_with_replacement(range(nx), 2):
        for y0, y1 in itertools.combinations_with_replacement(range(ny), 2):
            for z0, z1 in itertools.combinations_with_replacement(range(nz), 2):
                size = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
                if size > best and grid[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1].all():
                    best, corner = size, ((x0, y0, z0), (x1, y1, z1))
    return best, corner


def test_the_largest_cuboid_is_the_one_a_brute_force_search_finds():
    """
    Description:
        On 40 random 6 x 5 x 4 masks (seed 20261121, 70 % qualifying) the cuboid found has the most points of
        any all-qualifying cuboid, as an exhaustive search over every cuboid counts them, and all its points
        qualify; on a mask with a known 3 x 2 x 2 block and scattered points it is that block, offset by the
        lattice's lowest index; a mask without a qualifying point is refused.
    """

    rng = np.random.default_rng(20261121)
    index = np.stack(np.meshgrid(np.arange(6), np.arange(5), np.arange(4), indexing='ij'), axis=-1).reshape(-1, 3) - [3, 7, 1]
    for _ in range(40):
        grid = rng.random((6, 5, 4)) < 0.7
        lower, upper = Largest_Cuboid(index, grid.reshape(-1))
        found = grid[tuple(slice(a, b + 1) for a, b in zip(lower + [3, 7, 1], upper + [3, 7, 1]))]
        assert found.all() and found.size == _Brute_Cuboid(grid)[0]
    grid = np.zeros((6, 5, 4), dtype=bool)
    grid[1:4, 2:4, 0:2] = True
    grid[5, 0, 3] = grid[0, 4, 3] = True
    lower, upper = Largest_Cuboid(index, grid.reshape(-1))
    assert lower.tolist() == [1 - 3, 2 - 7, 0 - 1] and upper.tolist() == [3 - 3, 3 - 7, 1 - 1]
    with pytest.raises(Catch_Error_Cls, match='No lattice point qualifies'):
        Largest_Cuboid(index, np.zeros(len(index), dtype=bool))


def test_the_2022_hemisphere_fails_the_area_check_and_a_qualifying_box_passes(context, axes, configuration):
    """
    Description:
        The forced failure of the area check: the lattice points (0.05 m) of the 2022 hemisphere, Areas.Robot
        (r 1.0 m about the shoulder), do not all qualify; the check stops at the first failing point and names
        its axis and criterion. Eight points around the good point (Part A's cuboid) all qualify.
    """

    area = configuration.Areas.Robot
    points, _ = Lattice(area.Centre, area.Radius, 0.05, area.Centre[2])
    checked, failed = Check_Area(points, axes, Catch_Context_Str=context)
    assert len(failed) == 1 and checked <= len(points) and failed[0][2] in CONST_CRITERIA
    box = np.array(CONST_GOOD_POINT) + 0.05 * np.array(list(itertools.product((0, 1), repeat=3)))
    assert Check_Area(box, axes, Catch_Context_Str=context) == (8, [])


def test_the_configured_ready_pose_is_a_catch_pose_of_the_cones_centre(context, configuration):
    """
    Description:
        Robot.Ready_Pose passes every criterion of a catch pose, and its cup's axis is the cone's centre
        direction (56 deg, 1 deg) within 1e-6: the search picks it among the centre direction's catch poses.
    """

    from Robot.Core import Forward_Kinematics
    import mujoco
    q = np.array(configuration.Robot.Ready_Pose)
    assert Check_Pose(q, Catch_Context_Str=context).Failed == ''
    _, frame = Forward_Kinematics(context.Model, mujoco.MjData(context.Model), q, Robot_Str=context.Robot)
    centre = Arrival_Axes(configuration.Catch.Arrival_Elevation, configuration.Catch.Arrival_Azimuth)[12]
    assert np.allclose(frame[:, 2], centre, atol=1e-6)


@pytest.mark.slow
def test_the_configured_catch_area_is_the_largest_cuboid_reached_in_time_shrunk_by_one_step(context, configuration):
    """
    Description:
        The pin of Areas.Catch (F4; the rule B800): its faces lie on the lattice
        of Catch.Workspace_Step; grown by one step on every face it is a cuboid of lattice points that all
        qualify for the 25 directions of the configured cone and that the robot reaches from Robot.Ready_Pose,
        at rest, within CONST_REACH_TIME at the nominal acceleration; one more layer beyond each of its six faces
        holds a point that does not, so no face can move out. The ready pose's cup lies inside the area. All
        is re-derived here from the scene, not read from the workspace's files.
    """

    from Catch.Workspace import Derive_Workspace, Reach_Times, Joint_Wrap
    from Robot.Core import Forward_Kinematics
    import mujoco
    step = configuration.Catch.Workspace_Step
    axes = Arrival_Axes(configuration.Catch.Arrival_Elevation, configuration.Catch.Arrival_Azimuth)
    centre, size = np.array(configuration.Areas.Catch.Centre), np.array(configuration.Areas.Catch.Size)
    lower, upper = np.round((centre - size / 2.0) / step).astype(int) - 1, np.round((centre + size / 2.0) / step).astype(int) + 1
    assert np.allclose((centre - size / 2.0) / step, lower + 1, atol=1e-9) and np.allclose((centre + size / 2.0) / step, upper - 1, atol=1e-9)
    grid = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lower, upper)], indexing='ij'), axis=-1).reshape(-1, 3)
    layers = []
    for axis in range(3):
        for side, layer in ((-1, lower[axis] - 1), (1, upper[axis] + 1)):
            face = grid[grid[:, axis] == (lower[axis] if side < 0 else upper[axis])].copy()
            face[:, axis] = layer
            layers.append(face)
    index = np.vstack([grid] + layers)
    w = Derive_Workspace(index * step, index, step, axes, Catch_Context_Str=context)
    q = np.array(configuration.Robot.Ready_Pose)
    reach = Reach_Times(w, q[:5], np.array(configuration.Robot.Joint_Speed[:5]), configuration.Robot.Joint_Accelerations[0], Joint_Wrap(context))
    good = w.Qualifies & (reach <= CONST_REACH_TIME)
    assert np.all(good[:len(grid)]) and len(grid) > 1000
    start = len(grid)
    for number, face in enumerate(layers):
        assert not np.all(good[start:start + len(face)]), number
        start += len(face)
    tcp, _ = Forward_Kinematics(context.Model, mujoco.MjData(context.Model), q, Robot_Str=context.Robot)
    assert np.all(np.abs(tcp - centre) <= size / 2.0 + 1e-9)
