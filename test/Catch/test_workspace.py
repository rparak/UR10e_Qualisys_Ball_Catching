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
# File Name: ../test/Catch/test_workspace.py
## =========================================================================== ##


# System (Default)
import os
# CSV (The summary's file)
import csv
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Catch/Workspace
from Catch.Workspace import (Workspace_Str, Derive_Workspace, Configured_Cuboid, Flight_Samples, Cell_Visits, Through_Share, Box_Share,
                             Reach_Times, Catchable_Share, Reached_Cuboid, Search_Catch_Area, Write_Workspace_Map, Write_Workspace_Summary,
                             CONST_SAMPLE_STEP, CONST_ALLOWANCE)
#   ../Catch/Core - the largest cuboid
from Catch.Core import Largest_Cuboid
#   ../Catch/Core - the catch pose
from Catch.Core import Catch_Context, Criteria_Str, Arrival_Axes, Check_Pose, Catch_Error_Cls, CONST_CRITERIA
#   ../Robot/Core - the move time
from Robot.Core import Move_Time
#   ../Flight/Core - a vacuum flight
from Flight.Core import Flight_Model_Str
#   ../Scene/Core - the robot of the shipped scene
from Scene.Core import Scene_Robot

"""
Description:
    Initialization of constants.

    The workspace map, the reach and the ready pose's search (src/Catch/Workspace.py), on constructed maps and
    flights with known answers; the map of the shipped scene only on a handful of points.
"""
# The UR10e's joint speeds of joints 1 to 5 and the nominal acceleration (SPEC §4.3; assumed).
#   Unit [Vector<float> 1x5, rad/s], [rad/s^2]
CONST_SPEED = np.radians([120.0, 120.0, 180.0, 180.0, 180.0])
CONST_ACCELERATION = np.radians(800.0)
CONST_WRAP = np.array([True, True, False, True, True])


def _Map(poses, index, step=0.1):
    """
    Description:
        A constructed map: the lattice points of `index`, the first len(poses) of them qualifying with the
        given poses (M x D x 8 x 5).
    """

    index = np.asarray(index, dtype=np.int64)
    qualifies = np.zeros(len(index), dtype=bool)
    qualifies[:len(poses)] = True
    return Workspace_Str(Step=step, Points=index * step, Index=index, Axes=np.zeros((np.asarray(poses).shape[1], 3)), Qualifies=qualifies,
                         Failed_Axis=np.where(qualifies, -1, 0).astype(np.int8), Failed_Criterion=np.where(qualifies, -1, 0).astype(np.int8),
                         Rows=np.arange(len(poses)), Poses=np.asarray(poses, dtype=np.float32))


def test_the_map_of_a_few_points_keeps_every_qualifying_pose_and_why_the_others_fail(scene):
    """
    Description:
        On three points of the shipped scene (inside Part A's cuboid, beyond the reach, inside the stand's
        box) the map marks the first as qualifying with every qualifying pose of every axis kept (each passes
        the check, NaN elsewhere), the second as failing at axis 0 for 'ik', the third for a clearance or 'ik'.
    """

    context = Catch_Context(scene, Scene_Robot(scene), Criteria_Str=Criteria_Str(Joint_Margin=np.radians(10.0), Clearance=0.05,
                                                                                 Speed_Margin=0.5,
                                                                                 Joint_Speed=np.radians([120.0] * 2 + [180.0] * 4)))
    axes = Arrival_Axes(np.radians([44.0, 68.0]), np.radians([-25.0, 27.0]))
    index = np.array([[0, -15, 17], [50, 0, 10], [0, 20, 9]])
    w = Derive_Workspace(index * 0.05, index, 0.05, axes, Catch_Context_Str=context)
    assert w.Qualifies.tolist() == [True, False, False] and w.Rows.tolist() == [0]
    assert w.Poses.shape == (1, 25, 8, 5) and w.Poses.dtype == np.float32
    assert w.Failed_Axis[1] == 0 and CONST_CRITERIA[w.Failed_Criterion[1]] == 'ik' and w.Failed_Criterion[0] == -1
    assert CONST_CRITERIA[w.Failed_Criterion[2]] in ('ik', 'environment', 'self')
    kept = w.Poses[0].reshape(-1, 5)
    finite = np.all(np.isfinite(kept), axis=1)
    assert np.all(np.isfinite(w.Poses[0, :, :, 0]).sum(axis=1) >= 1) and np.isfinite(w.Poses[0, :, :, 0]).sum() > 25
    for q in kept[finite]:
        assert Check_Pose(np.append(q.astype(np.float64), 0.0), Catch_Context_Str=context).Failed == ''


def test_the_configured_cuboid_is_the_largest_shrunk_by_one_step_on_every_face():
    """
    Description:
        The largest cuboid from (-7, -18, 10) to (7, -12, 24) lattice steps of 0.05 m becomes the area from
        (-0.30, -0.85, 0.55) to (0.30, -0.65, 1.15) m: centre (0, -0.75, 0.85), size (0.6, 0.2, 0.6); a
        cuboid three points deep in one axis (one step after shrinking would be 0) is refused.
    """

    centre, size = Configured_Cuboid(np.array([-7, -18, 10]), np.array([7, -12, 24]), 0.05)
    assert centre == pytest.approx([0.0, -0.75, 0.85], abs=1e-12) and size == pytest.approx([0.6, 0.2, 0.6], abs=1e-12)
    assert Configured_Cuboid(np.array([0, 0, 0]), np.array([3, 3, 3]), 0.05)[1] == pytest.approx([0.05] * 3, abs=1e-12)
    with pytest.raises(Catch_Error_Cls, match='too small'):
        Configured_Cuboid(np.array([0, 0, 0]), np.array([4, 2, 4]), 0.05)


def test_the_flight_samples_count_only_the_descent_above_the_floor_before_the_floor_time():
    """
    Description:
        A vacuum throw from (0, -4, 1.5) m at (0, 4, 3) m/s, sampled every 10 ms: the positions are the parabola
        within 1e-9 m, and a sample counts only while the ball descends (after its apex at 0.306 s), its
        centre at or above 0.030 m and not after the floor time given (1.2 s).
    """

    model = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.0, Drag_Std=0.0)
    times, states, valid = Flight_Samples(np.array([[0.0, -4.0, 1.5, 0.0, 4.0, 3.0]]), np.array([1.2]), Flight_Model_Str=model)
    assert np.allclose(np.diff(times), CONST_SAMPLE_STEP) and times[-1] >= 1.2
    z = 1.5 + 3.0 * times - 0.5 * 9.81 * times ** 2
    assert np.allclose(states[0, :, 2], z, atol=1e-9) and np.allclose(states[0, :, 1], -4.0 + 4.0 * times, atol=1e-9)
    expected = (3.0 - 9.81 * times < 0.0) & (z >= 0.030) & (times <= 1.2)
    assert valid[0].tolist() == expected.tolist() and expected.sum() > 50


def test_the_visits_take_each_sample_to_its_nearest_lattice_point_or_drop_it():
    """
    Description:
        Samples at 0.04 and 0.06 m from lattice points go to the nearest point of a 0.1 m lattice; a sample
        whose nearest point is not on the lattice and a sample that does not count are dropped; with a stride
        of 2 a sample goes to the nearest point of the 0.2 m sub-lattice.
    """

    index = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]])
    w = _Map(np.zeros((0, 1, 8, 5)), index)
    positions = np.array([[[0.04, 0.0, 0.0], [0.06, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.11, 0.0], [0.13, 0.0, 0.0]]])
    valid = np.array([[True, True, True, True, False]])
    flight, row, sample = Cell_Visits(positions, valid, w)
    assert flight.tolist() == [0, 0, 0] and row.tolist() == [0, 1, 3] and sample.tolist() == [0, 1, 3]
    _, row, _ = Cell_Visits(np.array([[[0.09, 0.0, 0.0], [0.11, 0.0, 0.0]]]), np.array([[True, True]]), w, stride=2)
    assert row.tolist() == [0, 2]


def test_the_shares_count_each_throw_once():
    """
    Description:
        Of 4 throws, 2 pass through the chosen cells (one of them twice): 0.5. A box from x 0 to 1 m holds the
        counting samples of 2 throws (one on its face; a sample 1e-9 m beyond it and a sample that does not
        count are not inside): 0.5; a box from x 0 to 0.5 m holds 1: 0.25.
    """

    visit_flight, visit_row = np.array([0, 0, 1, 2]), np.array([3, 3, 4, 1])
    cells = np.zeros(5, dtype=bool)
    cells[[3, 4]] = True
    assert Through_Share(visit_flight, visit_row, cells, 4) == 0.5
    positions = np.array([[[1.0, 0.0, 0.0]], [[1.0 + 1e-9, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]])
    valid = np.array([[True], [True], [False], [True]])
    assert Box_Share(positions, valid, (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)) == 0.5
    assert Box_Share(positions, valid, (0.25, 0.0, 0.0), (0.5, 1.0, 1.0)) == 0.25


def test_the_reach_time_is_the_slowest_axis_of_its_fastest_pose():
    """
    Description:
        For a point with two axes, the reach time is the larger over the axes of the smaller over each axis'
        qualifying poses, as Robot.Core.Move_Time gives them; a branch that does not qualify (NaN) is never
        the fastest; a point that does not qualify takes infinity; a subset of the rows leaves the others at
        infinity.
    """

    poses = np.full((2, 2, 8, 5), np.nan)
    poses[0, 0, 0] = [0.1, 0, 0, 0, 0]
    poses[0, 0, 1] = [0.5, 0, 0, 0, 0]
    poses[0, 1, 3] = [0.0, 0.0, 1.0, 0, 0]
    poses[1, :, 0] = [0.2, 0, 0, 0, 0]
    w = _Map(poses, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    times = Reach_Times(w, np.zeros(5), CONST_SPEED, CONST_ACCELERATION, CONST_WRAP)
    single = lambda q: float(Move_Time(np.zeros(5), np.array(q), CONST_SPEED, CONST_ACCELERATION, CONST_WRAP))
    assert times[0] == pytest.approx(max(single([0.1, 0, 0, 0, 0]), single([0, 0, 1.0, 0, 0])), rel=1e-6)
    assert times[1] == pytest.approx(single([0.2, 0, 0, 0, 0]), rel=1e-6) and times[2] == np.inf
    assert Reach_Times(w, np.zeros(5), CONST_SPEED, CONST_ACCELERATION, CONST_WRAP, rows=np.array([1]))[0] == np.inf


def test_a_sample_is_catchable_only_when_the_reach_fits_in_the_time_left():
    """
    Description:
        The time left at a visit is its time minus the decision's minus the allowance (0.10 s): a cell whose
        reach time is 1e-9 s less is catchable, 1e-6 s more is not; a throw without its decision sample is
        not; a cell that is not allowed is not.
    """

    times = np.array([0.0, 1.0])
    visit_flight, visit_row, visit_sample = np.array([0, 1, 2]), np.array([0, 1, 0]), np.array([1, 1, 1])
    decision = np.array([0.2, 0.2, np.nan])
    left = 1.0 - 0.2 - CONST_ALLOWANCE
    allowed = np.array([True, True])
    reach = np.array([left - 1e-9, left + 1e-6])
    assert Catchable_Share(visit_flight, visit_row, visit_sample, times, decision, reach, allowed, 3) == pytest.approx(1.0 / 3.0)
    assert Catchable_Share(visit_flight, visit_row, visit_sample, times, decision, reach, np.array([False, True]), 3) == 0.0


def _Stepped_Map():
    """
    Description:
        A constructed map on a 0.1 m lattice, x 0...8, y 0...7, z 0...4 steps, whose points qualify except
        y 4...7 at x 0...3; each has one pose of its single axis (branch 2) with joint 1 = 0.5 x rad, the
        other joints 0. At 120 deg/s and 800 deg/s^2 joint 1 moves 1.0 rad (2 steps of x) in 0.63 s and
        1.5 rad (3 steps) in 0.87 s, so from x0 the points within 2 steps of x are reached within 0.7 s.
    """

    grid = np.stack(np.meshgrid(np.arange(9), np.arange(8), np.arange(5), indexing='ij'), axis=-1).reshape(-1, 3)
    index = grid[~((grid[:, 0] <= 3) & (grid[:, 1] >= 4))]
    poses = np.full((len(index), 1, 8, 5), np.nan)
    poses[:, 0, 2, 0] = 0.5 * index[:, 0]
    poses[:, 0, 2, 1:] = 0.0
    return _Map(poses, index)


def test_the_reached_cuboid_is_the_largest_of_the_points_reached_in_time_shrunk_by_one_step():
    """
    Description:
        From x0 = 6 on the stepped map the points reached within 0.7 s are x 4...8: their largest cuboid is
        x 4...8, y 0...7, z 0...4 steps (200 points, all reached and qualifying), its shrunk volume
        2 x 5 x 2 steps of 0.1 m = 0.020 m^3; on the sub-lattice of every second point the cuboid is
        x 4...8, y 0...6, z 0...4 and the estimate 2 x 4 x 2 = 0.016 m^3; from x0 = 7 (reached x 5...8) the
        sub-lattice holds x 6 and 8 only, a cuboid x 6...8 too thin to shrink: 0. From x0 = 0 the reached points are
        x 0...2, y 0...3: a cuboid 3 points deep in x, too thin to shrink, so 0. A start from which no point is
        reached in time (joint 3 20 rad away) gives (0, None, None).
    """

    w = _Stepped_Map()
    start = np.array([3.0, 0.0, 0.0, 0.0, 0.0])
    volume, lower, upper = Reached_Cuboid(w, start, CONST_SPEED, CONST_ACCELERATION, CONST_WRAP)
    assert lower.tolist() == [4, 0, 0] and upper.tolist() == [8, 7, 4] and volume == pytest.approx(0.020, abs=1e-12)
    within = Reach_Times(w, start, CONST_SPEED, CONST_ACCELERATION, CONST_WRAP) <= 0.7
    assert within.sum() == 200 and [c.tolist() for c in Largest_Cuboid(w.Index, within)] == [[4, 0, 0], [8, 7, 4]]
    volume, lower, upper = Reached_Cuboid(w, start, CONST_SPEED, CONST_ACCELERATION, CONST_WRAP, stride=2)
    assert lower.tolist() == [4, 0, 0] and upper.tolist() == [8, 6, 4] and volume == pytest.approx(0.016, abs=1e-12)
    volume, lower, upper = Reached_Cuboid(w, np.array([3.5, 0.0, 0.0, 0.0, 0.0]), CONST_SPEED, CONST_ACCELERATION, CONST_WRAP, stride=2)
    assert lower.tolist() == [6, 0, 0] and upper.tolist() == [8, 6, 4] and volume == 0.0
    volume, lower, upper = Reached_Cuboid(w, np.zeros(5), CONST_SPEED, CONST_ACCELERATION, CONST_WRAP)
    assert lower.tolist() == [0, 0, 0] and upper.tolist() == [2, 3, 4] and volume == 0.0
    assert Reached_Cuboid(w, np.array([0.0, 0.0, 20.0, 0.0, 0.0]), CONST_SPEED, CONST_ACCELERATION, CONST_WRAP) == (0.0, None, None)


def test_the_joint_search_finds_the_ready_pose_with_the_largest_area_beyond_the_coarse_lattice():
    """
    Description:
        On the stepped map every candidate on the coarse lattice (x 0, 4, 8) scores 0, and its whole-lattice
        volume is 0 too; the refinement around the coarse best (the later positions first: x 8) reaches
        x0 = 6, and the search returns a pose there with the largest cuboid x 4...8, y 0...7, z 0...4 and
        0.020 m^3, the largest over every candidate of the map. A map without a candidate on the coarse
        lattice is refused; so is one from whose best candidates no point is reached in time.
    """

    w = _Stepped_Map()
    ready, row, lower, upper, volume = Search_Catch_Area(w, CONST_SPEED, CONST_ACCELERATION, CONST_WRAP, 0)
    assert ready[0] == pytest.approx(3.0) and np.all(ready[1:] == 0.0) and w.Index[row, 0] == 6
    assert lower.tolist() == [4, 0, 0] and upper.tolist() == [8, 7, 4] and volume == pytest.approx(0.020, abs=1e-12)
    every = [Reached_Cuboid(w, w.Poses[p, 0, 2], CONST_SPEED, CONST_ACCELERATION, CONST_WRAP)[0] for p in range(len(w.Rows))]
    assert max(every) == pytest.approx(volume, abs=1e-12)
    coarse = np.flatnonzero(np.all(w.Index % 4 == 0, axis=1))
    assert max(every[p] for p in coarse) == 0.0
    empty = _Map(np.full((1, 1, 8, 5), 0.0), np.array([[1, 0, 0]]))
    with pytest.raises(Catch_Error_Cls, match='coarse lattice'):
        Search_Catch_Area(empty, CONST_SPEED, CONST_ACCELERATION, CONST_WRAP, 0)
    slow = _Map(np.full((1, 2, 8, 5), 0.0), np.array([[0, 0, 0]]))
    slow.Poses[0, 1] = np.nan
    slow.Poses[0, 1, 0] = [0.0, 0.0, 20.0, 0.0, 0.0]
    with pytest.raises(Catch_Error_Cls, match='reached within 0.70 s'):
        Search_Catch_Area(slow, CONST_SPEED, CONST_ACCELERATION, CONST_WRAP, 0)


def test_the_map_and_the_summary_are_written_the_same_bytes_twice(tmp_path):
    """
    Description:
        The map's file holds its format version, every array of the map and the extra arrays, readable by
        np.load, and two writes of the same map give the same bytes; the summary is a CSV with the columns
        quantity, value, unit, its rows in order.
    """

    poses = np.zeros((1, 2, 8, 5))
    w = _Map(poses, [[0, 0, 0], [1, 0, 0]])
    first, second = str(tmp_path / 'a.npz'), str(tmp_path / 'b.npz')
    for path in (first, second):
        Write_Workspace_Map(path, w, {'ready_pose': np.arange(6.0)})
    with open(first, 'rb') as a, open(second, 'rb') as b:
        assert a.read() == b.read()
    with np.load(first) as data:
        assert int(data['format_version']) == 1 and data['poses'].dtype == np.float32 and data['ready_pose'].tolist() == list(range(6))
        assert data['qualifies'].tolist() == [True, False] and list(data['criteria']) == list(CONST_CRITERIA)
    summary = str(tmp_path / 'summary.csv')
    Write_Workspace_Summary(summary, [('lattice_points', '2', '-'), ('catch_area_volume', '0.1000', 'm^3')])
    with open(summary, newline='', encoding='ascii') as file:
        assert list(csv.reader(file)) == [['quantity', 'value', 'unit'], ['lattice_points', '2', '-'], ['catch_area_volume', '0.1000', 'm^3']]
    assert os.path.getsize(summary) > 0
