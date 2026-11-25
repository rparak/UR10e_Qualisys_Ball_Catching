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
# File Name: ../test/Catch/test_decision.py
## =========================================================================== ##

# CSV (The decision's files)
import csv
# Math (The closed forms of the intervals)
import math
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Catch/Decision - the truth, the margins, the decision, the rates and the files under test
from Catch.Decision import (Area_Lattice, Reach_Grid, Reach_Bound, Truth_Area, Truth_Samples, Exact_Reach, Catchable, Horizon_Bins,
                            Margin_Errors, Margins, Candidates, Decide, Stage_Rule, Clopper_Pearson, Rates, Write_Margins, Read_Margins,
                            Write_Per_Throw, Write_Summary, CONST_HORIZON_BINS, CONST_MARGIN_COLUMNS, CONST_PER_THROW_COLUMNS,
                            CONST_SUMMARY_COLUMNS)
#   ../Catch/Workspace, Core - a constructed map, the catch pose
from Catch.Workspace import Workspace_Str, CONST_ALLOWANCE
from Catch.Core import Catch_Context, Criteria_Str, Catch_Error_Cls
#   ../Robot/Core, ../Scene/Core, ../Flight/Core - the move time, the robot of the shipped scene, a vacuum flight
from Robot.Core import Move_Time, Forward_Kinematics
from Scene.Core import Scene_Robot
from Flight.Core import Flight_Model_Str

"""
Description:
    Initialization of constants.

    The catch decision (src/Catch/Decision.py; F9) on constructed inputs with known answers; the
    shipped scene only for the exact reach of a handful of samples. Expectations come from the rules of SPEC §7.11,
    closed forms or the owner's decisions, never from the module.
"""
# The UR10e's joint speeds of joints 1 to 5 and the two accelerations (SPEC §4.3; assumed).
#   Unit [Vector<float> 1x5, rad/s], [rad/s^2]
CONST_SPEED = np.radians([120.0, 120.0, 180.0, 180.0, 180.0])
CONST_ACCELERATIONS = [np.radians(800.0), np.radians(400.0)]
CONST_WRAP = np.array([True, True, False, True, True])
# Areas.Catch (SPEC §4.10) and a vacuum flight model.
#   Unit [Vector<float> 1x3, m], [Flight_Model_Str]
CONST_CENTRE, CONST_SIZE = [0.7, -0.175, 0.9], [0.3, 0.75, 0.5]
CONST_VACUUM = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]))


def test_the_area_lattice_holds_every_multiple_of_the_step_with_the_faces():
    """
    Description:
        Areas.Catch, x 0.55-0.85, y -0.55-0.20, z 0.65-1.15 m, holds 7 x 16 x 11 = 1232 points of the 0.05 m lattice, its
        faces included; a box between two lattice planes is refused.
    """

    points, index = Area_Lattice(CONST_CENTRE, CONST_SIZE, 0.05)
    assert len(points) == 1232 and np.array_equal(points, index * 0.05)
    assert index.min(axis=0).tolist() == [11, -11, 13] and index.max(axis=0).tolist() == [17, 4, 23]
    with pytest.raises(Catch_Error_Cls, match='holds no lattice point'):
        Area_Lattice([0.025, 0.0, 1.0], [0.02, 1.0, 1.0], 0.05)


def test_the_reach_grid_takes_each_points_slowest_axis_of_its_fastest_pose_and_infinity_elsewhere():
    """
    Description:
        On a constructed map of two lattice points, the first qualifying with two axes (each two branches) and the
        second not: the grid cell of the first holds the slower axis's faster branch from the start (Robot.Core.Move_Time),
        the second's cell infinity; the grid's first cell is the smallest index.
    """

    start = np.zeros(5)
    poses = np.full((1, 2, 8, 5), np.nan, dtype=np.float32)
    poses[0, 0, 0], poses[0, 0, 1] = [0.5, 0, 0, 0, 0], [0.2, 0, 0, 0, 0]
    poses[0, 1, 0], poses[0, 1, 1] = [0, 0.9, 0, 0, 0], [0, 0.6, 0, 0, 0]
    index = np.array([[3, 1, 2], [4, 1, 2]])
    w = Workspace_Str(Step=0.05, Points=index * 0.05, Index=index, Axes=np.zeros((2, 3)), Qualifies=np.array([True, False]),
                      Failed_Axis=np.array([-1, 0], dtype=np.int8), Failed_Criterion=np.array([-1, 0], dtype=np.int8), Rows=np.array([0]),
                      Poses=poses)
    grid, low = Reach_Grid(w, start, CONST_SPEED, CONST_ACCELERATIONS[0], CONST_WRAP)
    expected = max(Move_Time(start, np.array([0.2, 0, 0, 0, 0]), CONST_SPEED, CONST_ACCELERATIONS[0], CONST_WRAP),
                   Move_Time(start, np.array([0, 0.6, 0, 0, 0]), CONST_SPEED, CONST_ACCELERATIONS[0], CONST_WRAP))
    assert low.tolist() == [3, 1, 2] and grid.shape == (2, 1, 1)
    assert grid[0, 0, 0] == pytest.approx(float(expected), rel=1e-6) and np.isinf(grid[1, 0, 0])


def test_the_reach_bound_is_the_largest_time_within_the_steps_and_infinity_beyond_the_grid():
    """
    Description:
        On a 5 x 1 x 1 grid of 1, 2, 3, 2, 1 s: within 0 steps each cell keeps its time; within 1 step the middle three
        take 3 s and the two end cells infinity (a neighbour beyond the grid is outside the area); within 5 steps every
        cell is infinite. Along an axis of one cell, 1 step is infinite too.
    """

    grid = np.array([1.0, 2.0, 3.0, 2.0, 1.0]).reshape(5, 1, 1)
    assert np.array_equal(Reach_Bound(grid, (0, 0, 0)), grid)
    assert Reach_Bound(grid, (1, 0, 0)).ravel().tolist() == [np.inf, 3.0, 3.0, 3.0, np.inf]
    assert np.all(np.isinf(Reach_Bound(grid, (5, 0, 0)))) and np.all(np.isinf(Reach_Bound(grid, (0, 1, 0))))


def test_the_truth_area_is_the_catch_area_grown_by_one_step_and_holds_the_shell():
    """
    Description:
        F9 Q1 amended: the truth credits a catch in Areas.Catch grown by one 0.05 m step on every face,
        the unshrunk qualifying cuboid of B800: the same centre, 0.4 x 0.85 x 0.6 m. A vacuum throw that passes 0.03 m
        beyond the area's +x face (x 0.88 m) has samples inside the truth area and none inside Areas.Catch.
    """

    centre, size = Truth_Area(CONST_CENTRE, CONST_SIZE, 0.05)
    assert centre == CONST_CENTRE and size == pytest.approx([0.4, 0.85, 0.6], abs=1e-12)
    beside = np.array([[0.88, -4.8, 1.2, 0.0, 4.2, 5.5]])
    assert len(Truth_Samples(beside, np.array([1.5]), centre, size, Flight_Model_Str=CONST_VACUUM)[0]) >= 3
    assert len(Truth_Samples(beside, np.array([1.5]), CONST_CENTRE, CONST_SIZE, Flight_Model_Str=CONST_VACUUM)[0]) == 0


def test_the_true_samples_are_the_descending_ones_inside_the_area():
    """
    Description:
        A vacuum throw from (0.7, -4.8, 1.2) m aimed through the area's centre: every sample returned lies inside
        Areas.Catch (faces included) and descends, its time a multiple of 10 ms from the release; a throw to the other
        side of the robot (x -0.7 m) has none; a throw that rises through the area from (0.7, -0.5, 0.66) m and falls
        back into it gives only its descending samples.
    """

    through = [0.7, -4.8, 1.2, 0.0, 4.2, 5.5]
    beside = [-0.7, -4.8, 1.2, 0.0, 4.2, 5.5]
    rising = [0.7, -0.5, 0.66, 0.0, 1.0, 3.0]
    flight, t, position, velocity = Truth_Samples(np.array([through, beside]), np.array([1.5, 1.5]), CONST_CENTRE, CONST_SIZE,
                                                  Flight_Model_Str=CONST_VACUUM)
    assert len(flight) >= 3 and np.all(flight == 0)
    up_flight, up_t, _, up_velocity = Truth_Samples(np.array([rising]), np.array([1.5]), CONST_CENTRE, CONST_SIZE, Flight_Model_Str=CONST_VACUUM)
    assert len(up_flight) >= 5 and np.all(up_velocity[:, 2] < 0.0) and up_t.min() > 3.0 / 9.81
    assert np.all(np.abs(position - CONST_CENTRE) <= np.array(CONST_SIZE) / 2.0 + 1e-12) and np.all(velocity[:, 2] < 0.0)
    assert np.allclose(t * 100.0, np.round(t * 100.0), atol=1e-9)
    assert position[:, 0] == pytest.approx(0.7) and np.allclose(velocity[:, 2], 5.5 - 9.81 * t)


def test_the_exact_reach_of_the_ready_pose_is_zero_and_infinite_without_a_catch_pose(scene, configuration):
    """
    Description:
        On the shipped scene (F9 Q1): at the ready pose's cup, for the ball arriving along the ready pose's own cup axis
        (the cone's centre), the fastest qualifying pose is the ready pose itself, reached in 0 s at both accelerations;
        a point beyond the robot's reach and a ball rising into the cup (no pose meets the criteria) take infinity.
    """

    robot = Scene_Robot(scene)
    catch = configuration.Catch
    context = Catch_Context(scene, robot, Criteria_Str=Criteria_Str(Joint_Margin=catch.Joint_Margin, Clearance=catch.Clearance,
                                                                    Speed_Margin=catch.Speed_Margin,
                                                                    Joint_Speed=np.array(configuration.Robot.Joint_Speed)))
    ready = np.array(configuration.Robot.Ready_Pose)
    import mujoco
    data = mujoco.MjData(scene)
    tcp, rotation = Forward_Kinematics(scene, data, ready, Robot_Str=robot)
    axis = np.asarray(rotation).reshape(3, 3)[:, 2]
    reach = Exact_Reach(np.array([tcp, [3.0, 0.0, 1.0], tcp]), np.array([-axis * 7.0, [0.0, 7.0, -5.0], axis * 7.0]), ready[:5], CONST_SPEED,
                        CONST_ACCELERATIONS, CONST_WRAP, Catch_Context_Str=context)
    assert reach.shape == (3, 2)
    assert reach[0].tolist() == pytest.approx([0.0, 0.0], abs=2e-3)
    assert np.all(np.isinf(reach[1])) and np.all(np.isinf(reach[2]))


def test_a_throw_is_catchable_only_when_a_sample_is_reached_in_the_time_left():
    """
    Description:
        F9 Q1: catchable when reach <= the sample's time - the decision time - 0.10 s. A sample at 1.0 s reached in
        0.5 s: from a decision at 0.3 s (0.6 s left) catchable, at 0.45 s (0.45 s left) not; a throw without its
        decision time (NaN) or with an infinite reach is not catchable; a throw with two samples needs one.
    """

    assert CONST_ALLOWANCE == 0.10
    flight = np.array([0, 1, 2, 3, 3])
    t = np.array([1.0, 1.0, 1.0, 0.5, 1.0])
    reach = np.array([0.5, 0.5, np.inf, 0.5, 0.5])
    decision = np.array([0.3, 0.45, 0.3, 0.3, np.nan, 0.3])
    assert Catchable(flight, t, reach, decision[[0, 1, 2, 3]]).tolist() == [True, False, False, True]
    assert Catchable(np.array([0]), np.array([1.0]), np.array([0.5]), np.array([np.nan])).tolist() == [False]
    assert Catchable(flight, t, reach, np.array([0.3, 0.3, 0.3, 0.3, 0.3])).tolist() == [True, True, False, True, False]


def test_the_horizon_bins_are_50_ms_wide_and_the_last_holds_every_longer_horizon():
    """
    Description:
        F9 Q6: the margins are per 0.05 s of horizon; 0.049 s is bin 0, 0.051 s bin 1, 0.3 s (on an edge) bin 6, 1.99 s
        bin 39 and 5 s bin 39 (40 bins, 2 s).
    """

    assert CONST_HORIZON_BINS == 40
    assert Horizon_Bins(np.array([0.049, 0.051, 0.3, 1.99, 5.0])).tolist() == [0, 1, 6, 39, 39]


def test_the_errors_split_into_the_cross_track_part_and_the_time_the_ball_is_early():
    """
    Description:
        F9 Q6: the truth moving at 7 m/s along +y, descending: a prediction 0.07 m behind it along the track is 0.01 s
        early and has no cross-track error; one 0.07 m ahead is 0 s early; one 0.03 m off in x has 0.03 m cross-track in
        x. A truth sample that rises, one below the floor and one at h <= 0 are left out.
    """

    truth = np.array([[0.7, 0.0, 1.0, 0.0, 7.0, -1e-9]] * 3 + [[0.7, 0.0, 1.0, 0.0, 7.0, 1.0], [0.7, 0.0, 0.02, 0.0, 7.0, -1.0],
                                                                [0.7, 0.0, 1.0, 0.0, 7.0, -1.0]])
    predicted = truth.copy()
    predicted[0, 1] -= 0.07
    predicted[1, 1] += 0.07
    predicted[2, 0] += 0.03
    bins, cross, early = Margin_Errors(predicted, truth, np.array([0.12, 0.3, 0.6, 0.5, 0.5, 0.0]))
    assert bins.tolist() == [2, 6, 12]
    assert early == pytest.approx([0.01, 0.0, 0.0], abs=1e-12)
    assert cross[0] == pytest.approx([0, 0, 0], abs=1e-9) and cross[2] == pytest.approx([0.03, 0, 0], abs=1e-12)


def test_the_margins_are_the_quantile_per_bin_and_infinite_below_50_errors():
    """
    Description:
        F9 Q6 (q 0.95, Part B: a bin with fewer than 50 errors makes no go): bin 3 holds 100 errors 0.001-0.100 m in x
        and 0-0.099 s early, so its margin is their 0.95 quantile (0.09505 m, 0.09405 s); bin 4 with 49 errors and every
        other bin take infinity; the counts are returned.
    """

    values = np.arange(1, 101) / 1000.0
    bins = np.array([3] * 100 + [4] * 49)
    cross = np.zeros((149, 3))
    cross[:100, 0] = values
    early = np.concatenate([values - 0.001, np.zeros(49)])
    cross_margin, time_margin, counts = Margins(bins, cross, early, 0.95)
    assert cross_margin.shape == (40, 3) and counts[3] == 100 and counts[4] == 49
    assert cross_margin[3] == pytest.approx([0.09505, 0.0, 0.0]) and time_margin[3] == pytest.approx(0.09405)
    assert np.all(np.isinf(cross_margin[4])) and np.isinf(time_margin[4]) and np.all(np.isinf(np.delete(time_margin, 3)))


def test_the_candidates_are_the_descending_predicted_samples_inside_the_area_after_the_decision():
    """
    Description:
        Of five predicted samples: one before the decision time, one rising, one below the floor, one outside the
        area and one inside descending; only the last is a candidate, with its horizon from the decision.
    """

    times = np.array([0.2, 0.5, 0.6, 0.7, 0.9])
    states = np.array([[0.7, -0.1, 0.9, 0, 6, -3], [0.7, -0.1, 0.9, 0, 6, 3], [0.7, -0.1, 0.01, 0, 6, -3], [0.0, -0.1, 0.9, 0, 6, -3],
                       [0.7, -0.1, 0.9, 0, 6, -3]], dtype=float)
    h, position = Candidates(times, states, 0.3, CONST_CENTRE, CONST_SIZE)
    assert h == pytest.approx([0.6]) and position.tolist() == [[0.7, -0.1, 0.9]]


def _Uniform_Grid(value=0.3):
    points, index = Area_Lattice(CONST_CENTRE, CONST_SIZE, 0.05)
    low = index.min(axis=0)
    return np.full(index.max(axis=0) - low + 1, value), low


def test_a_go_needs_the_shrunk_area_and_the_conservative_reach_in_the_time_left():
    """
    Description:
        F9 Q6, every cell reached in 0.3 s: a candidate at the area's centre with zero margins goes at h 0.5 s
        (0.5 - 0.1 >= 0.3) and not at 0.35 s; with a time margin of 0.15 s it does not go at 0.5 s; a candidate 0.10 m
        from the x face with a 0.12 m cross-track margin is outside the shrunk area; one cell with infinity within the
        margin's steps blocks a candidate beside it; a throw without a candidate is no-go.
    """

    grid, low = _Uniform_Grid()
    zero, no_time = np.zeros((40, 3)), np.zeros(40)
    centre = np.array([CONST_CENTRE])
    go = Decide(np.array([0, 1]), np.array([0.5, 0.35]), np.vstack([centre, centre]), 3, zero, no_time, grid, low, 0.05, CONST_CENTRE, CONST_SIZE)
    assert go.tolist() == [True, False, False]
    late = np.full(40, 0.15)
    assert Decide(np.array([0]), np.array([0.5]), centre, 1, zero, late, grid, low, 0.05, CONST_CENTRE, CONST_SIZE).tolist() == [False]
    wide = np.zeros((40, 3))
    wide[:, 0] = 0.12
    near_face = np.array([[0.75, -0.175, 0.9]])
    assert Decide(np.array([0]), np.array([0.5]), near_face, 1, wide, no_time, grid, low, 0.05, CONST_CENTRE, CONST_SIZE).tolist() == [False]
    one = np.zeros((40, 3))
    one[:, 1] = 0.05
    blocked = grid.copy()
    cell = np.round(np.array(CONST_CENTRE) / 0.05).astype(int) - low
    blocked[cell[0], cell[1] + 1, cell[2]] = np.inf
    assert Decide(np.array([0]), np.array([0.5]), centre, 1, one, no_time, blocked, low, 0.05, CONST_CENTRE, CONST_SIZE).tolist() == [False]
    assert Decide(np.array([0]), np.array([0.5]), centre, 1, zero, no_time, blocked, low, 0.05, CONST_CENTRE, CONST_SIZE).tolist() == [True]
    empty = Decide(np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros((0, 3)), 2, zero, no_time, grid, low, 0.05, CONST_CENTRE, CONST_SIZE)
    assert empty.tolist() == [False, False]


def test_a_throw_goes_at_the_first_stage_that_says_go_and_is_judged_there():
    """
    Description:
        F9 Q8, stages N = 20 then 40: throw 0 goes at 20 (judged by its truth at 20), throw 1 at 40 only (judged at
        40), throw 2 at both (the first, 20), throw 3 never (judged at 20, the most time left).
    """

    go = [np.array([True, False, True, False]), np.array([False, True, True, False])]
    truth = [np.array([True, True, False, True]), np.array([False, False, True, False])]
    made, stage, judged = Stage_Rule(go, truth)
    assert made.tolist() == [True, True, True, False] and stage.tolist() == [0, 1, 0, -1]
    assert judged.tolist() == [True, False, False, True]


def test_the_clopper_pearson_bounds_match_their_closed_forms():
    """
    Description:
        F9 Q10 and Q11: with no event in n the upper bound is 1 - 0.025^(1/n) (0.3682 % for n = 1000; from n = 3688 at
        or below 0.1 %, at 3687 above it), with n events in n the lower bound is 0.025^(1/n); for 5 of 10 the bounds are 0.187086 and 0.812914
        (the beta quantiles); k > n is refused.
    """

    assert Clopper_Pearson(0, 1000) == (0.0, pytest.approx(1.0 - 0.025 ** (1.0 / 1000), rel=1e-9))
    assert Clopper_Pearson(0, 3688)[1] < 0.001 < Clopper_Pearson(0, 3687)[1]
    assert Clopper_Pearson(20, 20) == (pytest.approx(0.025 ** (1.0 / 20), rel=1e-9), 1.0)
    lower, upper = Clopper_Pearson(5, 10)
    assert (lower, upper) == (pytest.approx(0.187086, abs=1e-6), pytest.approx(0.812914, abs=1e-6))
    assert math.isclose(lower + upper, 1.0, abs_tol=1e-9)
    with pytest.raises(Catch_Error_Cls, match='0 <= k <= n'):
        Clopper_Pearson(3, 2)


def test_the_rates_count_false_go_among_the_uncatchable_and_false_no_go_among_the_catchable():
    """
    Description:
        F9 Q10: of 6 throws, 4 catchable (3 go) and 2 not (1 go): false-go 1 of 2, false-no-go 1 of 4, the false-go share
        among the 4 go decisions 1/4, each rate with its interval; without an uncatchable throw the false-go rate is NaN.
    """

    rates = Rates(np.array([True, True, True, False, True, False]), np.array([True, True, True, True, False, False]))
    assert (rates['throws'], rates['catchable'], rates['not_catchable'], rates['go']) == (6, 4, 2, 4)
    assert (rates['false_go'], rates['false_go_rate'], rates['false_no_go'], rates['false_no_go_rate']) == (1, 0.5, 1, 0.25)
    assert rates['false_go_of_go'] == 0.25
    assert (rates['false_go_lower'], rates['false_go_upper']) == Clopper_Pearson(1, 2)
    alone = Rates(np.array([True, False]), np.array([True, True]))
    assert alone['not_catchable'] == 0 and math.isnan(alone['false_go_rate']) and math.isnan(alone['false_go_upper'])


def test_the_margins_file_reads_back_and_the_files_have_their_columns(tmp_path):
    """
    Description:
        SPEC §9.12: margins.csv read back gives per (seed, N) the margins written, an infinite one included; a file with
        other columns is refused; per_throw.csv and summary.csv have their headers.
    """

    rows = [('0', 20, b, 0.05 * b, 100, 0.01 * b, 0.02, 0.05, 0.001 * b) for b in range(40)]
    rows[39] = ('0', 20, 39, 1.95, 3, np.inf, np.inf, np.inf, np.inf)
    Write_Margins(str(tmp_path / 'margins.csv'), rows)
    read = Read_Margins(str(tmp_path / 'margins.csv'))
    cross, time_margin = read[('0', 20)]
    assert list(read) == [('0', 20)] and cross[3].tolist() == [0.03, 0.02, 0.05] and time_margin[3] == 0.003
    assert np.all(np.isinf(cross[39])) and np.isinf(time_margin[39])
    (tmp_path / 'other.csv').write_text('seed,n\n0,20\n', encoding='ascii')
    with pytest.raises(Catch_Error_Cls, match='does not hold the columns'):
        Read_Margins(str(tmp_path / 'other.csv'))
    Write_Per_Throw(str(tmp_path / 'per_throw.csv'), [('Catch_In', '7', '0', '800', 'true', 'go', '20', 'true_go', '')])
    Write_Summary(str(tmp_path / 'summary.csv'), [])
    for name, columns in (('margins.csv', CONST_MARGIN_COLUMNS), ('per_throw.csv', CONST_PER_THROW_COLUMNS), ('summary.csv', CONST_SUMMARY_COLUMNS)):
        with open(str(tmp_path / name), newline='', encoding='ascii') as file:
            assert tuple(next(csv.reader(file))) == columns
