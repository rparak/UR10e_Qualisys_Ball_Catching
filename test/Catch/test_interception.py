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
# File Name: ../test/Catch/test_interception.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Flight/Core - a vacuum flight for the crossing
from Flight.Core import Flight_Model_Str
#   ../Robot/Core - the forward kinematics
from Robot.Core import Forward_Kinematics
#   ../Robot/Motion - a profile, the profiles' states and the lever bounds
from Robot.Motion import Profile, Profile_State, Lever_Bounds
#   ../Scene/Core - the scene's robot
from Scene.Core import Scene_Robot
#   ../Catch/Core - the catch-pose context and its criteria
from Catch.Core import Catch_Context, Criteria_Str
#   ../Catch/Decision - the decision Candidate_Slack follows
from Catch.Decision import Decide, CONST_HORIZON_BINS
#   ../Catch/Interception
from Catch.Interception import (Ladder_Stage, Candidate_Slack, Wrapped_Goal, Crossing, Moving_Crossing, Plan_Go, Plan_State, Plan_Ticks, Plan_Str,
                                Cup_Depth, Decision_Lines, Summary_Rows, Write_Per_Throw, Write_Summary, Interception_Setup_Str,
                                CONST_PER_THROW_COLUMNS, CONST_SUMMARY_COLUMNS, CONST_MOVING_COLUMNS, CONST_MOVING_SUMMARY_COLUMNS)

"""
Description:
    Initialization of constants.

    The interception (SPEC §7.11; F10).
"""
# The shipped ladder (Interception.Stages, SPEC §8.1).
#   Unit [List<int>]
CONST_LADDER = [20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150]
# The robot's limits and the ready pose (SPEC §8.1).
#   Unit [rad/s], [rad/s^2], [rad]
CONST_SPEED = np.radians([120.0, 120.0, 180.0, 180.0, 180.0, 180.0])
CONST_ACCELERATION = np.radians(800.0)
CONST_READY = np.radians([29.972673, -104.916569, -88.564625, -57.375802, -61.349913, 0.0])
# The arrival cone's centre direction (Catch.Arrival_Elevation 44-68 deg, Arrival_Azimuth -25-27 deg): elevation 56 deg,
# azimuth 1 deg, the ball's velocity of 6 m/s along it.
#   Unit [Vector<float> 1x3, m/s]
CONST_VELOCITY = 6.0 * np.array([np.cos(np.radians(56.0)) * np.sin(np.radians(1.0)), np.cos(np.radians(56.0)) * np.cos(np.radians(1.0)),
                                 -np.sin(np.radians(56.0))])


def test_a_decision_at_n_samples_takes_the_largest_stage_at_or_below_n():
    """
    Description:
        Q3: below the first stage there is no decision (-1); at a stage its own margins; between two stages the lower's;
        beyond the last the last's.
    """

    cases = {19: -1, 20: 20, 24: 20, 25: 25, 49: 40, 50: 50, 149: 120, 150: 150, 400: 150}
    assert {n: Ladder_Stage(n, CONST_LADDER) for n in cases} == cases


def test_the_candidates_that_pass_are_exactly_where_decide_goes():
    """
    Description:
        Candidate_Slack follows Catch.Decision.Decide: on 300 random sets of candidates (horizons, positions in and
        around a 0.3 m cube, random margins, a random reach grid) a set has a passing candidate exactly when Decide goes,
        a candidate's slack is >= 0 exactly when it passes, and one outside the shrunk cube has the slack -inf.
    """

    rng = np.random.default_rng(20261121)
    centre, size, step = np.array([0.7, -0.175, 0.9]), np.array([0.3, 0.3, 0.3]), 0.05
    low = np.round((centre - size / 2) / step).astype(np.int64)
    for _ in range(300):
        grid = rng.uniform(0.2, 0.9, (7, 7, 7))
        cross, time = rng.uniform(0.0, 0.06, (CONST_HORIZON_BINS, 3)), rng.uniform(0.0, 0.02, CONST_HORIZON_BINS)
        h = rng.uniform(0.05, 1.2, 12)
        position = centre + rng.uniform(-0.2, 0.2, (12, 3))
        passing, slack = Candidate_Slack(h, position, cross, time, grid, low, step, centre, size, {})
        go = Decide(np.zeros(12, dtype=np.int64), h, position, 1, cross, time, grid, low, step, centre, size, bounds={})
        assert bool(passing.any()) == bool(go[0])
        assert np.array_equal(passing, slack >= 0.0)
    outside = Candidate_Slack(np.array([0.5]), (centre + 0.2)[None], cross, time, grid, low, step, centre, size, {})
    assert (bool(outside[0][0]), outside[1][0]) == (False, -np.inf)


def test_a_wrapping_joints_goal_is_reached_the_nearer_way_round():
    """
    Description:
        A goal 350 deg ahead on a wrapping joint is reached 10 deg back; a non-wrapping joint (the elbow) keeps its angle.
    """

    wrap = np.array([True, True, False, True, True, False])
    goal = Wrapped_Goal(np.radians([350.0, 0.0, 170.0, -350.0, 0.0, 0.0]), np.zeros(6), wrap)
    assert np.degrees(goal) == pytest.approx([-10.0, 0.0, 170.0, 10.0, 0.0, 0.0], abs=1e-9)


def test_the_crossing_of_the_cups_mouth_plane_gives_its_time_the_lateral_miss_and_the_angle():
    """
    Description:
        Q1: in vacuum (the drag 0) a ball from a known release crosses, at 0.6 s, the plane of a cup whose mouth lies
        0.020 m from the trajectory across the velocity: the crossing's time is 0.6 s, the lateral miss 0.020 m and the
        angle 0 (within 1e-6); the same cup tilted by 5 deg about an axis normal to the velocity gives 5 deg; a cup the
        ball never reaches (behind the release) has no crossing (NaN).
    """

    model = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.0)
    release = np.array([0.0, -4.5, 1.3, 0.2, 6.0, 4.0])
    t_star = 0.6
    p = release[:3] + release[3:] * t_star + 0.5 * np.array([0.0, 0.0, -9.81]) * t_star ** 2
    v = release[3:] + np.array([0.0, 0.0, -9.81]) * t_star
    u = v / np.linalg.norm(v)
    normal = np.cross(u, [1.0, 0.0, 0.0])
    normal /= np.linalg.norm(normal)
    t_c, lateral, angle = Crossing(release, 1.5, 0.1, p + 0.020 * normal, -u, model)
    assert (t_c, lateral, angle) == (pytest.approx(t_star, abs=1e-6), pytest.approx(0.020, abs=1e-6), pytest.approx(0.0, abs=1e-6))
    k = np.cross(u, normal)
    tilted = -(np.cos(np.radians(5.0)) * u + np.sin(np.radians(5.0)) * normal)
    _, _, angle_5 = Crossing(release, 1.5, 0.1, p, tilted / np.linalg.norm(tilted), model)
    assert np.degrees(angle_5) == pytest.approx(5.0, abs=1e-6) and abs(k @ tilted) < 1e-12
    assert np.isnan(Crossing(release, 1.5, 0.1, release[:3] - np.array([0.0, 2.0, 0.0]), -u, model)[0])


class _Trajectory_Cls:
    """
    Description:
        A stand-in prediction: the same state (a point in the catch area, the cone's centre velocity) at every time of
        [T_0, T_End].

    Initialization of the Class:
        Args:
            (1) t_0 [float]: Its last observed time [s].
            (2) point [Vector<float> 1x3]: The point [m].
    """

    def __init__(self, t_0, point):
        self.T_0, self.T_End, self.point = t_0, 2.0, np.asarray(point, dtype=np.float64)

    def State(self, t):
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        return np.tile(np.concatenate([self.point, CONST_VELOCITY]), (len(t), 1))


def _Setup(configuration, budget=0.05, guard=0.03, scales=None, stroke_budget=0.10):
    criteria = Criteria_Str(Joint_Margin=configuration.Catch.Joint_Margin, Clearance=configuration.Catch.Clearance,
                            Speed_Margin=configuration.Catch.Speed_Margin, Joint_Speed=np.array(configuration.Robot.Joint_Speed))
    setup = Interception_Setup_Str(Areas=configuration.Areas, Criteria=criteria, Ready=CONST_READY, Speed=CONST_SPEED,
                                   Accelerations=[CONST_ACCELERATION], Latency=0.004, Compute_Budget=budget, Guard=guard, Path_Step=0.0025)
    if scales is not None:
        setup.Catch_Mode, setup.Stroke_Scales, setup.Stroke_Budget = 'moving', list(scales), stroke_budget
    return setup


def _Plan(scene, setup, t_hat, samples=120, rate=300.0):
    context = Catch_Context(scene, Scene_Robot(scene), Criteria_Str=setup.Criteria)
    lever = Lever_Bounds(scene, context.Pairs, Robot_Str=context.Robot)
    t_obs = 0.2 + np.arange(samples) / rate
    asked = []

    def Predict(k):
        asked.append(k)
        return _Trajectory_Cls(t_obs[k - 1], [0.75, -0.05, 1.0])

    wrap = np.array([True, True, False, True, True, False])
    plan = Plan_Go(t_obs, 20, t_hat, Predict, CONST_ACCELERATION, wrap, lever, Interception_Setup_Str=setup, Catch_Context_Str=context)
    return plan, t_obs, asked, context


def test_plans_start_at_the_newest_sample_seen_one_at_a_time_and_end_at_rest_at_the_goal(scene, configuration):
    """
    Description:
        Q2, Q8: after a go at the 20th sample (300 Hz from 0.2 s) with the catch time 0.9 s, the first plan is ready the
        compute budget after the 20th sample is seen (its capture time + 4 ms), at the next 2 ms tick; each later plan
        the budget after the previous one, on the newest sample seen then; no plan is ready later than the catch time
        minus the guard; the robot rests at the cup's catch pose for the point (its mouth there, its axis against the
        velocity) before the catch time minus the guard.
    """

    setup = _Setup(configuration)
    plan, t_obs, asked, context = _Plan(scene, setup, 0.9)
    starts = np.array([p.Start_Time for p in plan.Profiles])
    first = np.ceil((t_obs[19] + 0.004 + 0.05) / 0.002 - 1e-9) * 0.002
    assert starts[0] == pytest.approx(first, abs=1e-12) and len(starts) == len(asked) and plan.Rejected == 0
    assert np.all(np.diff(starts) >= 0.05 - 1e-12) and starts[-1] < 0.9 - 0.03
    assert asked[0] == 20
    for k, start in zip(asked[1:], starts[:-1]):
        assert t_obs[k - 1] + 0.004 <= start + 1e-12 and (k == len(t_obs) or t_obs[k] + 0.004 > start)
    last = plan.Profiles[-1]
    assert last.Start_Time + last.Duration <= 0.9 - 0.03
    point, orientation = Forward_Kinematics(scene, context.Data, last.Goal, Robot_Str=context.Robot)
    assert np.linalg.norm(point - [0.75, -0.05, 1.0]) < 1e-9
    assert orientation[:, 2] == pytest.approx(-CONST_VELOCITY / 6.0, abs=1e-9)
    t, q, qd, _ = Plan_Ticks(plan, 0.9, 0.002)
    assert np.array_equal(q[-1], last.Goal) and not qd[-1].any() and t[0] == starts[0]


def test_no_plan_is_accepted_when_the_robot_cannot_rest_the_guard_before_the_catch_time(scene, configuration):
    """
    Description:
        Q9: with the catch time 0.15 s after the go's plan is ready the move from the ready pose (about 0.3 s at 800
        deg/s^2) cannot end the guard before it: no profile, every goal rejected; with a guard and a budget of 0 s the
        same catch time still allows none, and 1 s later one is accepted.
    """

    setup = _Setup(configuration)
    t_hat = 0.2 + 19 / 300.0 + 0.004 + 0.05 + 0.15
    plan, _, asked, _ = _Plan(scene, setup, t_hat)
    assert plan.Profiles == [] and plan.Rejected == len(asked) >= 1
    later, _, _, _ = _Plan(scene, setup, t_hat + 1.0)
    assert len(later.Profiles) >= 1


def _Cup(scene, context, plan, t):
    """
    Description:
        The cup on a plan at a time, from the forward kinematics alone: the mouth's centre, the cup's axis and the
        mouth's velocity (a central difference over 2e-6 s).
    """

    q, _, _ = Plan_State(plan, np.array([t - 1e-6, t, t + 1e-6]))
    points = [Forward_Kinematics(scene, context.Data, row, Robot_Str=context.Robot) for row in q]
    return points[1][0], points[1][1][:, 2], (points[2][0] - points[0][0]) / 2e-6


def test_the_cups_depth_is_the_mouth_less_the_bottoms_face_and_the_balls_radius(scene):
    """
    Description:
        F15: a ball on the cup's bottom has its centre 0.190 - 0.005 - 0.030 = 0.155 m below the mouth's plane (SPEC
        §4.4: the cup's length, the bottom's face, the ball's radius), read from the scene's cup and ball.
    """

    assert Cup_Depth(scene) == pytest.approx(0.155, abs=1e-9)


def test_a_moving_plan_rests_at_the_strokes_start_and_peaks_at_the_predicted_bottom_contact(scene, configuration):
    """
    Description:
        F15: in moving mode, after a go at the 20th sample with the catch time 0.9 s, the
        first plan is ready the stroke's compute budget (0.10 s) after the sample is seen, each later one 0.10 s after
        the one before, and none is ready after the stroke starts. The last profile is the stroke, at the full scale;
        the profile before it ends at rest in the stroke's own start joints at or before the stroke begins. At the peak
        one joint is at its speed limit (the largest speed the joint speeds allow) and the fastest joint brakes at once
        at the acceleration limit (the peak in the stroke's middle). The peak's joint velocities, taken at the catch
        pose by the forward kinematics alone, move the cup along the ball's velocity (within 1e-6 of the speed) at w
        between 1.5 and 2.5 m/s with its axis held; the peak is 0.155 / (6 - w) s after the catch time, the ball's way
        to the bottom at the relative speed, and the mouth is then within 5 mm of where a cup moving at w from the
        catch point would be. On the joint line the cup's velocity is the ball's direction only at the catch pose: at
        the catch time it is within 1 deg of it, at the peak within 10 deg (5.7 deg here: the Jacobian has changed
        over the 0.06 rad to the peak). The plan holds one catch pose per profile, the last with the mouth at the
        predicted point and the cup's axis against the velocity.
    """

    setup = _Setup(configuration, scales=[1.0, 0.75, 0.5, 0.25])
    plan, t_obs, asked, context = _Plan(scene, setup, 0.9)
    starts = np.array([p.Start_Time for p in plan.Profiles])
    stroke, approach = plan.Profiles[-1], plan.Profiles[-2]
    first = np.ceil((t_obs[19] + 0.004 + 0.10) / 0.002 - 1e-9) * 0.002
    assert plan.Stroke and plan.Scale == 1.0 and plan.Rejected == 0 and len(plan.Poses) == len(plan.Profiles) == len(asked) + 1
    assert starts[0] == pytest.approx(first, abs=1e-12) and np.all(np.diff(starts[:-1]) >= 0.10 - 1e-12) and starts[-2] < stroke.Start_Time
    assert approach.Start_Time + approach.Duration <= stroke.Start_Time and np.array_equal(approach.Goal, stroke.Start)
    assert not stroke.Velocity.any() and stroke.Durations[0, 1] == 0.0
    peak_time = stroke.Start_Time + 0.5 * stroke.Duration
    _, qd, qdd = Plan_State(plan, np.array([peak_time - 1e-9]))
    assert np.max(np.abs(qd[0]) / CONST_SPEED) == pytest.approx(1.0, abs=1e-6) and np.max(np.abs(qdd)) == pytest.approx(CONST_ACCELERATION, rel=1e-9)
    u = CONST_VELOCITY / 6.0
    catch_point, orientation = Forward_Kinematics(scene, context.Data, plan.Poses[-1], Robot_Str=context.Robot)
    assert np.linalg.norm(catch_point - [0.75, -0.05, 1.0]) < 1e-9 and orientation[:, 2] == pytest.approx(-u, abs=1e-9)
    moved, turned = Forward_Kinematics(scene, context.Data, plan.Poses[-1] + 1e-6 * qd[0], Robot_Str=context.Robot)
    w = float(np.linalg.norm(moved - catch_point)) / 1e-6
    assert 1.5 < w < 2.5 and np.allclose((moved - catch_point) / 1e-6, w * u, rtol=0.0, atol=1e-6 * w + 1e-5)
    assert np.allclose(turned[:, 2], orientation[:, 2], rtol=0.0, atol=1e-9)
    inside = 0.155 / (6.0 - w)
    assert peak_time == pytest.approx(0.9 + inside, abs=1e-6)
    point, _, velocity = _Cup(scene, context, plan, peak_time - 2e-6)
    assert np.linalg.norm(point - (np.array([0.75, -0.05, 1.0]) + w * inside * u)) < 0.005
    angle = (lambda v: np.degrees(np.arccos(np.clip(v @ u / np.linalg.norm(v), -1.0, 1.0))))
    assert angle(_Cup(scene, context, plan, 0.9)[2]) < 1.0 and 1.0 < angle(velocity) < 10.0
    t, q, qd, _ = Plan_Ticks(plan, stroke.Start_Time + stroke.Duration + 0.002, 0.002)
    assert np.array_equal(q[-1], stroke.Goal) and not qd[-1].any()


def test_a_moving_plan_takes_the_first_scale_in_time_and_without_one_the_static_plan(scene, configuration):
    """
    Description:
        F15: with the ladder 1, 0.75, 0.5, 0.25 the stroke's scale falls as the catch time
        nears: 1 at 0.9 s, 0.75 at 0.8 s, 0.5 at 0.75 s, 0.25 at 0.7 s (the stroke's run-up T = scale x 0.225 s must
        start after the robot rests at its start); with the scale 0.5 alone a catch at 0.9 s has the scale 0.5 (the
        scales are tried in their order, not the largest possible). With the scale 1 alone a catch at 0.75 s has no
        stroke in time, and the plan is the static one: no stroke, its profiles those of the static mode planned with
        the same budget (0.10 s), ending at rest in the catch pose the guard before the catch time. At 0.65 s neither
        is in time: no profile.
    """

    ladder = [1.0, 0.75, 0.5, 0.25]
    scales = {t_hat: _Plan(scene, _Setup(configuration, scales=ladder), t_hat)[0].Scale for t_hat in (0.9, 0.8, 0.75, 0.7)}
    assert scales == {0.9: 1.0, 0.8: 0.75, 0.75: 0.5, 0.7: 0.25}
    half = _Plan(scene, _Setup(configuration, scales=[0.5]), 0.9)[0]
    assert half.Stroke and half.Scale == 0.5 and half.Profiles[-1].Duration == pytest.approx(0.225, abs=1e-3)
    fallback = _Plan(scene, _Setup(configuration, scales=[1.0]), 0.75)[0]
    static = _Plan(scene, _Setup(configuration, budget=0.10), 0.75)[0]
    assert not fallback.Stroke and fallback.Scale == 0.0 and len(fallback.Profiles) == len(static.Profiles) >= 1
    assert len(fallback.Poses) == len(fallback.Profiles) and static.Poses == []
    for mine, theirs in zip(fallback.Profiles, static.Profiles):
        assert mine.Start_Time == theirs.Start_Time and np.array_equal(mine.Goal, theirs.Goal) and np.array_equal(mine.Durations, theirs.Durations)
    last = fallback.Profiles[-1]
    assert last.Start_Time + last.Duration <= 0.75 - 0.03 and np.array_equal(fallback.Poses[-1], last.Goal)
    none = _Plan(scene, _Setup(configuration, scales=ladder), 0.65)[0]
    assert none.Profiles == [] and not none.Stroke and none.Poses == [] and none.Rejected >= 1


def test_the_crossing_of_a_moving_cups_mouth_plane_gives_the_cups_and_the_relative_speed(scene, configuration):
    """
    Description:
        F15: a ball without gravity and drag flies at 6 m/s along the cone's centre direction
        through a point 10 mm beside the predicted catch point at the catch time 0.9 s, and the robot runs the moving
        plan for that prediction. The crossing of the moving mouth plane, searched from the stroke's start: within 2 ms
        of the catch time and inside the stroke; the lateral miss 10 mm within 0.5 mm; the relative velocity within
        0.5 deg of the cup's axis; the cup's speed along the ball that of the forward kinematics at the crossing
        (within 1e-3 m/s), between 1 and 2.5 m/s and less than at the peak; the relative speed 6 m/s less that; the
        cup's velocity within 1 deg of the ball's. For a cup at rest (the static plan) the crossing's time, lateral
        miss and angle are Crossing's own (within 1e-9), the cup's speed 0, the relative speed 6 m/s. A ball flying
        away from the cup has no crossing (NaN).
    """

    from Flight.Core import Flight_Model_Str as Model_Str
    still = Model_Str(Gravity=np.zeros(3), Drag=0.0)
    u = CONST_VELOCITY / 6.0
    side = np.cross(u, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    release = np.concatenate([np.array([0.75, -0.05, 1.0]) + 0.010 * side - 0.9 * CONST_VELOCITY, CONST_VELOCITY])
    plan, _, _, context = _Plan(scene, _Setup(configuration, scales=[1.0, 0.75, 0.5, 0.25]), 0.9)
    stroke = plan.Profiles[-1]
    t_c, lateral, angle, cup_speed, relative_speed, direction = Moving_Crossing(release, 2.0, stroke.Start_Time, plan, scene, context.Data, still,
                                                                               Robot_Str=context.Robot)
    assert abs(t_c - 0.9) < 0.002 and stroke.Start_Time < t_c < stroke.Start_Time + 0.5 * stroke.Duration
    assert lateral == pytest.approx(0.010, abs=5e-4) and np.degrees(angle) < 0.5 and np.degrees(direction) < 1.0
    _, _, velocity = _Cup(scene, context, plan, t_c)
    assert cup_speed == pytest.approx(float(velocity @ u), abs=1e-3) and 1.0 < cup_speed < 2.5
    assert cup_speed < np.linalg.norm(_Cup(scene, context, plan, stroke.Start_Time + 0.5 * stroke.Duration - 2e-6)[2])
    assert relative_speed == pytest.approx(6.0 - cup_speed, abs=1e-3)
    rest, _, _, _ = _Plan(scene, _Setup(configuration), 0.9)
    goal = rest.Profiles[-1].Goal
    point, orientation = Forward_Kinematics(scene, context.Data, goal, Robot_Str=context.Robot)
    expected = Crossing(release, 2.0, rest.Profiles[-1].Start_Time + rest.Profiles[-1].Duration, point.copy(), orientation[:, 2].copy(), still)
    held = Plan_Str(Profiles=[Profile(goal, np.zeros(6), goal, CONST_SPEED, CONST_ACCELERATION, start_time=0.5)])
    found = Moving_Crossing(release, 2.0, 0.5, held, scene, context.Data, still, Robot_Str=context.Robot)
    assert found[:3] == pytest.approx(expected, abs=1e-9) and found[3] == 0.0 and found[4] == pytest.approx(6.0, abs=1e-12) and found[5] == 0.0
    away = np.concatenate([release[:3], -CONST_VELOCITY])
    assert all(np.isnan(value) for value in Moving_Crossing(away, 2.0, 0.5, held, scene, context.Data, still, Robot_Str=context.Robot))


def _Row(**values):
    row = dict.fromkeys(CONST_PER_THROW_COLUMNS, '')
    row.update(dataset='Set', flight='0', seed='0', acceleration_deg_s2='800', refused='', **values)
    return tuple(row[name] for name in CONST_PER_THROW_COLUMNS)


def _Go(catchable, intercepted, certified='true', within='true', at_rest='true', crossing=0.9, lateral=0.01):
    return _Row(catchable=catchable, decision='go', go_n='25', outcome='true_go' if catchable == 'true' else 'false_go',
                stages_decision='no-go', stages_outcome='false_no_go' if catchable == 'true' else 'true_no_go', plans=3, rejected=1,
                arrival_s=0.85, crossing_s=crossing, margin_s=crossing - 0.85, lateral_m=lateral, angle_deg=0.5, clear_environment_m=0.12,
                clear_self_m=0.058, clear_exempt_m=0.046, poses=400, certified=certified, within_limits=within, at_rest=at_rest,
                intercepted=intercepted, tracking_m=0.0001, tracking_joint_rad=0.001, force_limit_ticks=0, control_limit_ticks=0)


def test_the_summary_counts_the_rates_the_interceptions_and_r2s_violations():
    """
    Description:
        Q13: of 6 throws (4 catchable at the first stage), the per-sample decision goes on 4 (3 catchable, 1 not: one
        false go of 2 not catchable), 1 catchable is a no-go (false-no-go 1 of 4); F9's stages went on none (their
        false-no-go 4 of 4); 2 goes are intercepted (2 of the 4 catchable, 2 of the 4 goes); R2's violations are the
        go not certified and the go not at rest before its crossing (2), not the intercepted ones; the goes after N 40
        are counted (4, each at N 25 is not: 0).
    """

    rows = [_Go('true', 'true'), _Go('true', 'true'), _Go('true', 'false', certified='false'), _Go('false', 'false', at_rest='false'),
            _Row(catchable='true', decision='no-go', outcome='false_no_go', stages_decision='no-go', stages_outcome='false_no_go'),
            _Row(catchable='false', decision='no-go', outcome='true_no_go', stages_decision='no-go', stages_outcome='true_no_go')]
    summary = dict(zip(CONST_SUMMARY_COLUMNS, Summary_Rows(rows, 'validation', [20, 40])[0]))
    assert (summary['throws'], summary['catchable'], summary['not_catchable'], summary['go']) == (6, 4, 2, 4)
    assert (summary['false_go'], summary['false_no_go'], summary['false_no_go_rate'], summary['stages_false_no_go_rate']) == (1, 1, 0.25, 1.0)
    assert (summary['intercepted'], summary['intercepted_of_catchable'], summary['intercepted_of_go']) == (2, 0.5, 0.5)
    assert (summary['r2_violations'], summary['go_after_stages'], summary['unplanned'], summary['no_crossing']) == (2, 0, 0, 0)
    assert summary['lateral_max_m'] == 0.01 and summary['margin_min_s'] == pytest.approx(0.05, abs=1e-12)


def _Moving(row, mode='', scale='', in_stroke='', cup='', relative=''):
    return row + (mode, scale, 0.70 if mode == 'moving' else '', 1.15 if mode == 'moving' else '', cup, relative, 0.5 if mode == 'moving' else '',
                  in_stroke)


def test_the_summary_of_a_moving_catch_reads_the_new_r2_and_counts_the_strokes():
    """
    Description:
        F15: rows with the moving columns. Of five planned goes, three have a stroke (scales 1,
        1 and 0.5): one sound; one whose crossing is not inside the stroke; one that did not rest at the stroke's start.
        Two are planned static: one at rest before its crossing; one without a crossing, which F10's R2 does not hold
        to the rest. R2's violations are the two strokes, and a stroke without a crossing inside it is one even though
        it rested (the static rule would pass it). The summary has the five further columns after the others: 3
        strokes, their mean scale 0.8333, the medians of the cup's speed and of the relative speed and its 95th
        percentile over the four goes with a crossing. Rows without the moving columns give the summary without them.
    """

    rows = [_Moving(_Go('true', 'true'), 'moving', 1.0, 'true', 1.4, 5.6),
            _Moving(_Go('true', 'false'), 'moving', 1.0, 'false', 1.2, 5.8),
            _Moving(_Go('true', 'false', at_rest='false'), 'moving', 0.5, 'true', 0.8, 6.2),
            _Moving(_Go('true', 'true'), 'static', 0.0, '', 0.0, 7.0),
            _Moving(_Go('true', 'false', at_rest='false', crossing=float('nan')), 'static', 0.0, '', 0.0, ''),
            _Moving(_Row(catchable='false', decision='no-go', outcome='true_no_go', stages_decision='no-go', stages_outcome='true_no_go'))]
    summary = Summary_Rows(rows, 'validation', [20, 40])[0]
    assert len(summary) == len(CONST_SUMMARY_COLUMNS) + len(CONST_MOVING_SUMMARY_COLUMNS) == 44
    named = dict(zip(CONST_SUMMARY_COLUMNS + CONST_MOVING_SUMMARY_COLUMNS, summary))
    assert (named['go'], named['r2_violations'], named['no_crossing'], named['strokes']) == (5, 2, 1, 3)
    assert named['stroke_scale_mean'] == pytest.approx(2.5 / 3.0, abs=1e-12) and named['cup_speed_p50_m_s'] == pytest.approx(1.0, abs=1e-12)
    assert named['relative_speed_p50_m_s'] == pytest.approx(6.0, abs=1e-12) and named['relative_speed_p95_m_s'] == pytest.approx(6.88, abs=1e-12)
    plain = Summary_Rows([row[:len(CONST_PER_THROW_COLUMNS)] for row in rows], 'validation', [20, 40])[0]
    assert len(plain) == len(CONST_SUMMARY_COLUMNS) and plain[CONST_SUMMARY_COLUMNS.index('r2_violations')] == 1


def test_a_moving_catchs_files_carry_its_columns_last_and_the_decisions_lines_are_their_first_thirteen_fields(tmp_path):
    """
    Description:
        F15: per_throw.csv and summary.csv of a moving catch have the static files' header
        followed by the 8 and the 5 further columns, and only when written as a moving catch's. The decision's lines
        (R1) are the first 13 fields of each line of per_throw.csv, from the dataset to the catch time, as the file
        holds them (a float by its repr), for the moving file and for the static one alike.
    """

    go = _Go('true', 'true')[:12] + (1.27,) + _Go('true', 'true')[13:]
    rows = [_Moving(go, 'moving', 1.0, 'true', 1.4, 5.6),
            _Moving(_Row(catchable='false', decision='no-go', outcome='true_no_go', stages_decision='no-go', stages_outcome='true_no_go'))]
    Write_Per_Throw(str(tmp_path / 'moving.csv'), rows, True)
    Write_Per_Throw(str(tmp_path / 'static.csv'), [row[:len(CONST_PER_THROW_COLUMNS)] for row in rows])
    Write_Summary(str(tmp_path / 'summary.csv'), Summary_Rows(rows, 'validation', [20, 40]), True)
    moving, static = ((tmp_path / name).read_text(encoding='ascii').splitlines() for name in ('moving.csv', 'static.csv'))
    assert static[0] == ','.join(CONST_PER_THROW_COLUMNS) and moving[0] == static[0] + ',' + ','.join(CONST_MOVING_COLUMNS)
    assert CONST_MOVING_COLUMNS == ('plan_mode', 'stroke_scale', 'stroke_start_s', 'stroke_end_s', 'cup_speed_m_s', 'relative_speed_m_s',
                                    'direction_deg', 'in_stroke')
    assert (tmp_path / 'summary.csv').read_text(encoding='ascii').splitlines()[0] == ','.join(CONST_SUMMARY_COLUMNS + CONST_MOVING_SUMMARY_COLUMNS)
    lines = Decision_Lines(rows)
    assert lines == [','.join(line.split(',')[:13]) for line in moving[1:]] == [','.join(line.split(',')[:13]) for line in static[1:]]
    assert lines[0] == 'Set,0,0,800,true,go,25,,true_go,no-go,,false_no_go,1.27' and CONST_PER_THROW_COLUMNS[12] == 'catch_time_s'
