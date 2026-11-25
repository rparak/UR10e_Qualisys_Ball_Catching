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
# File Name: ../test/Robot/test_motion.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation) [pip3 install mujoco]
import mujoco
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Robot/Core - the forward kinematics, the move time, the collision pairs, the clearance
from Robot.Core import Forward_Kinematics, Move_Time, Collision_Pairs, Clearance
#   ../Robot/Motion
from Robot.Motion import (Profile, Profile_State, Profile_Extremes, Stroke, Lever_Bounds, Check_Profile, Servos, Servo_Command, Track,
                          Motion_Error_Cls)
#   ../Scene/Core - the robot of the shipped scene
from Scene.Core import Scene_Robot

"""
Description:
    Initialization of constants.

    The robot's motion (SPEC §7.9; F10): the time-optimal joint profile, its path check, the servos' command.
"""
# The UR10e's joint speeds (SPEC §4.3) and the nominal acceleration (800 deg/s^2, assumed).
#   Unit [Vector<float> 1x6, rad/s], [rad/s^2]
CONST_SPEED = np.radians([120.0, 120.0, 180.0, 180.0, 180.0, 180.0])
CONST_ACCELERATION = np.radians(800.0)
# Three profiles from moving starts and the duration of each joint that Ruckig 0.19.4 (MIT; second order, jerk unlimited,
# Synchronization.No, its independent_min_durations) gives at these limits: an independent implementation, run in a
# scratch environment outside the repository (2026-11-21, F10 Part B; 2 000 random cases agreed within 2.2e-15 s).
#   Unit [List<Tuple>: start rad, velocity rad/s, goal rad, durations s]
CONST_RUCKIG = [([0.5, -1.8, -1.5, -1.0, -1.1, 0.0], [0.0, 1.5, -2.0, 3.0, -0.5, 0.0], [1.2, -1.0, -1.6, -1.2, -0.2, 0.0],
                 [0.48422538049298025, 0.46301266374771965, 0.2545372008651518, 0.6016727046167457, 0.5501384180510288, 0.0]),
                ([0.0] * 6, [2.0, -2.0, 3.0, -3.0, 1.0, 0.0], [0.0] * 6,
                 [0.3458106199180547, 0.3458106199180547, 0.5187159298770819, 0.5187159298770819, 0.17290530995902734, 0.0]),
                ([0.3, -0.2, 0.1, 0.4, -0.3, 0.0], [-1.0, 0.5, 2.5, 0.0, 3.1, 0.0], [0.31, -0.25, 0.6, 0.4, -0.25, 0.0],
                 [0.18617782676495553, 0.16576615230591796, 0.27634708936203184, 0.0, 0.5123017569542198, 0.0])]
# The ready pose (Robot.Ready_Pose, SPEC §8.1).
#   Unit [Vector<float> 1x6, rad]
CONST_READY = np.radians([29.972673, -104.916569, -88.564625, -57.375802, -61.349913, 0.0])
# A move of joints 1-3 from the ready pose whose ends are clear and whose path passes 0.046 m into the environment: the
# first of a seeded search (2026-11-21, F10 Part B: numpy default_rng(20261121), joints 1-3 moved by U(-2, 2) rad, both ends
# >= 0.055 m to the environment and between the parts, the deepest of 80 poses on the way below -0.02 m).
#   Unit [Vector<float> 1x6, rad]
CONST_THROUGH_STAND = np.array([1.5164, -1.1694, 1.903, 0.0, 0.0, 0.0])


def test_a_profile_from_rest_takes_each_joints_rest_to_rest_move_time():
    """
    Description:
        From rest the profile is the rest-to-rest trapezoid per joint: each joint's duration equals Robot.Core.Move_Time
        for that joint alone (within 1e-12 s) at 300 random moves, and one hand-computed case holds: 1 rad at 120 deg/s
        and 800 deg/s^2 cruises, 1 / v + v / a = 0.4775 + 0.15 = 0.6275 s.
    """

    rng = np.random.default_rng(20261121)
    for _ in range(300):
        start, goal = rng.uniform(-3.0, 3.0, 6), rng.uniform(-3.0, 3.0, 6)
        profile = Profile(start, np.zeros(6), goal, CONST_SPEED, CONST_ACCELERATION)
        for j in range(6):
            alone = Move_Time(start[j:j + 1], goal[None, j:j + 1], CONST_SPEED[j:j + 1], CONST_ACCELERATION, np.zeros(1, dtype=bool))[0]
            assert profile.Durations[j].sum() == pytest.approx(alone, abs=1e-12)
    one = Profile(np.zeros(1), np.zeros(1), np.ones(1), CONST_SPEED[:1], CONST_ACCELERATION)
    assert one.Duration == pytest.approx(1.0 / CONST_SPEED[0] + CONST_SPEED[0] / CONST_ACCELERATION, abs=1e-12)
    assert one.Duration == pytest.approx(0.6275, abs=1e-4)


def test_a_profile_from_a_moving_start_takes_ruckigs_time_optimal_durations():
    """
    Description:
        From moving starts (a joint moving away from its goal, one cruising, a return to the start, very short moves) each
        joint's duration equals Ruckig's time-optimal one (CONST_RUCKIG, an independent implementation) within 1e-12 s.
    """

    for start, velocity, goal, durations in CONST_RUCKIG:
        profile = Profile(np.array(start), np.array(velocity), np.array(goal), CONST_SPEED, CONST_ACCELERATION)
        assert profile.Durations.sum(axis=1) == pytest.approx(durations, abs=1e-12)
        assert profile.Duration == pytest.approx(max(durations), abs=1e-12)


def test_a_profile_ends_at_rest_at_its_goal_within_the_limits_and_continuous():
    """
    Description:
        At 300 random moving starts the profile's state at its start is the start state, after its end every joint is
        exactly at its goal at rest, its speed never exceeds a joint's limit and its acceleration never the limit, its
        position is continuous (a 0.1 ms step moves it by at most the speed limit times the step), and its extremes
        (Profile_Extremes) bound every sampled position.
    """

    rng = np.random.default_rng(7)
    for _ in range(300):
        start, velocity = rng.uniform(-3.0, 3.0, 6), rng.uniform(-1.0, 1.0, 6) * CONST_SPEED
        goal = start + rng.uniform(-1.0, 1.0, 6) * rng.choice([0.01, 1.0], 6)
        profile = Profile(start, velocity, goal, CONST_SPEED, CONST_ACCELERATION, start_time=0.5)
        t = 0.5 + np.arange(0.0, profile.Duration + 0.01, 1e-4)
        q, qd, qdd = Profile_State(profile, t)
        assert np.allclose(q[0], start, atol=1e-12) and np.allclose(qd[0], velocity, atol=1e-12)
        after, after_v, after_a = Profile_State(profile, np.array([0.5 + profile.Duration + 1e-9]))
        assert np.array_equal(after[0], goal) and not after_v.any() and not after_a.any()
        assert np.all(np.abs(qd) <= CONST_SPEED * (1.0 + 1e-12)) and np.all(np.abs(qdd) <= CONST_ACCELERATION * (1.0 + 1e-12))
        assert np.all(np.abs(np.diff(q, axis=0)) <= CONST_SPEED * 1e-4 * (1.0 + 1e-9) + 1e-12)
        low, high = Profile_Extremes(profile)
        assert np.all(q.min(axis=0) >= low - 1e-12) and np.all(q.max(axis=0) <= high + 1e-12)


def test_a_start_beyond_a_speed_limit_is_refused_naming_the_joint():
    """
    Description:
        A start velocity beyond a joint's speed limit by more than the rounding tolerance is refused (never clipped),
        naming the joint, its velocity and its limit; one exactly at the limit is accepted.
    """

    velocity = np.zeros(6)
    velocity[2] = CONST_SPEED[2] * 1.001
    with pytest.raises(Motion_Error_Cls, match='joint 3 starts at .* beyond its speed limit'):
        Profile(np.zeros(6), velocity, np.ones(6), CONST_SPEED, CONST_ACCELERATION)
    velocity[2] = CONST_SPEED[2]
    assert Profile(np.zeros(6), velocity, np.ones(6), CONST_SPEED, CONST_ACCELERATION).Duration > 0.0


def test_the_extremes_hold_the_point_where_a_joint_moving_away_stops():
    """
    Description:
        A joint moving away from its goal at 2 rad/s brakes before it turns back: its extreme is start + v^2 / (2 a)
        (0.1432 rad beyond the start at 800 deg/s^2), which is neither the start nor the goal, and the sampled profile
        reaches it.
    """

    profile = Profile(np.zeros(1), np.array([2.0]), np.array([-0.5]), CONST_SPEED[:1], CONST_ACCELERATION)
    low, high = Profile_Extremes(profile)
    assert (low[0], high[0]) == (-0.5, pytest.approx(4.0 / (2.0 * CONST_ACCELERATION), abs=1e-15))
    q, _, _ = Profile_State(profile, np.linspace(0.0, profile.Duration, 20001))
    assert q.max() == pytest.approx(high[0], abs=1e-8)


def test_a_stroke_passes_its_peak_with_the_velocity_given_from_rest_to_rest_on_one_joint_line():
    """
    Description:
        F15: at 100 random peaks and velocities within the speed limits a stroke is at its
        peak with the velocity given at the peak time (within 1e-12), at rest at its start and at its end, T = max |v| /
        a before and after the peak; the fastest joint accelerates and brakes at the limit and no joint beyond it, the
        speeds never exceed the peak's; every sampled position lies on the joint line through the peak along the
        velocity (one scalar for all joints, within 1e-12 rad); after its end every joint is exactly at its goal, at
        rest; the extremes are its two ends. One hand-computed case: 2 rad/s at 800 deg/s^2 takes 2 / 13.963 = 0.1432 s
        each way and starts 2 x 0.1432 / 2 = 0.1432 rad before the peak.
    """

    rng = np.random.default_rng(20261123)
    for _ in range(100):
        peak, velocity = rng.uniform(-3.0, 3.0, 6), rng.uniform(-1.0, 1.0, 6) * CONST_SPEED
        stroke = Stroke(peak, velocity, CONST_SPEED, CONST_ACCELERATION, 0.8)
        duration = np.max(np.abs(velocity)) / CONST_ACCELERATION
        assert stroke.Start_Time == pytest.approx(0.8 - duration, abs=1e-12) and stroke.Duration == pytest.approx(2.0 * duration, abs=1e-12)
        q, qd, qdd = Profile_State(stroke, np.array([stroke.Start_Time, 0.8, 0.8 + duration - 1e-9, 0.8 + duration + 1e-9]))
        assert np.allclose(q[1], peak, rtol=0.0, atol=1e-12) and np.allclose(qd[1], velocity, rtol=0.0, atol=1e-12)
        assert not qd[0].any() and np.allclose(qd[2], 0.0, atol=1e-7) and not qd[3].any() and np.array_equal(q[3], stroke.Goal)
        assert np.allclose(q[0], peak - 0.5 * velocity * duration, rtol=0.0, atol=1e-12)
        assert np.allclose(stroke.Goal, peak + 0.5 * velocity * duration, rtol=0.0, atol=1e-12)
        t = stroke.Start_Time + np.linspace(0.0, stroke.Duration, 201)
        q, qd, qdd = Profile_State(stroke, t)
        assert np.max(np.abs(qdd)) == pytest.approx(CONST_ACCELERATION, rel=1e-12)
        assert np.all(np.abs(qdd[:-1]) <= CONST_ACCELERATION * np.abs(velocity) / np.max(np.abs(velocity)) * (1.0 + 1e-12))
        assert np.all(np.abs(qd) <= np.abs(velocity) * (1.0 + 1e-12))
        scalar = ((q - peak) @ velocity) / (velocity @ velocity)
        assert np.allclose(q - peak, scalar[:, None] * velocity[None, :], rtol=0.0, atol=1e-12) and np.all(np.diff(scalar) >= -1e-15)
        low, high = Profile_Extremes(stroke)
        assert np.array_equal(low, np.minimum(stroke.Start, stroke.Goal)) and np.array_equal(high, np.maximum(stroke.Start, stroke.Goal))
    one = Stroke(np.zeros(1), np.array([2.0]), CONST_SPEED[:1], CONST_ACCELERATION, 1.0)
    assert one.Duration / 2.0 == pytest.approx(0.1432, abs=1e-4) and one.Start[0] == pytest.approx(-0.1432, abs=1e-4)


def test_a_stroke_beyond_a_speed_limit_and_one_that_moves_no_joint_are_refused():
    """
    Description:
        A velocity at the peak beyond a joint's speed limit is refused, naming the joint and the limit, never clipped;
        at the limit it is taken; a velocity of 0 for every joint is refused: a stroke passes its peak in motion.
    """

    velocity = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(Motion_Error_Cls, match=r'joint 3 would pass the stroke\'s peak at 3\.2'):
        Stroke(np.zeros(6), np.array([0.5, 0.0, 3.2, 0.0, 0.0, 0.0]), CONST_SPEED, CONST_ACCELERATION, 0.0)
    assert Stroke(np.zeros(6), np.array([0.5, 0.0, CONST_SPEED[2], 0.0, 0.0, 0.0]), CONST_SPEED, CONST_ACCELERATION, 0.0).Duration > 0.0
    with pytest.raises(Motion_Error_Cls, match='moves no joint'):
        Stroke(np.zeros(6), 0.0 * velocity, CONST_SPEED, CONST_ACCELERATION, 0.0)


def test_the_path_check_takes_a_stroke_and_finds_one_through_the_stand(scene):
    """
    Description:
        A stroke is a profile like any other for the path check (F15): along the move of CONST_THROUGH_STAND from the
        ready pose (joints 1-3, the fastest at its speed limit), a stroke whose peak lies at 0.85 of the move has both
        ends clear (>= 0.055 m to the environment and between the parts) and passes into the stand between them: not
        certified, 'environment', the deepest pose below 0 m; the same stroke with its peak at 0.05 of the move is
        certified with its 0.05 m, and its poses are no more than 2.5 mm apart (at least 300 of them for the 0.38 s).
    """

    robot = Scene_Robot(scene)
    pairs = Collision_Pairs(scene, Robot_Str=robot)
    lever = Lever_Bounds(scene, pairs, Robot_Str=robot)
    data = mujoco.MjData(scene)
    velocity = CONST_THROUGH_STAND * np.min(CONST_SPEED[:3] / np.abs(CONST_THROUGH_STAND[:3]))
    through = Stroke(CONST_READY + 0.85 * CONST_THROUGH_STAND, velocity, CONST_SPEED, CONST_ACCELERATION, 1.0)
    for q in (through.Start, through.Goal):
        Forward_Kinematics(scene, data, q, Robot_Str=robot)
        assert Clearance(scene, data, pairs.Environment_Set, 1.0)[0] >= 0.055 and Clearance(scene, data, pairs.Self_Set, 1.0)[0] >= 0.055
    check = Check_Profile(scene, data, through, through.Start_Time, np.inf, lever, 0.0025, 0.05, 1.0, pairs, stop=False, Robot_Str=robot)
    assert (check.Certified, check.Within_Range, check.Failed) == (False, True, 'environment') and check.Environment < 0.0
    clear = Stroke(CONST_READY + 0.05 * CONST_THROUGH_STAND, velocity, CONST_SPEED, CONST_ACCELERATION, 1.0)
    ok = Check_Profile(scene, data, clear, clear.Start_Time, np.inf, lever, 0.0025, 0.05, 1.0, pairs, stop=False, Robot_Str=robot)
    assert ok.Certified and ok.Within_Range and ok.Failed == '' and ok.Environment >= 0.0525 and ok.Poses >= 300


def test_the_lever_bounds_bound_the_motion_of_every_robot_geom(scene):
    """
    Description:
        The lever bounds hold in every pose: at 200 random poses and joint steps of up to 0.01 rad per joint, no robot
        collision geom's centre moves more than sum_j bound_j |dq_j|; and each bound is at least the largest distance
        from its joint's axis measured at 20 000 random poses in F10 Part A (1.544, 1.534, 0.922, 0.351, 0.317, 0.122 m,
        without the bounding radius of the farthest geom, so the bound lies above).
    """

    robot = Scene_Robot(scene)
    pairs = Collision_Pairs(scene, Robot_Str=robot)
    lever = Lever_Bounds(scene, pairs, Robot_Str=robot)
    assert np.all(lever >= np.array([1.544, 1.534, 0.922, 0.351, 0.317, 0.122]) - 1e-3)
    geoms = np.unique(np.concatenate([pairs.Environment[:, 0], pairs.Self.ravel()]))
    data = mujoco.MjData(scene)
    rng = np.random.default_rng(11)
    for _ in range(200):
        q = rng.uniform(-np.pi, np.pi, 6) * [1, 1, 0.95, 1, 1, 1]
        dq = rng.uniform(-0.01, 0.01, 6)
        Forward_Kinematics(scene, data, q, Robot_Str=robot)
        before = data.geom_xpos[geoms].copy()
        Forward_Kinematics(scene, data, q + dq, Robot_Str=robot)
        assert np.max(np.linalg.norm(data.geom_xpos[geoms] - before, axis=1)) <= np.sum(lever * np.abs(dq))


def test_the_path_check_finds_a_collision_between_two_clear_poses(scene):
    """
    Description:
        A profile from the ready pose to CONST_THROUGH_STAND has both ends clear by more than the clearance plus two steps
        (to the environment and between the parts) but passes 0.046 m into the environment on the way: Check_Profile fails
        it 'environment', not certified, stopping at the first pose within the clearance plus a step; checked to its end
        it finds the penetration (below -0.04 m). The forced failure: both ends pass alone (the catch pose's distances of F4), so the
        path check is what finds it. A short move from the ready pose is certified.
    """

    robot = Scene_Robot(scene)
    pairs = Collision_Pairs(scene, Robot_Str=robot)
    lever = Lever_Bounds(scene, pairs, Robot_Str=robot)
    data = mujoco.MjData(scene)
    goal = CONST_READY + CONST_THROUGH_STAND
    for q in (CONST_READY, goal):
        Forward_Kinematics(scene, data, q, Robot_Str=robot)
        assert Clearance(scene, data, pairs.Environment_Set, 1.0)[0] >= 0.055 and Clearance(scene, data, pairs.Self_Set, 1.0)[0] >= 0.055
    profile = Profile(CONST_READY, np.zeros(6), goal, CONST_SPEED, CONST_ACCELERATION)
    check = Check_Profile(scene, data, profile, 0.0, np.inf, lever, 0.0025, 0.05, 0.0, pairs, Robot_Str=robot)
    assert (check.Certified, check.Within_Range, check.Failed) == (False, True, 'environment') and check.Environment < 0.0525
    whole = Check_Profile(scene, data, profile, 0.0, np.inf, lever, 0.0025, 0.05, 1.0, pairs, stop=False, Robot_Str=robot)
    assert not whole.Certified and whole.Environment < -0.04 and whole.Poses > check.Poses
    short = Profile(CONST_READY, np.zeros(6), CONST_READY + 0.02, CONST_SPEED, CONST_ACCELERATION)
    ok = Check_Profile(scene, data, short, 0.0, np.inf, lever, 0.0025, 0.05, 0.0, pairs, Robot_Str=robot)
    assert ok.Certified and ok.Within_Range and ok.Failed == '' and ok.Poses >= 2


def test_the_path_check_keeps_the_clearance_plus_its_steps_at_every_pose(scene):
    """
    Description:
        The certificate's thresholds (Q6): a short move whose poses keep 0.10 m to the environment and 0.058 m between
        the robot's parts passes at the clearance 0.05 m with a 2.5 mm step, and fails 'self' when the clearance asked is
        within two steps of the self distance, so a pose with less than clearance + 2 steps between the parts fails.
    """

    robot = Scene_Robot(scene)
    pairs = Collision_Pairs(scene, Robot_Str=robot)
    lever = Lever_Bounds(scene, pairs, Robot_Str=robot)
    data = mujoco.MjData(scene)
    profile = Profile(CONST_READY, np.zeros(6), CONST_READY + 0.01, CONST_SPEED, CONST_ACCELERATION)
    first = Check_Profile(scene, data, profile, 0.0, np.inf, lever, 0.0025, 0.05, 1.0, pairs, stop=False, Robot_Str=robot)
    assert first.Certified
    edge = first.Self - 2 * 0.0025
    inside = Check_Profile(scene, data, profile, 0.0, np.inf, lever, 0.0025, edge - 1e-6, 1.0, pairs, stop=False, Robot_Str=robot)
    beyond = Check_Profile(scene, data, profile, 0.0, np.inf, lever, 0.0025, edge + 1e-6, 1.0, pairs, stop=False, Robot_Str=robot)
    assert inside.Certified and not beyond.Certified and beyond.Failed in ('self', 'environment')


def test_a_path_beyond_a_joints_range_is_refused(scene):
    """
    Description:
        A profile whose elbow, moving at 3 rad/s towards its +180 deg limit, overshoots it while it brakes (its extreme
        beyond the range, the goal inside) is refused 'range' before any distance is measured.
    """

    robot = Scene_Robot(scene)
    pairs = Collision_Pairs(scene, Robot_Str=robot)
    lever = Lever_Bounds(scene, pairs, Robot_Str=robot)
    start, velocity = CONST_READY.copy(), np.zeros(6)
    start[2], velocity[2] = robot.Upper[2] - 0.05, 3.0
    goal = start.copy()
    goal[2] = robot.Upper[2] - 0.3
    check = Check_Profile(scene, mujoco.MjData(scene), Profile(start, velocity, goal, CONST_SPEED, CONST_ACCELERATION), 0.0, np.inf, lever,
                          0.0025, 0.05, 0.0, pairs, Robot_Str=robot)
    assert (check.Within_Range, check.Certified, check.Failed, check.Poses) == (False, False, 'range', 0)


def test_the_servos_are_the_vendored_position_servos(scene):
    """
    Description:
        The vendored UR10e's servos (models/universal_robots_ur10e/ur10e.xml): kp 5000, kv 500 on every joint, the force
        ranges 330, 330, 150, 56, 56, 56 N m, the control ranges +-6.2831 rad (the elbow +-3.1415 rad).
    """

    servos = Servos(scene, Robot_Str=Scene_Robot(scene))
    assert np.array_equal(servos.Kp, [5000.0] * 6) and np.array_equal(servos.Kv, [500.0] * 6)
    assert np.array_equal(servos.Force_Range[:, 1], [330.0, 330.0, 150.0, 56.0, 56.0, 56.0])
    assert np.array_equal(servos.Control_Range[:, 1], [6.2831, 6.2831, 3.1415, 6.2831, 6.2831, 6.2831])


def test_a_servo_of_another_form_is_refused(configuration):
    """
    Description:
        A model whose first servo has another bias than (0, -kp, -kv) is refused, naming the joint: the command holds for
        the vendored form only.
    """

    from Scene.Core import Build_Scene
    model = Build_Scene(Areas_Str=configuration.Areas)
    robot = Scene_Robot(model)
    model.actuator_biasprm[Servos(model, Robot_Str=robot).Actuators[0], 1] = -1000.0
    with pytest.raises(Motion_Error_Cls, match='joint 1\'s actuator is not a force-limited position servo'):
        Servos(model, Robot_Str=robot)


def _Motion(scene, robot):
    profile = Profile(CONST_READY, np.zeros(6), CONST_READY + np.array([0.6, -0.4, 0.5, -0.3, 0.4, 0.0]), CONST_SPEED, CONST_ACCELERATION)
    t = 0.002 * np.arange(int(np.ceil(profile.Duration / 0.002)) + 50)
    return Profile_State(profile, t)


def _Tracked(scene, robot, command):
    q, qd, qdd = _Motion(scene, robot)
    data = mujoco.MjData(scene)
    ball = scene.jnt_qposadr[scene.body_jntadr[mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_BODY, 'ball')]]
    data.qpos[ball:ball + 3] = (0.0, -3.0, 0.03)
    data.qpos[robot.Position_Address] = q[0]
    servos = Servos(scene, Robot_Str=robot)
    if command == 'feed-forward':
        return q, Track(scene, data, servos, q, qd, qdd, 250, Robot_Str=robot)
    positions = np.zeros(q.shape)
    for k in range(-250, len(q)):
        if k >= 0:
            positions[k] = data.qpos[robot.Position_Address]
        data.ctrl[servos.Actuators] = q[max(k, 0)]
        mujoco.mj_step(scene, data)
    return q, positions


def test_the_feed_forward_command_tracks_a_planned_motion_within_2_mm(scene):
    """
    Description:
        A 0.5 s motion from the ready pose at 800 deg/s^2, tracked by the vendored servos with
        the feed-forward of their damping and of the inverse dynamics (Track), keeps every joint within 0.003 rad of the
        plan and the cup's mouth within 2 mm of the plan's at the end, no force or command at its range. The forced
        failure (the plain position command of the servos): the same motion leaves the cup more than 5 mm off at the end
        (Part A: 12.7 mm at the median of 120 throws).
    """

    robot = Scene_Robot(scene)
    q, tracked = _Tracked(scene, robot, 'feed-forward')
    data = mujoco.MjData(scene)
    planned = Forward_Kinematics(scene, data, q[-1], Robot_Str=robot)[0]
    assert np.max(np.abs(tracked.Positions - q)) < 0.003
    assert np.linalg.norm(Forward_Kinematics(scene, data, tracked.Positions[-1], Robot_Str=robot)[0] - planned) < 0.002
    assert (tracked.Force_Limit_Ticks, tracked.Control_Limit_Ticks) == (0, 0)
    _, plain = _Tracked(scene, robot, 'position')
    assert np.linalg.norm(Forward_Kinematics(scene, data, plain[-1], Robot_Str=robot)[0] - planned) > 0.005


def test_the_command_is_the_position_plus_the_damping_and_the_inverse_dynamics_over_kp(scene):
    """
    Description:
        Servo_Command at a planned state is q + (kv qd + tau) / kp with tau MuJoCo's inverse dynamics at that state: the
        servo force kp (ctrl - q) - kv qd it gives equals tau (within 1e-9 N m), and at rest it is the gravity torque.
    """

    robot = Scene_Robot(scene)
    servos = Servos(scene, Robot_Str=robot)
    data = mujoco.MjData(scene)
    q, qd, qdd = CONST_READY, np.array([0.5, -0.3, 0.2, 0.1, -0.4, 0.0]), np.array([2.0, 1.0, -3.0, 0.5, 0.0, 0.0])
    command = Servo_Command(scene, data, servos, q, qd, qdd, Robot_Str=robot)
    check = mujoco.MjData(scene)
    check.qpos[robot.Position_Address], check.qvel[robot.Velocity_Address], check.qacc[robot.Velocity_Address] = q, qd, qdd
    mujoco.mj_inverse(scene, check)
    assert servos.Kp * (command - q) - servos.Kv * qd == pytest.approx(check.qfrc_inverse[robot.Velocity_Address], abs=1e-9)
    rest = Servo_Command(scene, data, servos, q, np.zeros(6), np.zeros(6), Robot_Str=robot)
    check.qvel[:], check.qacc[:] = 0.0, 0.0
    mujoco.mj_inverse(scene, check)
    assert servos.Kp * (rest - q) == pytest.approx(check.qfrc_inverse[robot.Velocity_Address], abs=1e-9)
