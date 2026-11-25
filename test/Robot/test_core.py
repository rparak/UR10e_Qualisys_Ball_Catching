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
# File Name: ../test/Robot/test_core.py
## =========================================================================== ##

# System (Default)
import os
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation) [pip3 install mujoco]
import mujoco
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Robot/Core
from Robot.Core import Find_Robot, Check_Joints, Forward_Kinematics, Robot_Error_Cls, CONST_JOINT_NAMES
#   ../Scene/Core - the robot of the shipped scene
from Scene.Core import Scene_Robot

"""
Description:
    Initialization of constants.

    The robot in a compiled model (src/Robot/Core.py). Expectations come from the vendored model's file
    (its link offsets, its joint ranges) and from the owner's decisions, never from the code under test.
"""
# The height of the robot's mounting face above the floor (the stand's CAD).
#   Unit [m]
CONST_MOUNT_HEIGHT = 0.180
# The cup's length along its axis, from the flange to the centre of its mouth (CAD, SPEC §4.4).
#   Unit [m]
CONST_CUP_LENGTH = 0.190
# The joint ranges of the vendored ur10e.xml: 6.28319 rad for every joint but the elbow's 3.1415 rad.
#   Unit [Vector<float> 1x6, rad]
CONST_RANGE = np.array([6.28319, 6.28319, 3.1415, 6.28319, 6.28319, 6.28319])
# The vendored model alone, for the poses the scene must reproduce.
#   Unit [string]
CONST_VENDORED = os.path.join('models', 'universal_robots_ur10e', 'ur10e.xml')


def test_the_robot_is_found_by_its_six_joints_in_their_order(scene):
    """
    Description:
        The robot's joints are found in the order base to flange, behind the ball's free joint (7
        positions, 6 velocities), with the vendored ranges, and its two sites are distinct.
    """

    robot = Scene_Robot(scene)
    assert [scene.joint(int(joint)).name for joint in robot.Joint_Id] == ['ur10e/' + name for name in CONST_JOINT_NAMES]
    assert robot.Position_Address.tolist() == [7, 8, 9, 10, 11, 12]
    assert robot.Velocity_Address.tolist() == [6, 7, 8, 9, 10, 11]
    assert np.array_equal(robot.Lower, -CONST_RANGE) and np.array_equal(robot.Upper, CONST_RANGE)
    assert scene.site(robot.Flange_Site).name == 'ur10e/attachment_site'
    assert scene.site(robot.Tcp_Site).name == 'ur10e/cup_mouth'


def test_a_model_without_the_robot_is_refused_naming_what_is_missing(root):
    """
    Description:
        A model that lacks the robot is refused once, with every missing joint and site named; a model
        that lacks only the tool centre point names only that.
    """

    empty = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="elbow_joint"/><geom size="0.1"/></body></worldbody></mujoco>')
    with pytest.raises(Robot_Error_Cls) as error:
        Find_Robot(empty, '', 'tool')
    for name in CONST_JOINT_NAMES:
        assert (('the joint "%s"' % name) in str(error.value)) == (name != 'elbow_joint')
    assert 'the site "attachment_site"' in str(error.value) and 'the site "tool"' in str(error.value)

    vendored = mujoco.MjModel.from_xml_path(os.path.join(root, CONST_VENDORED))
    with pytest.raises(Robot_Error_Cls) as error:
        Find_Robot(vendored, '', 'cup_mouth')
    assert 'the site "cup_mouth" missing' in str(error.value) and 'the joint' not in str(error.value)
    assert Find_Robot(vendored, '', 'attachment_site').Position_Address.tolist() == [0, 1, 2, 3, 4, 5]


def test_the_forward_kinematics_at_zero_is_the_arm_stretched_along_minus_x(scene):
    """
    Description:
        At q = 0 the vendored arm lies stretched along -x of the scene (its base body is turned half a
        turn about z in its file). From the file's link offsets: x = -(0.613 + 0.571) = -1.184 m, y =
        -(0.176 - 0.137 + 0.135 + 0.1) = -0.274 m at the flange, z = 0.181 - 0.12 = 0.061 m above the
        mounting face. The cup's axis is the flange's, -y there, so the tool centre point is 0.190 m
        further along -y.
    """

    robot = Scene_Robot(scene)
    data = mujoco.MjData(scene)
    position, orientation = Forward_Kinematics(scene, data, np.zeros(6), Robot_Str=robot)
    assert np.allclose(data.site_xpos[robot.Flange_Site], [-1.184, -0.274, CONST_MOUNT_HEIGHT + 0.061], atol=1e-9)
    assert np.allclose(position, [-1.184, -0.274 - CONST_CUP_LENGTH, CONST_MOUNT_HEIGHT + 0.061], atol=1e-9)
    assert np.allclose(orientation[:, 2], [0.0, -1.0, 0.0], atol=1e-12)
    assert np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-12)


def test_the_scene_reproduces_the_vendored_models_flange_pose_raised_by_the_mount(scene, root):
    """
    Description:
        The flange of the robot in the scene is where the vendored model, compiled alone, has it, raised
        by the mounting height: attaching the robot changes nothing of its kinematics. Checked at 20
        random joint positions (seed 20261120).
    """

    robot = Scene_Robot(scene)
    data = mujoco.MjData(scene)
    vendored = mujoco.MjModel.from_xml_path(os.path.join(root, CONST_VENDORED))
    alone = mujoco.MjData(vendored)
    site = mujoco.mj_name2id(vendored, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')
    rng = np.random.default_rng(20261120)
    for _ in range(20):
        q = rng.uniform(-CONST_RANGE, CONST_RANGE)
        alone.qpos[:] = q
        mujoco.mj_kinematics(vendored, alone)
        position, orientation = Forward_Kinematics(scene, data, q, Robot_Str=robot)
        assert np.allclose(data.site_xpos[robot.Flange_Site], alone.site_xpos[site] + [0.0, 0.0, CONST_MOUNT_HEIGHT], atol=1e-12)
        assert np.allclose(orientation, alone.site_xmat[site].reshape(3, 3), atol=1e-12)
        assert np.allclose(position, data.site_xpos[robot.Flange_Site] + CONST_CUP_LENGTH * orientation[:, 2], atol=1e-12)


def test_turning_the_base_joint_turns_the_tool_about_the_vertical_through_the_base(scene):
    """
    Description:
        Joint 1 turns the whole arm about the vertical axis through the origin: the tool centre point at
        q_1 + 0.7 rad is the one at q_1 turned by 0.7 rad about z. A wrong joint order or address would
        turn something else.
    """

    robot = Scene_Robot(scene)
    data = mujoco.MjData(scene)
    q = np.array([0.3, -1.2, 1.4, -0.9, 1.1, 0.2])
    before, axes_before = Forward_Kinematics(scene, data, q, Robot_Str=robot)
    after, axes_after = Forward_Kinematics(scene, data, q + [0.7, 0, 0, 0, 0, 0], Robot_Str=robot)
    turn = np.array([[np.cos(0.7), -np.sin(0.7), 0.0], [np.sin(0.7), np.cos(0.7), 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(after, turn @ before, atol=1e-12) and np.allclose(axes_after, turn @ axes_before, atol=1e-12)
    assert np.linalg.norm(after - before) > 0.1


def test_a_joint_vector_is_refused_outside_its_range_and_accepted_on_it(scene):
    """
    Description:
        A joint on its limit is accepted and one 1e-6 rad beyond is refused with the joint, its value
        and the range; nothing is clamped. A vector of another length and a value that is not a number
        are refused too, every problem in one message.
    """

    robot = Scene_Robot(scene)
    on_limit = np.array([0.0, 0.0, 3.1415, 0.0, 0.0, -6.28319])
    assert np.array_equal(Check_Joints(on_limit, Robot_Str=robot), on_limit)
    with pytest.raises(Robot_Error_Cls) as error:
        Check_Joints(on_limit + [0, 0, 1e-6, 0, 0, -1e-6], Robot_Str=robot)
    assert 'elbow_joint = 3.141501000 rad is outside its range [-3.141500000, 3.141500000] rad' in str(error.value)
    assert 'wrist_3_joint = -6.283191000 rad' in str(error.value) and 'shoulder' not in str(error.value)
    with pytest.raises(Robot_Error_Cls, match='6 values'):
        Check_Joints(np.zeros(5), Robot_Str=robot)
    with pytest.raises(Robot_Error_Cls, match='wrist_1_joint is nan'):
        Check_Joints([0, 0, 0, np.nan, 0, 0], Robot_Str=robot)
    data = mujoco.MjData(scene)
    with pytest.raises(Robot_Error_Cls, match='elbow_joint'):
        Forward_Kinematics(scene, data, [0, 0, 3.2, 0, 0, 0], Robot_Str=robot)


def test_the_forward_kinematics_moves_the_robot_only_and_forgets_the_last_position(scene):
    """
    Description:
        The forward kinematics writes the robot's six joint positions and nothing else (the ball stays
        where the data has it), and its result depends on q alone, not on what the data held before.
    """

    robot = Scene_Robot(scene)
    data = mujoco.MjData(scene)
    data.qpos[:7] = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    q = np.array([0.5, -1.0, 1.0, 0.3, -0.4, 2.0])
    first, _ = Forward_Kinematics(scene, data, q, Robot_Str=robot)
    assert data.qpos[:7].tolist() == [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0] and np.array_equal(data.qpos[7:], q)
    Forward_Kinematics(scene, data, -q, Robot_Str=robot)
    again, _ = Forward_Kinematics(scene, data, q, Robot_Str=robot)
    assert np.array_equal(first, again)


def _Geometry(scene):
    from Robot.Core import Robot_Geometry
    robot = Scene_Robot(scene)
    return robot, Robot_Geometry(scene, Robot_Str=robot)


def _Targets(scene, robot, count, seed):
    """
    Description:
        Reachable targets: the tool centre point's position and axis at random joint vectors (joint 6 at 0,
        the elbow inside its range).
    """

    rng = np.random.default_rng(seed)
    data = mujoco.MjData(scene)
    joints = rng.uniform(-np.pi, np.pi, (count, 6))
    joints[:, 2] = rng.uniform(-3.1, 3.1, count)
    joints[:, 5] = 0.0
    positions, axes = np.zeros((count, 3)), np.zeros((count, 3))
    for i, q in enumerate(joints):
        positions[i], orientation = Forward_Kinematics(scene, data, q, Robot_Str=robot)
        axes[i] = orientation[:, 2]
    return joints, positions, axes


def _Wrapped(difference):
    return np.pi - np.mod(np.pi - difference, 2.0 * np.pi)


def test_the_ur_geometry_is_read_from_the_model(scene):
    """
    Description:
        The closed form's numbers come from the compiled model, as the vendored file has them: the arm's
        lateral offset 0.176 - 0.137 + 0.135 = 0.174 m, joint 5's 0.12 m, the tool 0.1 + 0.19 = 0.29 m beyond
        the wrist centre along joint 6's axis, joint 1's anchor on the mounting face's axis.
    """

    _, geometry = _Geometry(scene)
    assert abs(geometry.Lateral - 0.174) < 1e-12 and abs(geometry.Wrist_Offset - 0.12) < 1e-12
    assert abs(geometry.Tool_Offset - (0.1 + CONST_CUP_LENGTH)) < 1e-12 and geometry.Tool_Sign == 1.0
    assert np.allclose(geometry.Base_Point, [0.0, 0.0, CONST_MOUNT_HEIGHT + 0.181], atol=1e-12)
    assert np.allclose(geometry.Base_Axis, [0.0, 0.0, 1.0], atol=1e-12)


def test_a_model_that_is_not_a_ur_geometry_is_refused_naming_the_residual(root):
    """
    Description:
        A copy of the vendored robot whose elbow axis is tilted by 1 degree is not a UR geometry: the
        geometry is refused, naming joint 3's axis against joint 2's, so the closed form is never used on a
        robot it does not hold for.
    """

    from Robot.Core import Robot_Geometry
    spec = mujoco.MjSpec.from_file(os.path.join(root, CONST_VENDORED))
    spec.joint('elbow_joint').axis = [0.0, np.cos(np.radians(1.0)), np.sin(np.radians(1.0))]
    model = spec.compile()
    robot = Find_Robot(model, '', 'attachment_site')
    with pytest.raises(Robot_Error_Cls) as error:
        Robot_Geometry(model, Robot_Str=robot)
    assert 'not a UR geometry' in str(error.value) and "joint 3's axis against joint 2's" in str(error.value)
    assert Robot_Geometry(mujoco.MjModel.from_xml_path(os.path.join(root, CONST_VENDORED)), Robot_Str=robot).Tool_Offset == pytest.approx(0.1)


def test_the_closed_forms_solutions_round_trip_to_1e_9_m(scene):
    """
    Description:
        For 300 random poses the closed form gives up to 8 joint vectors, in pairs (a branch whose two-link arm
        cannot reach the wrist point has none: 227 targets have 8, 12 have 6, 49 have 4, 12 have 2, measured
        2026-11-21 with this seed); a branch without one is NaN as a whole row; every solution puts the tool
        centre point on the target within 1e-9 m with its axis within 1e-9; the pose the target came from is
        among them (joints 1 to 5 within 1e-7 rad, joint 6 free).
    """

    from Robot.Core import Inverse_Kinematics
    robot, geometry = _Geometry(scene)
    joints, positions, axes = _Targets(scene, robot, 300, 20261121)
    solutions = Inverse_Kinematics(positions, axes, Robot_Geometry_Str=geometry)
    finite = np.all(np.isfinite(solutions), axis=2)
    assert solutions.shape == (300, 8, 6) and np.array_equal(finite, np.any(np.isfinite(solutions), axis=2))
    assert np.bincount(finite.sum(axis=1), minlength=9).tolist() == [0, 0, 12, 0, 49, 0, 12, 0, 227]
    data = mujoco.MjData(scene)
    worst = 0.0
    for i in range(300):
        for q in solutions[i][finite[i]]:
            position, orientation = Forward_Kinematics(scene, data, q, Robot_Str=robot)
            worst = max(worst, np.linalg.norm(position - positions[i]), np.linalg.norm(orientation[:, 2] - axes[i]))
        assert np.any(np.all(np.abs(_Wrapped(solutions[i, :, :5] - joints[i, :5])) < 1e-7, axis=1)), i
    assert worst < 1e-9


def _Numerical(scene, robot, position, axis, seed_q, iterations=200):
    """
    Description:
        The numerical inverse kinematics of the test: damped least squares on the compiled model's Jacobian
        of the tool centre point and of its axis' two tilts, joints 1 to 5, from a seed; None when it does
        not converge to 1e-11.
    """

    data = mujoco.MjData(scene)
    q = np.array(seed_q, dtype=np.float64)
    q[5] = 0.0
    translation, rotation = np.zeros((3, scene.nv)), np.zeros((3, scene.nv))
    columns = robot.Velocity_Address[:5]
    for _ in range(iterations):
        q[:5] = _Wrapped(q[:5])
        q[2] = np.clip(q[2], -3.1415, 3.1415)
        data.qpos[robot.Position_Address] = q
        mujoco.mj_kinematics(scene, data)
        mujoco.mj_comPos(scene, data)
        p = data.site_xpos[robot.Tcp_Site]
        frame = data.site_xmat[robot.Tcp_Site].reshape(3, 3)
        a = frame[:, 2]
        tilt = np.cross(a, axis)
        error = np.concatenate([position - p, frame[:, :2].T @ tilt])
        if np.linalg.norm(error) < 1e-11 and a @ axis > 0.0:
            return q.copy()
        mujoco.mj_jacSite(scene, data, translation, rotation, robot.Tcp_Site)
        jacobian = np.vstack([translation[:, columns], frame[:, :2].T @ rotation[:, columns]])
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-8 * np.eye(5), error)
        q[:5] += step * min(1.0, 0.3 / max(np.abs(step).max(), 1e-12))
    return None


def test_the_closed_form_matches_the_numerical_multi_seed_ik_on_the_compiled_model(scene):
    """
    Description:
        At 6 random targets, the numerical inverse kinematics of the test (damped least squares on MuJoCo's
        Jacobian, independent of the closed form) is started from 40 random seeds and from each closed-form
        solution moved by 0.05 rad: every vector it converges to matches a closed-form solution within 1e-6
        rad (joints 1 to 5), so the closed form misses none, and it finds each closed-form solution again.
    """

    from Robot.Core import Inverse_Kinematics
    robot, geometry = _Geometry(scene)
    _, positions, axes = _Targets(scene, robot, 6, 7)
    rng = np.random.default_rng(20261121)
    for i in range(6):
        closed = Inverse_Kinematics(positions[i:i + 1], axes[i:i + 1], Robot_Geometry_Str=geometry)[0]
        closed = closed[np.all(np.isfinite(closed), axis=1)]
        seeds = [rng.uniform(-np.pi, np.pi, 6) for _ in range(40)] + [q + rng.uniform(-0.05, 0.05, 6) for q in closed]
        found = set()
        for seed in seeds:
            q = _Numerical(scene, robot, positions[i], axes[i], seed)
            if q is None:
                continue
            distance = np.max(np.abs(_Wrapped(closed[:, :5] - q[:5])), axis=1)
            assert distance.min() < 1e-6, (i, q)
            found.add(int(np.argmin(distance)))
        assert found == set(range(len(closed))) and len(closed) >= 2, (i, found)


def test_an_unreachable_target_and_the_wrist_singularity_have_no_solution(scene):
    """
    Description:
        A target 3 m from the shoulder and a wrist centre on joint 1's axis (closer than the arm's lateral
        offset) give no solution (NaN rows), never a nearest pose. At the wrist singularity (joint 5 at 0:
        joint 6's axis along the arm's axes) the four branches of that pose's joint 1 have none, the four of
        the other joint 1 have theirs; a reachable target in the same call still gets its solutions.
    """

    from Robot.Core import Inverse_Kinematics
    robot, geometry = _Geometry(scene)
    _, position, axis = _Targets(scene, robot, 1, 3)
    data = mujoco.MjData(scene)
    singular, frame = Forward_Kinematics(scene, data, [0.4, -1.2, 1.5, -0.8, 0.0, 0.0], Robot_Str=robot)
    down = np.array([0.0, 0.0, -1.0])
    targets = np.array([[3.0, 0.0, 0.4], geometry.Base_Point + [0.0, 0.0, 0.3] + 0.29 * down, position[0], singular])
    tool = np.array([down, down, axis[0], frame[:, 2]])
    solutions = Inverse_Kinematics(targets, tool, Robot_Geometry_Str=geometry)
    assert np.all(np.isnan(solutions[0])) and np.all(np.isnan(solutions[1]))
    assert np.all(np.isfinite(solutions[2]), axis=1).sum() >= 2
    branch = np.isnan(solutions[3, :, 0])
    assert branch.sum() == 4 and (branch[:4].all() or branch[4:].all())
    other = solutions[3][~branch]
    assert np.all(np.abs(_Wrapped(other[:, 0] - 0.4)) > 1e-3)


def test_the_joint_margin_is_tested_on_both_sides_of_it(scene):
    """
    Description:
        With a margin of 10 deg the elbow (range +-180 deg) passes at 170 deg - 1e-9 rad and fails 1e-6 rad
        beyond; a joint of +-360 deg passes anywhere in (-180, 180] deg; a NaN vector fails.
    """

    from Robot.Core import Within_Margin
    robot = Scene_Robot(scene)
    margin = np.radians(10.0)
    inside = np.array([3.0, -3.0, 3.1415 - margin - 1e-9, np.pi, -np.pi, 0.0])
    outside = inside + [0, 0, 1e-6 + 1e-9, 0, 0, 0]
    assert Within_Margin(np.stack([inside, outside, np.full(6, np.nan)]), margin, Robot_Str=robot).tolist() == [True, False, False]


def test_the_move_time_is_the_trapezoid_on_both_sides_of_the_speed_and_takes_the_nearer_way(scene):
    """
    Description:
        A move below v^2 / a never reaches the speed and takes 2 sqrt(d / a); one above takes d / v + v / a;
        the two agree at d = v^2 / a. A wrapping joint goes 350 deg as -10 deg, a joint that does not wrap the
        whole way. The slowest joint decides.
    """

    from Robot.Core import Move_Time, Wrapping_Joints
    robot = Scene_Robot(scene)
    wrap = Wrapping_Joints(Robot_Str=robot)
    assert wrap.tolist() == [True, True, False, True, True, True]
    v, a = np.radians(120.0), np.radians(800.0)
    speed = np.full(6, v)
    for d, expected in ((0.1, 2.0 * np.sqrt(0.1 / a)), (v * v / a, 2.0 * v / a), (2.0, 2.0 / v + v / a)):
        goal = np.zeros(6)
        goal[0] = d
        assert Move_Time(np.zeros(6), goal, speed, a, wrap) == pytest.approx(expected, rel=1e-12)
    goal = np.radians([350.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert Move_Time(np.zeros(6), goal, speed, a, wrap) == pytest.approx(2.0 * np.sqrt(np.radians(10.0) / a), rel=1e-12)
    elbow = np.radians([0.0, 0.0, 350.0, 0.0, 0.0, 0.0])
    assert Move_Time(np.zeros(6), elbow, speed, a, wrap) == pytest.approx(np.radians(350.0) / v + v / a, rel=1e-12)
    both = np.array([0.1, 0.0, 0.0, 0.0, 2.0, 0.0])
    assert Move_Time(np.zeros(6), both, speed, a, wrap) == pytest.approx(2.0 / v + v / a, rel=1e-12)


def _Finite_Jacobian(scene, robot, q, h=1e-6):
    data = mujoco.MjData(scene)
    base, frame = Forward_Kinematics(scene, data, q, Robot_Str=robot)
    columns = []
    for j in range(5):
        moved = np.array(q, dtype=np.float64)
        moved[j] += h
        position, orientation = Forward_Kinematics(scene, data, moved, Robot_Str=robot)
        tilt = np.cross(frame[:, 2], orientation[:, 2]) / h
        columns.append(np.concatenate([(position - base) / h, frame[:, :2].T @ tilt]))
    return np.array(columns).T


def test_the_speed_margin_is_the_joint_speeds_over_the_inverse_jacobians_rows(scene):
    """
    Description:
        At a pose away from singularities the speed margin equals min_j v_j / |row j of J^-1|, with J from a
        finite difference of the forward kinematics (an independent Jacobian) within 1e-4; it scales with the
        speed limits; at the wrist singularity (joint 5 at 0, joints 4 and 6 aligned) it is 0.
    """

    from Robot.Core import Speed_Margin
    robot = Scene_Robot(scene)
    data = mujoco.MjData(scene)
    speed = np.radians([120.0, 120.0, 180.0, 180.0, 180.0, 180.0])
    q = np.array([0.4, -1.2, 1.5, -0.8, 1.1, 0.0])
    rates = np.linalg.norm(np.linalg.inv(_Finite_Jacobian(scene, robot, q))[:, :3], axis=1)
    margin = Speed_Margin(scene, data, q, speed, 5, Robot_Str=robot)
    assert margin == pytest.approx(np.min(speed[:5] / rates), rel=1e-4)
    assert Speed_Margin(scene, data, q, 2.0 * speed, 5, Robot_Str=robot) == pytest.approx(2.0 * margin, rel=1e-12)
    assert Speed_Margin(scene, data, [0.4, -1.2, 1.5, -0.8, 0.0, 0.0], speed, 5, Robot_Str=robot) == 0.0


def test_the_line_rates_move_the_tool_along_the_direction_with_its_axis_held(scene):
    """
    Description:
        F15: the joint rates of a unit speed along a direction, checked against the forward
        kinematics alone: a step of 1e-6 along the rates moves the tool centre point by 1e-6 m along the direction (within
        1e-9 m) and leaves its z axis where it was (within 1e-9); the direction's length does not matter; joint 6 is not
        used (five rates); at the wrist singularity there are none; a direction of length 0 is refused.
    """

    from Robot.Core import Line_Rates, Robot_Error_Cls
    robot = Scene_Robot(scene)
    data = mujoco.MjData(scene)
    q = np.array([0.4, -1.2, 1.5, -0.8, 1.1, 0.0])
    direction = np.array([0.1, 0.56, -0.82])
    rates = Line_Rates(scene, data, q, direction, 5, Robot_Str=robot)
    assert rates.shape == (5,)
    point, orientation = Forward_Kinematics(scene, data, q, Robot_Str=robot)
    moved, turned = Forward_Kinematics(scene, data, q + 1e-6 * np.append(rates, 0.0), Robot_Str=robot)
    assert np.allclose(moved - point, 1e-6 * direction / np.linalg.norm(direction), rtol=0.0, atol=1e-9)
    assert np.allclose(turned[:, 2], orientation[:, 2], rtol=0.0, atol=1e-9)
    assert np.allclose(Line_Rates(scene, data, q, 3.0 * direction, 5, Robot_Str=robot), rates, rtol=0.0, atol=1e-12)
    assert Line_Rates(scene, data, [0.4, -1.2, 1.5, -0.8, 0.0, 0.0], direction, 5, Robot_Str=robot) is None
    with pytest.raises(Robot_Error_Cls, match='length above 0'):
        Line_Rates(scene, data, q, np.zeros(3), 5, Robot_Str=robot)


def test_the_collision_pairs_leave_out_only_what_no_joint_changes(scene):
    """
    Description:
        The pairs join the robot's colliding geoms (the cup included) with the world's and with each other,
        never on one body or a parent and its child, never the ball's. A pair left out as constant keeps its
        distance within 1e-9 m at 20 further random poses (another seed than the one that sorted it); every
        other pair moves. The exempt pairs are exactly the forearm's against wrist 2's.
    """

    from Robot.Core import Collision_Pairs
    robot = Scene_Robot(scene)
    pairs = Collision_Pairs(scene, Robot_Str=robot)
    body = lambda geom: scene.body(int(scene.geom_bodyid[geom])).name
    ball = scene.body('ball').id
    rows = np.vstack([pairs.Environment, pairs.Self, pairs.Exempt, pairs.Constant])
    assert len({tuple(r) for r in rows}) == len(rows) and not np.any(scene.geom_bodyid[rows] == ball)
    for a, b in np.vstack([pairs.Self, pairs.Exempt]):
        assert scene.geom_bodyid[a] != scene.geom_bodyid[b]
        assert scene.body_parentid[scene.geom_bodyid[a]] != scene.geom_bodyid[b] and scene.body_parentid[scene.geom_bodyid[b]] != scene.geom_bodyid[a]
    assert all(scene.geom_bodyid[b] == 0 for _, b in pairs.Environment)
    assert {frozenset((body(a), body(b))) for a, b in pairs.Exempt} == {frozenset(('ur10e/forearm_link', 'ur10e/wrist_2_link'))}
    assert len(pairs.Exempt) == 4 and len(pairs.Constant) > 0
    rng = np.random.default_rng(99)
    data = mujoco.MjData(scene)
    measured = {name: [] for name in ('constant', 'moving')}
    for _ in range(20):
        Forward_Kinematics(scene, data, np.append(rng.uniform(-np.pi, np.pi, 5), 0.0) * [1, 1, 0.99, 1, 1, 1], Robot_Str=robot)
        measured['constant'].append([mujoco.mj_geomDistance(scene, data, int(a), int(b), 10.0, None) for a, b in pairs.Constant])
        measured['moving'].append([mujoco.mj_geomDistance(scene, data, int(a), int(b), 10.0, None)
                                   for a, b in np.vstack([pairs.Environment, pairs.Self, pairs.Exempt])])
    assert np.ptp(np.array(measured['constant']), axis=0).max() < 1e-9
    assert np.ptp(np.array(measured['moving']), axis=0).min() > 1e-6
    assert np.allclose(np.array(measured['constant'])[0], pairs.Constant_Distance, atol=1e-9)


def test_the_cups_shape_for_the_ball_takes_no_part_in_the_clearance(scene):
    """
    Description:
        The cup's shape for the ball (F11: 49 geoms that only take contacts, contype 0) is in no pair of the
        clearance, and the pairs are those of the scene before it: 41 against the world, 23 of the robot
        with itself, 4 exempt, 11 constant (measured at aaac598, before the cone; with the cone's geoms
        taken in they were 286, 365, 4 and 61, F11 Part A). The cup's cylinder and the wrist's two shapes
        that pass the ball (contype 1, conaffinity 0) stay in.
    """

    from Robot.Core import Collision_Pairs
    pairs = Collision_Pairs(scene, Robot_Str=Scene_Robot(scene))
    assert (len(pairs.Environment), len(pairs.Self), len(pairs.Exempt), len(pairs.Constant)) == (41, 23, 4, 11)
    rows = np.vstack([pairs.Environment, pairs.Self, pairs.Exempt, pairs.Constant])
    shape = [index for index in range(scene.ngeom) if scene.geom_contype[index] == 0 and scene.geom_conaffinity[index] != 0]
    assert len(shape) == 49 and not np.any(np.isin(rows, shape))
    passing = [index for index in range(scene.ngeom) if scene.geom_contype[index] != 0 and scene.geom_conaffinity[index] == 0]
    assert len(passing) == 3 and all(np.any(rows == index) for index in passing)


def test_the_model_numbers_every_parent_body_before_its_child_and_the_geoms_by_body(scene):
    """
    Description:
        The order Collision_Pairs relies on to test a parent and its child one way only: in the compiled scene
        every body but the world has a parent with a smaller number, and the geoms' bodies never decrease, so
        of two geoms the later one's body is never the parent of the earlier one's.
    """

    body = np.asarray(scene.geom_bodyid)
    parent = np.asarray(scene.body_parentid)
    assert scene.nbody == 10 and scene.ngeom > 10 and parent[0] == 0
    assert np.all(parent[1:] < np.arange(1, scene.nbody))
    assert np.all(np.diff(body) >= 0) and len(np.unique(body)) == scene.nbody
    earlier, later = np.triu_indices(scene.ngeom, 1)
    apart = body[earlier] != body[later]
    assert apart.sum() > 100 and not np.any(parent[body[earlier][apart]] == body[later][apart])


def test_the_clearance_with_its_broad_phase_equals_the_brute_force_minimum(scene):
    """
    Description:
        At 60 random poses the clearance (a broad phase on bounding spheres, then mj_geomDistance) equals the
        minimum of mj_geomDistance over every pair, brute force, at the cut of 0.05 m and of 0.5 m, and names
        a pair that has that distance.
    """

    from Robot.Core import Collision_Pairs, Clearance, Pair_Set
    robot = Scene_Robot(scene)
    pairs = Collision_Pairs(scene, Robot_Str=robot)
    every = np.vstack([pairs.Environment, pairs.Self])
    prepared = Pair_Set(scene, every)
    data = mujoco.MjData(scene)
    rng = np.random.default_rng(5)
    cut_below = 0
    for _ in range(60):
        Forward_Kinematics(scene, data, np.append(rng.uniform(-np.pi, np.pi, 5), 0.0) * [1, 1, 0.99, 1, 1, 1], Robot_Str=robot)
        for cut in (0.05, 0.5):
            brute = [mujoco.mj_geomDistance(scene, data, int(a), int(b), cut, None) for a, b in every]
            distance, row = Clearance(scene, data, prepared, cut)
            assert distance == min(cut, min(brute))
            if row >= 0:
                assert brute[row] == distance
                cut_below += 1
    assert cut_below > 20
