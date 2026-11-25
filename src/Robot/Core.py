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
# File Name: ../Robot/Core.py
## =========================================================================== ##

# Dataclasses (Data classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation; here its kinematics) [pip3 install mujoco]
import mujoco
# Typing (Support for type hints)
import typing as tp

"""
Description:
    Initialization of constants.

    The UR10e in a compiled MuJoCo model (SPEC §7.9): where its six joints and its two tool sites are, its
    forward kinematics, its closed-form inverse kinematics with the UR geometry read from the model, the
    joint limits with a margin, the exact clearance of a pose, the speed margin and the rest-to-rest move
    time. The model is passed in: this layer is below Scene and never builds one. Pure computation on the
    model and the data given: no file, no configuration, no printing.
"""
# The six joints of the vendored UR10e (models/universal_robots_ur10e/ur10e.xml, MuJoCo Menagerie), base
# to flange, without the prefix a scene gives them. This order is the order of every joint vector q.
#   Unit [Tuple<string>]
CONST_JOINT_NAMES = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
                     'wrist_3_joint')
# The vendored model's site on the flange (its z axis leaves the flange along joint 6's axis), without
# the prefix.
#   Unit [string]
CONST_FLANGE_SITE = 'attachment_site'
# How far a joint value may lie beyond its range before it is refused: the rounding of a value computed
# onto a limit, far below any angle that matters (1e-9 rad is 1.3e-9 m at the UR10e's reach).
#   Unit [rad]
CONST_RANGE_EPSILON = 1e-9
# How far the model may be from a UR geometry (parallel axes, perpendicular axes, intersecting lines, the
# tool on joint 6's axis) for the closed form to hold: a cosine or a length in m. The vendored model's
# residuals are below 1e-15 (measured 2026-11-21); 1e-9 m is the round trip the closed form is held to.
#   Unit [-], [m]
CONST_GEOMETRY_TOLERANCE = 1e-9
# Below this |n x a'| the tool axis is parallel to the arm's axes (the wrist singularity): joint 5's axis
# is not defined, and that branch has no solution.
#   Unit [-]
CONST_SINGULAR_TOLERANCE = 1e-9
# The test of a pair whose distance no joint changes (Collision_Pairs): the number of random joint vectors,
# their seed, the largest variation of a constant distance, and the cut of that measurement (far above any
# distance in the cell, so that a far pair is measured, not cut).
#   Unit [int], [int], [m], [m]
CONST_CONSTANT_POSES = 12
CONST_CONSTANT_SEED = 20261121
CONST_CONSTANT_TOLERANCE = 1e-9
CONST_CONSTANT_CUT = 10.0


class Robot_Error_Cls(ValueError):
    """
    Description:
        The robot cannot be found in the model, or a joint vector cannot be used. The message names the
        element or the joint, its value and the limit; a script prints it verbatim as an [ERROR] line.
    """


@dataclass
class Robot_Str:
    """
    Description:
        Where the robot is in one compiled model: the addresses of its joints, their ranges and its two
        tool sites. Valid for the model it was found in only.
    """

    # The joints' ids in the model, in the order of CONST_JOINT_NAMES.
    #   Unit [Vector<int> 1x6]
    Joint_Id: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # The joints' addresses in the model's position vector (qpos).
    #   Unit [Vector<int> 1x6]
    Position_Address: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # The joints' addresses in the model's velocity vector (qvel; the columns of a Jacobian).
    #   Unit [Vector<int> 1x6]
    Velocity_Address: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # The joints' lower position limits, from the model.
    #   Unit [Vector<float> 1x6, rad]
    Lower: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # The joints' upper position limits, from the model.
    #   Unit [Vector<float> 1x6, rad]
    Upper: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # The id of the site on the flange (CONST_FLANGE_SITE).
    #   Unit [int]
    Flange_Site: int = -1
    # The id of the tool centre point's site (the centre of the cup's mouth, its z axis out of the cup).
    #   Unit [int]
    Tcp_Site: int = -1


def Find_Robot(model: mujoco.MjModel, prefix: str, tcp_site: str) -> Robot_Str:
    """
    Description:
        Find the UR10e in a compiled model by the names of its joints and sites. Every element that is
        missing is named, and the robot is refused once.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) prefix [string]: The prefix the scene gave the vendored model's names ('' for the vendored
                             model alone).
        (3) tcp_site [string]: The full name of the tool centre point's site.

    Returns:
        (1) parameter [Robot_Str(object)]: The robot's addresses in this model.
    """

    missing = []
    joints = []
    for name in CONST_JOINT_NAMES:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
        if joint < 0:
            missing.append('the joint "%s"' % (prefix + name))
        joints.append(joint)
    flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + CONST_FLANGE_SITE)
    if flange < 0:
        missing.append('the site "%s"' % (prefix + CONST_FLANGE_SITE))
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
    if tcp < 0:
        missing.append('the site "%s"' % tcp_site)
    if missing:
        raise Robot_Error_Cls('The model does not hold the robot: %s missing. Build it with Scene.Core.Build_Scene.'
                              % ', '.join(missing))

    joints = np.array(joints, dtype=np.int64)
    return Robot_Str(Joint_Id=joints, Position_Address=np.array(model.jnt_qposadr[joints], dtype=np.int64),
                     Velocity_Address=np.array(model.jnt_dofadr[joints], dtype=np.int64),
                     Lower=np.array(model.jnt_range[joints, 0], dtype=np.float64),
                     Upper=np.array(model.jnt_range[joints, 1], dtype=np.float64), Flange_Site=int(flange), Tcp_Site=int(tcp))


def Check_Joints(q: tp.Sequence[float], *, Robot_Str: Robot_Str) -> np.ndarray:
    """
    Description:
        A joint vector as the kinematics takes it: six finite numbers, each within its joint's range.
        Every problem is named with its joint, its value and the range, and the vector is refused once;
        nothing is clamped.

    Args:
        (1) q [Vector<float> 1x6]: The joint positions [rad], in the order of CONST_JOINT_NAMES.
        (2) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Vector<float> 1x6]: The joint positions [rad].
    """

    q = np.asarray(q, dtype=np.float64)
    if q.shape != (len(CONST_JOINT_NAMES),):
        raise Robot_Error_Cls('A joint vector has %d values in the order %s; one of shape %s was given.'
                              % (len(CONST_JOINT_NAMES), ', '.join(CONST_JOINT_NAMES), q.shape))
    problems = []
    for name, value, lower, upper in zip(CONST_JOINT_NAMES, q, Robot_Str.Lower, Robot_Str.Upper):
        if not np.isfinite(value):
            problems.append('%s is %s' % (name, value))
        elif value < lower - CONST_RANGE_EPSILON or value > upper + CONST_RANGE_EPSILON:
            problems.append('%s = %.9f rad is outside its range [%.9f, %.9f] rad' % (name, value, lower, upper))
    if problems:
        raise Robot_Error_Cls('The joint vector cannot be used: %s. Give a position inside the ranges.' % '; '.join(problems))
    return q


def Forward_Kinematics(model: mujoco.MjModel, data: mujoco.MjData, q: tp.Sequence[float], *,
                       Robot_Str: Robot_Str) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        The pose of the tool centre point at the joint positions q, in the model's world frame. The
        robot's joints in `data` are set to q and the positions of every body and site are computed
        (mj_kinematics), so `data` also holds the flange's site and the links at q afterwards; nothing
        else in `data` is changed.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) data [mujoco.MjData]: Its data; the robot's joint positions are overwritten.
        (3) q [Vector<float> 1x6]: The joint positions [rad], in the order of CONST_JOINT_NAMES.
        (4) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Vector<float> 1x3]: The position of the tool centre point [m].
        (2) parameter [Matrix<float> 3x3]: Its orientation: the columns are its x, y, z axes in the world
                                           frame.
    """

    data.qpos[Robot_Str.Position_Address] = Check_Joints(q, Robot_Str=Robot_Str)
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[Robot_Str.Tcp_Site].copy(), data.site_xmat[Robot_Str.Tcp_Site].reshape(3, 3).copy()


@dataclass
class Robot_Geometry_Str:
    """
    Description:
        The UR geometry of the robot in one compiled model, read from it at q = 0 (Robot_Geometry): the
        points and axes the closed-form inverse kinematics needs, in the model's world frame. Valid for
        that model only.
    """

    # A point on joint 1's axis (its anchor) and the axis.
    #   Unit [Vector<float> 1x3, m], [Vector<float> 1x3, -]
    Base_Point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Base_Axis: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # The common direction n of the parallel axes of joints 2, 3 and 4 (joint 2's axis), and the signs of
    # joint 3's and joint 4's axes against it.
    #   Unit [Vector<float> 1x3, -], [float, -]
    Arm_Axis: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Sign_3: float = 1.0
    Sign_4: float = 1.0
    # Points on the axes of joints 2 and 3, and the point P on joint 4's axis from which the wrist centre W
    # (where the axes of joints 5 and 6 meet) lies along joint 5's axis.
    #   Unit [Vector<float> 1x3, m]
    Shoulder_Point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Elbow_Point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Wrist_Point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Joint 5's and joint 6's axes at q = 0.
    #   Unit [Vector<float> 1x3, -]
    Axis_5: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Axis_6: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # The lateral offset D of the arm along n from joint 1's axis (W . n = D in every pose), the distance
    # d5 from P to W along joint 5's axis, and the distance c from W to the tool centre point along joint
    # 6's axis.
    #   Unit [m]
    Lateral: float = 0.0
    Wrist_Offset: float = 0.0
    Tool_Offset: float = 0.0
    # The tool axis (the tool centre point's z axis) as +1 or -1 times joint 6's axis.
    #   Unit [float, -]
    Tool_Sign: float = 1.0


def _Signed_Angle(a: np.ndarray, b: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Description:
        The signed angle that turns a onto b about the axis k, for vectors perpendicular to k:
        atan2(k . (a x b), a . b). Row-wise for arrays.

    Args:
        (1 - 2) a, b [Matrix<float> Nx3]: The vectors.
        (3) k [Vector<float> 1x3 or Matrix<float> Nx3]: The unit axis.

    Returns:
        (1) parameter [Vector<float> 1xN]: The angles [rad].
    """

    return np.arctan2(np.sum(np.cross(a, b) * k, axis=-1), np.sum(a * b, axis=-1))


def _Rotate(v: np.ndarray, k: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Description:
        Rotate vectors about a unit axis (Rodrigues): v cos + (k x v) sin + k (k . v)(1 - cos). Row-wise.

    Args:
        (1) v [Matrix<float> Nx3]: The vectors.
        (2) k [Vector<float> 1x3 or Matrix<float> Nx3]: The unit axis.
        (3) angle [Vector<float> 1xN]: The angles [rad].

    Returns:
        (1) parameter [Matrix<float> Nx3]: The rotated vectors.
    """

    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    k = np.broadcast_to(k, v.shape)
    return v * c + np.cross(k, v) * s + k * np.sum(k * v, axis=-1, keepdims=True) * (1.0 - c)


def _Plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Description:
        Vectors projected onto the plane perpendicular to the unit vector n. Row-wise.

    Args:
        (1) v [Matrix<float> Nx3]: The vectors.
        (2) n [Vector<float> 1x3]: The plane's unit normal.

    Returns:
        (1) parameter [Matrix<float> Nx3]: The projections.
    """

    return v - np.sum(v * n, axis=-1, keepdims=True) * n


def _Wrap(angle: np.ndarray) -> np.ndarray:
    """
    Description:
        Angles brought into (-pi, pi].

    Args:
        (1) angle [Array<float>]: The angles [rad].

    Returns:
        (1) parameter [Array<float>]: The same angles in (-pi, pi] [rad].
    """

    return np.pi - np.mod(np.pi - angle, 2.0 * np.pi)


def Robot_Geometry(model: mujoco.MjModel, *, Robot_Str: Robot_Str) -> Robot_Geometry_Str:
    """
    Description:
        Read the robot's UR geometry from the compiled model at q = 0 and check that it is one: joint 1's
        axis perpendicular to the parallel axes of joints 2, 3 and 4, joint 5's axis perpendicular to them,
        joint 6's axis perpendicular to joint 5's and meeting it in a point W, W reached from joint 4's axis
        along joint 5's axis, and the tool centre point on joint 6's axis with its z axis along it. Every
        check that fails is named, with its residual, and the geometry is refused once.

        Note:
            Nothing of the UR10e's dimensions is written here: the closed form works for any model with this
            structure, and its numbers come from the model (MuJoCo Menagerie's file differs from UR's
            Denavit-Hartenberg values by up to 1.5 mm, SPEC §7.9).

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Robot_Geometry_Str(object)]: The geometry.
    """

    data = mujoco.MjData(model)
    Forward_Kinematics(model, data, np.zeros(len(CONST_JOINT_NAMES)), Robot_Str=Robot_Str)
    anchors = np.array(data.xanchor[Robot_Str.Joint_Id], dtype=np.float64)
    axes = np.array(data.xaxis[Robot_Str.Joint_Id], dtype=np.float64)
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    tcp = np.array(data.site_xpos[Robot_Str.Tcp_Site], dtype=np.float64)
    tool = np.array(data.site_xmat[Robot_Str.Tcp_Site], dtype=np.float64).reshape(3, 3)[:, 2]
    u1, n = axes[0], axes[1]
    sign_3, sign_4 = float(np.sign(axes[2] @ n)), float(np.sign(axes[3] @ n))
    u5, u6 = axes[4], axes[5]

    # The point W where joint 5's and joint 6's axes meet: the closest points of the two lines.
    w0 = anchors[5] - anchors[4]
    b = u5 @ u6
    denominator = 1.0 - b * b
    s = (u5 @ w0 - b * (u6 @ w0)) / denominator if denominator > 0.0 else 0.0
    t = (b * (u5 @ w0) - u6 @ w0) / denominator if denominator > 0.0 else 0.0
    on_5, on_6 = anchors[4] + s * u5, anchors[5] + t * u6
    wrist = 0.5 * (on_5 + on_6)
    # W - A4 = lambda n + d5 u5 (+ a residual along n x u5 that a UR geometry does not have).
    offset = wrist - anchors[3]
    across = np.cross(n, u5)
    across = across / np.linalg.norm(across) if np.linalg.norm(across) > 0.0 else across
    point = anchors[3] + (offset @ n) * n
    tool_offset = (tcp - wrist) @ u6

    residuals = {'joint 3\'s axis against joint 2\'s': 1.0 - abs(axes[2] @ n), 'joint 4\'s axis against joint 2\'s': 1.0 - abs(axes[3] @ n),
                 'joint 1\'s axis . joint 2\'s': abs(u1 @ n), 'joint 5\'s axis . joint 2\'s': abs(u5 @ n), 'joint 6\'s axis . joint 5\'s': abs(b),
                 'the distance of joint 5\'s and joint 6\'s axes [m]': float(np.linalg.norm(on_5 - on_6)),
                 'W off the plane of joint 4\'s and joint 5\'s axes [m]': abs(offset @ across),
                 'the tool centre point off joint 6\'s axis [m]': float(np.linalg.norm(tcp - wrist - tool_offset * u6)),
                 'the tool axis against joint 6\'s': 1.0 - abs(tool @ u6)}
    wrong = ['%s %.3g' % (name, value) for name, value in residuals.items() if value > CONST_GEOMETRY_TOLERANCE]
    if wrong:
        raise Robot_Error_Cls('The robot in the model is not a UR geometry, so the closed-form inverse kinematics does not hold: %s '
                              '(tolerance %.0e).' % ('; '.join(wrong), CONST_GEOMETRY_TOLERANCE))
    return Robot_Geometry_Str(Base_Point=anchors[0], Base_Axis=u1, Arm_Axis=n, Sign_3=sign_3, Sign_4=sign_4, Shoulder_Point=anchors[1],
                              Elbow_Point=anchors[2], Wrist_Point=point, Axis_5=u5, Axis_6=u6,
                              Lateral=float((point - anchors[0]) @ n), Wrist_Offset=float(offset @ u5), Tool_Offset=float(tool_offset),
                              Tool_Sign=float(np.sign(tool @ u6)))


def Inverse_Kinematics(position: np.ndarray, axis: np.ndarray, *, Robot_Geometry_Str: Robot_Geometry_Str) -> np.ndarray:
    """
    Description:
        Every joint vector that puts the tool centre point at `position` with its z axis along `axis`, in
        closed form: the wrist centre W = p - c a' (a' joint 6's axis), joint 1 from W . n = D (two
        branches), joint 5's axis along +-(n x a') (two), the two-link arm of joints 2 and 3 to the point
        P = W - d5 z5 (two), joint 4 from joint 5's axis, joint 5 from joint 6's. Joint 6 turns the tool
        about its own axis, which this target leaves free: it is set to 0. Up to 8 solutions per target.

        Note:
            A target with W closer to joint 1's axis than D, a P out of the arm's reach, or a tool axis
            parallel to n (the wrist singularity, where joint 5's axis is not defined) has no solution in
            that branch: its rows are NaN, never a nearest pose. The angles are in (-pi, pi]; whether they
            lie within the joints' ranges is not checked here (Within_Margin).

    Args:
        (1) position [Matrix<float> Nx3]: The targets of the tool centre point [m], world frame.
        (2) axis [Matrix<float> Nx3]: The tool's z axis at each target (unit vectors).
        (3) Robot_Geometry_Str [Robot_Geometry_Str(object)]: REQUIRED, keyword-only: the geometry (Robot_Geometry).

    Returns:
        (1) parameter [Array<float> Nx8x6]: The joint vectors [rad], in the order of CONST_JOINT_NAMES; the
                                            solution of branch (joint 1 i, joint 5 j, elbow k) at 4 i + 2 j + k;
                                            NaN where a branch has none.
    """

    g = Robot_Geometry_Str
    position, axis = np.atleast_2d(np.asarray(position, dtype=np.float64)), np.atleast_2d(np.asarray(axis, dtype=np.float64))
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    count = len(position)
    result = np.full((count, 8, 6), np.nan)
    axis_6 = g.Tool_Sign * axis
    wrist = position - g.Tool_Offset * axis_6
    n, e2 = g.Arm_Axis, np.cross(g.Base_Axis, g.Arm_Axis)

    # 1\ Joint 1: W . n(q1) = D with n(q1) = cos q1 n + sin q1 (u1 x n).
    relative = wrist - g.Base_Point
    x, y = relative @ n, relative @ e2
    radius = np.hypot(x, y)
    with np.errstate(invalid='ignore', divide='ignore'):
        spread = np.arccos(g.Lateral / radius)
    phi = np.arctan2(y, x)
    # The two-link arm at q = 0, in the plane perpendicular to n.
    upper = _Plane(g.Elbow_Point - g.Shoulder_Point, n)
    lower = _Plane(g.Wrist_Point - g.Elbow_Point, n)
    length_2, length_3 = np.linalg.norm(upper), np.linalg.norm(lower)
    gamma = _Signed_Angle(upper, lower, n)
    for i, side in enumerate((1.0, -1.0)):
        q1 = phi + side * spread
        # The targets turned back into the frame of q1 = 0.
        back = -q1
        wrist_0 = g.Base_Point + _Rotate(wrist - g.Base_Point, g.Base_Axis, back)
        axis_0 = _Rotate(axis_6, g.Base_Axis, back)
        normal = np.cross(np.broadcast_to(n, axis_0.shape), axis_0)
        size = np.linalg.norm(normal, axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            normal = np.where(size > CONST_SINGULAR_TOLERANCE, normal / size, np.nan)
        for j, flip in enumerate((1.0, -1.0)):
            # 2\ Joint 5's axis z5 perpendicular to n and to joint 6's axis.
            z5 = flip * normal
            target = _Plane(wrist_0 - g.Wrist_Offset * z5 - g.Shoulder_Point, n)
            reach = np.linalg.norm(target, axis=1)
            with np.errstate(invalid='ignore'):
                bend = np.arccos((reach ** 2 - length_2 ** 2 - length_3 ** 2) / (2.0 * length_2 * length_3))
            for k, elbow in enumerate((1.0, -1.0)):
                # 3\ The two-link arm: |upper + R(q3') lower| = |target|, then q2' turns it onto the target.
                q3 = -gamma + elbow * bend
                arm = upper + _Rotate(np.broadcast_to(lower, target.shape), n, q3)
                q2 = _Signed_Angle(arm, target, n)
                # 4\ Joint 4 turns joint 5's axis onto z5, joint 5 turns joint 6's axis onto the target's.
                turned = q2 + q3
                q4 = _Signed_Angle(_Rotate(np.broadcast_to(g.Axis_5, target.shape), n, turned), z5, n)
                axis_5 = _Rotate(np.broadcast_to(g.Axis_5, target.shape), n, turned + q4)
                q5 = _Signed_Angle(_Rotate(np.broadcast_to(g.Axis_6, target.shape), n, turned + q4), axis_0, axis_5)
                solution = np.stack([q1, q2, g.Sign_3 * q3, g.Sign_4 * q4, q5, np.zeros(count)], axis=1)
                # A branch with no solution is NaN as a whole row, joint 1 and joint 6 included.
                solution[~np.all(np.isfinite(solution), axis=1)] = np.nan
                result[:, 4 * i + 2 * j + k] = _Wrap(solution)
    return result


def Within_Margin(q: np.ndarray, margin: float, *, Robot_Str: Robot_Str) -> np.ndarray:
    """
    Description:
        Whether each joint vector keeps every joint at least `margin` inside its range (a NaN vector does
        not).

    Args:
        (1) q [Array<float> ...x6]: The joint vectors [rad].
        (2) margin [float]: The margin to each limit [rad].
        (3) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Array<bool> ...]: True for a vector inside the ranges by the margin.
    """

    q = np.asarray(q, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        return np.all((q >= Robot_Str.Lower + margin) & (q <= Robot_Str.Upper - margin), axis=-1)


def Wrapping_Joints(*, Robot_Str: Robot_Str) -> np.ndarray:
    """
    Description:
        The joints whose range spans two full turns or more (the UR10e's +-360 deg): such a joint reaches a
        target angle the nearer way round, and both ways stay inside its range for angles in (-pi, pi].

    Args:
        (1) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Vector<bool> 1x6]: True for a wrapping joint.
    """

    return (Robot_Str.Upper - Robot_Str.Lower) >= 4.0 * np.pi - 1e-4


def Move_Time(start: np.ndarray, goal: np.ndarray, speed: np.ndarray, acceleration: float, wrap: np.ndarray) -> np.ndarray:
    """
    Description:
        The shortest rest-to-rest time from `start` to each `goal` under each joint's speed and one
        acceleration limit: per joint a trapezoid, |d| / v + v / a when it reaches its speed (|d| >= v^2 / a),
        else 2 sqrt(|d| / a); the slowest joint decides. A wrapping joint takes the nearer way round. NaN
        goals give NaN.

    Args:
        (1) start [Vector<float> 1xJ]: The joint vector at rest [rad].
        (2) goal [Array<float> ...xJ]: The joint vectors to reach [rad].
        (3) speed [Vector<float> 1xJ]: The joints' speed limits [rad/s].
        (4) acceleration [float]: The joints' acceleration limit [rad/s^2].
        (5) wrap [Vector<bool> 1xJ]: Which joints wrap (Wrapping_Joints).

    Returns:
        (1) parameter [Array<float> ...]: The times [s].
    """

    difference = np.asarray(goal, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    difference = np.abs(np.where(wrap, _Wrap(difference), difference))
    speed = np.asarray(speed, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        times = np.where(difference >= speed ** 2 / acceleration, difference / speed + speed / acceleration,
                         2.0 * np.sqrt(difference / acceleration))
    return np.max(times, axis=-1)


def Tool_Jacobian(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, joints: int, *, Robot_Str: Robot_Str) -> np.ndarray:
    """
    Description:
        The Jacobian of the tool centre point's velocity and of its z axis' two tilting rates at q, for the first
        `joints` joints: the three rows of the point's velocity, then the rotation's rows along the tool's x and y
        axes. With five joints and a tool symmetric about joint 6 (the cup) it is 5 x 5.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) data [mujoco.MjData]: Its data; the robot's joint positions are overwritten.
        (3) q [Vector<float> 1x6]: The joint vector [rad].
        (4) joints [int]: The joints used, from joint 1.
        (5) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Matrix<float> 5xJ]: The Jacobian [m/rad, 1], J = joints.
    """

    _, orientation = Forward_Kinematics(model, data, q, Robot_Str=Robot_Str)
    mujoco.mj_comPos(model, data)
    translation, rotation = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, translation, rotation, Robot_Str.Tcp_Site)
    columns = Robot_Str.Velocity_Address[:joints]
    tilt = orientation[:, :2].T
    return np.vstack([translation[:, columns], tilt @ rotation[:, columns]])


def Speed_Margin(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, speed: np.ndarray, joints: int, *,
                 Robot_Str: Robot_Str) -> float:
    """
    Description:
        The speed the tool centre point can reach in every direction at q with its z axis held, within the
        joints' speed limits, using the first `joints` joints: with J the Jacobian of the point's velocity and
        of the axis' two tilting rates, the joint rates of a unit velocity v are J^-1 [v, 0, 0], and the
        largest a joint j needs over all directions is the norm of row j of J^-1's velocity columns; the
        margin is min_j v_max,j over that norm. A singular J (smallest singular value below 1e-9) gives 0:
        some direction cannot be reached at all.

        Note:
            With five joints and a tool symmetric about joint 6 (the cup) J is 5 x 5. This is the measure of
            the distance from a singularity chosen for the catch pose, in m/s.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) data [mujoco.MjData]: Its data; the robot's joint positions are overwritten.
        (3) q [Vector<float> 1x6]: The joint vector [rad].
        (4) speed [Vector<float> 1x6]: The joints' speed limits [rad/s].
        (5) joints [int]: The joints used, from joint 1 (5 for the symmetric cup).
        (6) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [float]: The speed margin [m/s].
    """

    jacobian = Tool_Jacobian(model, data, q, joints, Robot_Str=Robot_Str)
    if np.linalg.svd(jacobian, compute_uv=False)[-1] < 1e-9:
        return 0.0
    rates = np.linalg.norm(np.linalg.inv(jacobian)[:, :3], axis=1)
    return float(np.min(np.asarray(speed, dtype=np.float64)[:joints] / rates))


def Line_Rates(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, direction: np.ndarray, joints: int, *,
               Robot_Str: Robot_Str) -> tp.Optional[np.ndarray]:
    """
    Description:
        The joint rates that move the tool centre point at q with a unit speed along a direction while its z axis is
        held (F15): J^-1 [d, 0, 0] with the Jacobian of Tool_Jacobian, using the first `joints`
        joints. Times a speed they are the joint velocities of that motion; the largest speed within the joints' speed
        limits is min_j v_max,j / |rate_j|. None at a singular J (smallest singular value below 1e-9, as Speed_Margin):
        some direction cannot be followed.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) data [mujoco.MjData]: Its data; the robot's joint positions are overwritten.
        (3) q [Vector<float> 1x6]: The joint vector [rad].
        (4) direction [Vector<float> 1x3]: The direction of the motion (any length > 0; it is normalised).
        (5) joints [int]: The joints used, from joint 1 (5 for the symmetric cup).
        (6) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Vector<float> 1xJ or None]: The joint rates per unit speed [rad/m], J = joints.
    """

    direction = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if not (direction.shape == (3,) and np.isfinite(length) and length > 0.0):
        raise Robot_Error_Cls('The direction %s is not three finite numbers with a length above 0: the tool moves along a direction.'
                              % (direction.tolist(),))
    jacobian = Tool_Jacobian(model, data, q, joints, Robot_Str=Robot_Str)
    if np.linalg.svd(jacobian, compute_uv=False)[-1] < 1e-9:
        return None
    return np.linalg.inv(jacobian) @ np.concatenate([direction / length, np.zeros(joints - 3)])


@dataclass
class Pair_Set_Str:
    """
    Description:
        A set of geom pairs prepared for Clearance in one compiled model (Pair_Set): each side's geoms as an array
        and as Python integers, their bounding radii and whether a pair holds a plane, read from the model once
        instead of at every call.
    """

    # The pairs, each a row (geom, geom).
    #   Unit [Matrix<int> Kx2]
    Pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    # The first and the second geom of each pair, as arrays (the broad phase) and as integers (mj_geomDistance).
    #   Unit [Vector<int> 1xK], [List<int>]
    First: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    Second: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    First_Ids: tp.List[int] = field(default_factory=list)
    Second_Ids: tp.List[int] = field(default_factory=list)
    # The geoms' bounding-sphere radii (geom_rbound; 0 for a plane).
    #   Unit [Vector<float> 1xK, m]
    First_Radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Second_Radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Whether a pair holds a plane, which the broad phase always measures.
    #   Unit [Vector<bool> 1xK]
    Plane: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


def Pair_Set(model: mujoco.MjModel, pairs: np.ndarray) -> Pair_Set_Str:
    """
    Description:
        Prepare geom pairs for Clearance: what the broad phase reads of the model's static geometry, read once.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) pairs [Matrix<int> Kx2]: The geom pairs.

    Returns:
        (1) parameter [Pair_Set_Str(object)]: The prepared pairs.
    """

    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    first, second = pairs[:, 0].copy(), pairs[:, 1].copy()
    return Pair_Set_Str(Pairs=pairs, First=first, Second=second, First_Ids=[int(g) for g in first], Second_Ids=[int(g) for g in second],
                        First_Radius=model.geom_rbound[first].copy(), Second_Radius=model.geom_rbound[second].copy(),
                        Plane=(model.geom_type[pairs] == int(mujoco.mjtGeom.mjGEOM_PLANE)).any(axis=1))


@dataclass
class Collision_Pairs_Str:
    """
    Description:
        The geom pairs whose exact distance the clearance of a robot pose is made of (Collision_Pairs), each a
        row (geom, geom).
    """

    # The robot's geoms against the environment's (the world body's colliding geoms: the floor, the stand).
    #   Unit [Matrix<int> Kx2]
    Environment: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    # The robot's geoms against each other, neither on the same body nor on a parent and its child.
    #   Unit [Matrix<int> Kx2]
    Self: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    # The self pairs between the bodies of joint 3 and joint 5 (the vendored forearm's and wrist 2's
    # capsules), held apart (> 0) but not to the clearance: they are 0.036-0.046 m apart in every pose
    # (F4 Part A).
    #   Unit [Matrix<int> Kx2]
    Exempt: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    # The pairs left out because no joint changes their distance (the base joint turning the shoulder's
    # capsule about its own axis, joint 6 turning the cup about its axis), with that distance.
    #   Unit [Matrix<int> Kx2], [Vector<float> 1xK, m]
    Constant: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    Constant_Distance: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Environment, Self and Exempt prepared for Clearance (Pair_Set).
    #   Unit [Pair_Set_Str]
    Environment_Set: Pair_Set_Str = field(default_factory=Pair_Set_Str)
    Self_Set: Pair_Set_Str = field(default_factory=Pair_Set_Str)
    Exempt_Set: Pair_Set_Str = field(default_factory=Pair_Set_Str)


def Collision_Pairs(model: mujoco.MjModel, *, Robot_Str: Robot_Str) -> Collision_Pairs_Str:
    """
    Description:
        Sort the colliding geoms' pairs for the clearance: the robot's geoms are the colliding geoms of the
        bodies of the robot's tree (the cup's cylinder included); the environment's those of the world body; other
        bodies (the ball) take no part. A colliding geom is one with a contype: a geom that only takes
        contacts (contype 0, a conaffinity: the cup's shape for the ball, F11) touches nothing of the robot or the
        world and is left out, so the pairs are those of the scene without it. A pair on one body or on a parent and its child is left out (the
        joint between them overlaps them by design), and so is a pair whose distance no joint changes:
        measured at CONST_CONSTANT_POSES joint vectors drawn with the seed CONST_CONSTANT_SEED, it varies by
        less than CONST_CONSTANT_TOLERANCE.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Collision_Pairs_Str(object)]: The pairs.
    """

    colliding = [g for g in range(model.ngeom) if model.geom_contype[g]]
    root = model.body_rootid[model.jnt_bodyid[Robot_Str.Joint_Id[0]]]
    robot = [g for g in colliding if model.body_rootid[model.geom_bodyid[g]] == root and model.geom_bodyid[g] != 0]
    world = [g for g in colliding if model.geom_bodyid[g] == 0]
    candidates = [(a, b) for a in robot for b in world]
    # MuJoCo numbers a parent body before its children and the geoms in the order of their bodies, so a later
    # geom's body is never the parent of an earlier one's: of a parent and its child, only the earlier geom's
    # body can be the parent (asserted by a per-commit test on the scene).
    for i, a in enumerate(robot):
        for b in robot[i + 1:]:
            body_a, body_b = model.geom_bodyid[a], model.geom_bodyid[b]
            if body_a != body_b and model.body_parentid[body_b] != body_a:
                candidates.append((a, b))

    data = mujoco.MjData(model)
    rng = np.random.default_rng(CONST_CONSTANT_SEED)
    distances = np.zeros((CONST_CONSTANT_POSES, len(candidates)))
    for row in range(CONST_CONSTANT_POSES):
        Forward_Kinematics(model, data, rng.uniform(np.maximum(Robot_Str.Lower, -np.pi), np.minimum(Robot_Str.Upper, np.pi)), Robot_Str=Robot_Str)
        distances[row] = [mujoco.mj_geomDistance(model, data, a, b, CONST_CONSTANT_CUT, None) for a, b in candidates]
    constant = np.ptp(distances, axis=0) < CONST_CONSTANT_TOLERANCE
    forearm, wrist_2 = model.jnt_bodyid[Robot_Str.Joint_Id[2]], model.jnt_bodyid[Robot_Str.Joint_Id[4]]
    environment, own, exempt, fixed, fixed_distance = [], [], [], [], []
    for index, (a, b) in enumerate(candidates):
        if constant[index]:
            fixed.append((a, b))
            fixed_distance.append(distances[0, index])
        elif b in world:
            environment.append((a, b))
        elif {model.geom_bodyid[a], model.geom_bodyid[b]} == {forearm, wrist_2}:
            exempt.append((a, b))
        else:
            own.append((a, b))
    as_array = (lambda rows: np.array(rows, dtype=np.int64).reshape(-1, 2))
    environment, own, exempt = as_array(environment), as_array(own), as_array(exempt)
    return Collision_Pairs_Str(Environment=environment, Self=own, Exempt=exempt, Constant=as_array(fixed), Constant_Distance=np.array(fixed_distance),
                               Environment_Set=Pair_Set(model, environment), Self_Set=Pair_Set(model, own), Exempt_Set=Pair_Set(model, exempt))


def Clearance(model: mujoco.MjModel, data: mujoco.MjData, pairs: Pair_Set_Str, cut: float) -> tp.Tuple[float, int]:
    """
    Description:
        The smallest exact distance (mj_geomDistance) over the pairs in the pose `data` holds (its geom
        positions must be current: Forward_Kinematics), up to `cut`. A broad phase skips a pair whose
        bounding spheres are farther apart than the cut (their distance is then above it); a plane, whose
        bounding radius is 0, is always measured. A negative distance is a penetration.

        Note:
            The pairs come prepared (Pair_Set), so a call reads only the geoms' current positions; the broad
            phase computes |c_1 - c_2| - r_1 - r_2 in this order, as before the preparation (F10 Part B, step 1:
            byte for byte the same distance and row).

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) data [mujoco.MjData]: Its data in the pose.
        (3) pairs [Pair_Set_Str(object)]: The geom pairs, prepared for this model (Pair_Set).
        (4) cut [float]: The distance above which a pair is not measured [m].

    Returns:
        (1) parameter [float]: The smallest distance, `cut` when every pair is at least that far apart [m].
        (2) parameter [int]: Its pair's row, -1 when none is below the cut.
    """

    if len(pairs.First_Ids) == 0:
        return cut, -1
    xpos = data.geom_xpos
    near = pairs.Plane | (np.linalg.norm(xpos[pairs.First] - xpos[pairs.Second], axis=1) - pairs.First_Radius - pairs.Second_Radius < cut)
    best, which = cut, -1
    first, second = pairs.First_Ids, pairs.Second_Ids
    for row in np.flatnonzero(near).tolist():
        distance = mujoco.mj_geomDistance(model, data, first[row], second[row], cut, None)
        if distance < best:
            best, which = distance, row
    return best, which
