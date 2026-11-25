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
# File Name: ../Robot/Motion.py
## =========================================================================== ##

# Dataclasses (Data classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation; here its kinematics, inverse dynamics and time stepping) [pip3 install mujoco]
import mujoco
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Robot/Core - the robot in the model, its forward kinematics, its collision pairs and their clearance
from Robot.Core import Robot_Str, Collision_Pairs_Str, Forward_Kinematics, Clearance

"""
Description:
    Initialization of constants.

    The robot's motion (SPEC §7.9; F10): the time-optimal joint profile
    from a moving state to rest at a goal (each joint on its own, second order: the joint's speed limit and one
    acceleration limit, no jerk limit, no synchronisation), the stroke of a moving catch (F15: every joint on one
    straight joint line through a peak with a given velocity), the check of a profile's path by exact distances with a
    certificate between the poses checked, the joints' ranges along it, and the command of the vendored position
    servos with the feed-forward of their damping and of the inverse dynamics. Pure computation on the model and the
    data given: no file, no configuration, no printing.
"""
# A start velocity may exceed a joint's speed limit by this fraction (the rounding of a profile evaluated at a time
# inside it); beyond it the start is refused, never clipped.
#   Unit [-]
CONST_SPEED_TOLERANCE = 1e-9
# The phases of a joint's profile: accelerate (or brake and reverse) towards the goal, cruise at the speed limit,
# brake to rest.
#   Unit [int]
CONST_PHASES = 3


class Motion_Error_Cls(ValueError):
    """
    Description:
        A motion cannot be computed: a start beyond a limit, a servo the command cannot be written for. The message
        names the value and what it must be; a script prints it verbatim as an [ERROR] line.
    """


@dataclass
class Profile_Str:
    """
    Description:
        A time-optimal joint profile (Profile): each joint from its start state to rest at its goal in at most three
        phases of constant acceleration, from Start_Time on; after its own end a joint rests at its goal.
    """

    # The time the profile starts at.
    #   Unit [float, s]
    Start_Time: float = 0.0
    # The joints' start positions and velocities, and the goal.
    #   Unit [Vector<float> 1xJ, rad], [Vector<float> 1xJ, rad/s], [Vector<float> 1xJ, rad]
    Start: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Velocity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Goal: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Per joint the durations and the accelerations of the three phases.
    #   Unit [Matrix<float> Jx3, s], [Matrix<float> Jx3, rad/s^2]
    Durations: np.ndarray = field(default_factory=lambda: np.zeros((0, CONST_PHASES)))
    Accelerations: np.ndarray = field(default_factory=lambda: np.zeros((0, CONST_PHASES)))
    # The time from Start_Time to the last joint's rest (the slowest joint).
    #   Unit [float, s]
    Duration: float = 0.0


def Profile(start: np.ndarray, velocity: np.ndarray, goal: np.ndarray, speed: np.ndarray, acceleration: float,
            start_time: float = 0.0) -> Profile_Str:
    """
    Description:
        The time-optimal profile of each joint from (start, velocity) to rest at goal under |v| <= speed and |a| <=
        acceleration (second order). With s = v |v| / (2 a) the distance to stop, the joint heads in the direction
        sigma of goal - start - s (the direction of its velocity when that is 0); with u_0 = sigma v and D = sigma
        (goal - start) the peak speed is u_p = sqrt(a D + u_0^2 / 2), capped at the speed limit: phase 1 at sigma a
        from u_0 to u_p (it brakes and reverses when u_0 < 0), phase 2 cruises at the limit, phase 3 brakes at -sigma a
        to rest. At rest (velocity 0) this is the rest-to-rest trapezoid of Robot.Core.Move_Time per joint.

    Args:
        (1) start [Vector<float> 1xJ]: The start positions [rad].
        (2) velocity [Vector<float> 1xJ]: The start velocities [rad/s], each within its speed limit.
        (3) goal [Vector<float> 1xJ]: The goals [rad] (a wrapping joint's goal is given on the side it is reached).
        (4) speed [Vector<float> 1xJ]: The speed limits [rad/s], each > 0.
        (5) acceleration [float]: The acceleration limit [rad/s^2], > 0.
        (6) start_time [float]: The time the profile starts at [s].

    Returns:
        (1) parameter [Profile_Str(object)]: The profile.
    """

    start, velocity, goal = (np.asarray(v, dtype=np.float64).copy() for v in (start, velocity, goal))
    speed = np.asarray(speed, dtype=np.float64)
    over = np.flatnonzero(np.abs(velocity) > speed * (1.0 + CONST_SPEED_TOLERANCE))
    if over.size:
        j = int(over[0])
        raise Motion_Error_Cls('The joint %d starts at %r rad/s, beyond its speed limit %r rad/s: a profile starts within the limits. Plan '
                               'from a state of a profile within them.' % (j + 1, float(velocity[j]), float(speed[j])))
    v = np.clip(velocity, -speed, speed)
    a = float(acceleration)
    d = goal - start
    stop = v * np.abs(v) / (2.0 * a)
    sigma = np.sign(d - stop)
    sigma = np.where(sigma == 0.0, np.sign(v), sigma)
    u_0, D = sigma * v, sigma * d
    u_p = np.sqrt(np.maximum(a * D + 0.5 * u_0 ** 2, 0.0))
    peak = np.minimum(u_p, speed)
    t_1 = np.maximum((peak - u_0) / a, 0.0)
    t_3 = peak / a
    cruise = np.where(u_p > speed, D - (speed ** 2 - u_0 ** 2) / (2.0 * a) - speed ** 2 / (2.0 * a), 0.0)
    t_2 = np.maximum(cruise, 0.0) / speed
    durations = np.stack([t_1, t_2, t_3], axis=1)
    accelerations = np.stack([sigma * a, np.zeros_like(sigma), -sigma * a], axis=1)
    return Profile_Str(Start_Time=float(start_time), Start=start, Velocity=v, Goal=goal, Durations=durations, Accelerations=accelerations,
                       Duration=float(np.max(durations.sum(axis=1))) if len(start) else 0.0)


def Stroke(peak: np.ndarray, velocity: np.ndarray, speed: np.ndarray, acceleration: float, peak_time: float) -> Profile_Str:
    """
    Description:
        A stroke (F15): every joint on one straight joint line through `peak`, from rest to
        the velocity `velocity` there at `peak_time` and at once back to rest, all joints in the same two phases of
        the same length T = max_j |velocity_j| / acceleration, so that the fastest joint accelerates and brakes at the
        limit and the others in proportion: q(t) = peak + velocity g(t). It starts at peak - velocity T / 2 at
        peak_time - T and ends at peak + velocity T / 2 at peak_time + T. A profile like any other (its middle phase
        has no length): Profile_State, Profile_Extremes and Check_Profile take it as it is. A velocity beyond a joint's
        speed limit, and one that moves no joint, are refused, never clipped.

    Args:
        (1) peak [Vector<float> 1xJ]: The joint positions at the peak [rad].
        (2) velocity [Vector<float> 1xJ]: The joint velocities at the peak [rad/s], each within its speed limit.
        (3) speed [Vector<float> 1xJ]: The speed limits [rad/s], each > 0.
        (4) acceleration [float]: The acceleration limit [rad/s^2], > 0.
        (5) peak_time [float]: The time of the peak [s].

    Returns:
        (1) parameter [Profile_Str(object)]: The stroke.
    """

    peak, velocity = (np.asarray(v, dtype=np.float64).copy() for v in (peak, velocity))
    speed = np.asarray(speed, dtype=np.float64)
    over = np.flatnonzero(np.abs(velocity) > speed * (1.0 + CONST_SPEED_TOLERANCE))
    if over.size:
        j = int(over[0])
        raise Motion_Error_Cls('The joint %d would pass the stroke\'s peak at %r rad/s, beyond its speed limit %r rad/s: a stroke stays '
                               'within the limits. Lower the speed along the line.' % (j + 1, float(velocity[j]), float(speed[j])))
    fastest = float(np.max(np.abs(velocity))) if velocity.size else 0.0
    if not fastest > 0.0:
        raise Motion_Error_Cls('The stroke\'s velocity at its peak moves no joint (%s rad/s): a stroke passes its peak in motion. A catch '
                               'at rest is a profile to the pose (Profile).' % (velocity.tolist(),))
    duration = fastest / float(acceleration)
    durations = np.tile([duration, 0.0, duration], (len(peak), 1))
    accelerations = np.stack([velocity / duration, np.zeros_like(velocity), -velocity / duration], axis=1)
    return Profile_Str(Start_Time=float(peak_time) - duration, Start=peak - 0.5 * velocity * duration, Velocity=np.zeros_like(peak),
                       Goal=peak + 0.5 * velocity * duration, Durations=durations, Accelerations=accelerations, Duration=2.0 * duration)


def Profile_State(profile: Profile_Str, t: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        The joints' positions, velocities and accelerations at the times given: the phases integrated in closed form;
        before Start_Time the start state (at rest only if the start velocity is 0), after a joint's end its goal at
        rest, exactly.

    Args:
        (1) profile [Profile_Str(object)]: The profile.
        (2) t [Vector<float> 1xT]: The times [s].

    Returns:
        (1) parameter [Matrix<float> TxJ]: The positions [rad].
        (2) parameter [Matrix<float> TxJ]: The velocities [rad/s].
        (3) parameter [Matrix<float> TxJ]: The accelerations [rad/s^2].
    """

    tau = np.maximum(np.asarray(t, dtype=np.float64).reshape(-1, 1) - profile.Start_Time, 0.0)
    p, v = profile.Start[None, :], profile.Velocity[None, :]
    q, qd, qdd = np.zeros((len(tau), len(p[0]))), np.zeros((len(tau), len(p[0]))), np.zeros((len(tau), len(p[0])))
    elapsed = np.zeros_like(profile.Start)[None, :]
    done = np.zeros(q.shape, dtype=bool)
    for k in range(CONST_PHASES):
        duration, a = profile.Durations[:, k][None, :], profile.Accelerations[:, k][None, :]
        inside = ~done & (tau < elapsed + duration)
        s = tau - elapsed
        q = np.where(inside, p + v * s + 0.5 * a * s ** 2, q)
        qd = np.where(inside, v + a * s, qd)
        qdd = np.where(inside, a, qdd)
        done |= inside
        p, v = p + v * duration + 0.5 * a * duration ** 2, v + a * duration
        elapsed = elapsed + duration
    goal = np.broadcast_to(profile.Goal[None, :], q.shape)
    return np.where(done, q, goal), np.where(done, qd, 0.0), np.where(done, qdd, 0.0)


def Profile_Extremes(profile: Profile_Str) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        Each joint's smallest and largest position over the profile: its start, its goal and, when phase 1 brakes and
        reverses (the joint starts moving away from where it heads), the point where it stops, start + v |v| / (2 a).

    Args:
        (1) profile [Profile_Str(object)]: The profile.

    Returns:
        (1) parameter [Vector<float> 1xJ]: The smallest positions [rad].
        (2) parameter [Vector<float> 1xJ]: The largest positions [rad].
    """

    a = np.abs(profile.Accelerations[:, 0])
    reverses = (profile.Velocity * np.sign(profile.Accelerations[:, 0]) < 0.0) & (a > 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        turn = np.where(reverses, profile.Start + profile.Velocity * np.abs(profile.Velocity) / (2.0 * np.where(a > 0.0, a, 1.0)), profile.Start)
    points = np.stack([profile.Start, profile.Goal, turn])
    return points.min(axis=0), points.max(axis=0)


def Lever_Bounds(model: mujoco.MjModel, pairs: Collision_Pairs_Str, *, Robot_Str: Robot_Str) -> np.ndarray:
    """
    Description:
        Per joint an upper bound of the distance of any point of the robot's colliding geoms it moves from its axis,
        valid in every pose: along the chain from the joint's anchor, the distances between successive joints'
        anchors (each pair fixed in the body between them) and from the last joint's anchor to the geom's centre (fixed
        in its body), plus the geom's bounding radius. A point then moves at most sum_j bound_j |dq_j| for a joint
        motion dq (each joint turns it about its own axis).

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) pairs [Collision_Pairs_Str(object)]: The robot's collision pairs (Robot.Core.Collision_Pairs).
        (3) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Vector<float> 1xJ]: The bounds [m].
    """

    data = mujoco.MjData(model)
    Forward_Kinematics(model, data, np.zeros(len(Robot_Str.Joint_Id)), Robot_Str=Robot_Str)
    anchors = np.array([data.xanchor[j] for j in Robot_Str.Joint_Id])
    joint_body = [int(model.jnt_bodyid[j]) for j in Robot_Str.Joint_Id]
    geoms = sorted({int(g) for g in np.concatenate([pairs.Environment[:, 0], pairs.Self.ravel(), pairs.Exempt.ravel()])})
    bounds = np.zeros(len(joint_body))
    for g in geoms:
        chain, body = set(), int(model.geom_bodyid[g])
        while body != 0:
            chain.add(body)
            body = int(model.body_parentid[body])
        moved = [j for j, b in enumerate(joint_body) if b in chain]
        if not moved:
            continue
        last = moved[-1]
        for j in moved:
            reach = sum(np.linalg.norm(anchors[i + 1] - anchors[i]) for i in range(j, last)) + np.linalg.norm(data.geom_xpos[g] - anchors[last])
            bounds[j] = max(bounds[j], reach + model.geom_rbound[g])
    return bounds


@dataclass
class Path_Check_Str:
    """
    Description:
        A profile's path checked by exact distances (Check_Profile): the smallest distance at the poses checked per
        pair set (cut at the cut given), the poses checked, and whether the path is certified (every pose far enough
        that no point between two poses can come closer than the clearance) and within the joints' ranges.
    """

    # The smallest distance to the environment, between the robot's parts and of the exempt pair at the poses checked.
    #   Unit [float, m]
    Environment: float = float('inf')
    Self: float = float('inf')
    Exempt: float = float('inf')
    # The poses checked.
    #   Unit [int]
    Poses: int = 0
    # The path is certified collision-free with the clearance, and every joint stays within its range.
    #   Unit [bool]
    Certified: bool = False
    Within_Range: bool = False
    # The first criterion failed ('' when certified and within the ranges): 'range', 'environment', 'self', 'exempt'.
    #   Unit [string]
    Failed: str = ''


def Check_Profile(model: mujoco.MjModel, data: mujoco.MjData, profile: Profile_Str, t_from: float, t_to: float, lever: np.ndarray,
                  step: float, clearance: float, cut: float, pairs: Collision_Pairs_Str, stop: bool = True, *,
                  Robot_Str: Robot_Str) -> Path_Check_Str:
    """
    Description:
        Check a profile's path from t_from to t_to (or its end) by exact distances: poses at
        times chosen so that no robot point moves more than `step` from one to the next (a joint's speed over an
        interval is at most its speed at the interval's start plus the acceleration limit times its length, and a point
        moves at most sum_j lever_j |dq_j|); at each pose the clearance to the environment must be >= clearance + step,
        between the robot's parts >= clearance + 2 step (both geoms move), and the exempt pair > 2 step, so that between
        two poses the clearance is kept (>= clearance, > 0 for the exempt pair). Every joint's extreme positions
        (Profile_Extremes) must lie within its range. The robot and its surroundings only: the ball is not checked.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) data [mujoco.MjData]: A data for the poses; the robot's joint positions are overwritten.
        (3) profile [Profile_Str(object)]: The profile.
        (4) t_from, t_to [float]: The times checked between [s]; t_to beyond the profile's end checks to its end.
        (5) lever [Vector<float> 1xJ]: The joints' lever bounds (Lever_Bounds) [m].
        (6) step [float]: The largest motion of any robot point between two poses [m].
        (7) clearance [float]: The clearance kept to the environment and between the robot's parts [m].
        (8) cut [float]: The cut of the distances measured (at least clearance + 2 step) [m].
        (9) pairs [Collision_Pairs_Str(object)]: The collision pairs, prepared (Robot.Core.Collision_Pairs).
        (10) stop [bool]: Stop at the first pose that fails.
        (11) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Path_Check_Str(object)]: The check.
    """

    check = Path_Check_Str()
    low, high = Profile_Extremes(profile)
    check.Within_Range = bool(np.all(low >= Robot_Str.Lower) and np.all(high <= Robot_Str.Upper))
    if not check.Within_Range:
        check.Failed = 'range'
        if stop:
            return check
    thresholds = (('environment', pairs.Environment_Set, clearance + step), ('self', pairs.Self_Set, clearance + 2.0 * step),
                  ('exempt', pairs.Exempt_Set, 2.0 * step))
    cut = max(cut, clearance + 2.0 * step)
    end = min(t_to, profile.Start_Time + profile.Duration)
    a = float(np.max(np.abs(profile.Accelerations))) if profile.Accelerations.size else 0.0
    total_lever = float(np.sum(lever))
    t = max(t_from, profile.Start_Time)
    certified = True
    while True:
        q, qd, _ = Profile_State(profile, np.array([t]))
        Forward_Kinematics(model, data, q[0], Robot_Str=Robot_Str)
        check.Poses += 1
        for name, pair_set, threshold in thresholds:
            distance, _ = Clearance(model, data, pair_set, cut)
            setattr(check, name.capitalize(), min(getattr(check, name.capitalize()), distance))
            if not (distance >= threshold if name != 'exempt' else distance > threshold):
                certified = False
                check.Failed = check.Failed or name
        if (not certified and stop) or t >= end:
            break
        # The next time: sum_j lever_j (|qd_j| + a dt) dt <= step.
        b = float(np.sum(lever * np.abs(qd[0])))
        dt = step / b if a * total_lever == 0.0 else (-b + np.sqrt(b * b + 4.0 * a * total_lever * step)) / (2.0 * a * total_lever)
        if not np.isfinite(dt):
            break
        t = min(t + dt, end)
    check.Certified = certified and check.Poses > 0
    return check


@dataclass
class Servo_Str:
    """
    Description:
        The robot's position servos in the model (Servos): per joint its actuator, the gain and the damping of the
        affine servo force = kp (ctrl - q) - kv qd, the force range and the control range.
    """

    # The actuators of the joints, in the robot's joint order.
    #   Unit [Vector<int> 1xJ]
    Actuators: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # The gains and the dampings.
    #   Unit [Vector<float> 1xJ, N m/rad], [Vector<float> 1xJ, N m s/rad]
    Kp: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Kv: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # The force ranges and the control ranges, [min, max] per joint.
    #   Unit [Matrix<float> Jx2, N m], [Matrix<float> Jx2, rad]
    Force_Range: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    Control_Range: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def Servos(model: mujoco.MjModel, *, Robot_Str: Robot_Str) -> Servo_Str:
    """
    Description:
        The position servos of the robot's joints as the vendored model defines them (MuJoCo Menagerie's UR10e: a
        general actuator with a fixed gain and the affine bias (0, -kp, -kv) on each joint). A joint without exactly one
        such actuator is refused, naming it: the command (Servo_Command) holds only for this form.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Servo_Str(object)]: The servos.
    """

    actuators, problems = [], []
    for k, joint in enumerate(Robot_Str.Joint_Id):
        found = [a for a in range(model.nu) if model.actuator_trntype[a] == int(mujoco.mjtTrn.mjTRN_JOINT) and model.actuator_trnid[a, 0] == joint]
        if len(found) != 1:
            problems.append('the joint %d has %d actuators' % (k + 1, len(found)))
            continue
        a = found[0]
        kp, bias = float(model.actuator_gainprm[a, 0]), model.actuator_biasprm[a, :3]
        if model.actuator_gaintype[a] != int(mujoco.mjtGain.mjGAIN_FIXED) or model.actuator_biastype[a] != int(mujoco.mjtBias.mjBIAS_AFFINE) \
                or bias[0] != 0.0 or bias[1] != -kp or not model.actuator_forcelimited[a]:
            problems.append('the joint %d\'s actuator is not a force-limited position servo with the bias (0, -kp, -kv)' % (k + 1))
        actuators.append(a)
    if problems:
        raise Motion_Error_Cls('The robot\'s actuators cannot be commanded: %s. The command holds for the vendored UR10e\'s servos (SPEC '
                               '§7.9).' % '; '.join(problems))
    actuators = np.array(actuators, dtype=np.int64)
    return Servo_Str(Actuators=actuators, Kp=model.actuator_gainprm[actuators, 0].copy(), Kv=-model.actuator_biasprm[actuators, 2].copy(),
                     Force_Range=model.actuator_forcerange[actuators].copy(), Control_Range=model.actuator_ctrlrange[actuators].copy())


def Servo_Command(model: mujoco.MjModel, data: mujoco.MjData, servos: Servo_Str, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, *,
                  Robot_Str: Robot_Str) -> np.ndarray:
    """
    Description:
        The servos' command for a planned state: ctrl = q + (kv qd + tau) / kp, with tau the
        inverse dynamics of the robot at (q, qd, qdd) (mj_inverse: the inertia, the bias forces, the joints' damping and
        armature; the cup on the flange), so that the servo force kp (ctrl - q) - kv qd equals tau on the plan. The
        rest of the model (the ball) is taken from `data` at rest.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) data [mujoco.MjData]: A data for the inverse dynamics; its state is overwritten.
        (3) servos [Servo_Str(object)]: The servos (Servos).
        (4 - 6) q, qd, qdd [Vector<float> 1xJ]: The planned positions [rad], velocities [rad/s], accelerations [rad/s^2].
        (7) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Vector<float> 1xJ]: The command [rad].
    """

    data.qvel[:], data.qacc[:] = 0.0, 0.0
    data.qpos[Robot_Str.Position_Address] = q
    data.qvel[Robot_Str.Velocity_Address] = qd
    data.qacc[Robot_Str.Velocity_Address] = qdd
    mujoco.mj_inverse(model, data)
    tau = data.qfrc_inverse[Robot_Str.Velocity_Address]
    return np.asarray(q, dtype=np.float64) + (servos.Kv * np.asarray(qd, dtype=np.float64) + tau) / servos.Kp


@dataclass
class Track_Str:
    """
    Description:
        A planned motion tracked by the servos in MuJoCo (Track): the joints at every control tick, and the ticks at
        which a servo's force or command reached its range.
    """

    # The joint positions at each tick, before its command is applied (the state the tick's plan is compared with).
    #   Unit [Matrix<float> TxJ, rad]
    Positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # The ticks with a servo force at its range, and with a command beyond its control range (MuJoCo clamps it).
    #   Unit [int]
    Force_Limit_Ticks: int = 0
    Control_Limit_Ticks: int = 0


def Track(model: mujoco.MjModel, data: mujoco.MjData, servos: Servo_Str, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, hold: int, *,
          Robot_Str: Robot_Str) -> Track_Str:
    """
    Description:
        Track a planned motion given at the control ticks (one step of the model's time step per tick): `hold` ticks
        holding the first planned state, then each tick's command (Servo_Command) and one mj_step. `data` holds the
        start (the robot at the first planned state at rest, the rest of the model placed by the caller).

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) data [mujoco.MjData]: The simulation's data, stepped.
        (3) servos [Servo_Str(object)]: The servos (Servos).
        (4 - 6) q, qd, qdd [Matrix<float> TxJ]: The planned states at the ticks [rad, rad/s, rad/s^2].
        (7) hold [int]: The ticks holding the first state before the motion.
        (8) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [Track_Str(object)]: The tracked joints and the ticks at a limit.
    """

    inverse = mujoco.MjData(model)
    inverse.qpos[:] = data.qpos
    out = Track_Str(Positions=np.zeros((len(q), len(Robot_Str.Joint_Id))))
    rest = np.zeros(len(Robot_Str.Joint_Id))
    for k in range(-hold, len(q)):
        row = max(k, 0)
        if k >= 0:
            out.Positions[k] = data.qpos[Robot_Str.Position_Address]
        command = Servo_Command(model, inverse, servos, q[row], qd[row] if k >= 0 else rest, qdd[row] if k >= 0 else rest, Robot_Str=Robot_Str)
        out.Control_Limit_Ticks += int(np.any((command < servos.Control_Range[:, 0]) | (command > servos.Control_Range[:, 1])))
        data.ctrl[servos.Actuators] = command
        mujoco.mj_step(model, data)
        force = data.actuator_force[servos.Actuators]
        out.Force_Limit_Ticks += int(np.any((force <= servos.Force_Range[:, 0]) | (force >= servos.Force_Range[:, 1])))
    return out
