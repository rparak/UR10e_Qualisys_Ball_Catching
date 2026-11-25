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
# File Name: ../Catch/Interception.py
## =========================================================================== ##

# CSV (The interception's files)
import csv
# Dataclasses (Data classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation; here the tracking's data) [pip3 install mujoco]
import mujoco
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the areas' structure and the catch modes
from Configuration.Core import Areas_Str, CONST_CATCH_MODES
#   ../Flight/Core - the true trajectory
from Flight.Core import Flight_Model_Str, Simulate
#   ../Observation/Core - M4 (a numerical predictor's weights)
from Observation.Core import Observation_Model_Str
#   ../Dataset/Core - the dataset's reader
from Dataset.Core import Read_Dataset
#   ../Prediction/Core - a refused prediction
from Prediction.Core import Prediction_Error_Cls
#   ../Learning/Predictor - a run's predictor as a function of a prefix
from Learning.Predictor import Run_Predictor
#   ../Robot/Core - the forward kinematics, the move time and the joint rates along a line
from Robot.Core import Forward_Kinematics, Move_Time, Line_Rates
#   ../Robot/Motion - the profile, the stroke, the path check and the servos
from Robot.Motion import Profile, Profile_State, Profile_Str, Stroke, Lever_Bounds, Check_Profile, Path_Check_Str, Servos, Track
#   ../Scene/Core - the scene, its robot, its ball and its cup
from Scene.Core import Build_Scene, Scene_Robot, CONST_BALL_BODY, CONST_ROBOT_PREFIX, CONST_CUP_BOTTOM_GEOM, CONST_CUP_MOUTH_SITE
#   ../Catch/Core - the catch pose and the catch's error
from Catch.Core import Catch_Context, Catch_Poses, Check_Pose, Criteria_Str, Catch_Error_Cls, Read_Table, CONST_JOINTS_USED
#   ../Catch/Workspace - the throws' samples, the allowance and the wrapping joints
from Catch.Workspace import Flight_Samples, Joint_Wrap, CONST_SAMPLE_STEP, CONST_FLOOR, CONST_ALLOWANCE
#   ../Catch/Decision - the truth, the margins, the candidates, the decision, the rates
from Catch.Decision import (Truth_Samples, Exact_Reach, Margin_Errors, Margins, Candidates, Decide, Horizon_Bins, Reach_Bound, Rates,
                            Clopper_Pearson, Stage_Rule)

"""
Description:
    Initialization of constants.

    The interception (SPEC §2.9, §7.11; F10): the catch decision at every
    observed sample from the first stage on, with the margins of the largest stage of a ladder at or below the
    sample's N (calibrated on the train partitions, Q3), and after a go the robot moved to the cup's catch pose at the
    catch time of the candidate with the most slack (Q8), re-planned one plan at a time on the newest sample (Q2): a
    plan is ready a compute budget after the moment it starts, from the robot's state then, the time-optimal profile
    of each joint to the pose for the newest prediction at the catch time (Robot.Motion.Profile), accepted when the
    robot rests at the goal a guard before the catch time and the profile's path is certified (Q6). The result is
    kinematic (Q1): at the true crossing of the cup's mouth plane the ball's centre within the mouth's radius minus the
    ball's from the cup's axis, its velocity within the angle tolerance of the axis, the robot at rest; the motion
    tracked by the vendored servos with the feed-forward (Q7). No contact is simulated (F11).

    A moving catch (F15; Interception_Setup_Str.Catch_Mode 'moving'): the robot goes rest to
    rest to the start of a stroke and waits there; the stroke (Robot.Motion.Stroke) runs every joint on one straight
    joint line through the catch pose, so that the cup moves along the ball's direction, peaks at the predicted contact
    with the cup's bottom and brakes at the limit at once. Its speed is the largest scale of Stroke_Scales whose stroke
    is in time and certified; without one the plan is the static one. The result is read at the true crossing of the
    moving mouth plane, and the rows carry further columns (CONST_MOVING_COLUMNS), in moving mode only. Every number
    is a simulation result; nothing is run on, or validated on, the real UR10e.
"""
# The catch mode of a moving catch (Configuration.Core.CONST_CATCH_MODES; the other is the static catch of F10).
#   Unit [string]
CONST_MOVING = CONST_CATCH_MODES[1]
# The inner radius of the cup's mouth: the opening of the lip ring, 62.5 mm (63.5 mm at its chamfer; SPEC §4.4, measured
# on its STL, corrected 2026-11-22); the smaller value. With the ball's radius (the scene's ball) it gives the lateral
# tolerance (Q1).
#   Unit [m]
CONST_MOUTH_RADIUS = 0.0625
# The true trajectory's time step for the crossing of the cup's mouth plane (linearly interpolated between).
#   Unit [s]
CONST_CROSSING_STEP = 0.0005
# The ticks the tracking holds the ready pose before the motion (0.5 s at the vendored time step).
#   Unit [int]
CONST_HOLD_TICKS = 250
# Where the ball is parked for the tracking: on the floor at the flight area's centre, 3 m from the robot (it takes no
# part in the tracking; F11 simulates it).
#   Unit [Vector<float> 1x2, m]
CONST_BALL_PARK = (0.0, -3.0)
# The cut of the distances reported along an executed path (the planner's check cuts at its thresholds).
#   Unit [m]
CONST_REPORT_CUT = 0.2
# The columns of the interception's files (SPEC §9.13).
#   Unit [Tuple<string>]
CONST_PER_THROW_COLUMNS = ('dataset', 'flight', 'seed', 'acceleration_deg_s2', 'catchable', 'decision', 'go_n', 'go_time_s', 'outcome',
                           'stages_decision', 'stages_go_n', 'stages_outcome', 'catch_time_s', 'plans', 'rejected', 'arrival_s', 'crossing_s',
                           'margin_s', 'lateral_m', 'angle_deg', 'clear_environment_m', 'clear_self_m', 'clear_exempt_m', 'poses',
                           'certified', 'within_limits', 'at_rest', 'intercepted', 'tracking_m', 'tracking_joint_rad', 'force_limit_ticks',
                           'control_limit_ticks', 'refused')
CONST_SUMMARY_COLUMNS = ('dataset', 'partition', 'seed', 'acceleration_deg_s2', 'throws', 'catchable', 'not_catchable', 'go', 'false_go',
                         'false_go_rate', 'false_go_lower', 'false_go_upper', 'false_no_go', 'false_no_go_rate', 'false_no_go_lower',
                         'false_no_go_upper', 'stages_false_no_go_rate', 'go_after_stages', 'intercepted', 'intercepted_of_catchable',
                         'intercepted_lower', 'intercepted_upper', 'intercepted_of_go', 'unplanned', 'no_crossing', 'r2_violations',
                         'lateral_p50_m', 'lateral_p95_m', 'lateral_max_m', 'margin_min_s', 'margin_p50_s', 'clear_environment_min_m',
                         'clear_self_min_m', 'clear_exempt_min_m', 'tracking_p50_m', 'tracking_max_m', 'force_limit_ticks',
                         'control_limit_ticks', 'refused')
# The further columns of a moving catch's files, after the columns above, in moving mode only (SPEC §9.13).
# Per throw: how the go was planned ('moving' with a stroke, 'static' the fallback), the stroke's
# scale, start and end, the cup's speed along the ball's velocity and the ball's speed relative to the cup at the
# crossing, the angle between the cup's velocity and the ball's, and whether the crossing lies inside the stroke.
#   Unit [Tuple<string>]
CONST_MOVING_COLUMNS = ('plan_mode', 'stroke_scale', 'stroke_start_s', 'stroke_end_s', 'cup_speed_m_s', 'relative_speed_m_s',
                        'direction_deg', 'in_stroke')
CONST_MOVING_SUMMARY_COLUMNS = ('strokes', 'stroke_scale_mean', 'cup_speed_p50_m_s', 'relative_speed_p50_m_s', 'relative_speed_p95_m_s')
# The first columns of a row are the decision's (the dataset to the catch time): what F9's truth and the decision at
# every sample give, whatever the motion (the reading R1 of F15 compares them with the static catch's stored rows).
#   Unit [int]
CONST_DECISION_COLUMNS = CONST_PER_THROW_COLUMNS.index('catch_time_s') + 1


@dataclass
class Interception_Setup_Str:
    """
    Description:
        Everything a chunk of throws needs (Intercept_Chunk, Calibrate_Chunk), picklable for a worker process: the
        areas, the criteria, the models, the run, the robot's limits, the area's reach grids, the decision's ladder
        and margins, and the planner's parameters.
    """

    # The areas (the scene draws them; Areas.Catch the decision's area) and the truth area's centre and size.
    #   Unit [Areas_Str], [List<float> m], [List<float> m]
    Areas: Areas_Str = field(default_factory=Areas_Str)
    Truth_Centre: tp.List[float] = field(default_factory=list)
    Truth_Size: tp.List[float] = field(default_factory=list)
    # The catch pose's criteria (the clearance also kept along a path).
    #   Unit [Criteria_Str]
    Criteria: Criteria_Str = field(default_factory=Criteria_Str)
    # The flight model and M4.
    #   Unit [Flight_Model_Str], [Observation_Model_Str]
    Flight: Flight_Model_Str = field(default_factory=Flight_Model_Str)
    Observation: Observation_Model_Str = field(default_factory=Observation_Model_Str)
    # The run: its predictor's name and folder.
    #   Unit [string], [string]
    Model_Name: str = ''
    Run: str = ''
    # The ready pose, the joints' speeds and the accelerations (the nominal first).
    #   Unit [Vector<float> 1x6, rad], [Vector<float> 1x6, rad/s], [List<float> rad/s^2]
    Ready: np.ndarray = field(default_factory=lambda: np.zeros(6))
    Speed: np.ndarray = field(default_factory=lambda: np.zeros(6))
    Accelerations: tp.List[float] = field(default_factory=list)
    # Per acceleration the reach grid of the area's lattice cells and its first cell (Catch.Decision.Reach_Grid), and
    # the lattice's step.
    #   Unit [List<Tuple<Array<float>, Vector<int>>>], [float, m]
    Grids: tp.List[tp.Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    Step: float = 0.0
    # The ladder's stages, F9's stages (the baseline) and the margins per (seed, stage).
    #   Unit [List<int>], [List<int>], [Dict<Tuple<string, int>, Tuple>]
    Stages: tp.List[int] = field(default_factory=list)
    Decision_Stages: tp.List[int] = field(default_factory=list)
    Margins: tp.Dict[tp.Tuple[str, int], tp.Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    # The planner: the latency, the compute budget, the guard, the path check's step, the angle tolerance.
    #   Unit [float, s], [float, s], [float, s], [float, m], [float, rad]
    Latency: float = 0.0
    Compute_Budget: float = 0.0
    Guard: float = 0.0
    Path_Step: float = 0.0
    Angle_Tolerance: float = 0.0
    # How the robot catches (Configuration.Core.CONST_CATCH_MODES; the static catch of F10 by default), and for a moving
    # catch (F15) the stroke's speed scales in the order they are tried and the compute budget of one plan.
    #   Unit [string], [List<float>], [float, s]
    Catch_Mode: str = CONST_CATCH_MODES[0]
    Stroke_Scales: tp.List[float] = field(default_factory=list)
    Stroke_Budget: float = 0.0


def Ladder_Stage(n: int, stages: tp.Sequence[int]) -> int:
    """
    Description:
        The ladder's stage whose margins a decision at N observed samples uses: the largest stage <= N; -1 below the
        first stage (no decision).

    Args:
        (1) n [int]: The observed samples.
        (2) stages [Sequence<int>]: The ladder, increasing.

    Returns:
        (1) parameter [int]: The stage, or -1.
    """

    below = [s for s in stages if s <= n]
    return int(max(below)) if below else -1


def Prediction_Grid(trajectory: tp.Any, t_0: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        A trajectory's states on the truth's grid (every CONST_SAMPLE_STEP from the release) after t_0 to its end, as
        the decision samples it (scripts/catch.py decide).

    Args:
        (1) trajectory [Trajectory_Str or Query_Trajectory_Str(object)]: The predicted trajectory.
        (2) t_0 [float]: Its last observed time [s].

    Returns:
        (1) parameter [Vector<float> 1xK]: The times [s].
        (2) parameter [Matrix<float> Kx6]: The states [m, m/s].
    """

    first = int(np.floor(trajectory.T_0 / CONST_SAMPLE_STEP)) + 1
    last = int(np.floor(trajectory.T_End / CONST_SAMPLE_STEP + 1e-9))
    times = np.arange(first, last + 1) * CONST_SAMPLE_STEP
    times = times[times <= trajectory.T_End]
    return times, (trajectory.State(times) if len(times) else np.zeros((0, 6)))


def Candidate_Slack(h: np.ndarray, position: np.ndarray, cross_margin: np.ndarray, time_margin: np.ndarray, grid: np.ndarray,
                    low: np.ndarray, step: float, centre: tp.Sequence[float], size: tp.Sequence[float],
                    bounds: tp.Dict[tp.Tuple[int, int, int], np.ndarray]) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        Per candidate of one prediction (Catch.Decision.Candidates) whether it passes the decision's test
        (Catch.Decision.Decide: inside the area shrunk by its bin's cross-track margin, its conservative reach <= its
        horizon - CONST_ALLOWANCE - its bin's time margin, the same expressions) and its slack, that right side minus
        the reach; the go of Decide is the any() of the first (pinned by a test).

    Args:
        (1) h [Vector<float> 1xC]: The candidates' horizons [s].
        (2) position [Matrix<float> Cx3]: Their predicted positions [m].
        (3 - 4) cross_margin, time_margin: The stage's margins (Catch.Decision.Margins) [m], [s].
        (5 - 6) grid, low: The reach grid and its first cell (Catch.Decision.Reach_Grid).
        (7) step [float]: The lattice's step [m].
        (8 - 9) centre, size [Vector<float> 1x3]: Areas.Catch [m].
        (10) bounds [Dict]: A cache of Catch.Decision.Reach_Bound by the steps (filled here).

    Returns:
        (1) parameter [Vector<bool> 1xC]: Whether each candidate passes.
        (2) parameter [Vector<float> 1xC]: Its slack [s], -inf when it is not inside the shrunk area.
    """

    passing, slack = np.zeros(len(h), dtype=bool), np.full(len(h), -np.inf)
    if len(h) == 0:
        return passing, slack
    b = Horizon_Bins(h)
    m, tm = cross_margin[b], time_margin[b]
    with np.errstate(invalid='ignore'):
        inside = np.all(np.abs(np.asarray(position) - np.asarray(centre)) <= 0.5 * np.asarray(size) - m, axis=1)
    cell = np.round(np.asarray(position) / step).astype(np.int64) - low
    inside &= np.all((cell >= 0) & (cell < np.array(grid.shape)), axis=1)
    candidates = np.flatnonzero(inside)
    steps = np.ceil(m[candidates] / step - 1e-9).astype(np.int64)
    for key in {tuple(int(v) for v in s) for s in steps}:
        select = candidates[np.all(steps == key, axis=1)]
        if key not in bounds:
            bounds[key] = Reach_Bound(grid, key)
        reach = bounds[key][tuple(cell[select].T)]
        allowed = h[select] - CONST_ALLOWANCE - tm[select]
        passing[select] = reach <= allowed
        slack[select] = allowed - reach
    return passing, slack


def Wrapped_Goal(q: np.ndarray, reference: np.ndarray, wrap: np.ndarray) -> np.ndarray:
    """
    Description:
        A joint vector of the closed form as a goal reached from `reference`: a wrapping joint the nearer way round.

    Args:
        (1) q [Vector<float> 1x6]: The joint vector [rad].
        (2) reference [Vector<float> 1x6]: The position it is reached from [rad].
        (3) wrap [Vector<bool> 1x6]: Which joints wrap.

    Returns:
        (1) parameter [Vector<float> 1x6]: The goal [rad].
    """

    d = np.asarray(q, dtype=np.float64) - reference
    return reference + np.where(wrap, (d + np.pi) % (2.0 * np.pi) - np.pi, d)


def Pose_Goal(point: np.ndarray, velocity: np.ndarray, reference: np.ndarray, nearest: bool, wrap: np.ndarray, speed: np.ndarray,
              acceleration: float, *, Catch_Context_Str: tp.Any) -> tp.Optional[np.ndarray]:
    """
    Description:
        The goal for a catch at a point with the cup's axis against a velocity: among the closed form's qualifying
        poses (Catch.Core.Check_Pose), the fastest from `reference` at rest (Robot.Core.Move_Time; the first plan), or
        the one nearest to `reference` in the largest joint change (a re-plan: the previous goal). None without one.

    Args:
        (1) point [Vector<float> 1x3]: The cup mouth's point [m].
        (2) velocity [Vector<float> 1x3]: The ball's velocity there [m/s].
        (3) reference [Vector<float> 1x6]: The ready pose or the previous goal [rad].
        (4) nearest [bool]: The nearest instead of the fastest.
        (5) wrap [Vector<bool> 1x6]: Which joints wrap.
        (6 - 7) speed, acceleration: The joints' limits [rad/s], [rad/s^2].
        (8) Catch_Context_Str [Catch_Context_Str(object)]: REQUIRED, keyword-only: the catch-pose context.

    Returns:
        (1) parameter [Vector<float> 1x6 or None]: The goal [rad].
    """

    axis = -np.asarray(velocity, dtype=np.float64) / np.linalg.norm(velocity)
    best, best_value = None, np.inf
    for q in Catch_Poses(point, axis[None], Catch_Context_Str=Catch_Context_Str)[0]:
        if not np.all(np.isfinite(q)):
            continue
        goal = Wrapped_Goal(q, reference, wrap)
        value = float(np.max(np.abs(goal - reference))) if nearest else \
            float(Move_Time(reference[:CONST_JOINTS_USED], goal[None, :CONST_JOINTS_USED], speed[:CONST_JOINTS_USED], acceleration,
                            np.zeros(CONST_JOINTS_USED, dtype=bool))[0])
        if value < best_value and Check_Pose(q, Catch_Context_Str=Catch_Context_Str).Failed == '':
            best, best_value = goal, value
    return best


@dataclass
class Plan_Str:
    """
    Description:
        The executed motion of one go (Plan_Go): the accepted profiles in order (each runs from its start to the next
        one's), the goals rejected, and whether a first plan was accepted at all.
    """

    # The accepted profiles.
    #   Unit [List<Profile_Str>]
    Profiles: tp.List[Profile_Str] = field(default_factory=list)
    # The goals tried and not accepted (no prediction, no qualifying pose, too late, a path not certified).
    #   Unit [int]
    Rejected: int = 0
    # A moving catch (F15): whether the last profile is a stroke, its scale, and the catch pose each profile serves (the
    # pose its plan aims the cup at: an approach ends at its stroke's start, not in the pose). Empty for the static
    # catch, whose profiles end in their poses.
    #   Unit [bool], [float], [List<Vector<float> 1x6, rad>]
    Stroke: bool = False
    Scale: float = 0.0
    Poses: tp.List[np.ndarray] = field(default_factory=list)


def Plan_Go(t_obs: np.ndarray, n_go: int, t_hat: float, predict: tp.Callable[[int], tp.Any], acceleration: float, wrap: np.ndarray,
            lever: np.ndarray, *, Interception_Setup_Str: Interception_Setup_Str, Catch_Context_Str: tp.Any) -> Plan_Str:
    """
    Description:
        The re-planned motion after a go at the n_go-th sample: one plan at a time,
        each started at the newest sample seen (its capture time + the latency) when the previous one is done and ready
        the compute budget later, at the next control tick; its goal the catch pose for the prediction of that sample's
        prefix at t_hat (the first: the fastest from the ready pose; later: the qualifying pose nearest the previous
        goal); its profile from the robot's state at that tick; accepted when the robot rests at the goal at least the
        guard before t_hat and its path is certified (Robot.Motion.Check_Profile), else the current motion continues.
        Plans stop when one could no longer rest the guard before t_hat.

        A moving catch (F15; Catch_Mode 'moving'): a plan is ready Stroke_Budget after it starts and is first tried as a
        stroke (Stroke_Plan): its profile then ends at rest at the stroke's start, and the stroke of the last plan
        accepted so is appended as the last profile. Plans stop when one could not be ready before that stroke starts.
        A plan without a stroke in time and certified is the static one above, as long as no stroke is pending.

    Args:
        (1) t_obs [Vector<float> 1xM]: The observed samples' capture times [s].
        (2) n_go [int]: The go's N.
        (3) t_hat [float]: The catch time [s].
        (4) predict [Callable[[int], trajectory or None]]: The prediction of the first k samples (None when refused).
        (5) acceleration [float]: The acceleration limit [rad/s^2].
        (6) wrap [Vector<bool> 1x6]: Which joints wrap.
        (7) lever [Vector<float> 1x6]: The lever bounds (Robot.Motion.Lever_Bounds) [m].
        (8) Interception_Setup_Str [Interception_Setup_Str(object)]: REQUIRED, keyword-only: the setup.
        (9) Catch_Context_Str [Catch_Context_Str(object)]: REQUIRED, keyword-only: the catch-pose context.

    Returns:
        (1) parameter [Plan_Str(object)]: The accepted profiles.
    """

    s = Interception_Setup_Str
    c = Catch_Context_Str
    tick = float(c.Model.opt.timestep)
    seen = np.asarray(t_obs, dtype=np.float64) + s.Latency
    moving = s.Catch_Mode == CONST_MOVING
    budget, depth = (s.Stroke_Budget, Cup_Depth(c.Model)) if moving else (s.Compute_Budget, 0.0)
    plan = Plan_Str()
    stroke: tp.Optional[Profile_Str] = None
    goal, k, t_free = None, n_go, float(seen[n_go - 1])
    while True:
        t_ready = np.ceil((t_free + budget) / tick - 1e-9) * tick
        if t_ready >= (t_hat - s.Guard if stroke is None else stroke.Start_Time):
            break
        trajectory = predict(k)
        candidate, state = None, None
        if trajectory is not None and trajectory.T_0 <= t_hat <= trajectory.T_End:
            state = trajectory.State(np.array([t_hat]))[0]
            reference = s.Ready if goal is None else goal
            candidate = Pose_Goal(state[:3], state[3:], reference, goal is not None, wrap, s.Speed, acceleration, Catch_Context_Str=c)
        accepted = False
        if candidate is not None:
            if plan.Profiles:
                q, qd, _ = Profile_State(plan.Profiles[-1], np.array([t_ready]))
                start, velocity = q[0], qd[0]
            else:
                start, velocity = s.Ready, np.zeros(len(s.Ready))
            found = Stroke_Plan(candidate, state, start, velocity, t_ready, t_hat, acceleration, lever, depth, Interception_Setup_Str=s,
                                Catch_Context_Str=c) if moving else None
            if found is not None:
                plan.Profiles.append(found[0])
                plan.Poses.append(candidate)
                stroke, plan.Scale, goal, accepted = found[1], found[2], candidate, True
            elif stroke is None:
                profile = Profile(start, velocity, candidate, s.Speed, acceleration, start_time=t_ready)
                if t_ready + profile.Duration <= t_hat - s.Guard:
                    check = Check_Profile(c.Model, c.Data, profile, t_ready, np.inf, lever, s.Path_Step, s.Criteria.Clearance, 0.0, c.Pairs,
                                          Robot_Str=c.Robot)
                    if check.Certified and check.Within_Range:
                        plan.Profiles.append(profile)
                        goal, accepted = candidate, True
                        if moving:
                            plan.Poses.append(candidate)
        plan.Rejected += int(not accepted)
        # The next plan: the newest sample seen when this one is done, or the next sample to come.
        t_free = t_ready
        newer = np.flatnonzero((seen <= t_free) & (np.arange(1, len(seen) + 1) > k))
        if len(newer):
            k = int(newer[-1]) + 1
        else:
            later = np.flatnonzero(np.arange(1, len(seen) + 1) > k)
            if not len(later):
                break
            k = int(later[0]) + 1
            t_free = float(seen[k - 1])
    if stroke is not None:
        plan.Profiles.append(stroke)
        plan.Poses.append(goal)
        plan.Stroke = True
    return plan


def Cup_Depth(model: mujoco.MjModel) -> float:
    """
    Description:
        How far below the plane of the cup's mouth the centre of a ball lying on the cup's bottom is: the mouth's height
        above the flange less the height of the bottom's face and the ball's radius, from the scene's cup and ball
        (SPEC §4.4: 0.190 - 0.005 - 0.030 = 0.155 m). A ball that crosses the mouth plane along the cup's axis meets the
        bottom after this distance relative to the cup. A scene without the cup's bottom is refused.

    Args:
        (1) model [mujoco.MjModel]: The compiled scene.

    Returns:
        (1) parameter [float]: The depth [m].
    """

    bottom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CONST_ROBOT_PREFIX + CONST_CUP_BOTTOM_GEOM)
    mouth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CONST_ROBOT_PREFIX + CONST_CUP_MOUTH_SITE)
    if bottom < 0 or mouth < 0:
        raise Catch_Error_Cls('The scene holds no geom "%s" or no site "%s": a moving catch aims its stroke at the contact with the cup\'s '
                              'bottom, which the scene of scripts/Maintenance/build_scene.py holds (SPEC §7.10).'
                              % (CONST_ROBOT_PREFIX + CONST_CUP_BOTTOM_GEOM, CONST_ROBOT_PREFIX + CONST_CUP_MOUTH_SITE))
    return float(model.site_pos[mouth][2]) - float(model.geom_pos[bottom][2] + model.geom_size[bottom][1]) - _Ball_Radius(model)


def Stroke_Plan(pose: np.ndarray, state: np.ndarray, start: np.ndarray, velocity: np.ndarray, t_ready: float, t_hat: float,
                acceleration: float, lever: np.ndarray, depth: float, *, Interception_Setup_Str: Interception_Setup_Str,
                Catch_Context_Str: tp.Any) -> tp.Optional[tp.Tuple[Profile_Str, Profile_Str, float]]:
    """
    Description:
        One plan of a moving catch (F15): for the catch pose and the ball's predicted state
        at the catch time, the stroke and the approach to its start. The cup moves along the ball's velocity at the
        speed w = scale x the largest speed the joint speeds allow at the pose (Robot.Core.Line_Rates), the joint
        velocities w x the rates; the ball then meets the bottom depth / (|v| - w) after the catch time, and the stroke
        (Robot.Motion.Stroke) peaks then, at the joints a cup moving at w from the pose at the catch time would have,
        and brakes at once. The approach is the time-optimal profile from the robot's state when the plan is ready to
        rest at the stroke's start (Robot.Motion.Profile). The scales of Stroke_Scales are tried in their order; the
        first whose approach ends at or before the stroke's start, with both paths certified and within the joints'
        ranges (Robot.Motion.Check_Profile), is taken. None without one, or at a pose where the cup cannot follow the
        direction (a singular Jacobian).

    Args:
        (1) pose [Vector<float> 1x6]: The catch pose [rad].
        (2) state [Vector<float> 1x6]: The ball's predicted state at the catch time [m, m/s].
        (3 - 4) start, velocity [Vector<float> 1x6]: The robot's state when the plan is ready [rad], [rad/s].
        (5) t_ready [float]: The time the plan is ready [s].
        (6) t_hat [float]: The catch time [s].
        (7) acceleration [float]: The acceleration limit [rad/s^2].
        (8) lever [Vector<float> 1x6]: The lever bounds (Robot.Motion.Lever_Bounds) [m].
        (9) depth [float]: The ball's way from the mouth plane to the cup's bottom (Cup_Depth) [m].
        (10) Interception_Setup_Str [Interception_Setup_Str(object)]: REQUIRED, keyword-only: the setup.
        (11) Catch_Context_Str [Catch_Context_Str(object)]: REQUIRED, keyword-only: the catch-pose context.

    Returns:
        (1) parameter [Tuple<Profile_Str, Profile_Str, float> or None]: The approach, the stroke and its scale.
    """

    s, c = Interception_Setup_Str, Catch_Context_Str
    ball_speed = float(np.linalg.norm(state[3:]))
    rates = Line_Rates(c.Model, c.Data, pose, state[3:], CONST_JOINTS_USED, Robot_Str=c.Robot)
    if rates is None:
        return None
    fastest = float(np.min(s.Speed[:CONST_JOINTS_USED] / np.maximum(np.abs(rates), 1e-12)))
    rates = np.append(rates, np.zeros(len(pose) - CONST_JOINTS_USED))
    for scale in s.Stroke_Scales:
        w = scale * fastest
        if not w < ball_speed:
            continue
        inside = depth / (ball_speed - w)
        stroke = Stroke(pose + rates * w * inside, rates * w, s.Speed, acceleration, t_hat + inside)
        if stroke.Start_Time <= t_ready:
            continue
        approach = Profile(start, velocity, stroke.Start, s.Speed, acceleration, start_time=t_ready)
        if t_ready + approach.Duration > stroke.Start_Time:
            continue
        certified = True
        for profile in (approach, stroke):
            check = Check_Profile(c.Model, c.Data, profile, profile.Start_Time, np.inf, lever, s.Path_Step, s.Criteria.Clearance, 0.0, c.Pairs,
                                  Robot_Str=c.Robot)
            certified = check.Certified and check.Within_Range
            if not certified:
                break
        if certified:
            return approach, stroke, float(scale)
    return None


def Plan_State(plan: Plan_Str, t: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        The executed motion at the times given: each time in the profile active then (a profile runs from its start to
        the next one's; before the first profile's start, that profile's start state).

    Args:
        (1) plan [Plan_Str(object)]: The plan, with at least one profile.
        (2) t [Vector<float> 1xT]: The times [s].

    Returns:
        (1 - 3) parameter [Matrix<float> TxJ]: The positions, velocities and accelerations [rad, rad/s, rad/s^2].
    """

    starts = np.array([p.Start_Time for p in plan.Profiles])
    active = np.maximum(np.searchsorted(starts, t + 1e-12, side='right') - 1, 0)
    q, qd, qdd = (np.zeros((len(t), len(plan.Profiles[0].Start))) for _ in range(3))
    for k in np.unique(active):
        select = active == k
        q[select], qd[select], qdd[select] = Profile_State(plan.Profiles[k], t[select])
    return q, qd, qdd


def Plan_Ticks(plan: Plan_Str, t_end: float, tick: float) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        The executed motion at the control ticks from the first profile's start to t_end: each tick in the profile
        active then (a profile runs from its start to the next one's).

    Args:
        (1) plan [Plan_Str(object)]: The plan, with at least one profile.
        (2) t_end [float]: The last time [s].
        (3) tick [float]: The control period [s].

    Returns:
        (1) parameter [Vector<float> 1xT]: The ticks' times [s].
        (2 - 4) parameter [Matrix<float> TxJ]: The positions, velocities and accelerations [rad, rad/s, rad/s^2].
    """

    start = plan.Profiles[0].Start_Time
    t = start + tick * np.arange(int(np.floor((t_end - start) / tick + 1e-9)) + 1)
    return (t,) + Plan_State(plan, t)


def Crossing(release_state: np.ndarray, floor_time: float, t_from: float, point: np.ndarray, axis: np.ndarray,
             Flight_Model_Str: Flight_Model_Str) -> tp.Tuple[float, float, float]:
    """
    Description:
        The true ball's first crossing of the cup's mouth plane after t_from (the plane through the mouth's centre
        normal to the cup's axis, crossed against the axis), every CONST_CROSSING_STEP linearly interpolated: its time,
        the ball centre's distance from the axis and the angle of its velocity against the axis (Q1). NaN without one.

    Args:
        (1) release_state [Vector<float> 1x6]: The throw's release state [m, m/s].
        (2) floor_time [float]: Its floor time [s].
        (3) t_from [float]: The time after which the crossing counts [s].
        (4) point [Vector<float> 1x3]: The mouth's centre [m].
        (5) axis [Vector<float> 1x3]: The cup's axis, out of the mouth.
        (6) Flight_Model_Str [Flight_Model_Str(object)]: The flight model.

    Returns:
        (1) parameter [float]: The crossing's time [s].
        (2) parameter [float]: The lateral distance [m].
        (3) parameter [float]: The angle [rad].
    """

    times = np.arange(0.0, floor_time, CONST_CROSSING_STEP)
    true = Simulate(np.asarray(release_state, dtype=np.float64), times, Flight_Model_Str=Flight_Model_Str)
    along = (true[:, :3] - point) @ axis
    found = np.flatnonzero((times[1:] > t_from) & (along[:-1] > 0.0) & (along[1:] <= 0.0))
    if not len(found):
        return float('nan'), float('nan'), float('nan')
    k = int(found[0])
    w = along[k] / (along[k] - along[k + 1])
    state = true[k] + w * (true[k + 1] - true[k])
    relative = state[:3] - point
    lateral = float(np.linalg.norm(relative - (relative @ axis) * axis))
    angle = float(np.arccos(np.clip(-state[3:] @ axis / np.linalg.norm(state[3:]), -1.0, 1.0)))
    return float(times[k] + w * CONST_CROSSING_STEP), lateral, angle


def Moving_Crossing(release_state: np.ndarray, floor_time: float, t_from: float, plan: Plan_Str, model: mujoco.MjModel, data: mujoco.MjData,
                    Flight_Model_Str: Flight_Model_Str, *, Robot_Str: tp.Any) -> tp.Tuple[float, float, float, float, float, float]:
    """
    Description:
        The true ball's first crossing, at or after t_from, of the plane of the cup's mouth while the cup moves on its
        plan (F15; Crossing's grid and interpolation, the plane that of the cup at each time): its time, the ball
        centre's distance from the cup's axis, the angle of the ball's velocity relative to the cup against the axis,
        the cup's speed along the ball's velocity (at the ball's centre: the mouth's velocity plus the cup's rotation),
        the ball's speed relative to the cup, and the angle between the cup's velocity and the ball's (0 for a cup at
        rest). NaN without a crossing.

    Args:
        (1) release_state [Vector<float> 1x6]: The throw's release state [m, m/s].
        (2) floor_time [float]: Its floor time [s].
        (3) t_from [float]: The time from which the crossing counts (the stroke's start) [s].
        (4) plan [Plan_Str(object)]: The executed plan.
        (5) model [mujoco.MjModel]: The compiled scene.
        (6) data [mujoco.MjData]: A data for the poses; the robot's joint positions are overwritten.
        (7) Flight_Model_Str [Flight_Model_Str(object)]: The flight model.
        (8) Robot_Str [Robot_Str(object)]: REQUIRED, keyword-only: the robot in the model.

    Returns:
        (1) parameter [float]: The crossing's time [s].
        (2) parameter [float]: The lateral distance [m].
        (3) parameter [float]: The angle of the relative velocity against the cup's axis [rad].
        (4) parameter [float]: The cup's speed along the ball's velocity [m/s].
        (5) parameter [float]: The ball's speed relative to the cup [m/s].
        (6) parameter [float]: The angle between the cup's velocity and the ball's [rad].
    """

    nothing = (float('nan'),) * 6
    times = np.arange(0.0, floor_time, CONST_CROSSING_STEP)
    times = times[times >= t_from]
    if len(times) < 2:
        return nothing
    true = Simulate(np.asarray(release_state, dtype=np.float64), times, Flight_Model_Str=Flight_Model_Str)
    q, qd, _ = Plan_State(plan, times)
    before = float('nan')
    for k in range(len(times)):
        point, orientation = Forward_Kinematics(model, data, q[k], Robot_Str=Robot_Str)
        axis = orientation[:, 2]
        along = float((true[k, :3] - point) @ axis)
        if before > 0.0 and along <= 0.0:
            w = before / (before - along)
            state = true[k - 1] + w * (true[k] - true[k - 1])
            relative = state[:3] - point
            lateral = float(np.linalg.norm(relative - (relative @ axis) * axis))
            mujoco.mj_comPos(model, data)
            translation, rotation = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, translation, rotation, Robot_Str.Tcp_Site)
            columns = Robot_Str.Velocity_Address
            cup = translation[:, columns] @ qd[k] + np.cross(rotation[:, columns] @ qd[k], relative)
            approach = state[3:] - cup
            ball_speed, cup_speed = float(np.linalg.norm(state[3:])), float(np.linalg.norm(cup))
            angle = float(np.arccos(np.clip(-approach @ axis / np.linalg.norm(approach), -1.0, 1.0)))
            direction = float(np.arccos(np.clip(cup @ state[3:] / (cup_speed * ball_speed), -1.0, 1.0))) if cup_speed > 0.0 else 0.0
            return (float(times[k - 1] + w * CONST_CROSSING_STEP), lateral, angle, float(cup @ state[3:] / ball_speed),
                    float(np.linalg.norm(approach)), direction)
        before = along
    return nothing


def Calibrate_Chunk(task: tp.Tuple[Interception_Setup_Str, str, np.ndarray, str]) -> tp.Dict[int, tp.List[tp.Any]]:
    """
    Description:
        The margin errors of a chunk of throws at each stage of the ladder (Catch.Decision.Margin_Errors; on the train
        partitions of Interception.Calibration), for one seed.

    Args:
        (1) task [Tuple]: (the setup, the dataset's flights.npz, the throws' indices, the seed's label).

    Returns:
        (1) parameter [Dict<int, List>]: Per stage the (bins, cross-track, early) of each prediction.
    """

    setup, path, indices, seed = task
    data = Read_Dataset(path)
    predict = Run_Predictor(setup.Model_Name, setup.Run, int(seed) if seed else None, Flight_Model_Str=setup.Flight,
                            Observation_Model_Str=setup.Observation)
    times, states, _ = Flight_Samples(data.Release_State[indices], data.Floor_Time[indices], Flight_Model_Str=setup.Flight)
    errors: tp.Dict[int, tp.List[tp.Any]] = {n: [] for n in setup.Stages}
    for row, i in enumerate(indices):
        t_obs, x_obs = data.Observations(int(i))
        for n in setup.Stages:
            if len(t_obs) < n:
                break
            try:
                trajectory = predict(t_obs[:n], x_obs[:n], float(data.Rate[i]))
            except Prediction_Error_Cls:
                continue
            first = int(np.floor(trajectory.T_0 / CONST_SAMPLE_STEP)) + 1
            last = min(int(np.floor(trajectory.T_End / CONST_SAMPLE_STEP + 1e-9)), len(times) - 1)
            while last >= first and times[last] > trajectory.T_End:
                last -= 1
            if last < first:
                continue
            grid = times[first:last + 1]
            errors[n].append(Margin_Errors(trajectory.State(grid), states[row, first:last + 1], grid - float(t_obs[n - 1])))
    return errors


def Ladder_Margins(errors: tp.Dict[int, tp.List[tp.Any]], quantile: float) -> tp.Dict[int, tp.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Description:
        The ladder's margins of one seed from its pooled errors (Catch.Decision.Margins per stage).

    Args:
        (1) errors [Dict<int, List>]: Per stage the errors (Calibrate_Chunk, the chunks' lists joined in order).
        (2) quantile [float]: The quantile (Decision.Quantile).

    Returns:
        (1) parameter [Dict<int, Tuple>]: Per stage the cross-track margins Bx3 [m], the time margins 1xB [s], the counts.
    """

    out = {}
    for n, parts in errors.items():
        if parts:
            bins, cross, early = (np.concatenate([part[c] for part in parts]) for c in range(3))
        else:
            bins, cross, early = np.zeros(0, dtype=np.int64), np.zeros((0, 3)), np.zeros(0)
        out[n] = Margins(bins, cross, early, quantile)
    return out


def _Context(setup: Interception_Setup_Str) -> tp.Tuple[tp.Any, np.ndarray, tp.Any, np.ndarray]:
    """
    Description:
        A worker's scene: the catch-pose context, the lever bounds, the servos, the wrapping joints (6, joint 6 not).

    Args:
        (1) setup [Interception_Setup_Str(object)]: The setup.

    Returns:
        (1) parameter [Catch_Context_Str(object)]: The context.
        (2) parameter [Vector<float> 1x6]: The lever bounds [m].
        (3) parameter [Servo_Str(object)]: The servos.
        (4) parameter [Vector<bool> 1x6]: The wrapping joints.
    """

    model = Build_Scene(Areas_Str=setup.Areas)
    context = Catch_Context(model, Scene_Robot(model), Criteria_Str=setup.Criteria)
    lever = Lever_Bounds(model, context.Pairs, Robot_Str=context.Robot)
    return context, lever, Servos(model, Robot_Str=context.Robot), np.append(Joint_Wrap(context), False)


def Intercept_Chunk(task: tp.Tuple[Interception_Setup_Str, str, str, np.ndarray, str], *,
                    predictor: tp.Optional[tp.Callable[[int], tp.Callable[..., tp.Any]]] = None,
                    on_plan: tp.Optional[tp.Callable[[int, int, Plan_Str], None]] = None) -> tp.List[tp.Tuple[tp.Any, ...]]:
    """
    Description:
        The interception of a chunk of throws for one seed (F10): per throw the truth (F9's, exact
        per sample), the decision at every sample from the ladder's first stage (and F9's stages with the ladder's
        margins, the baseline), and per acceleration with a go the plan (Plan_Go), its result at the true crossing
        (Crossing), the executed path checked again (Robot.Motion.Check_Profile, the distances cut at
        CONST_REPORT_CUT), the joints' limits, and the tracking by the servos (Robot.Motion.Track): one row per throw
        and acceleration in CONST_PER_THROW_COLUMNS' order. The two keyword arguments are F11's:
        without them the interception is F10's, the run's predictor and nothing kept. In moving mode (F15) a go planned
        with a stroke has its result at the crossing of the moving mouth plane (_Moving_Result), and every row carries
        CONST_MOVING_COLUMNS after the others (empty for a no-go and for a go without a plan).

    Args:
        (1) task [Tuple]: (the setup, the dataset's label, its flights.npz, the throws' indices, the seed's label).
        (2) predictor [Callable[[int], Callable] or None]: Keyword-only: None for the run's predictor; otherwise a
                                                          function giving, for a throw's index, the predictor of its
                                                          prefixes (the robustness sweep's injected error).
        (3) on_plan [Callable[[int, int, Plan_Str], None] or None]: Keyword-only: called with each go's throw index,
                                                                    acceleration index and plan (the physical catch).

    Returns:
        (1) parameter [List<Tuple>]: The rows.
    """

    setup, label, path, indices, seed = task
    data = Read_Dataset(path)
    context, lever, servos, wrap6 = _Context(setup)
    model, robot = context.Model, context.Robot
    tick = float(model.opt.timestep)
    moving = setup.Catch_Mode == CONST_MOVING
    blank = [''] * len(CONST_MOVING_COLUMNS) if moving else []
    run_predict = Run_Predictor(setup.Model_Name, setup.Run, int(seed) if seed else None, Flight_Model_Str=setup.Flight,
                                Observation_Model_Str=setup.Observation) if predictor is None else None
    centre, size = setup.Areas.Catch.Centre, setup.Areas.Catch.Size
    z_low = centre[2] - 0.5 * size[2]
    tolerance = CONST_MOUTH_RADIUS - _Ball_Radius(model)
    track_data = mujoco.MjData(model)
    ball = model.jnt_qposadr[model.body_jntadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONST_BALL_BODY)]]
    caches = [{} for _ in setup.Accelerations]
    rows = []
    for i in (int(v) for v in indices):
        t_obs, x_obs = data.Observations(i)
        rate = float(data.Rate[i])
        _, s_time, s_point, s_velocity = Truth_Samples(data.Release_State[i:i + 1], data.Floor_Time[i:i + 1], setup.Truth_Centre,
                                                       setup.Truth_Size, Flight_Model_Str=setup.Flight)
        reach = Exact_Reach(s_point, s_velocity, setup.Ready[:CONST_JOINTS_USED], setup.Speed[:CONST_JOINTS_USED], setup.Accelerations,
                            wrap6[:CONST_JOINTS_USED], Catch_Context_Str=context)
        predictions: tp.Dict[int, tp.Any] = {}
        refused: tp.List[int] = []
        throw_predict = run_predict if predictor is None else predictor(i)

        def Predict(k: int) -> tp.Any:
            """
            Description:
                The prediction of the first k samples, once per throw; None when refused.

            Args:
                (1) k [int]: The samples.

            Returns:
                (1) parameter [trajectory or None]: The trajectory.
            """

            if k not in predictions:
                try:
                    predictions[k] = throw_predict(t_obs[:k], x_obs[:k], rate)
                except Prediction_Error_Cls:
                    predictions[k] = None
                    refused.append(k)
            return predictions[k]

        def Catchable(t_d: float, a: int) -> bool:
            """
            Description:
                F9's truth: a truth sample reached from the ready pose within its time - t_d - CONST_ALLOWANCE.

            Args:
                (1) t_d [float]: The decision time [s].
                (2) a [int]: The acceleration's index.

            Returns:
                (1) parameter [bool]: Catchable.
            """

            return bool(np.any(reach[:, a] <= s_time - t_d - CONST_ALLOWANCE)) if len(s_time) else False

        # 1\ The decision at every sample (go: (N, t_d, t_hat)), and at F9's stages (the baseline).
        go: tp.List[tp.Optional[tp.Tuple[int, float, float]]] = [None for _ in setup.Accelerations]
        stages_go: tp.List[tp.List[bool]] = [[False] * len(setup.Decision_Stages) for _ in setup.Accelerations]
        last_stage = max(setup.Decision_Stages)
        for n in range(setup.Stages[0], len(t_obs) + 1):
            if x_obs[n - 1, 2] < z_low and n > last_stage:
                break
            stage = Ladder_Stage(n, setup.Stages)
            trajectory = Predict(n)
            if trajectory is None:
                continue
            t_d = float(t_obs[n - 1])
            times, states = Prediction_Grid(trajectory, t_d)
            h, position = Candidates(times, states, t_d, centre, size)
            cross_margin, time_margin = setup.Margins[(seed, stage)]
            for a in range(len(setup.Accelerations)):
                grid, low = setup.Grids[a]
                says = bool(Decide(np.zeros(len(h), dtype=np.int64), h, position, 1, cross_margin, time_margin, grid, low, setup.Step, centre,
                                   size, bounds=caches[a])[0])
                if n in setup.Decision_Stages:
                    stages_go[a][setup.Decision_Stages.index(n)] = says
                if says and go[a] is None:
                    passing, slack = Candidate_Slack(h, position, cross_margin, time_margin, grid, low, setup.Step, centre, size, caches[a])
                    best = np.flatnonzero(passing)[int(np.argmax(slack[passing]))]
                    go[a] = (n, t_d, t_d + float(h[best]))
            if all(g is not None for g in go) and n >= last_stage:
                break
            if len(h) and float(np.max(h)) < CONST_ALLOWANCE and n >= last_stage:
                break
        # 2\ Per acceleration: the truth, the plan, the result.
        first_time = float(t_obs[setup.Stages[0] - 1]) if len(t_obs) >= setup.Stages[0] else float('nan')
        stage_times = [float(t_obs[n - 1]) if len(t_obs) >= n else float('nan') for n in setup.Decision_Stages]
        for a, acceleration in enumerate(setup.Accelerations):
            truth_first = Catchable(first_time, a) if np.isfinite(first_time) else False
            made, stage_index, judged = Stage_Rule([np.array([v]) for v in stages_go[a]],
                                                   [np.array([Catchable(t, a) if np.isfinite(t) else False]) for t in stage_times])
            stages_decision = 'go' if made[0] else 'no-go'
            stages_outcome = _Outcome(bool(made[0]), bool(judged[0]))
            stages_n = '%d' % setup.Decision_Stages[int(stage_index[0])] if made[0] else ''
            row = [label, '%d' % i, seed, '%.0f' % np.degrees(acceleration)]
            if go[a] is None:
                rows.append(tuple(row + ['true' if truth_first else 'false', 'no-go', '', '', _Outcome(False, truth_first), stages_decision,
                                         stages_n, stages_outcome] + [''] * 20 + [';'.join('%d' % k for k in refused)] + blank))
                continue
            n_go, t_go, t_hat = go[a]
            truth = Catchable(t_go, a)
            plan = Plan_Go(t_obs, n_go, t_hat, Predict, acceleration, wrap6, lever, Interception_Setup_Str=setup, Catch_Context_Str=context)
            if on_plan is not None:
                on_plan(i, a, plan)
            if plan.Stroke:
                result, extras = _Moving_Result(plan, data.Release_State[i], float(data.Floor_Time[i]), t_hat, lever, tolerance, tick, servos,
                                                track_data, ball, setup, context)
            else:
                result = _Result(plan, data.Release_State[i], float(data.Floor_Time[i]), t_go, t_hat, lever, tolerance, tick, servos, track_data,
                                 ball, setup, context)
                extras = _Static_Extras(plan, data.Release_State[i], result, setup) if moving else []
            rows.append(tuple(row + ['true' if truth else 'false', 'go', '%d' % n_go, t_go, _Outcome(True, truth), stages_decision, stages_n,
                                     stages_outcome, t_hat, len(plan.Profiles), plan.Rejected] + result
                              + [';'.join('%d' % k for k in refused)] + extras))
    return rows


def _Ball_Radius(model: mujoco.MjModel) -> float:
    """
    Description:
        The radius of the scene's ball: its colliding sphere (Scene.Core, R 30 mm).

    Args:
        (1) model [mujoco.MjModel]: The compiled scene.

    Returns:
        (1) parameter [float]: The radius [m].
    """

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONST_BALL_BODY)
    spheres = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_SPHERE)
               and (model.geom_contype[g] or model.geom_conaffinity[g])]
    return float(model.geom_size[spheres[0], 0])


def _Outcome(go: bool, truth: bool) -> str:
    """
    Description:
        A decision's outcome against its truth, as F9's files name it.

    Args:
        (1) go [bool]: The decision.
        (2) truth [bool]: The truth judging it.

    Returns:
        (1) parameter [string]: true_go, false_go, false_no_go or true_no_go.
    """

    return ('true_go' if truth else 'false_go') if go else ('false_no_go' if truth else 'true_no_go')


def _Result(plan: Plan_Str, release_state: np.ndarray, floor_time: float, t_go: float, t_hat: float, lever: np.ndarray, tolerance: float,
            tick: float, servos: tp.Any, track_data: tp.Any, ball: int, setup: Interception_Setup_Str, context: tp.Any) -> tp.List[tp.Any]:
    """
    Description:
        A go's result (Q1, Q7, R2): from arrival_s to control_limit_ticks of CONST_PER_THROW_COLUMNS. Without an accepted
        plan the robot does not move (every value empty, not intercepted).

    Args:
        (1) plan [Plan_Str(object)]: The plan.
        (2) release_state, floor_time: The throw [m, m/s], [s].
        (3) t_go, t_hat [float]: The go's and the catch's times [s].
        (4) lever [Vector<float> 1x6]: The lever bounds [m].
        (5) tolerance [float]: The lateral tolerance [m].
        (6) tick [float]: The control period [s].
        (7) servos, track_data, ball: The servos, the tracking's data and the ball's qpos address.
        (8) setup [Interception_Setup_Str(object)]: The setup.
        (9) context [Catch_Context_Str(object)]: The context.

    Returns:
        (1) parameter [List<Any>]: The values.
    """

    if not plan.Profiles:
        return [''] * 8 + ['', 'false', 'false', 'false', 'false', '', '', '', '']
    model, robot = context.Model, context.Robot
    last = plan.Profiles[-1]
    arrival = last.Start_Time + last.Duration
    point, orientation = Forward_Kinematics(model, context.Data, last.Goal, Robot_Str=robot)
    axis = orientation[:, 2]
    t_c, lateral, angle = Crossing(release_state, floor_time, t_go, point, axis, setup.Flight)
    total = _Executed_Path(plan, lever, setup, context)
    t_end = t_c if np.isfinite(t_c) else t_hat
    t, q, qd, qdd = Plan_Ticks(plan, max(t_end, arrival), tick)
    within = total.Within_Range and bool(np.all(np.abs(qd) <= setup.Speed * (1.0 + 1e-9))) \
        and bool(np.all(np.abs(np.concatenate([p.Accelerations for p in plan.Profiles])) <= max(setup.Accelerations) * (1.0 + 1e-12)))
    at_rest = bool(np.isfinite(t_c) and arrival <= t_c)
    intercepted = bool(at_rest and lateral <= tolerance and angle <= setup.Angle_Tolerance and total.Certified and within)
    # The tracking by the servos, from the ready pose at rest, the ball parked.
    mujoco.mj_resetData(model, track_data)
    track_data.qpos[ball:ball + 3] = (CONST_BALL_PARK[0], CONST_BALL_PARK[1], _Ball_Radius(model))
    track_data.qpos[robot.Position_Address] = q[0]
    tracked = Track(model, track_data, servos, q, qd, qdd, CONST_HOLD_TICKS, Robot_Str=robot)
    at = min(int(np.searchsorted(t, t_end - 1e-12)), len(t) - 1)
    tracked_point, _ = Forward_Kinematics(model, context.Data, tracked.Positions[at], Robot_Str=robot)
    return [arrival, t_c, t_c - arrival, lateral, float(np.degrees(angle)), total.Environment, total.Self, total.Exempt, total.Poses,
            _Flag(total.Certified), _Flag(within), _Flag(at_rest), _Flag(intercepted), float(np.linalg.norm(tracked_point - point)),
            float(np.max(np.abs(tracked.Positions - q))), tracked.Force_Limit_Ticks, tracked.Control_Limit_Ticks]


def _Executed_Path(plan: Plan_Str, lever: np.ndarray, setup: Interception_Setup_Str, context: tp.Any) -> Path_Check_Str:
    """
    Description:
        The executed path of a plan, checked again with the distances reported (cut at CONST_REPORT_CUT): each profile
        from its start to the next one's (the last to its end), every pose checked (Robot.Motion.Check_Profile).

    Args:
        (1) plan [Plan_Str(object)]: The plan, with at least one profile.
        (2) lever [Vector<float> 1x6]: The lever bounds [m].
        (3) setup [Interception_Setup_Str(object)]: The setup.
        (4) context [Catch_Context_Str(object)]: The context.

    Returns:
        (1) parameter [Path_Check_Str(object)]: The smallest distances, the poses, certified and within the ranges.
    """

    total = Path_Check_Str(Certified=True, Within_Range=True)
    for k, profile in enumerate(plan.Profiles):
        t_to = plan.Profiles[k + 1].Start_Time if k + 1 < len(plan.Profiles) else np.inf
        check = Check_Profile(context.Model, context.Data, profile, profile.Start_Time, t_to, lever, setup.Path_Step, setup.Criteria.Clearance,
                              CONST_REPORT_CUT, context.Pairs, stop=False, Robot_Str=context.Robot)
        total.Environment, total.Self = min(total.Environment, check.Environment), min(total.Self, check.Self)
        total.Exempt, total.Poses = min(total.Exempt, check.Exempt), total.Poses + check.Poses
        total.Certified &= check.Certified
        total.Within_Range &= check.Within_Range
    return total


def _Moving_Result(plan: Plan_Str, release_state: np.ndarray, floor_time: float, t_hat: float, lever: np.ndarray, tolerance: float,
                   tick: float, servos: tp.Any, track_data: tp.Any, ball: int, setup: Interception_Setup_Str, context: tp.Any
                   ) -> tp.Tuple[tp.List[tp.Any], tp.List[tp.Any]]:
    """
    Description:
        The result of a go planned with a stroke (F15; the new R2): the columns of _Result, read
        for a moving cup, and CONST_MOVING_COLUMNS. arrival_s is the time the robot rests at the stroke's start;
        at_rest whether it rests there, at the stroke's own start joints, at or before the stroke begins; the crossing
        is the true ball's with the moving mouth plane from the stroke's start on (Moving_Crossing), the angle that of
        the ball's velocity relative to the cup; in_stroke whether the crossing lies between the stroke's start and its
        end; intercepted needs both, the lateral miss and the angle within their tolerances, and the executed path (the
        stroke to its end at rest included) certified and within the limits. The tracking runs to the stroke's end,
        its distance read at the crossing against the planned cup there.

    Args:
        (1) plan [Plan_Str(object)]: The plan, its last profile the stroke.
        (2) release_state, floor_time: The throw [m, m/s], [s].
        (3) t_hat [float]: The catch time [s].
        (4) lever [Vector<float> 1x6]: The lever bounds [m].
        (5) tolerance [float]: The lateral tolerance [m].
        (6) tick [float]: The control period [s].
        (7) servos, track_data, ball: The servos, the tracking's data and the ball's qpos address.
        (8) setup [Interception_Setup_Str(object)]: The setup.
        (9) context [Catch_Context_Str(object)]: The context.

    Returns:
        (1) parameter [List<Any>]: The values from arrival_s to control_limit_ticks.
        (2) parameter [List<Any>]: The values of CONST_MOVING_COLUMNS.
    """

    model, robot = context.Model, context.Robot
    stroke, approach = plan.Profiles[-1], plan.Profiles[-2]
    arrival, end = approach.Start_Time + approach.Duration, stroke.Start_Time + stroke.Duration
    t_c, lateral, angle, cup_speed, relative_speed, direction = Moving_Crossing(release_state, floor_time, stroke.Start_Time, plan, model,
                                                                               context.Data, setup.Flight, Robot_Str=robot)
    total = _Executed_Path(plan, lever, setup, context)
    t, q, qd, qdd = Plan_Ticks(plan, end, tick)
    within = total.Within_Range and bool(np.all(np.abs(qd) <= setup.Speed * (1.0 + 1e-9))) \
        and bool(np.all(np.abs(np.concatenate([p.Accelerations for p in plan.Profiles])) <= max(setup.Accelerations) * (1.0 + 1e-12)))
    at_rest = bool(arrival <= stroke.Start_Time and np.array_equal(approach.Goal, stroke.Start))
    in_stroke = bool(np.isfinite(t_c) and stroke.Start_Time <= t_c <= end)
    intercepted = bool(at_rest and in_stroke and lateral <= tolerance and angle <= setup.Angle_Tolerance and total.Certified and within)
    mujoco.mj_resetData(model, track_data)
    track_data.qpos[ball:ball + 3] = (CONST_BALL_PARK[0], CONST_BALL_PARK[1], _Ball_Radius(model))
    track_data.qpos[robot.Position_Address] = q[0]
    tracked = Track(model, track_data, servos, q, qd, qdd, CONST_HOLD_TICKS, Robot_Str=robot)
    at = min(int(np.searchsorted(t, (t_c if np.isfinite(t_c) else t_hat) - 1e-12)), len(t) - 1)
    planned_point, _ = Forward_Kinematics(model, context.Data, q[at], Robot_Str=robot)
    planned_point = planned_point.copy()
    tracked_point, _ = Forward_Kinematics(model, context.Data, tracked.Positions[at], Robot_Str=robot)
    return ([arrival, t_c, t_c - arrival, lateral, float(np.degrees(angle)), total.Environment, total.Self, total.Exempt, total.Poses,
             _Flag(total.Certified), _Flag(within), _Flag(at_rest), _Flag(intercepted), float(np.linalg.norm(tracked_point - planned_point)),
             float(np.max(np.abs(tracked.Positions - q))), tracked.Force_Limit_Ticks, tracked.Control_Limit_Ticks],
            [CONST_MOVING, plan.Scale, stroke.Start_Time, end, cup_speed, relative_speed, float(np.degrees(direction)), _Flag(in_stroke)])


def _Static_Extras(plan: Plan_Str, release_state: np.ndarray, result: tp.List[tp.Any], setup: Interception_Setup_Str) -> tp.List[tp.Any]:
    """
    Description:
        CONST_MOVING_COLUMNS of a go of a moving catch that is not planned with a stroke: empty without an accepted
        plan; for the static plan (the fallback) 'static', the scale 0, no stroke, the cup at rest (speed 0) and the
        ball's own speed at the crossing as the relative speed (empty without a crossing).

    Args:
        (1) plan [Plan_Str(object)]: The plan.
        (2) release_state [Vector<float> 1x6]: The throw's release state [m, m/s].
        (3) result [List<Any>]: The go's result (_Result): its second value is the crossing's time.
        (4) setup [Interception_Setup_Str(object)]: The setup.

    Returns:
        (1) parameter [List<Any>]: The values.
    """

    if not plan.Profiles:
        return [''] * len(CONST_MOVING_COLUMNS)
    t_c = result[1]
    speed: tp.Any = ''
    if np.isfinite(t_c):
        speed = float(np.linalg.norm(Simulate(np.asarray(release_state, dtype=np.float64), np.array([t_c]), Flight_Model_Str=setup.Flight)[0, 3:]))
    return [CONST_CATCH_MODES[0], 0.0, '', '', 0.0, speed, '', '']


def _Flag(value: bool) -> str:
    """
    Description:
        A boolean as the files write it.

    Args:
        (1) value [bool]: The value.

    Returns:
        (1) parameter [string]: 'true' or 'false'.
    """

    return 'true' if value else 'false'


def Decision_Lines(rows: tp.Iterable[tp.Sequence[tp.Any]]) -> tp.List[str]:
    """
    Description:
        The decision's columns of each row (the first CONST_DECISION_COLUMNS: the dataset, the flight, the seed, the
        acceleration, the truth, the decision with its N and time, its outcome, F9's stages and the catch time) as
        per_throw.csv holds them, one line per row: what the reading R1 of a moving catch compares with the leading
        fields of the static catch's stored lines (F15).

    Args:
        (1) rows [Iterable<Sequence<Any>>]: The per-throw rows (Intercept_Chunk).

    Returns:
        (1) parameter [List<string>]: The lines, without a line end.
    """

    return [','.join(repr(float(v)) if isinstance(v, (float, np.floating)) else '%s' % (v,) for v in row[:CONST_DECISION_COLUMNS]) for row in rows]


def Summary_Rows(rows: tp.Sequence[tp.Sequence[tp.Any]], partition: str, decision_stages: tp.Sequence[int]) -> tp.List[tp.Tuple[tp.Any, ...]]:
    """
    Description:
        The summary per dataset, seed and acceleration (SPEC §9.13): the per-sample decision's
        rates (F9's: false-go among the throws not catchable at their go, false-no-go among those catchable at the first
        stage, Clopper-Pearson 95 %), the false-no-go of F9's stages with the same margins (the baseline), the goes
        after F9's last stage, the interceptions (of the catchable, with their interval, and of the goes), the goes
        without a plan or a crossing, R2's violations (a go whose motion is not certified, not within the limits or not
        at rest before the crossing), and the distributions of the lateral miss, the margin, the clearances and the
        tracking. Rows of a moving catch (they carry CONST_MOVING_COLUMNS; F15) are read by the
        new R2: a go planned with a stroke violates it when its motion is not certified or not within the limits, the
        robot does not rest at the stroke's start before it begins, or the crossing does not lie inside the stroke; a
        go planned static as above. Their summary carries CONST_MOVING_SUMMARY_COLUMNS after the others: the goes with a
        stroke and their mean scale, the cup's speed along the ball (median) and the relative speed at the crossing
        (median, 95th percentile) over the planned goes with a crossing.

    Args:
        (1) rows [Sequence<Sequence<Any>>]: The per-throw rows (Intercept_Chunk).
        (2) partition [string]: The partition.
        (3) decision_stages [Sequence<int>]: F9's stages (Decision.Stages): a go after the last counts as late.

    Returns:
        (1) parameter [List<Tuple>]: The rows, in CONST_SUMMARY_COLUMNS' order (a moving catch's: then
                                     CONST_MOVING_SUMMARY_COLUMNS).
    """

    moving = bool(rows) and len(rows[0]) == len(CONST_PER_THROW_COLUMNS) + len(CONST_MOVING_COLUMNS)
    column = {name: k for k, name in enumerate(CONST_PER_THROW_COLUMNS + (CONST_MOVING_COLUMNS if moving else ()))}
    groups: tp.Dict[tp.Tuple[str, str, str], tp.List[tp.Sequence[tp.Any]]] = {}
    for row in rows:
        groups.setdefault((row[0], row[2], row[3]), []).append(row)
    out = []
    for (label, seed, acceleration), group in groups.items():
        get = (lambda name: [r[column[name]] for r in group])
        go = np.array([v == 'go' for v in get('decision')])
        truth = np.array([v == 'true' for v in get('catchable')])
        rates = Rates(go, truth)
        stages_go = np.array([v == 'go' for v in get('stages_decision')])
        stages_truth = np.array([o in ('true_go', 'false_no_go') for o in get('stages_outcome')])
        stages = Rates(stages_go, stages_truth)
        late = int(sum(1 for v in get('go_n') if v != '' and int(v) > max(decision_stages)))
        intercepted = np.array([v == 'true' for v in get('intercepted')])
        hits = int(np.sum(intercepted & truth))
        low, high = Clopper_Pearson(hits, int(truth.sum())) if truth.any() else (float('nan'), float('nan'))
        gone = [r for r in group if r[column['decision']] == 'go']
        planned = [r for r in gone if r[column['plans']] not in ('', 0)]
        crossing = [r for r in planned if np.isfinite(float(r[column['crossing_s']]))]
        stroke = (lambda r: moving and r[column['plan_mode']] == CONST_MOVING)
        violations = sum(1 for r in planned if not (r[column['certified']] == 'true' and r[column['within_limits']] == 'true'
                                                     and ((r[column['at_rest']] == 'true' and r[column['in_stroke']] == 'true') if stroke(r) else
                                                          (r[column['at_rest']] == 'true' or not np.isfinite(float(r[column['crossing_s']]))))))
        values = (lambda name, rs: np.array([float(r[column[name]]) for r in rs], dtype=np.float64))
        lateral, margin = values('lateral_m', crossing), values('margin_s', crossing)
        q = (lambda x, p: float(np.percentile(x, p)) if len(x) else float('nan'))
        m = (lambda x, f: float(f(x)) if len(x) else float('nan'))
        out.append((label, partition, seed, acceleration, rates['throws'], rates['catchable'], rates['not_catchable'], rates['go'],
                    rates['false_go'], rates['false_go_rate'], rates['false_go_lower'], rates['false_go_upper'], rates['false_no_go'],
                    rates['false_no_go_rate'], rates['false_no_go_lower'], rates['false_no_go_upper'], stages['false_no_go_rate'], late,
                    int(intercepted.sum()), hits / max(int(truth.sum()), 1), low, high, int(intercepted.sum()) / max(len(gone), 1),
                    len(gone) - len(planned), len(planned) - len(crossing), violations, q(lateral, 50), q(lateral, 95), m(lateral, np.max),
                    m(margin, np.min), q(margin, 50), m(values('clear_environment_m', planned), np.min), m(values('clear_self_m', planned), np.min),
                    m(values('clear_exempt_m', planned), np.min), q(values('tracking_m', planned), 50), m(values('tracking_m', planned), np.max),
                    int(sum(int(r[column['force_limit_ticks']]) for r in planned)),
                    int(sum(int(r[column['control_limit_ticks']]) for r in planned)), sum(1 for v in get('refused') if v)))
        if moving:
            stroked, relative = [r for r in planned if stroke(r)], values('relative_speed_m_s', crossing)
            out[-1] = out[-1] + (len(stroked), m(values('stroke_scale', stroked), np.mean), q(values('cup_speed_m_s', crossing), 50),
                                 q(relative, 50), q(relative, 95))
    return out


def _Write_Rows(path: str, columns: tp.Sequence[str], rows: tp.Iterable[tp.Sequence[tp.Any]]) -> None:
    """
    Description:
        Write a CSV with its header, floats by repr (17 significant digits; NaN as 'nan'), one row per sequence.

    Args:
        (1) path [string]: The file.
        (2) columns [Sequence<string>]: The header.
        (3) rows [Iterable<Sequence<Any>>]: The rows.
    """

    with open(path, 'w', newline='', encoding='ascii') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(columns)
        for row in rows:
            writer.writerow([repr(float(v)) if isinstance(v, (float, np.floating)) else v for v in row])


def Write_Per_Throw(path: str, rows: tp.Iterable[tp.Sequence[tp.Any]], moving: bool = False) -> None:
    """
    Description:
        Write the interception's per_throw.csv (SPEC §9.13): one row per throw, seed and acceleration; a moving catch's
        rows carry CONST_MOVING_COLUMNS after the others.

    Args:
        (1) path [string]: The file.
        (2) rows [Iterable<Sequence<Any>>]: The rows, in the order of CONST_PER_THROW_COLUMNS (then CONST_MOVING_COLUMNS).
        (3) moving [bool]: The rows are a moving catch's.
    """

    _Write_Rows(path, CONST_PER_THROW_COLUMNS + (CONST_MOVING_COLUMNS if moving else ()), rows)


def Write_Summary(path: str, rows: tp.Iterable[tp.Sequence[tp.Any]], moving: bool = False) -> None:
    """
    Description:
        Write the interception's summary.csv (SPEC §9.13): one row per dataset, seed and acceleration; a moving catch's
        rows carry CONST_MOVING_SUMMARY_COLUMNS after the others.

    Args:
        (1) path [string]: The file.
        (2) rows [Iterable<Sequence<Any>>]: The rows, in the order of CONST_SUMMARY_COLUMNS (then CONST_MOVING_SUMMARY_COLUMNS).
        (3) moving [bool]: The rows are a moving catch's.
    """

    _Write_Rows(path, CONST_SUMMARY_COLUMNS + (CONST_MOVING_SUMMARY_COLUMNS if moving else ()), rows)


def Read_Per_Throw(path: str) -> tp.List[tp.Dict[str, str]]:
    """
    Description:
        An interception's per_throw.csv (SPEC §9.13; Write_Per_Throw), a static or a moving catch's, read back for the
        figures, as text by column.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [List<Dict<string, string>>]: The rows.
    """

    return Read_Table(path, [CONST_PER_THROW_COLUMNS, CONST_PER_THROW_COLUMNS + CONST_MOVING_COLUMNS], '§9.13')


def Read_Summary(path: str) -> tp.List[tp.Dict[str, str]]:
    """
    Description:
        An interception's summary.csv (SPEC §9.13; Write_Summary), a static or a moving catch's, read back for the
        figures, as text by column.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [List<Dict<string, string>>]: The rows.
    """

    return Read_Table(path, [CONST_SUMMARY_COLUMNS, CONST_SUMMARY_COLUMNS + CONST_MOVING_SUMMARY_COLUMNS], '§9.13')
