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
# File Name: ../Catch/Physical.py
## =========================================================================== ##


# CSV (The catch's files)
import csv
# Zlib (A dataset's label as a number, for the markers' orientation of a throw)
import zlib
# Dataclasses (Data classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation) [pip3 install mujoco]
import mujoco
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the areas' structure
from Configuration.Core import Areas_Str
#   ../Flight/Core - the flight model, and the ball's state at the first control tick
from Flight.Core import Flight_Model_Str, Simulate
#   ../Dataset/Core - the dataset's reader
from Dataset.Core import Read_Dataset
#   ../Robot/Core - the robot in the model
from Robot.Core import Robot_Str
#   ../Robot/Motion - the servos and their command
from Robot.Motion import Servos, Servo_Command, Servo_Str
#   ../Scene/Core - the scene, its robot, its cup and its ball
from Scene.Core import (Build_Scene, Scene_Spec, Compile_Scene, Scene_Robot, CONST_ROBOT_PREFIX, CONST_BALL_BODY, CONST_CUP_BODY,
                        CONST_CUP_MOUTH_SITE, CONST_CUP_BOTTOM_GEOM, CONST_CUP_WALL_GEOM, CONST_CUP_LIP_GEOM, CONST_BIT_WORLD, CONST_BIT_BALL,
                        CONST_GROUP_COLLISION)
#   ../Scene/Ball - the ball's start, its drag and its markers
from Scene.Ball import Ball_Str, Find_Ball, Find_Markers, Place_Ball, Apply_Drag
#   ../Catch/Core - the catch's error, the reader of its files
from Catch.Core import Catch_Error_Cls, Read_Table
#   ../Catch/Decision - the interval of a share
from Catch.Decision import Clopper_Pearson
#   ../Catch/Interception - the interception of a chunk, a plan at the control ticks, the opening's radius, the hold
from Catch.Interception import (Intercept_Chunk, Interception_Setup_Str, Plan_Str, Plan_Ticks, CONST_MOUTH_RADIUS, CONST_HOLD_TICKS,
                                CONST_BALL_PARK, CONST_MOVING, CONST_PER_THROW_COLUMNS as CONST_INTERCEPTION_COLUMNS,
                                CONST_MOVING_COLUMNS as CONST_INTERCEPTION_MOVING_COLUMNS)

"""
Description:
    Initialization of constants.

    The physical catch (SPEC §2.10, §7.11; F11): every go of
    the interception simulated in the scene. The robot is driven by the go's plan through the servos' command at the
    control ticks (as Robot.Motion.Track does), the ball flies from its release under the flight model (Scene.Ball) and
    meets the cup's own shape; the step and the ball's contact are set on the compiled scene in memory (Contact_Str),
    models/ keeps the scene's own. A throw is caught when the ball is retained: at the end of the settle window after
    the catch time its centre is below the cup's mouth plane, above the flange and within the opening's radius of the
    cup's axis (Q2); at rest (a speed relative to the cup below a bound for a time) is a secondary measure, because it
    depends on the rolling friction, which has no source. Every throw has exactly one class (CONST_CLASSES). Every
    number is a simulation result; nothing is run on, or validated on, the real UR10e.
"""
# The classes of a throw (Q7), each with SPEC §2.10's class: a no-go by the truth of its decision (not simulated); a go
# caught when the ball is retained, late without an accepted plan, otherwise by what the ball met first (a link of
# the robot or the cup's outside cylinder cannot be met: 'robot' is any other shape of the robot; the stand), then
# bounced out when it was inside the cup and touched it, on the rim when it touched the cup and was never inside,
# missed when it never touched the cup.
#   Unit [Dict<string, string>]
CONST_CLASSES = {'caught': 'caught', 'bounced_out': 'bounced off', 'rim': 'bounced off', 'hit_robot': 'bounced off', 'missed': 'missed',
                 'hit_stand': 'missed', 'late': 'late', 'no_go_catchable': 'missed', 'no_go': 'outside the workspace'}
# What a geom is to the catch, by its index in Catch_Scene_Str.Kind, and the names the files use: the ball, the cup's
# shape for the ball (bottom, wall, lip), any other shape of the robot, the stand, the floor, anything else.
#   Unit [Tuple<string>]
CONST_KINDS = ('other', 'ball', 'cup_bottom', 'cup_wall', 'cup_lip', 'robot', 'stand', 'floor')
CONST_KIND_BALL = CONST_KINDS.index('ball')
CONST_KIND_FLOOR = CONST_KINDS.index('floor')
CONST_KINDS_CUP = tuple(CONST_KINDS.index(name) for name in ('cup_bottom', 'cup_wall', 'cup_lip'))
# The ball's contact dimension: sliding, torsional and rolling friction (MuJoCo's condim 6), and its priority over
# every other geom, so that the ball's own parameters are those of each of its contacts.
#   Unit [int]
CONST_CONDIM = 6
CONST_PRIORITY = 1
# The columns of the catch's files (SPEC §9.14).
#   Unit [Tuple<string>]
CONST_PER_THROW_COLUMNS = ('dataset', 'flight', 'seed', 'acceleration_deg_s2', 'time_constant_s', 'damping_ratio', 'catchable', 'decision',
                           'intercepted', 'catch_time_s', 'class', 'outcome', 'retained', 'at_rest', 'rest_s', 'first_contact',
                           'first_contact_s', 'cup_contact_s', 'entered', 'entry_lateral_m', 'end_speed_m_s', 'penetration_m',
                           'robot_contact', 'tracking_joint_rad', 'force_limit_ticks')
CONST_SUMMARY_COLUMNS = ('dataset', 'partition', 'seed', 'acceleration_deg_s2', 'time_constant_s', 'damping_ratio', 'throws', 'catchable',
                         'go', 'false_go', 'intercepted', 'retained', 'retained_of_go', 'retained_of_go_lower', 'retained_of_go_upper',
                         'retained_catchable', 'retained_of_catchable', 'retained_of_catchable_lower', 'retained_of_catchable_upper',
                         'at_rest') + tuple(CONST_CLASSES) + ('robot_contacts', 'rest_p50_s', 'rest_max_s', 'penetration_max_m',
                                                              'force_limit_ticks')
# The sweep's files carry the injected error's source and scale before the same columns (SPEC §9.14).
#   Unit [Tuple<string>]
CONST_SWEEP_COLUMNS = ('error_source', 'error_scale')
# The further columns of a moving catch's files, after the columns above, in moving mode only (SPEC §9.14).
# Per throw: how the go was planned and its stroke's scale (the interception's), and the ball's
# speed relative to the cup at its first contact with the cup. Per summary row: the simulated goes with a stroke and
# that speed's median and 95th percentile.
#   Unit [Tuple<string>]
CONST_MOVING_COLUMNS = ('plan_mode', 'stroke_scale', 'contact_speed_m_s', 'markers')
CONST_MOVING_SUMMARY_COLUMNS = ('strokes', 'contact_speed_p50_m_s', 'contact_speed_p95_m_s', 'markers')
# The markers variant (F15): the ball's six markers as collision shapes, a sensitivity variant
# beside the nominal sphere, in memory only. A marker is a capsule of this half-length about its centre (the ball keeps
# its one sphere, Scene.Ball.Find_Ball; at 10 um the capsule is a sphere to the contact), named with the prefix; the
# ball's orientation per throw is drawn from the seed, the dataset's label and the throw's index.
#   Unit [m], [string], [int]
CONST_MARKER_HALF_LENGTH = 1e-5
CONST_MARKER_GEOM = 'ball_marker_'
CONST_MARKER_SEED = 20261123
# The positions of the first contact within a step that a restitution is averaged over (SPEC §4.14: 'the mean over 8
# positions of the first contact within a step'; F15's 0.347 and 0.560 measured so).
#   Unit [int]
CONST_REBOUND_PHASES = 8


@dataclass
class Contact_Str:
    """
    Description:
        The physical catch's settings (Configuration.Core.Contact_Parameters_Str, SPEC §8.1) and the contact levels a
        task simulates.
    """

    # The simulation step, a whole fraction of the scene's control period.
    #   Unit [float, s]
    Step: float = 0.0
    # The ball's friction: sliding, torsional, rolling.
    #   Unit [Vector<float> 1x3: -, m, m]
    Friction: tp.Sequence[float] = (0.0, 0.0, 0.0)
    # The settle window after the catch time.
    #   Unit [float, s]
    Settle_Time: float = 0.0
    # At rest: the speed relative to the cup below Rest_Speed for Rest_Time.
    #   Unit [float, m/s], [float, s]
    Rest_Speed: float = 0.0
    Rest_Time: float = 0.0
    # The contact levels simulated, each (time constant, damping ratio); the nominal first.
    #   Unit [List<Tuple<float, float>>: s, -]
    Levels: tp.List[tp.Tuple[float, float]] = field(default_factory=list)
    # Simulate every level a second time with the ball's markers as collision shapes (the markers variant; a moving
    # catch's only, F15).
    #   Unit [bool]
    Markers: bool = False


@dataclass
class Catch_Scene_Str:
    """
    Description:
        A worker's scene for the catch (Catch_Scene): the compiled model with the catch's step and the ball's contact,
        its data, and what the loop reads of it.
    """

    # The model (its time step is the catch's), the simulation's data and the data of the servos' feed-forward.
    #   Unit [mujoco.MjModel], [mujoco.MjData], [mujoco.MjData]
    Model: tp.Any = None
    Data: tp.Any = None
    Inverse: tp.Any = None
    # The robot, its servos, the ball and the flight model.
    #   Unit [Robot_Str], [Servo_Str], [Ball_Str], [Flight_Model_Str]
    Robot: Robot_Str = None
    Servos: Servo_Str = None
    Ball: Ball_Str = None
    Flight: Flight_Model_Str = None
    # The control period (the scene's own time step) and the steps per control tick.
    #   Unit [float, s], [int]
    Tick: float = 0.0
    Steps: int = 0
    # What each geom is to the catch (an index of CONST_KINDS), and each geom's name (its body's without one).
    #   Unit [Vector<int> 1xG], [List<string>]
    Kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    Names: tp.List[str] = field(default_factory=list)
    # The ball's geoms that take contacts: its sphere alone, or with the markers variant its sphere and the six markers.
    #   Unit [List<int>]
    Ball_Geoms: tp.List[int] = field(default_factory=list)
    # The ball's geom, the cup's body, the site in the centre of its mouth and the cup's length from the flange.
    #   Unit [int], [int], [int], [float, m]
    Ball_Geom: int = -1
    Cup_Body: int = -1
    Mouth_Site: int = -1
    Cup_Length: float = 0.0
    # The held robot's state by its first planned joints and the contact level (the hold is the same for every throw).
    #   Unit [Dict<bytes, Vector<float>>]
    Held: tp.Dict[bytes, np.ndarray] = field(default_factory=dict)


@dataclass
class Catch_Result_Str:
    """
    Description:
        What happened to one go's ball (Simulate_Catch). Times from the release.
    """

    # Retained at the end of the settle window (the primary criterion), and at rest in the cup (the secondary measure)
    # with the time it was reached.
    #   Unit [bool], [bool], [float, s]
    Retained: bool = False
    At_Rest: bool = False
    Rest_Time: float = float('nan')
    # What the ball met first (a name of CONST_KINDS, '' for nothing), when, and when it first met the cup's shape.
    #   Unit [string], [float, s], [float, s]
    First_Contact: str = ''
    First_Contact_Time: float = float('nan')
    Cup_Contact_Time: float = float('nan')
    # The ball's centre was inside the cup at a control tick, and its distance from the cup's axis at the first such tick.
    #   Unit [bool], [float, m]
    Entered: bool = False
    Entry_Lateral: float = float('nan')
    # The ball's speed relative to the cup at the last tick, and its deepest penetration into the cup's shape.
    #   Unit [float, m/s], [float, m]
    End_Speed: float = float('nan')
    Penetration: float = 0.0
    # The first contact of the robot with anything but the ball ('' for none): the two geoms' names.
    #   Unit [string]
    Robot_Contact: str = ''
    # The largest joint error against the plan before the ball met the cup, and the ticks with a servo at its force limit.
    #   Unit [float, rad], [int]
    Tracking: float = 0.0
    Force_Limit_Ticks: int = 0
    # The ball's speed relative to the cup (at the ball's centre) at the step of its first contact with the cup's shape;
    # NaN without one (F15).
    #   Unit [float, m/s]
    Contact_Speed: float = float('nan')


def Catch_Scene(areas: Areas_Str, contact: Contact_Str, *, Flight_Model_Str: Flight_Model_Str, markers: bool = False) -> Catch_Scene_Str:
    """
    Description:
        Build the scene for the catch: the shipped scene compiled once more (the interception keeps its own), its time
        step set to the catch's and the ball's friction, contact dimension and priority set, in memory. A step that is
        not a whole fraction of the scene's control period is refused. With `markers` (the markers variant) the ball
        also carries its six markers as collision shapes: the scene's specification with a
        capsule of negligible length at each marker measured on the ball's mesh (Scene.Ball.Find_Markers), on the ball's
        contact bits, compiled in memory; the ball's mass and inertia are the scene's (its inertial is given), and
        models/ and the fingerprint are untouched.

    Args:
        (1) areas [Areas_Str(object)]: The areas the scene draws.
        (2) contact [Contact_Str(object)]: The catch's settings.
        (3) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.
        (4) markers [bool]: Keyword-only: the ball with its markers as collision shapes.

    Returns:
        (1) parameter [Catch_Scene_Str(object)]: The scene.
    """

    model = Build_Scene(Areas_Str=areas)
    if markers:
        centres, radius = Find_Markers(model)
        specification = Scene_Spec(Areas_Str=areas)
        for number, centre in enumerate(centres):
            specification.body(CONST_BALL_BODY).add_geom(name='%s%d' % (CONST_MARKER_GEOM, number), type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                                         size=[radius, CONST_MARKER_HALF_LENGTH, 0.0], pos=[float(v) for v in centre],
                                                         contype=CONST_BIT_WORLD | CONST_BIT_BALL, conaffinity=CONST_BIT_BALL,
                                                         group=CONST_GROUP_COLLISION)
        model = Compile_Scene(specification)
    tick = float(model.opt.timestep)
    steps = int(round(tick / contact.Step))
    if steps < 1 or abs(steps * contact.Step - tick) > 1e-12:
        raise Catch_Error_Cls('Contact.Step %r s is not a whole fraction of the scene\'s control period %r s; the robot is commanded once '
                              'per control period (SPEC §7.11).' % (contact.Step, tick))
    robot = Scene_Robot(model)
    ball = Find_Ball(model)
    cup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONST_ROBOT_PREFIX + CONST_CUP_BODY)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CONST_ROBOT_PREFIX + CONST_CUP_MOUTH_SITE)
    root = model.body_rootid[cup]
    names, kind = [], np.zeros(model.ngeom, dtype=np.int64)
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        names.append(name)
        short = name[len(CONST_ROBOT_PREFIX):] if name.startswith(CONST_ROBOT_PREFIX) else ''
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        if body == ball.Body:
            kind[geom] = CONST_KIND_BALL
        elif short == CONST_CUP_BOTTOM_GEOM:
            kind[geom] = CONST_KINDS.index('cup_bottom')
        elif short.startswith(CONST_CUP_WALL_GEOM):
            kind[geom] = CONST_KINDS.index('cup_wall')
        elif short.startswith(CONST_CUP_LIP_GEOM):
            kind[geom] = CONST_KINDS.index('cup_lip')
        elif body != 0 and model.body_rootid[body] == root:
            kind[geom] = CONST_KINDS.index('robot')
        elif name == 'floor':
            kind[geom] = CONST_KIND_FLOOR
        elif body == 0:
            kind[geom] = CONST_KINDS.index('stand')
    balls = np.flatnonzero(kind == CONST_KIND_BALL)
    spheres = [int(g) for g in balls if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    others = [int(g) for g in balls if names[g].startswith(CONST_MARKER_GEOM)] if markers else []
    if len(spheres) != 1 or len(balls) != 1 + len(others) or not all(np.any(kind == k) for k in CONST_KINDS_CUP):
        raise Catch_Error_Cls('The scene holds %d colliding geoms of the ball and not all of the cup\'s shape for it (%s); the catch needs '
                              'the scene of scripts/Maintenance/build_scene.py (SPEC §7.10).'
                              % (len(balls), ', '.join(CONST_KINDS[k] for k in CONST_KINDS_CUP)))
    model.opt.timestep = contact.Step
    for geom in balls:
        model.geom_friction[geom] = contact.Friction
        model.geom_condim[geom] = CONST_CONDIM
        model.geom_priority[geom] = CONST_PRIORITY
    return Catch_Scene_Str(Model=model, Data=mujoco.MjData(model), Inverse=mujoco.MjData(model), Robot=robot, Servos=Servos(model, Robot_Str=robot),
                           Ball=ball, Flight=Flight_Model_Str, Tick=tick, Steps=steps, Kind=kind, Names=names, Ball_Geom=spheres[0],
                           Ball_Geoms=[int(g) for g in balls], Cup_Body=cup, Mouth_Site=site, Cup_Length=float(model.site_pos[site][2]))


def Set_Contact(scene: Catch_Scene_Str, time_constant: float, damping_ratio: float) -> None:
    """
    Description:
        Set the ball's contact for the throws that follow: MuJoCo's solref (the time constant, the damping ratio) on
        the ball's geom (with the markers variant on each of its geoms), whose priority makes it the contact of
        everything the ball touches. A time constant below two steps is refused: MuJoCo would raise it to two steps
        without a word.

    Args:
        (1) scene [Catch_Scene_Str(object)]: The scene, changed in place.
        (2) time_constant [float]: The time constant [s].
        (3) damping_ratio [float]: The damping ratio [-].
    """

    if not (time_constant >= 2.0 * scene.Model.opt.timestep and damping_ratio > 0.0):
        raise Catch_Error_Cls('The contact (time constant %r s, damping ratio %r) cannot be used: the time constant must be at least two '
                              'steps (%r s) and the damping ratio > 0 (SPEC §7.11).' % (time_constant, damping_ratio, 2.0 * scene.Model.opt.timestep))
    for geom in scene.Ball_Geoms:
        scene.Model.geom_solref[geom] = (time_constant, damping_ratio)


def _Hold(scene: Catch_Scene_Str, q: np.ndarray) -> None:
    """
    Description:
        The scene with the robot held at its first planned joints for CONST_HOLD_TICKS control ticks, the ball parked on
        the floor (as the interception's tracking starts). The held state is kept per joints and contact and restored
        for the next throw: it is the same for every throw.

    Args:
        (1) scene [Catch_Scene_Str(object)]: The scene; its data is left at the held state.
        (2) q [Vector<float> 1x6]: The joints held [rad].
    """

    model, data = scene.Model, scene.Data
    key = np.asarray(q, dtype=np.float64).tobytes() + model.geom_solref[scene.Ball_Geom].tobytes()
    mujoco.mj_resetData(model, data)
    if key in scene.Held:
        mujoco.mj_setState(model, data, scene.Held[key], mujoco.mjtState.mjSTATE_INTEGRATION)
        return
    address = scene.Ball.Position_Address
    data.qpos[address:address + 3] = (CONST_BALL_PARK[0], CONST_BALL_PARK[1], scene.Ball.Radius)
    data.qpos[scene.Robot.Position_Address] = q
    scene.Inverse.qpos[:] = data.qpos
    rest = np.zeros(len(scene.Robot.Joint_Id))
    data.ctrl[scene.Servos.Actuators] = Servo_Command(model, scene.Inverse, scene.Servos, q, rest, rest, Robot_Str=scene.Robot)
    for _ in range(CONST_HOLD_TICKS * scene.Steps):
        mujoco.mj_step(model, data)
    state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    scene.Held[key] = state


def Ball_In_Cup(scene: Catch_Scene_Str) -> tp.Tuple[bool, float]:
    """
    Description:
        Whether the ball's centre is inside the cup at the positions of the scene's data: below the plane of the cup's
        mouth, above the flange, and within the opening's radius (CONST_MOUTH_RADIUS) of the cup's axis; and its distance
        from the axis. The one definition of *inside* for the catch's loop and for the view.

    Args:
        (1) scene [Catch_Scene_Str(object)]: The scene; its data's kinematics are read.

    Returns:
        (1) parameter [bool]: Inside.
        (2) parameter [float]: The centre's distance from the cup's axis [m].
    """

    data = scene.Data
    mouth, axis = data.site_xpos[scene.Mouth_Site], data.site_xmat[scene.Mouth_Site].reshape(3, 3)[:, 2]
    offset = data.xpos[scene.Ball.Body] - mouth
    along = float(offset @ axis)
    lateral = float(np.linalg.norm(offset - along * axis))
    return -scene.Cup_Length < along < 0.0 and lateral < CONST_MOUTH_RADIUS, lateral


def Marker_Orientation(label: str, flight: int) -> np.ndarray:
    """
    Description:
        The ball's orientation for a throw of the markers variant: a unit quaternion drawn uniformly over the rotations
        (four normal numbers, normalised) from CONST_MARKER_SEED, the dataset's label and the throw's index, so that a
        throw has the same orientation at every acceleration and contact level and in every process.

    Args:
        (1) label [string]: The dataset's label.
        (2) flight [int]: The throw's index in its dataset.

    Returns:
        (1) parameter [Vector<float> 1x4]: The quaternion (w, x, y, z).
    """

    rng = np.random.default_rng(np.random.SeedSequence([CONST_MARKER_SEED, zlib.crc32(label.encode('utf-8')), int(flight)]))
    quaternion = rng.standard_normal(4)
    return quaternion / np.linalg.norm(quaternion)


def Simulate_Catch(scene: Catch_Scene_Str, release_state: np.ndarray, plan: Plan_Str, ready: np.ndarray, t_catch: float, *,
                   Contact_Str: Contact_Str, on_step: tp.Optional[tp.Callable[[float, np.ndarray], None]] = None,
                   orientation: tp.Optional[np.ndarray] = None) -> Catch_Result_Str:
    """
    Description:
        One go in the scene (F11's own loop; Q4): the robot held at the plan's first joints, the ball placed at its
        state at the first control tick at or after the release (the control ticks are the plan's: its first tick plus
        whole control periods), then per control tick the servos' command for the plan's state at that tick (the first
        state at rest before the plan, the last at rest after it; Robot.Motion.Servo_Command, as Robot.Motion.Track)
        and the steps of the catch's time step, the ball's drag set before each. The simulation ends at the catch time
        plus the settle window, or when the ball reaches the floor. Without an accepted plan the robot holds `ready`.
        The contact is the scene's (Set_Contact). `on_step` only reads (the view's
        recording): without it, and with it, the result is the same. With `orientation` the ball starts
        turned so, without spin (the markers variant: which way its markers point); without it as Scene.Ball places it.

    Args:
        (1) scene [Catch_Scene_Str(object)]: The scene.
        (2) release_state [Vector<float> 1x6]: The ball's state at the release, t = 0 [m, m/s].
        (3) plan [Plan_Str(object)]: The go's plan.
        (4) ready [Vector<float> 1x6]: The joints the robot waits in [rad].
        (5) t_catch [float]: The catch time [s].
        (6) Contact_Str [Contact_Str(object)]: REQUIRED, keyword-only: the settle window and the rest's speed and time.
        (7) on_step [Callable[[float, Vector<float>], None] or None]: Keyword-only: called with the time [s] and a copy
                                                                      of the scene's positions (qpos) at the start
                                                                      and after every step.
        (8) orientation [Vector<float> 1x4 or None]: Keyword-only: the ball's orientation at its start, a unit quaternion
                                                     (w, x, y, z).

    Returns:
        (1) parameter [Catch_Result_Str(object)]: What happened.
    """

    model, data, robot, servos, ball = scene.Model, scene.Data, scene.Robot, scene.Servos, scene.Ball
    tick, step = scene.Tick, float(model.opt.timestep)
    rest = np.zeros(len(robot.Joint_Id))
    # 1\ The plan at the control ticks; the first control tick at or after the release.
    if plan.Profiles:
        last = plan.Profiles[-1]
        t, q, qd, qdd = Plan_Ticks(plan, last.Start_Time + last.Duration + tick, tick)
        before = int(np.floor(t[0] / tick + 1e-9))
        start = float(t[0] - before * tick)
    else:
        q, qd, qdd = np.asarray(ready, dtype=np.float64)[None, :], rest[None, :], rest[None, :]
        before, start = 0, 0.0
    # 2\ The robot held, the ball at its state at the first control tick.
    _Hold(scene, q[0])
    state = np.asarray(release_state, dtype=np.float64)
    if start > 0.0:
        state = Simulate(state, np.array([start]), Flight_Model_Str=scene.Flight)[0]
    data.time = start
    Place_Ball(model, data, state, Ball_Str=ball, Flight_Model_Str=scene.Flight)
    if orientation is not None:
        data.qpos[ball.Position_Address + 3:ball.Position_Address + 7] = orientation
    if on_step is not None:
        on_step(float(data.time), data.qpos.copy())
    out = Catch_Result_Str()
    seen: tp.Set[int] = set()
    known: tp.Dict[bytes, bool] = {}
    slow_since = float('nan')
    inside, floor = False, False
    velocity = np.zeros(6)
    v = ball.Velocity_Address
    arriving = np.zeros(3)
    for k in range(int(np.ceil((t_catch + Contact_Str.Settle_Time - start) / tick - 1e-9))):
        row = k - before
        planned = 0 <= row < len(q)
        target = q[min(max(row, 0), len(q) - 1)]
        if planned and np.isnan(out.Cup_Contact_Time):
            out.Tracking = max(out.Tracking, float(np.max(np.abs(data.qpos[robot.Position_Address] - target))))
        data.ctrl[servos.Actuators] = Servo_Command(model, scene.Inverse, servos, target, qd[row] if planned else rest,
                                                    qdd[row] if planned else rest, Robot_Str=robot)
        for _ in range(scene.Steps):
            Apply_Drag(model, data, Ball_Str=ball, Flight_Model_Str=scene.Flight)
            free = np.isnan(out.Cup_Contact_Time)
            if free:
                arriving[:] = data.qvel[v:v + 3]
            mujoco.mj_step(model, data)
            if on_step is not None:
                on_step(float(data.time), data.qpos.copy())
            count = data.ncon
            if count:
                # The contacts are those of the state the step started from.
                geoms = data.contact.geom[:count]
                key = geoms.tobytes()
                if key not in known:
                    known[key] = _Read_Contacts(scene, geoms, data.time - step, seen, out)
                    if free and not np.isnan(out.Cup_Contact_Time):
                        # The velocities of that state: the ball's before the step, the cup's as the step computed it.
                        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, scene.Cup_Body, velocity, 0)
                        out.Contact_Speed = float(np.linalg.norm(arriving - velocity[3:] - np.cross(velocity[:3], data.xpos[ball.Body]
                                                                                                     - data.xpos[scene.Cup_Body])))
                if known[key]:
                    out.Penetration = max(out.Penetration, -float(data.contact.dist[:count].min()))
        force = data.actuator_force[servos.Actuators]
        out.Force_Limit_Ticks += int(np.any((force <= servos.Force_Range[:, 0]) | (force >= servos.Force_Range[:, 1])))
        # 3\ At the control tick: the ball against the cup.
        now = start + (k + 1) * tick
        inside, lateral = Ball_In_Cup(scene)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, scene.Cup_Body, velocity, 0)
        cup_velocity = velocity[3:] + np.cross(velocity[:3], data.xpos[ball.Body] - data.xpos[scene.Cup_Body])
        out.End_Speed = float(np.linalg.norm(data.qvel[v:v + 3] - cup_velocity))
        if inside and not out.Entered:
            out.Entered, out.Entry_Lateral = True, lateral
        if inside and out.End_Speed < Contact_Str.Rest_Speed:
            slow_since = now if np.isnan(slow_since) else slow_since
            if not out.At_Rest and now - slow_since >= Contact_Str.Rest_Time - 1e-12:
                out.At_Rest, out.Rest_Time = True, now
        else:
            slow_since = float('nan')
        if CONST_KIND_FLOOR in seen:
            floor = True
            break
    out.Retained = bool(inside and not floor)
    return out


def _Read_Contacts(scene: Catch_Scene_Str, geoms: np.ndarray, time: float, seen: tp.Set[int], out: Catch_Result_Str) -> bool:
    """
    Description:
        Read a set of contacts met for the first time in a throw: what the ball touches (the first of each kind, with
        the time), and a contact of the robot with anything but the ball.

    Args:
        (1) scene [Catch_Scene_Str(object)]: The scene.
        (2) geoms [Matrix<int> Cx2]: The contacts' geoms.
        (3) time [float]: The time of the state they belong to [s].
        (4) seen [Set<int>]: The kinds the ball has touched, extended in place.
        (5) out [Catch_Result_Str(object)]: The result, changed in place.

    Returns:
        (1) parameter [bool]: The ball touches the cup's shape in this set.
    """

    cup = False
    for first, second in geoms:
        kinds = (int(scene.Kind[first]), int(scene.Kind[second]))
        if CONST_KIND_BALL not in kinds:
            if not out.Robot_Contact:
                out.Robot_Contact = '%s + %s' % (scene.Names[first], scene.Names[second])
            continue
        other = kinds[1] if kinds[0] == CONST_KIND_BALL else kinds[0]
        cup = cup or other in CONST_KINDS_CUP
        if other not in seen:
            seen.add(other)
            if out.First_Contact == '':
                out.First_Contact, out.First_Contact_Time = CONST_KINDS[other], time
            if other in CONST_KINDS_CUP and np.isnan(out.Cup_Contact_Time):
                out.Cup_Contact_Time = time
    return cup


def Classify(decision: str, catchable: bool, planned: bool, result: tp.Optional[Catch_Result_Str]) -> str:
    """
    Description:
        The class of a throw (CONST_CLASSES), exactly one: a no-go by its truth; a go caught when the ball is retained,
        late without an accepted plan, then by what the ball met first, then by whether it was inside the cup.

    Args:
        (1) decision [string]: 'go' or 'no-go'.
        (2) catchable [bool]: The throw is catchable at its decision (F9's truth).
        (3) planned [bool]: The go has an accepted plan.
        (4) result [Catch_Result_Str(object) or None]: The go's simulation; None for a no-go.

    Returns:
        (1) parameter [string]: The class.
    """

    if decision != 'go':
        return 'no_go_catchable' if catchable else 'no_go'
    if result is None:
        raise Catch_Error_Cls('A go has no simulation to classify; every go is simulated (SPEC §7.11).')
    if result.Retained:
        return 'caught'
    if not planned:
        return 'late'
    if result.First_Contact in ('robot', 'stand'):
        return 'hit_' + result.First_Contact
    if not np.isnan(result.Cup_Contact_Time):
        return 'bounced_out' if result.Entered else 'rim'
    return 'missed'


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


def Catch_Row(row: tp.Sequence[tp.Any], time_constant: float, damping_ratio: float, planned: bool,
              result: tp.Optional[Catch_Result_Str], *, moving: bool = False, markers: bool = False) -> tp.Tuple[tp.Any, ...]:
    """
    Description:
        The catch's row of one throw, acceleration and contact, in CONST_PER_THROW_COLUMNS' order, from its
        interception row: a go with its simulation classified, a no-go classified by its truth (the columns after
        at_rest empty). A moving catch's row (its interception row carries the moving columns) ends with
        CONST_MOVING_COLUMNS: the plan's mode and the stroke's scale from the interception row, the relative speed at
        the first cup contact (empty for a no-go, and without a contact), and whether the row is the markers variant's.

    Args:
        (1) row [Sequence<Any>]: The interception's row (CONST_PER_THROW_COLUMNS of Catch.Interception).
        (2 - 3) time_constant, damping_ratio [float]: The contact [s], [-].
        (4) planned [bool]: The go has an accepted plan.
        (5) result [Catch_Result_Str(object) or None]: The go's simulation; None for a no-go.
        (6) moving [bool]: Keyword-only: the row is a moving catch's.
        (7) markers [bool]: Keyword-only: the row is the markers variant's (a moving catch's only).

    Returns:
        (1) parameter [Tuple<Any>]: The row.
    """

    column = {name: k for k, name in enumerate(CONST_INTERCEPTION_COLUMNS + (CONST_INTERCEPTION_MOVING_COLUMNS if moving else ()))}
    decision, catchable = row[column['decision']], row[column['catchable']] == 'true'
    head = [row[column[name]] for name in ('dataset', 'flight', 'seed', 'acceleration_deg_s2')] + [time_constant, damping_ratio]
    head += [row[column['catchable']], decision, row[column['intercepted']] or 'false', row[column['catch_time_s']]]
    extras = [row[column['plan_mode']], row[column['stroke_scale']]] if moving else []
    if decision != 'go':
        kind = Classify(decision, catchable, False, None)
        return tuple(head + [kind, CONST_CLASSES[kind], 'false', 'false'] + [''] * 11 + (extras + ['', _Flag(markers)] if moving else []))
    kind = Classify(decision, catchable, planned, result)
    extras += ['' if np.isnan(result.Contact_Speed) else result.Contact_Speed, _Flag(markers)] if moving else []
    return tuple(head + [kind, CONST_CLASSES[kind], _Flag(result.Retained), _Flag(result.At_Rest),
                         result.Rest_Time - float(row[column['catch_time_s']]), result.First_Contact, result.First_Contact_Time,
                         result.Cup_Contact_Time, _Flag(result.Entered), result.Entry_Lateral, result.End_Speed,
                         result.Penetration, result.Robot_Contact, result.Tracking, result.Force_Limit_Ticks] + extras)


def Catch_Chunk(task: tp.Tuple[Interception_Setup_Str, str, str, np.ndarray, str, Contact_Str, tp.Optional[tp.Callable[..., tp.Any]]]
                ) -> tp.Tuple[tp.List[tp.Tuple[tp.Any, ...]], tp.List[tp.Tuple[tp.Any, ...]]]:
    """
    Description:
        The physical catch of a chunk of throws for one seed: the interception of the chunk (Catch.Interception.
        Intercept_Chunk, with each go's plan kept), then per throw, acceleration and contact level one row in
        CONST_PER_THROW_COLUMNS' order: a go simulated (Simulate_Catch) and classified, a no-go classified by its truth.
        With the settings' Markers (the markers variant, a moving catch's only: refused otherwise) every level is
        simulated a second time in a scene whose ball carries its markers, each throw's ball turned by
        Marker_Orientation, and those rows follow the others.

    Args:
        (1) task [Tuple]: (the interception's setup, the dataset's label, its flights.npz, the throws' indices, the
                           seed's label, the catch's settings, None or a function giving a throw's predictor: the
                           sweep's, Catch.Robustness.Injected_Predictor).

    Returns:
        (1) parameter [List<Tuple>]: The interception's rows (CONST_PER_THROW_COLUMNS of Catch.Interception).
        (2) parameter [List<Tuple>]: The catch's rows.
    """

    setup, label, path, indices, seed, contact, predictor = task
    moving = setup.Catch_Mode == CONST_MOVING
    plans: tp.Dict[tp.Tuple[int, int], Plan_Str] = {}

    def Keep(flight: int, acceleration: int, plan: Plan_Str) -> None:
        """
        Description:
            Keep a go's plan.

        Args:
            (1) flight [int]: The throw's index.
            (2) acceleration [int]: The acceleration's index.
            (3) plan [Plan_Str(object)]: The plan.
        """

        plans[(flight, acceleration)] = plan

    if contact.Markers and not moving:
        raise Catch_Error_Cls('The markers variant is a moving catch\'s (Interception.Catch_Mode moving): the static catch\'s files have no '
                              'column for it (SPEC §9.14).')
    data = Read_Dataset(path)
    rows = Intercept_Chunk((setup, label, path, indices, seed), on_plan=Keep,
                           predictor=None if predictor is None else predictor(data, setup.Flight))
    column = {name: k for k, name in enumerate(CONST_INTERCEPTION_COLUMNS)}
    accelerations = ['%.0f' % np.degrees(a) for a in setup.Accelerations]
    out = []
    for markers in (False, True) if contact.Markers else (False,):
        scene = Catch_Scene(setup.Areas, contact, Flight_Model_Str=setup.Flight, markers=markers)
        for time_constant, damping_ratio in contact.Levels:
            Set_Contact(scene, time_constant, damping_ratio)
            for row in rows:
                if row[column['decision']] != 'go':
                    out.append(Catch_Row(row, time_constant, damping_ratio, False, None, moving=moving, markers=markers))
                    continue
                flight = int(row[column['flight']])
                plan = plans[(flight, accelerations.index(row[column['acceleration_deg_s2']]))]
                result = Simulate_Catch(scene, data.Release_State[flight], plan, setup.Ready, float(row[column['catch_time_s']]),
                                        Contact_Str=contact, orientation=Marker_Orientation(label, flight) if markers else None)
                out.append(Catch_Row(row, time_constant, damping_ratio, bool(plan.Profiles), result, moving=moving, markers=markers))
    return rows, out


def Summary_Rows(rows: tp.Sequence[tp.Sequence[tp.Any]], partition: str) -> tp.List[tp.Tuple[tp.Any, ...]]:
    """
    Description:
        The summary per dataset, seed, acceleration and contact level (SPEC §9.14): the throws, the catchable ones, the
        goes and the false goes, the interception's count, the retained balls with their share of the goes and, among
        the catchable throws, of the catchable (Clopper-Pearson 95 %; NaN without one), the balls at rest, the count
        of every class, the goes whose robot touched something but the ball (R2), the time at rest after the catch
        time, the deepest penetration and the ticks at a force limit. A moving catch's rows (they carry
        CONST_MOVING_COLUMNS) give CONST_MOVING_SUMMARY_COLUMNS after the others: the goes simulated with a stroke, and
        the median and the 95th percentile of the relative speed at the first cup contact, and whether the row is the
        markers variant's, which has its own rows.

    Args:
        (1) rows [Sequence<Sequence<Any>>]: The catch's rows (Catch_Chunk).
        (2) partition [string]: The partition.

    Returns:
        (1) parameter [List<Tuple>]: The rows, in CONST_SUMMARY_COLUMNS' order (a moving catch's: then
                                     CONST_MOVING_SUMMARY_COLUMNS).
    """

    moving = bool(rows) and len(rows[0]) == len(CONST_PER_THROW_COLUMNS) + len(CONST_MOVING_COLUMNS)
    column = {name: k for k, name in enumerate(CONST_PER_THROW_COLUMNS + (CONST_MOVING_COLUMNS if moving else ()))}
    groups: tp.Dict[tp.Tuple[tp.Any, ...], tp.List[tp.Sequence[tp.Any]]] = {}
    keys = ('dataset', 'seed', 'acceleration_deg_s2', 'time_constant_s', 'damping_ratio') + (('markers',) if moving else ())
    for row in rows:
        groups.setdefault(tuple(row[column[name]] for name in keys), []).append(row)
    out = []
    for key, group in groups.items():
        label, seed, acceleration, time_constant, damping_ratio = key[:5]
        get = (lambda name: [r[column[name]] for r in group])
        go = np.array([value == 'go' for value in get('decision')])
        truth = np.array([value == 'true' for value in get('catchable')])
        retained = np.array([value == 'true' for value in get('retained')])
        share = (lambda k, n: (k / n, *Clopper_Pearson(k, n)) if n else (float('nan'),) * 3)
        rest = np.array([float(r[column['rest_s']]) for r in group if r[column['at_rest']] == 'true'], dtype=np.float64)
        depth = [float(r[column['penetration_m']]) for r in group if r[column['decision']] == 'go']
        classes = get('class')
        out.append((label, partition, seed, acceleration, time_constant, damping_ratio, len(group), int(truth.sum()), int(go.sum()),
                    int(np.sum(go & ~truth)), sum(1 for value in get('intercepted') if value == 'true'), int(retained.sum()),
                    *share(int(retained.sum()), int(go.sum())), int(np.sum(retained & truth)),
                    *share(int(np.sum(retained & truth)), int(truth.sum())), sum(1 for value in get('at_rest') if value == 'true'),
                    *[classes.count(name) for name in CONST_CLASSES], sum(1 for value in get('robot_contact') if value),
                    float(np.median(rest)) if len(rest) else float('nan'), float(rest.max()) if len(rest) else float('nan'),
                    max(depth) if depth else float('nan'), sum(int(r[column['force_limit_ticks']]) for r in group if r[column['decision']] == 'go')))
        if moving:
            speeds = np.array([float(r[column['contact_speed_m_s']]) for r in group if r[column['contact_speed_m_s']] != ''], dtype=np.float64)
            out[-1] = out[-1] + (sum(1 for r in group if r[column['decision']] == 'go' and r[column['plan_mode']] == CONST_MOVING),
                                 float(np.percentile(speeds, 50)) if len(speeds) else float('nan'),
                                 float(np.percentile(speeds, 95)) if len(speeds) else float('nan'), key[5])
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
            writer.writerow([repr(float(value)) if isinstance(value, (float, np.floating)) else value for value in row])


def Write_Per_Throw(path: str, rows: tp.Iterable[tp.Sequence[tp.Any]], sweep: bool = False, moving: bool = False) -> None:
    """
    Description:
        Write per_throw.csv (SPEC §9.14): one row per throw, acceleration and contact level; the sweep's rows begin
        with the injected error's source and scale; a moving catch's rows end with CONST_MOVING_COLUMNS.

    Args:
        (1) path [string]: The file.
        (2) rows [Iterable<Sequence<Any>>]: The rows (Catch_Chunk; the sweep's with CONST_SWEEP_COLUMNS first).
        (3) sweep [bool]: The rows are the sweep's.
        (4) moving [bool]: The rows are a moving catch's.
    """

    _Write_Rows(path, (CONST_SWEEP_COLUMNS if sweep else ()) + CONST_PER_THROW_COLUMNS + (CONST_MOVING_COLUMNS if moving else ()), rows)


def Write_Summary(path: str, rows: tp.Iterable[tp.Sequence[tp.Any]], sweep: bool = False, moving: bool = False) -> None:
    """
    Description:
        Write summary.csv (SPEC §9.14): one row per dataset, seed, acceleration and contact level; the sweep's rows
        begin with the injected error's source and scale; a moving catch's rows end with CONST_MOVING_SUMMARY_COLUMNS.

    Args:
        (1) path [string]: The file.
        (2) rows [Iterable<Sequence<Any>>]: The rows (Summary_Rows; the sweep's with CONST_SWEEP_COLUMNS first).
        (3) sweep [bool]: The rows are the sweep's.
        (4) moving [bool]: The rows are a moving catch's.
    """

    _Write_Rows(path, (CONST_SWEEP_COLUMNS if sweep else ()) + CONST_SUMMARY_COLUMNS + (CONST_MOVING_SUMMARY_COLUMNS if moving else ()), rows)


def Read_Summary(path: str) -> tp.List[tp.Dict[str, str]]:
    """
    Description:
        A catch's summary.csv (SPEC §9.14; Write_Summary), a static, a moving or the sweep's, read back for the figures,
        as text by column.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [List<Dict<string, string>>]: The rows.
    """

    return Read_Table(path, [CONST_SUMMARY_COLUMNS, CONST_SUMMARY_COLUMNS + CONST_MOVING_SUMMARY_COLUMNS,
                             CONST_SWEEP_COLUMNS + CONST_SUMMARY_COLUMNS],
                      '§9.14')


def Rebound(scene: Catch_Scene_Str, speed: float, phase: float = 0.0) -> float:
    """
    Description:
        The restitution the scene's current contact gives (Set_Contact) in one bounce: the ball dropped on the floor far
        from the robot, 20 mm above its contact height plus `phase` of one step's travel, at `speed` straight down; the
        speed it leaves the floor with over the speed it arrives with. One bounce scatters by about 0.01 with the phase of
        the first contact within a step (F11's addendum); Mean_Rebound averages the phases. Moved from the tests for the
        figures' restitution axis (the visualisation, Q14); the scene's state is reset first and left after the bounce.

    Args:
        (1) scene [Catch_Scene_Str(object)]: The catch's scene, its contact set.
        (2) speed [float]: The arrival speed [m/s].
        (3) phase [float]: The fraction of one step's travel added to the drop's height [-], 0 to 1.

    Returns:
        (1) parameter [float]: The rebound ratio [-], 0 when the ball has not left the floor within 2000 steps.
    """

    model, data, ball = scene.Model, scene.Data, scene.Ball
    mujoco.mj_resetData(model, data)
    height = ball.Radius + 0.02 + phase * speed * model.opt.timestep
    Place_Ball(model, data, [0.0, -3.0, height, 0.0, 0.0, -speed], Ball_Str=ball, Flight_Model_Str=scene.Flight)
    arrival, touched = 0.0, False
    for _ in range(2000):
        before = data.qvel[ball.Velocity_Address + 2]
        Apply_Drag(model, data, Ball_Str=ball, Flight_Model_Str=scene.Flight)
        mujoco.mj_step(model, data)
        touching = data.qpos[ball.Position_Address + 2] < ball.Radius
        if touching and not touched:
            arrival, touched = -before, True
        if touched and not touching:
            return float(data.qvel[ball.Velocity_Address + 2] / arrival)
    return 0.0


def Mean_Rebound(scene: Catch_Scene_Str, speed: float, phases: int = CONST_REBOUND_PHASES) -> float:
    """
    Description:
        The restitution of the scene's current contact as SPEC §4.14 and the records of F11 and F15 measure it: the mean of
        Rebound over `phases` positions of the first contact within a step (the drop raised by k / phases of one step's
        travel, k = 0 ... phases - 1).

    Args:
        (1) scene [Catch_Scene_Str(object)]: The catch's scene, its contact set.
        (2) speed [float]: The arrival speed [m/s].
        (3) phases [int]: The positions within a step.

    Returns:
        (1) parameter [float]: The mean rebound ratio [-].
    """

    return float(np.mean([Rebound(scene, speed, k / phases) for k in range(phases)]))
