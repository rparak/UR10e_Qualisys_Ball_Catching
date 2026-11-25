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
# File Name: ../Scene/Ball.py
## =========================================================================== ##

# Dataclasses (Data classes)
from dataclasses import dataclass
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation; here the ball's free joint, its applied force and the model's options) [pip3 install mujoco]
import mujoco
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Flight/Core - the flight model and its acceleration
from Flight.Core import Flight_Model_Str, Acceleration
#   ../Scene/Core - the ball's body name in the scene
from Scene.Core import CONST_BALL_BODY

"""
Description:
    Initialization of constants.

    The ball's flight in the MuJoCo scene (SPEC §2.5, §7.10; F5): F3's flight
    model drives the scene's free ball, so one model drives the dataset, the predictors and the MuJoCo ball. The drag
    -m k |v| v is applied as the ball's external force before every step (data.xfrc_applied at its centre of mass);
    MuJoCo adds gravity itself. The scene's integrator (implicitfast, the vendored robot's) is semi-implicit Euler in
    the ball's free joint (v <- v + h a, then q <- q + h v), whose velocity is the velocity half a step back: a ball
    placed with v_0 - h/2 a(v_0) moves as the leapfrog scheme, second order in the step, and its synchronous velocity is
    qvel + h/2 a(qvel). Measured against Flight.Core.Simulate on the train partitions of Default, Catch_In and
    Catch_Near (SPEC §4.13): at most 0.98 um and 1.39 um/s over every flight to the floor at the scene's 0.002 s.
    Nothing in the scene changes: the force and the start live in the data. Pure computation on the model and the data
    given: no file, no configuration, no printing.
"""
# The ball's markers on its visual mesh (Find_Markers; F15): a vertex belongs to a marker when it
# lies this far beyond the ball's sphere (the mesh's vertices on the sphere are within 0.1 mm of it; the markers reach
# 16 mm beyond it), and two such vertices to the same marker when the cosine of the angle between them, seen from the
# ball's centre, is above the bound (the six markers are 90 deg apart; one spans 24 deg).
#   Unit [m], [-]
CONST_MARKER_SHELL = 0.003
CONST_MARKER_COSINE = 0.9


class Ball_Error_Cls(ValueError):
    """
    Description:
        The ball cannot be placed or driven as asked: a model without the scene's free ball, a model whose gravity or
        medium differs from the flight model, a state that is not six finite numbers. The message names the value and
        what it must be; a script prints it verbatim as an [ERROR] line.
    """


@dataclass
class Ball_Str:
    """
    Description:
        The scene's ball in a compiled model (Find_Ball): its body, the addresses of its free joint and its sphere.
    """

    # The ball's body.
    #   Unit [int]
    Body: int = 0
    # The first address of its free joint's position (x, y, z, then the quaternion) and of its velocity (linear, then
    # angular).
    #   Unit [int]
    Position_Address: int = 0
    Velocity_Address: int = 0
    # The mass of its body, read from the model (the force divided by it gives the flight model's acceleration
    # whatever the assumed mass, SPEC §7.4), and the radius of its sphere.
    #   Unit [float, kg], [float, m]
    Mass: float = 0.0
    Radius: float = 0.0


def Find_Ball(model: mujoco.MjModel) -> Ball_Str:
    """
    Description:
        The scene's ball in a compiled model: the body CONST_BALL_BODY with one free joint and one sphere geom (its
        collision sphere; the visual mesh is not a sphere). A model without one of them is refused, naming what is
        missing.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.

    Returns:
        (1) parameter [Ball_Str(object)]: The ball's addresses, mass and radius.
    """

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONST_BALL_BODY)
    if body < 0:
        raise Ball_Error_Cls('The model has no body "%s": the ball is the scene\'s (models/scene.xml). Build the model with '
                             'Scene.Core.Build_Scene.' % (CONST_BALL_BODY,))
    joint = int(model.body_jntadr[body])
    if model.body_jntnum[body] != 1 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise Ball_Error_Cls('The body "%s" has %d joints and no single free joint: the ball flies on one free joint '
                             '(models/scene.xml).' % (CONST_BALL_BODY, int(model.body_jntnum[body])))
    first = int(model.body_geomadr[body])
    spheres = [g for g in range(first, first + int(model.body_geomnum[body])) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    if len(spheres) != 1:
        raise Ball_Error_Cls('The body "%s" has %d sphere geoms and not one: the ball\'s collision sphere is the scene\'s '
                             '(models/scene.xml).' % (CONST_BALL_BODY, len(spheres)))
    geom = spheres[0]
    return Ball_Str(Body=int(body), Position_Address=int(model.jnt_qposadr[joint]), Velocity_Address=int(model.jnt_dofadr[joint]),
                    Mass=float(model.body_mass[body]), Radius=float(model.geom_size[geom][0]))


def Find_Markers(model: mujoco.MjModel) -> tp.Tuple[np.ndarray, float]:
    """
    Description:
        The ball's markers, measured on the scene's own visual mesh of the ball (the CAD's ball with its six markers,
        SPEC §4.4), in the ball's frame: the mesh's vertices more than CONST_MARKER_SHELL beyond the ball's sphere,
        grouped by direction, and per group the sphere through them (least squares). The scene's ball collides as
        its sphere alone; these are what a variant of the catch adds to it in memory (Catch.Physical). A ball without
        a mesh, or a mesh without a vertex beyond the sphere, is refused.

    Args:
        (1) model [mujoco.MjModel]: The compiled scene.

    Returns:
        (1) parameter [Matrix<float> Mx3]: The markers' centres in the ball's frame [m].
        (2) parameter [float]: Their radius, the mean over the markers [m].
    """

    ball = Find_Ball(model)
    first = int(model.body_geomadr[ball.Body])
    meshes = [g for g in range(first, first + int(model.body_geomnum[ball.Body])) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    if len(meshes) != 1:
        raise Ball_Error_Cls('The body "%s" has %d mesh geoms and not one: the markers are measured on the ball\'s visual mesh '
                             '(models/scene.xml).' % (CONST_BALL_BODY, len(meshes)))
    geom = meshes[0]
    mesh = int(model.geom_dataid[geom])
    vertices = np.asarray(model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]], dtype=np.float64)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    points = model.geom_pos[geom] + vertices @ rotation.reshape(3, 3).T
    distance = np.linalg.norm(points, axis=1)
    beyond = points[distance > ball.Radius + CONST_MARKER_SHELL]
    if not len(beyond):
        raise Ball_Error_Cls('The ball\'s mesh has no vertex more than %r m beyond its sphere of %r m: it carries no marker to measure '
                             '(models/meshes/ball.stl).' % (CONST_MARKER_SHELL, ball.Radius))
    directions = beyond / np.linalg.norm(beyond, axis=1)[:, None]
    left = np.ones(len(beyond), dtype=bool)
    centres, radii = [], []
    while left.any():
        select = left & (directions @ directions[int(np.flatnonzero(left)[0])] > CONST_MARKER_COSINE)
        group = beyond[select]
        solution = np.linalg.lstsq(np.c_[2.0 * group, np.ones(len(group))], np.sum(group ** 2, axis=1), rcond=None)[0]
        centres.append(solution[:3])
        radii.append(float(np.sqrt(solution[3] + solution[:3] @ solution[:3])))
        left &= ~select
    return np.array(centres), float(np.mean(radii))


def _Check_Model(model: mujoco.MjModel, Flight_Model_Str: Flight_Model_Str) -> None:
    """
    Description:
        Refuse a model whose gravity is not the flight model's, or whose medium (density, viscosity, wind) would add
        MuJoCo's fluid forces to the applied drag.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) Flight_Model_Str [Flight_Model_Str(object)]: The flight model.
    """

    if not np.array_equal(model.opt.gravity, Flight_Model_Str.Gravity):
        raise Ball_Error_Cls('The model\'s gravity %s m/s^2 is not the flight model\'s %s m/s^2 (Flight.Gravity): one flight model '
                             'drives the ball (SPEC §2.5).' % (model.opt.gravity.tolist(), np.asarray(Flight_Model_Str.Gravity).tolist()))
    if model.opt.density != 0.0 or model.opt.viscosity != 0.0 or np.any(model.opt.wind != 0.0):
        raise Ball_Error_Cls('The model has a medium (density %r kg/m^3, viscosity %r Pa s, wind %s m/s): MuJoCo\'s fluid forces '
                             'would add to the applied drag. The scene has none (SPEC §7.10).'
                             % (float(model.opt.density), float(model.opt.viscosity), model.opt.wind.tolist()))


def _Synchronous_Velocity(qvel: np.ndarray, h: float, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        The ball's velocity at the data's time from its free joint's velocity, which is half a step behind in free
        flight: qvel + h/2 a(qvel).

    Args:
        (1) qvel [Vector<float> 1x3]: The free joint's linear velocity [m/s].
        (2) h [float]: The model's time step [s].
        (3) Flight_Model_Str [Flight_Model_Str(object)]: The flight model.

    Returns:
        (1) parameter [Vector<float> 1x3]: The velocity [m/s].
    """

    return qvel + 0.5 * h * Acceleration(qvel, Flight_Model_Str=Flight_Model_Str)


def Place_Ball(model: mujoco.MjModel, data: mujoco.MjData, state: tp.Sequence[float], *, Ball_Str: Ball_Str,
               Flight_Model_Str: Flight_Model_Str) -> None:
    """
    Description:
        Place the ball at a state of its flight (the release, or any later state of Flight.Core.Simulate): its centre at
        the position, no rotation, no spin, and its free joint's velocity half a step back, v - h/2 a(v), so that the
        scene's semi-implicit step moves it as the leapfrog scheme. A state that is not six finite numbers, and a model
        whose gravity or medium differs from the flight model (_Check_Model), are refused.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) data [mujoco.MjData]: The simulation's data, changed in place.
        (3) state [Vector<float> 1x6]: The ball's position and velocity [m, m/s], in the scene's frame.
        (4) Ball_Str [Ball_Str(object)]: REQUIRED, keyword-only: the ball in the model (Find_Ball).
        (5) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.
    """

    x = np.asarray(state, dtype=np.float64)
    if x.shape != (6,) or not np.all(np.isfinite(x)):
        raise Ball_Error_Cls('The ball\'s state %s is not six finite numbers (x, y, z [m], v_x, v_y, v_z [m/s]).' % (np.asarray(state).tolist(),))
    _Check_Model(model, Flight_Model_Str)
    p, v = Ball_Str.Position_Address, Ball_Str.Velocity_Address
    data.qpos[p:p + 3] = x[:3]
    data.qpos[p + 3:p + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[v:v + 3] = x[3:] - 0.5 * model.opt.timestep * Acceleration(x[3:], Flight_Model_Str=Flight_Model_Str)
    data.qvel[v + 3:v + 6] = 0.0


def Apply_Drag(model: mujoco.MjModel, data: mujoco.MjData, *, Ball_Str: Ball_Str, Flight_Model_Str: Flight_Model_Str) -> None:
    """
    Description:
        Set the ball's external force for the next step: the mass times the flight model's acceleration less gravity,
        -m k |v| v, at the synchronous velocity, at the ball's centre of mass, no torque. Called before every mj_step;
        MuJoCo keeps data.xfrc_applied until it is set again.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) data [mujoco.MjData]: The simulation's data, changed in place.
        (3) Ball_Str [Ball_Str(object)]: REQUIRED, keyword-only: the ball in the model (Find_Ball).
        (4) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.
    """

    v = Ball_Str.Velocity_Address
    u = _Synchronous_Velocity(data.qvel[v:v + 3], model.opt.timestep, Flight_Model_Str)
    data.xfrc_applied[Ball_Str.Body, :3] = Ball_Str.Mass * (Acceleration(u, Flight_Model_Str=Flight_Model_Str) - Flight_Model_Str.Gravity)
    data.xfrc_applied[Ball_Str.Body, 3:] = 0.0


def Ball_State(model: mujoco.MjModel, data: mujoco.MjData, *, Ball_Str: Ball_Str, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        The ball's state at the data's time in free flight: its centre and its synchronous velocity, qvel + h/2 a(qvel).
        In contact the correction does not hold (the contact's acceleration is not the flight model's); how a state in
        contact is read is F11's (SPEC §7.10).

    Args:
        (1) model [mujoco.MjModel]: The compiled model.
        (2) data [mujoco.MjData]: The simulation's data.
        (3) Ball_Str [Ball_Str(object)]: REQUIRED, keyword-only: the ball in the model (Find_Ball).
        (4) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Vector<float> 1x6]: The position and the velocity [m, m/s].
    """

    p, v = Ball_Str.Position_Address, Ball_Str.Velocity_Address
    return np.concatenate((data.qpos[p:p + 3], _Synchronous_Velocity(data.qvel[v:v + 3], model.opt.timestep, Flight_Model_Str)))
