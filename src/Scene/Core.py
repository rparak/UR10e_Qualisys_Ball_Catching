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
# File Name: ../Scene/Core.py
## =========================================================================== ##

# OS (Operating system interfaces)
import os
# Hashlib (Secure hashes; the meshes of the fingerprint)
import hashlib
# JSON (The fingerprint's file)
import json
# Platform (The environment recorded beside a fingerprint)
import platform
# Warnings (MuJoCo's warnings while the scene is built)
import warnings
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation; the scene's model) [pip3 install mujoco]
import mujoco
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the areas' structures
from Configuration.Core import Areas_Str
#   ../Geometry/Core - the areas' rotation
from Geometry.Core import Rotation_Matrix_X, Rotation_Matrix_Y, Rotation_Matrix_Z, CONST_SPHERICAL_SHAPES
#   ../Robot/Core - the robot in the compiled scene
from Robot.Core import Find_Robot, Robot_Str, CONST_FLANGE_SITE, CONST_JOINT_NAMES

"""
Description:
    Initialization of constants.

    The MuJoCo scene (SPEC §7.10): built through MjSpec from models/scene.xml, the vendored UR10e and
    models/cup.xml, with the areas of the configuration drawn in; and the fingerprint that pins the
    compiled model. The vendored robot is never edited: it is attached in code. The only reader and
    writer of models/scene_fingerprint.json (SPEC §9.10). Nothing is printed.

    Frame (SPEC §14.6): the origin on the floor under the robot's base axis,
    z up, the thrower at -y. Lengths in m, angles in rad.
"""
# The models folder (SPEC §6), built from this file's location, so the scene builds from any working
# directory.
#   Unit [string]
CONST_MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models'))
# The files of the models folder that the scene is built from and its fingerprint, relative to it
# (written by scripts/Maintenance/build_scene.py; the robot is MuJoCo Menagerie's, vendored).
#   Unit [string]
CONST_SCENE_FILE = 'scene.xml'
CONST_CUP_FILE = 'cup.xml'
CONST_ROBOT_FILE = os.path.join('universal_robots_ur10e', 'ur10e.xml')
CONST_FINGERPRINT_FILE = 'scene_fingerprint.json'
# The site of scene.xml on which the robot's base is mounted: the centre of the stand's mounting face.
#   Unit [string]
CONST_MOUNT_SITE = 'robot_mount'
# The prefix of every name of the vendored robot and of the cup on it in the compiled scene.
#   Unit [string]
CONST_ROBOT_PREFIX = 'ur10e/'
# The names scene.xml and cup.xml give the ball's body (and its free joint), the cup's body and the site
# in the centre of the cup's mouth, the tool centre point.
#   Unit [string]
CONST_BALL_BODY = 'ball'
CONST_CUP_BODY = 'cup'
CONST_CUP_MOUTH_SITE = 'cup_mouth'
# The geom groups: the areas, the visual geometry and the collision geometry (2 and 3 are the vendored
# robot's own groups; a viewer shows 0 to 2 by default, so the collision geometry is hidden).
#   Unit [int]
CONST_GROUP_AREA = 1
CONST_GROUP_VISUAL = 2
CONST_GROUP_COLLISION = 3
# The contact bits (F11). MuJoCo generates a contact between two geoms when the contype
# of one shares a bit with the conaffinity of the other. The world bit: the floor, the stand and the robot's
# links have it in both (MuJoCo's default); the cup's cylinder and the wrist's shapes inside the cup have it in
# their contype only, so they meet the world and the robot, and the ball, which does not take it, passes them.
# The ball bit: the ball offers it (contype world | ball, conaffinity ball) and only the cup's own shape takes
# it (contype 0, conaffinity ball), so the ball lands on the cup's bottom and walls, and those shapes touch
# nothing else: they take no part in the clearance of the robot's motion (Robot.Core.Collision_Pairs).
#   Unit [int]
CONST_BIT_WORLD = 1
CONST_BIT_BALL = 2
# The names cup.xml gives the cup's shape for the ball: the bottom, the cone's staves and the lip's boxes
# (the last two followed by a two-digit number).
#   Unit [string]
CONST_CUP_BOTTOM_GEOM = 'cup_bottom'
CONST_CUP_WALL_GEOM = 'cup_wall_'
CONST_CUP_LIP_GEOM = 'cup_lip_'
# The vendored robot's collision shapes that reach into the cup's cavity, by their body and their position in
# it [m]: wrist 2's capsule along joint 6's axis and the flange's cylinder stand up to 12 mm above the cup's
# bottom (F11 Part A), so a ball in the cup would land on them. They keep the world bit in their contype
# only; a per-commit test measures that they are the shapes a ball on the cup's bottom overlaps, and all of them.
#   Unit [Dict<string, Vector<float> 1x3>]
CONST_CAVITY_GEOMS = {'wrist_2_link': (0.0, 0.028, 0.12), 'wrist_3_link': (0.0, 0.097, 0.0)}
# The areas drawn in the scene (SPEC §2.4), by the name of their geom: the field of Areas they come from
# and their colour (red, green, blue, alpha; the 2022 scene's red, blue and green, the catch cuboid
# yellow). They are visual only: nothing collides with an area.
#   Unit [Dict<string, Tuple<string, Vector<float> 1x4>>]
CONST_AREAS = {'area_throw': ('Throw', (1.0, 0.0, 0.0, 0.12)), 'area_flight': ('Flight', (0.0, 0.0, 1.0, 0.08)),
               'area_catch': ('Catch', (1.0, 0.8, 0.0, 0.12)), 'area_robot': ('Robot', (0.0, 1.0, 0.0, 0.15))}
# The drawing of a spherical area: its surface is sampled every 90 / RINGS degrees from its pole and
# every 360 / SEGMENTS degrees around it, and MuJoCo draws the convex hull of the samples (at these
# values the hull lies within 0.3 % of the radius inside the sphere).
#   Unit [int]
CONST_AREA_RINGS = 12
CONST_AREA_SEGMENTS = 48
# The version of the fingerprint's file; a reader refuses another.
#   Unit [int]
CONST_FINGERPRINT_VERSION = 1
# The significant digits a number of the fingerprint keeps: nine for the compiled model's float64 values,
# which hold every length to below a nanometre, and seven for its float32 values (colours), which carry
# no more.
#   Unit [int]
CONST_FINGERPRINT_DIGITS = 9
CONST_FINGERPRINT_DIGITS_SINGLE = 7
# The sizes of the model that the fingerprint holds (MjModel's fields).
#   Unit [Tuple<string>]
CONST_SIZES = ('nq', 'nv', 'nu', 'na', 'nbody', 'njnt', 'ngeom', 'nsite', 'nmesh', 'nmeshvert', 'nmeshface', 'nmat', 'ntex',
               'nkey', 'neq', 'nexclude', 'npair', 'ntendon', 'nsensor', 'ncam', 'nlight')
# The fields of the model's options that the fingerprint holds (MjOption's fields in MuJoCo 3.16.0).
#   Unit [Tuple<string>]
CONST_OPTIONS = ('timestep', 'impratio', 'tolerance', 'ls_tolerance', 'noslip_tolerance', 'ccd_tolerance', 'sleep_tolerance',
                 'gravity', 'wind', 'magnetic', 'density', 'viscosity', 'o_margin', 'o_solref', 'o_solimp', 'o_friction',
                 'integrator', 'cone', 'jacobian', 'solver', 'iterations', 'ls_iterations', 'noslip_iterations', 'ccd_iterations',
                 'disableflags', 'enableflags', 'disableactuator', 'sdf_initpoints', 'sdf_iterations')


class Scene_Error_Cls(ValueError):
    """
    Description:
        The scene cannot be built, or its fingerprint cannot be read. The message names the file or the
        element and what to do; a script prints it verbatim as an [ERROR] line.
    """


def _Spherical_Vertices(orientation: tp.Sequence[float], radius: float, shape: str) -> np.ndarray:
    """
    Description:
        Points on the surface of a spherical area about its centre, in the scene's frame: a sphere, a
        hemisphere (z >= 0 in the area's frame) or a quarter sphere (also x >= 0), turned by
        R = R_x(alpha) @ R_y(beta) @ R_z(gamma) as Geometry.Core.Spherical_Area_Cls turns it. Their convex
        hull is the area as it is drawn.

    Args:
        (1) orientation [Vector<float> 1x3]: The angles alpha, beta, gamma [rad].
        (2) radius [float]: The radius [m].
        (3) shape [string]: One of CONST_SPHERICAL_SHAPES.

    Returns:
        (1) parameter [Matrix<float> Nx3]: The points relative to the centre [m].
    """

    if shape not in CONST_SPHERICAL_SHAPES:
        raise Scene_Error_Cls('The shape "%s" of a spherical area is not one of %s.' % (shape, ', '.join(CONST_SPHERICAL_SHAPES)))
    polar = np.linspace(0.0, np.pi if shape == 'sphere' else np.pi / 2.0,
                        (2 if shape == 'sphere' else 1) * CONST_AREA_RINGS + 1)
    if shape == 'quarter_sphere':
        around = np.linspace(-np.pi / 2.0, np.pi / 2.0, CONST_AREA_SEGMENTS // 2 + 1)
    else:
        around = np.linspace(0.0, 2.0 * np.pi, CONST_AREA_SEGMENTS, endpoint=False)
    theta, phi = np.meshgrid(polar, around, indexing='ij')
    local = radius * np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1).reshape(-1, 3)
    alpha, beta, gamma = orientation
    return local @ (Rotation_Matrix_X(alpha) @ Rotation_Matrix_Y(beta) @ Rotation_Matrix_Z(gamma)).T


def _Add_Areas(scene: mujoco.MjSpec, areas: Areas_Str) -> None:
    """
    Description:
        Draw the areas of the configuration into the scene's specification: the three cuboids as boxes,
        the robot's area as the convex hull of points on its surface. Visual only: no contact bits.

    Args:
        (1) scene [mujoco.MjSpec]: The scene's specification.
        (2) areas [Areas_Str(object)]: The areas (SPEC §8.1).
    """

    for name, (field_name, colour) in CONST_AREAS.items():
        area = getattr(areas, field_name)
        if field_name == 'Robot':
            mesh = scene.add_mesh(name=name)
            mesh.uservert = _Spherical_Vertices(area.Orientation, area.Radius, area.Shape).ravel()
            scene.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=list(area.Centre), rgba=list(colour),
                                     contype=0, conaffinity=0, group=CONST_GROUP_AREA)
        else:
            scene.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(area.Centre),
                                     size=[0.5 * edge for edge in area.Size], rgba=list(colour), contype=0, conaffinity=0,
                                     group=CONST_GROUP_AREA)


def _Keep_Log_Out_Of_The_Working_Directory() -> None:
    """
    Description:
        MuJoCo writes each of its warnings to stderr, raises it as a Python warning and also appends it
        to a file MUJOCO_LOG.TXT in the working directory (MuJoCo 3.16.0, measured 2026-11-20). The file is
        switched off for this process (MjLogConfig), so that no run leaves it in a folder; the two other
        routes stay, and a warning is still seen.
    """

    config = mujoco.MjLogConfig.get()
    if config.logto_file:
        config.logto_file = False
        mujoco.MjLogConfig.set(config)


def _Model_Files(models: str) -> tp.List[str]:
    """
    Description:
        The three files the scene is built from, each present; MuJoCo's log file in the working directory is switched
        off for the process.

    Args:
        (1) models [string]: The models folder.

    Returns:
        (1) parameter [List<string>]: The scene's, the robot's and the cup's file.
    """

    paths = [os.path.join(models, name) for name in (CONST_SCENE_FILE, CONST_ROBOT_FILE, CONST_CUP_FILE)]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise Scene_Error_Cls('The scene cannot be built: %s missing. Restore the models folder from the repository '
                              '(git checkout -- models).' % ', '.join(missing))
    _Keep_Log_Out_Of_The_Working_Directory()
    return paths


def _Specification(paths: tp.List[str], areas: Areas_Str) -> mujoco.MjSpec:
    """
    Description:
        The scene's specification before it is compiled: models/scene.xml, the vendored UR10e mounted on the stand's
        mounting face, the cup on the robot's flange, and the areas of the configuration. The vendored file is read
        as it is; every change to it is made here: the two shapes of the wrist that reach into the cup's cavity take
        no contact from the ball (CONST_CAVITY_GEOMS).

    Args:
        (1) paths [List<string>]: The scene's, the robot's and the cup's file (_Model_Files).
        (2) areas [Areas_Str(object)]: The areas drawn (SPEC §8.1).

    Returns:
        (1) parameter [mujoco.MjSpec]: The specification.
    """

    scene, robot, cup = (mujoco.MjSpec.from_file(path) for path in paths)
    flange, mount = robot.site(CONST_FLANGE_SITE), scene.site(CONST_MOUNT_SITE)
    if flange is None or mount is None:
        raise Scene_Error_Cls('The scene cannot be built: the site "%s" of %s or the site "%s" of %s is missing. Restore the '
                              'models folder from the repository (git checkout -- models).'
                              % (CONST_FLANGE_SITE, paths[1], CONST_MOUNT_SITE, paths[0]))
    # 1\ The wrist's shapes inside the cup's cavity keep the world bit in their contype only (the ball
    #    passes them, F11); the vendored file is not changed.
    for body, position in CONST_CAVITY_GEOMS.items():
        link = robot.body(body)
        found = [geom for geom in (link.geoms if link is not None else [])
                 if geom.conaffinity and np.allclose(geom.pos, position, rtol=0.0, atol=1e-12)]
        if len(found) != 1:
            raise Scene_Error_Cls('The scene cannot be built: %s has %d colliding shapes at %s in its body "%s", not the one that '
                                  'reaches into the cup. Restore the models folder from the repository (git checkout -- models).'
                                  % (paths[1], len(found), position, body))
        found[0].conaffinity = 0
    # 2\ The cup on the flange, then the robot with its cup on the stand. The cup keeps its own names
    #    under the robot's prefix: without a prefix MuJoCo 3.16.0 puts "/" before each (measured
    #    2026-11-20), so the empty one is given.
    robot.attach(cup, prefix='', site=flange)
    scene.attach(robot, prefix=CONST_ROBOT_PREFIX, site=mount)
    # 3\ The areas of the configuration.
    _Add_Areas(scene, areas)
    return scene


def _Without_Warning(models: str, build: tp.Callable[[], tp.Any]) -> tp.Any:
    """
    Description:
        The result of a step of the scene's build, refused when MuJoCo raises an error or warns: the scene builds
        without a warning, so a warning is news.

    Args:
        (1) models [string]: The models folder (named in the message).
        (2) build [Callable[[], Any]]: The step.

    Returns:
        (1) parameter [Any]: What the step returns.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = build()
        except ValueError as error:
            if isinstance(error, Scene_Error_Cls):
                raise
            raise Scene_Error_Cls('MuJoCo could not build the scene from %s: %s' % (models, error)) from error
    if caught:
        raise Scene_Error_Cls('MuJoCo warned while the scene was built from %s, which it does not for the shipped scene: %s. '
                              'Compare the models folder with the repository (scripts/Maintenance/build_scene.py).'
                              % (models, '; '.join(str(warning.message) for warning in caught)))
    return result


def Scene_Spec(*, Areas_Str: Areas_Str, models: str = CONST_MODELS_DIR) -> mujoco.MjSpec:
    """
    Description:
        The scene's specification, not yet compiled: what Build_Scene
        compiles. A caller may add what the compiled model cannot take afterwards (a light, a texture, a material:
        the view's theme, View.Theme) and compiles it with Compile_Scene; the shipped scene is Build_Scene's. A
        warning of MuJoCo while the robot is attached is refused when the specification is compiled: MuJoCo 3.16.0
        prints it at once and raises it to Python at the compilation (measured 2026-11-22).

    Args:
        (1) Areas_Str [Areas_Str(object)]: REQUIRED, keyword-only: the areas drawn (SPEC §8.1).
        (2) models [string]: The models folder (the shipped one by default).

    Returns:
        (1) parameter [mujoco.MjSpec]: The specification.
    """

    paths = _Model_Files(models)
    return _Without_Warning(models, lambda: _Specification(paths, Areas_Str))


def Compile_Scene(scene: mujoco.MjSpec, models: str = CONST_MODELS_DIR) -> mujoco.MjModel:
    """
    Description:
        Compile a specification of the scene (Scene_Spec); a compilation that MuJoCo warns about is refused.

    Args:
        (1) scene [mujoco.MjSpec]: The specification.
        (2) models [string]: The models folder it was read from (named in a refusal).

    Returns:
        (1) parameter [mujoco.MjModel]: The compiled scene.
    """

    return _Without_Warning(models, scene.compile)


def Build_Scene(*, Areas_Str: Areas_Str, models: str = CONST_MODELS_DIR) -> mujoco.MjModel:
    """
    Description:
        Build and compile the scene: models/scene.xml (the floor, the stand, the rail, the cameras, the
        ball), the vendored UR10e mounted on the stand's mounting face, the cup on the robot's flange,
        and the areas of the configuration (the specification of Scene_Spec, compiled as it is). A build that
        MuJoCo warns about is refused: the scene builds without a warning, so a warning is news. MuJoCo's log
        file in the working directory is switched off for the process.

    Args:
        (1) Areas_Str [Areas_Str(object)]: REQUIRED, keyword-only: the areas drawn (SPEC §8.1).
        (2) models [string]: The models folder (the shipped one by default).

    Returns:
        (1) parameter [mujoco.MjModel]: The compiled scene.
    """

    paths = _Model_Files(models)
    return _Without_Warning(models, lambda: _Specification(paths, Areas_Str).compile())


def Scene_Robot(model: mujoco.MjModel) -> Robot_Str:
    """
    Description:
        The robot of a compiled scene, with the centre of the cup's mouth as its tool centre point.

    Args:
        (1) model [mujoco.MjModel]: A scene compiled by Build_Scene.

    Returns:
        (1) parameter [Robot_Str(object)]: The robot's addresses in the model.
    """

    return Find_Robot(model, CONST_ROBOT_PREFIX, CONST_ROBOT_PREFIX + CONST_CUP_MOUTH_SITE)


def _Value(value: tp.Any) -> tp.Any:
    """
    Description:
        A value of the model as the fingerprint holds it: an integer as it is, a real number rounded to
        CONST_FINGERPRINT_DIGITS significant digits (CONST_FINGERPRINT_DIGITS_SINGLE for a float32), an
        array as a list of those.

    Args:
        (1) value [number or array]: The value.

    Returns:
        (1) parameter [int, float or List]: The value to store.
    """

    array = np.asarray(value)
    if array.dtype.kind in 'iub':
        return array.astype(np.int64).tolist()
    digits = CONST_FINGERPRINT_DIGITS_SINGLE if array.dtype == np.float32 else CONST_FINGERPRINT_DIGITS
    rounded = np.array([float('%.*g' % (digits, x)) for x in array.astype(np.float64).ravel()])
    # No negative zero: it would differ from zero in the file only.
    return (rounded + 0.0).reshape(array.shape).tolist()


def _Name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    """
    Description:
        The name of an element, '' without one or for the index -1.

    Args:
        (1) model [mujoco.MjModel]: The model.
        (2) kind [mujoco.mjtObj]: The element's kind.
        (3) index [int]: Its id.

    Returns:
        (1) parameter [string]: The name.
    """

    return (mujoco.mj_id2name(model, kind, int(index)) or '') if index >= 0 else ''


def _Element_Keys(model: mujoco.MjModel, kind: mujoco.mjtObj, bodies: np.ndarray, body_name: tp.List[str]) -> tp.List[str]:
    """
    Description:
        The fingerprint's key of every geom or site: its name, or without one its body's name and its
        place among that body's elements of the kind ("ur10e/forearm_link#2").

    Args:
        (1) model [mujoco.MjModel]: The model.
        (2) kind [mujoco.mjtObj]: mjOBJ_GEOM or mjOBJ_SITE.
        (3) bodies [Vector<int> 1xN]: The body of each element.
        (4) body_name [List<string>]: The bodies' names by id.

    Returns:
        (1) parameter [List<string>]: The keys, in the elements' order.
    """

    seen: tp.Dict[int, int] = {}
    keys = []
    for index, body in enumerate(bodies):
        seen[int(body)] = seen.get(int(body), -1) + 1
        keys.append(_Name(model, kind, index) or '%s#%d' % (body_name[int(body)], seen[int(body)]))
    return keys


def Fingerprint(model: mujoco.MjModel) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    """
    Description:
        The fingerprint of a compiled model: its sizes, its options and, per element, the fields that
        decide what it computes and shows, keyed by the element's name so that a difference names the
        element and the field. An element without a name (the vendored robot's geoms) is keyed by its
        body and its place among that body's elements ("ur10e/forearm_link#2"). A mesh is held by the
        numbers of its vertices and faces and the SHA-256 of both arrays.

    Args:
        (1) model [mujoco.MjModel]: The compiled model.

    Returns:
        (1) parameter [Dict<string, Dict>]: The sections 'sizes', 'option', 'body', 'joint', 'geom', 'site',
                                            'mesh', 'material', 'actuator', 'key' and 'exclude'.
    """

    kinds = mujoco.mjtObj
    body_name = [_Name(model, kinds.mjOBJ_BODY, index) for index in range(model.nbody)]
    free, ball = int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)

    result: tp.Dict[str, tp.Dict[str, tp.Any]] = {'sizes': {name: int(getattr(model, name)) for name in CONST_SIZES},
                                                  'option': {name: _Value(getattr(model.opt, name)) for name in CONST_OPTIONS}}
    result['body'] = {body_name[i]: {'parent': body_name[model.body_parentid[i]], 'pos': _Value(model.body_pos[i]),
                                     'quat': _Value(model.body_quat[i]), 'mass': _Value(model.body_mass[i]),
                                     'inertia': _Value(model.body_inertia[i]), 'ipos': _Value(model.body_ipos[i]),
                                     'iquat': _Value(model.body_iquat[i]), 'gravcomp': _Value(model.body_gravcomp[i])}
                      for i in range(model.nbody)}
    result['joint'] = {}
    for i in range(model.njnt):
        kind, first, address = int(model.jnt_type[i]), int(model.jnt_dofadr[i]), int(model.jnt_qposadr[i])
        dofs = slice(first, first + {free: 6, ball: 3}.get(kind, 1))
        positions = slice(address, address + {free: 7, ball: 4}.get(kind, 1))
        result['joint'][_Name(model, kinds.mjOBJ_JOINT, i)] = {
            'type': kind, 'body': body_name[model.jnt_bodyid[i]], 'pos': _Value(model.jnt_pos[i]),
            'axis': _Value(model.jnt_axis[i]), 'limited': int(model.jnt_limited[i]), 'range': _Value(model.jnt_range[i]),
            'stiffness': _Value(model.jnt_stiffness[i]), 'margin': _Value(model.jnt_margin[i]), 'solref': _Value(model.jnt_solref[i]),
            'solimp': _Value(model.jnt_solimp[i]), 'damping': _Value(model.dof_damping[dofs]), 'armature': _Value(model.dof_armature[dofs]),
            'frictionloss': _Value(model.dof_frictionloss[dofs]), 'qpos0': _Value(model.qpos0[positions])}
    result['geom'] = {}
    for i, key in enumerate(_Element_Keys(model, kinds.mjOBJ_GEOM, model.geom_bodyid, body_name)):
        mesh = int(model.geom_dataid[i]) if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH) else -1
        result['geom'][key] = {
            'type': int(model.geom_type[i]), 'body': body_name[model.geom_bodyid[i]], 'size': _Value(model.geom_size[i]),
            'pos': _Value(model.geom_pos[i]), 'quat': _Value(model.geom_quat[i]), 'contype': int(model.geom_contype[i]),
            'conaffinity': int(model.geom_conaffinity[i]), 'condim': int(model.geom_condim[i]), 'priority': int(model.geom_priority[i]),
            'friction': _Value(model.geom_friction[i]), 'solmix': _Value(model.geom_solmix[i]), 'solref': _Value(model.geom_solref[i]),
            'solimp': _Value(model.geom_solimp[i]), 'margin': _Value(model.geom_margin[i]), 'gap': _Value(model.geom_gap[i]),
            'rgba': _Value(model.geom_rgba[i]), 'group': int(model.geom_group[i]), 'mesh': _Name(model, kinds.mjOBJ_MESH, mesh),
            'material': _Name(model, kinds.mjOBJ_MATERIAL, int(model.geom_matid[i]))}
    result['site'] = {key: {'body': body_name[model.site_bodyid[i]], 'type': int(model.site_type[i]), 'pos': _Value(model.site_pos[i]),
                            'quat': _Value(model.site_quat[i]), 'size': _Value(model.site_size[i]), 'group': int(model.site_group[i])}
                      for i, key in enumerate(_Element_Keys(model, kinds.mjOBJ_SITE, model.site_bodyid, body_name))}
    result['mesh'] = {}
    for i in range(model.nmesh):
        vertices = model.mesh_vert[model.mesh_vertadr[i]:model.mesh_vertadr[i] + model.mesh_vertnum[i]]
        faces = model.mesh_face[model.mesh_faceadr[i]:model.mesh_faceadr[i] + model.mesh_facenum[i]]
        result['mesh'][_Name(model, kinds.mjOBJ_MESH, i)] = {
            'vertices': int(model.mesh_vertnum[i]), 'faces': int(model.mesh_facenum[i]),
            'vertex_hash': hashlib.sha256(np.ascontiguousarray(vertices).tobytes()).hexdigest(),
            'face_hash': hashlib.sha256(np.ascontiguousarray(faces).tobytes()).hexdigest()}
    result['material'] = {_Name(model, kinds.mjOBJ_MATERIAL, i): {'rgba': _Value(model.mat_rgba[i]), 'specular': _Value(model.mat_specular[i]),
                                                                  'shininess': _Value(model.mat_shininess[i]),
                                                                  'reflectance': _Value(model.mat_reflectance[i]),
                                                                  'emission': _Value(model.mat_emission[i])}
                          for i in range(model.nmat)}
    result['actuator'] = {}
    for i in range(model.nu):
        joint = int(model.actuator_trnid[i, 0]) if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT) else -1
        result['actuator'][_Name(model, kinds.mjOBJ_ACTUATOR, i)] = {
            'trntype': int(model.actuator_trntype[i]), 'joint': _Name(model, kinds.mjOBJ_JOINT, joint),
            'dyntype': int(model.actuator_dyntype[i]), 'gaintype': int(model.actuator_gaintype[i]),
            'biastype': int(model.actuator_biastype[i]), 'dynprm': _Value(model.actuator_dynprm[i]),
            'gainprm': _Value(model.actuator_gainprm[i]), 'biasprm': _Value(model.actuator_biasprm[i]),
            'ctrllimited': int(model.actuator_ctrllimited[i]), 'ctrlrange': _Value(model.actuator_ctrlrange[i]),
            'forcelimited': int(model.actuator_forcelimited[i]), 'forcerange': _Value(model.actuator_forcerange[i]),
            'gear': _Value(model.actuator_gear[i])}
    result['key'] = {_Name(model, kinds.mjOBJ_KEY, i): {'time': _Value(model.key_time[i]), 'qpos': _Value(model.key_qpos[i]),
                                                        'qvel': _Value(model.key_qvel[i]), 'ctrl': _Value(model.key_ctrl[i])}
                     for i in range(model.nkey)}
    # A contact exclusion's signature holds its two bodies' ids, 16 bits each.
    result['exclude'] = {'%s|%s' % (body_name[int(signature) >> 16], body_name[int(signature) & 0xFFFF]): {'excluded': 1}
                         for signature in model.exclude_signature}
    return result


def _Flat(fingerprint: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
    """
    Description:
        A fingerprint as one table: "section/element/field" (the sizes and the options: "section/field")
        to its value.

    Args:
        (1) fingerprint [Dict<string, Dict>]: The fingerprint.

    Returns:
        (1) parameter [Dict<string, Any>]: The values by their path.
    """

    flat = {}
    for section, elements in fingerprint.items():
        for name, value in elements.items():
            if isinstance(value, dict):
                flat.update({'%s/%s/%s' % (section, name, field_name): field_value for field_name, field_value in value.items()})
            else:
                flat['%s/%s' % (section, name)] = value
    return flat


def Compare_Fingerprints(stored: tp.Dict[str, tp.Dict[str, tp.Any]], current: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.List[str]:
    """
    Description:
        Every difference between two fingerprints, one line each, named "section/element/field" (for the
        sizes and the options "section/field"), with the stored and the current value; an element or a
        field that only one of them holds is named as such. Equal fingerprints give no line.

    Args:
        (1 - 2) stored, current [Dict<string, Dict>]: The fingerprints (Fingerprint, Read_Fingerprint).

    Returns:
        (1) parameter [List<string>]: The differences, sorted.
    """

    old, new = _Flat(stored), _Flat(current)
    lines = []
    for key in sorted(set(old) | set(new)):
        if key not in new:
            lines.append('%s: stored %s, not in the model' % (key, json.dumps(old[key])))
        elif key not in old:
            lines.append('%s: not stored, the model has %s' % (key, json.dumps(new[key])))
        elif old[key] != new[key]:
            lines.append('%s: stored %s, the model has %s' % (key, json.dumps(old[key]), json.dumps(new[key])))
    return lines


def Write_Fingerprint(path: str, fingerprint: tp.Dict[str, tp.Dict[str, tp.Any]]) -> None:
    """
    Description:
        Write a fingerprint's file (SPEC §9.10): JSON with sorted keys holding the format's version, the
        fingerprint under 'model' and, under 'environment', the versions and the platform it was taken
        on, which a comparison leaves out.

    Args:
        (1) path [string]: The file.
        (2) fingerprint [Dict<string, Dict>]: The fingerprint (Fingerprint).
    """

    document = {'format_version': CONST_FINGERPRINT_VERSION, 'model': fingerprint,
                'environment': {'mujoco': mujoco.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
                                'platform': platform.platform(), 'machine': platform.machine()}}
    with open(path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(document, indent=1, sort_keys=True) + '\n')


def Read_Fingerprint(path: str) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    """
    Description:
        Read a fingerprint's file. A missing file, text that is not JSON, another format version and a
        file without its 'model' are refused, each named.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [Dict<string, Dict>]: The fingerprint stored under 'model'.
    """

    try:
        with open(path, 'r', encoding='utf-8') as file:
            document = json.load(file)
    except (OSError, ValueError) as error:
        raise Scene_Error_Cls('The fingerprint %s cannot be read: %s. Restore it from the repository (git checkout).'
                              % (path, error)) from error
    if not isinstance(document, dict) or document.get('format_version') != CONST_FINGERPRINT_VERSION:
        raise Scene_Error_Cls('The fingerprint %s has the format version %r; this code reads version %d. Write it again with '
                              'scripts/Maintenance/build_scene.py --write.'
                              % (path, document.get('format_version') if isinstance(document, dict) else None, CONST_FINGERPRINT_VERSION))
    if not isinstance(document.get('model'), dict):
        raise Scene_Error_Cls('The fingerprint %s holds no "model". Restore it from the repository (git checkout).' % path)
    return document['model']


def Summary_Lines(model: mujoco.MjModel) -> tp.List[str]:
    """
    Description:
        The scene in a few lines, for a report: its sizes and options, the robot's mounting height, its
        joints with their ranges, the tool centre point at the vendored home position, and the masses.
        The same model gives the same lines.

    Args:
        (1) model [mujoco.MjModel]: A scene compiled by Build_Scene.

    Returns:
        (1) parameter [List<string>]: The lines.
    """

    robot = Scene_Robot(model)
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, CONST_ROBOT_PREFIX + 'home')
    if home < 0:
        raise Scene_Error_Cls('The scene holds no key "%shome"; the vendored robot brings it. Restore the models folder from the '
                              'repository (git checkout -- models).' % CONST_ROBOT_PREFIX)
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_kinematics(model, data)
    tcp, axis = data.site_xpos[robot.Tcp_Site], data.site_xmat[robot.Tcp_Site].reshape(3, 3)[:, 2]
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONST_ROBOT_PREFIX + 'base')
    cup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONST_ROBOT_PREFIX + CONST_CUP_BODY)
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONST_BALL_BODY)
    links = [index for index in range(model.nbody) if _Name(model, mujoco.mjtObj.mjOBJ_BODY, index).startswith(CONST_ROBOT_PREFIX)
             and index != cup]
    collision = int(np.sum((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
    lines = ['Scene: %d bodies, %d joints (%d positions, %d velocities), %d actuators, %d geoms (%d collide), %d meshes'
             % (model.nbody, model.njnt, model.nq, model.nv, model.nu, model.ngeom, collision, model.nmesh),
             'Options: integrator %s, time step %.4f s, gravity (%.2f, %.2f, %.2f) m/s^2'
             % (mujoco.mjtIntegrator(model.opt.integrator).name[len('mjINT_'):].lower(), model.opt.timestep, *model.opt.gravity),
             'Frame: the origin on the floor under the base axis, z up, the thrower at -y',
             'Robot: mounting face at z %.3f m; joints, range [deg]:' % data.xpos[base][2]]
    lines += ['  %-20s %8.1f %8.1f' % (name, np.degrees(lower), np.degrees(upper))
              for name, lower, upper in zip(CONST_JOINT_NAMES, robot.Lower, robot.Upper)]
    lines += ['Home [deg]: %s' % ' '.join('%.1f' % angle for angle in np.degrees(data.qpos[robot.Position_Address])),
              'Tool centre point at home (the centre of the cup\'s mouth): (%.4f, %.4f, %.4f) m, the cup\'s axis (%.3f, %.3f, %.3f)'
              % (*tcp, *axis),
              'Masses: robot %.3f kg, cup %.3f kg, ball %.3f kg' % (float(model.body_mass[links].sum()), model.body_mass[cup],
                                                                   model.body_mass[ball])]
    for name, (field_name, _) in CONST_AREAS.items():
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        lines.append('Area %-7s %s' % (field_name + ':', 'not drawn' if geom < 0 else 'drawn, visual only' if model.geom_contype[geom] == 0
                                       and model.geom_conaffinity[geom] == 0 else 'drawn, COLLIDES'))
    return lines
