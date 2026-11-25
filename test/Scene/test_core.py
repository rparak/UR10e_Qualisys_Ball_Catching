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
# File Name: ../test/Scene/test_core.py
## =========================================================================== ##

# System (Default)
import os
import sys
# Copy (Copies of the shared model and configuration)
import copy
# JSON (The fingerprint's file)
import json
# Subprocess (Subprocess management)
import subprocess
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation) [pip3 install mujoco]
import mujoco
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Scene/Core
from Scene.Core import (Build_Scene, Scene_Spec, Compile_Scene, Scene_Robot, Fingerprint, Compare_Fingerprints, Write_Fingerprint,
                        Read_Fingerprint, Summary_Lines, Scene_Error_Cls)
#   ../Robot/Core - the tool centre point
from Robot.Core import Forward_Kinematics
#   ../Geometry/Core - the areas, independently of how the scene draws them
from Geometry.Core import Spherical_Area_Cls

"""
Description:
    Initialization of constants.

    The MuJoCo scene and its fingerprint (src/Scene/Core.py). Expectations come from the CAD's
    dimensions (SPEC §4.4), the vendored model's file, the configuration and the owner's decisions of
    2026-11-20, never from the scene under test.
"""
# The height of the robot's mounting face above the floor (the stand's CAD:
# pallet 144 mm, plate 16 mm, mounting plate 20 mm).
#   Unit [m]
CONST_MOUNT_HEIGHT = 0.180
# The stand's bounding box in the scene (SPEC §4.4: 800 x 1478 x 609 mm, the robot's axis 395 mm from
# its -x side and 330 mm from its -y side).
#   Unit [Vector<float> 1x3, m]
CONST_STAND_LOWER = (-0.395, -0.330, 0.0)
CONST_STAND_UPPER = (0.405, 1.149, 0.609)
# The cup (SPEC §4.4: 153 x 190 x 153 mm): its length along its axis and its outer radius.
#   Unit [m]
CONST_CUP_LENGTH = 0.190
CONST_CUP_RADIUS = 0.0765
# The cup's volume, measured on its CAD mesh in F4 Part A (383.6 cm^3), and the density assumed for it.
#   Unit [m^3], [kg/m^3]
CONST_CUP_VOLUME = 383.6e-6
CONST_CUP_DENSITY = 1240.0
# The ball (SPEC §7.4): radius, mass, and the markers' reach (the ball's 91.9 mm over the markers, halved).
#   Unit [m], [kg], [m]
CONST_BALL_RADIUS = 0.030
CONST_BALL_MASS = 0.122
CONST_MARKER_REACH = 0.0459
# The camera rail (SPEC §4.4: 11 000 x 5 000 mm of a 100 mm profile) at its assumed height and place.
#   Unit [m]
CONST_RAIL_HEIGHT = 3.0
CONST_RAIL_LOWER = (-2.5, -8.5)
CONST_RAIL_UPPER = (2.5, 2.5)
# The vendored robot: its file, its home key and the masses of its seven bodies (ur10e.xml).
#   Unit [string], [Vector<float> 1x6, rad], [kg]
CONST_VENDORED = os.path.join('models', 'universal_robots_ur10e', 'ur10e.xml')
CONST_HOME = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
CONST_ROBOT_MASS = 4.0 + 7.778 + 12.93 + 3.87 + 1.96 + 1.96 + 0.202
# The geometry that collides: the floor, the stand's four boxes, the ball's sphere, the cup's cylinder
# and the vendored robot's nine collision geoms (one on the shoulder, two each on the upper arm, the
# forearm and wrist 2, one each on wrist 1 and wrist 3).
#   Unit [Set<string>], [Dict<string, int>]
CONST_COLLIDING = {'floor', 'stand_pallet', 'stand_plate', 'stand_mount', 'stand_controller', 'ball_collision', 'ur10e/cup_collision'}
# The vendored robot's collision shapes inside the cup's cavity, by body and position in it [m]: wrist 2's capsule
# along joint 6's axis and the flange's cylinder (read from the vendored file's lines for wrist_2_link and wrist_3_link).
#   Unit [Dict<string, Vector<float> 1x3>]
CONST_CAVITY_GEOMS = {'wrist_2_link': (0.0, 0.028, 0.12), 'wrist_3_link': (0.0, 0.097, 0.0)}
# The cup's inner shape in its CAD mesh [m] (F11 Part A, measured on CAD/Gripper/STL/Assembly_Conical_Cylinder.stl,
# the flange face at 0): the bottom's face at 0.005 reaching the radius 0.04075, the cone from there to 0.0670 at
# 0.180, the opening of 0.0625 from 0.180 to the mouth at 0.190.
#   Unit [m]
CONST_CUP_BOTTOM = 0.005
CONST_CUP_BOTTOM_RADIUS = 0.04075
CONST_CUP_LIP = 0.180
CONST_CUP_CONE_RADIUS = 0.0670
CONST_CUP_MOUTH_RADIUS = 0.0625
# The cup's shape for the ball (F11): the bottom, 24 staves along the cone, 24 boxes around the lip.
CONST_CUP_SHAPE = ({'ur10e/cup_bottom'} | {'ur10e/cup_wall_%02d' % index for index in range(24)}
                   | {'ur10e/cup_lip_%02d' % index for index in range(24)})
CONST_ROBOT_COLLIDING = {'ur10e/shoulder_link': 1, 'ur10e/upper_arm_link': 2, 'ur10e/forearm_link': 2, 'ur10e/wrist_1_link': 1,
                         'ur10e/wrist_2_link': 2, 'ur10e/wrist_3_link': 1}


def _Vertices(model, geom):
    """
    Description:
        The vertices of a mesh geom in its body's frame.
    """

    index = model.geom(geom).id
    mesh = model.geom_dataid[index]
    vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]].astype(np.float64)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[index])
    return model.geom_pos[index] + vertices @ rotation.reshape(3, 3).T


def _Box(model, geom):
    """
    Description:
        The lower and the upper corner of an axis-aligned box geom of the world.
    """

    return model.geom(geom).pos - model.geom(geom).size, model.geom(geom).pos + model.geom(geom).size


def _Models_Copy(root, folder, scene_text=None, without=()):
    """
    Description:
        A models folder that links the shipped vendored robot and meshes, holds copies of the two MJCF
        files (the scene's text replaced on request) and leaves out the files named.
    """

    shipped = os.path.join(root, 'models')
    os.makedirs(folder)
    for name in ('universal_robots_ur10e', 'meshes'):
        os.symlink(os.path.join(shipped, name), os.path.join(folder, name))
    for name in ('scene.xml', 'cup.xml'):
        if name in without:
            continue
        with open(os.path.join(shipped, name), 'r', encoding='utf-8') as file:
            text = file.read()
        with open(os.path.join(folder, name), 'w', encoding='utf-8') as file:
            file.write(scene_text(text) if scene_text is not None and name == 'scene.xml' else text)
    return str(folder)


def test_the_scene_builds_from_another_working_directory_without_a_warning_or_a_log_file(root, tmp_path):
    """
    Description:
        Built the way a script builds it, in a fresh interpreter from an unrelated working directory,
        with every Python warning turned into an error: the scene compiles (13 positions: the ball's 7
        and the robot's 6), nothing is printed on stderr, and no MUJOCO_LOG.TXT appears in the working
        directory or in the repository. The mesh paths therefore do not depend on the working directory.
    """

    code = ('import sys; sys.path.insert(0, %r)\n'
            'from Configuration.Core import Load_Configuration\n'
            'from Scene.Core import Build_Scene\n'
            'print(Build_Scene(Areas_Str=Load_Configuration(%r).Areas).nq)\n'
            % (os.path.join(root, 'src'), os.path.join(root, 'Config', 'ball_catching.yaml')))
    result = subprocess.run([sys.executable, '-W', 'error', '-c', code], cwd=str(tmp_path), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '13' and result.stderr == ''
    for folder in (str(tmp_path), root, os.path.join(root, 'models'), os.path.join(root, 'src', 'Scene')):
        assert not os.path.exists(os.path.join(folder, 'MUJOCO_LOG.TXT')), folder


def test_a_build_that_mujoco_warns_about_is_refused_and_leaves_no_log_file(root, configuration, tmp_path, monkeypatch):
    """
    Description:
        A scene file without the vendored robot's integrator makes MuJoCo warn when the robot is
        attached (it would keep the scene's Euler). The build is refused with MuJoCo's message, and
        MuJoCo's own log file is not written into the working directory. The unchanged copy builds.
    """

    monkeypatch.chdir(tmp_path)
    same = _Models_Copy(root, tmp_path / 'same')
    assert Build_Scene(Areas_Str=configuration.Areas, models=same).nq == 13
    changed = _Models_Copy(root, tmp_path / 'changed', scene_text=lambda text: text.replace('  <option integrator="implicitfast"/>\n', ''))
    with pytest.raises(Scene_Error_Cls) as error:
        Build_Scene(Areas_Str=configuration.Areas, models=changed)
    assert 'MuJoCo warned' in str(error.value) and 'integrator' in str(error.value)
    assert not os.path.exists(tmp_path / 'MUJOCO_LOG.TXT')


def test_a_missing_or_broken_file_of_the_models_folder_is_refused_naming_it(root, configuration, tmp_path):
    """
    Description:
        A models folder without the cup's file is refused naming that file and how to restore it, and
        a scene file MuJoCo cannot read is refused with MuJoCo's reason, both as the scene's own error.
    """

    with pytest.raises(Scene_Error_Cls) as error:
        Build_Scene(Areas_Str=configuration.Areas, models=_Models_Copy(root, tmp_path / 'missing', without=('cup.xml',)))
    assert 'cup.xml' in str(error.value) and 'scene.xml' not in str(error.value) and 'git checkout' in str(error.value)
    broken = _Models_Copy(root, tmp_path / 'broken', scene_text=lambda text: text.replace('mesh="stand"', 'mesh="absent"'))
    with pytest.raises(Scene_Error_Cls, match='MuJoCo could not build the scene'):
        Build_Scene(Areas_Str=configuration.Areas, models=broken)


def test_the_robot_stands_on_the_mounting_face_with_its_shoulder_at_the_robot_areas_centre(scene, configuration):
    """
    Description:
        The robot's base is on the base axis at the height of the stand's mounting face, 0.180 m, as its
        file has it (turned half a turn about z); its shoulder is then 0.181 m higher, within 0.5 mm of
        the centre of the robot area, 0.3607 m (the 2022 "0.1807 + 0.18": the shoulder above the floor).
    """

    data = mujoco.MjData(scene)
    mujoco.mj_kinematics(scene, data)
    assert np.allclose(data.body('ur10e/base').xpos, [0.0, 0.0, CONST_MOUNT_HEIGHT], atol=1e-12)
    assert np.allclose(scene.body('ur10e/base').quat, [0.0, 0.0, 0.0, -1.0], atol=1e-12)
    shoulder = data.body('ur10e/shoulder_link').xpos
    assert np.allclose(shoulder, [0.0, 0.0, CONST_MOUNT_HEIGHT + 0.181], atol=1e-12)
    assert abs(shoulder[2] - configuration.Areas.Robot.Centre[2]) <= 0.0005
    assert configuration.Areas.Robot.Centre[:2] == [0.0, 0.0]


def test_the_vendored_robot_is_attached_unchanged(scene, root):
    """
    Description:
        Every body, joint, collision geom and actuator of the vendored model, compiled alone, is in the
        scene with the same masses, inertias, offsets, ranges, damping, armature, sizes and gains: the
        robot is patched in code, and nothing of it is changed but the conaffinity of the wrist's two shapes
        inside the cup's cavity (CONST_CAVITY_GEOMS: 0, so that the ball passes them, F11).
    """

    alone = mujoco.MjModel.from_xml_path(os.path.join(root, CONST_VENDORED))
    cavity = 0
    assert alone.nbody == 8 and alone.njnt == 6 and alone.nu == 6
    for index in range(1, alone.nbody):
        mine, theirs = scene.body('ur10e/' + alone.body(index).name), alone.body(index)
        for field in ('mass', 'inertia', 'ipos', 'iquat', 'quat'):
            assert np.array_equal(getattr(mine, field), getattr(theirs, field)), (theirs.name, field)
        if theirs.name != 'base':
            assert np.array_equal(mine.pos, theirs.pos), theirs.name
        mine_geoms = [g for g in range(scene.ngeom) if scene.geom_bodyid[g] == mine.id]
        their_geoms = [g for g in range(alone.ngeom) if alone.geom_bodyid[g] == theirs.id]
        assert len(mine_geoms) == len(their_geoms)
        for g, h in zip(mine_geoms, their_geoms):
            for field in ('geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_contype', 'geom_group', 'geom_friction'):
                assert np.array_equal(getattr(scene, field)[g], getattr(alone, field)[h]), (theirs.name, field)
            in_cavity = theirs.name in CONST_CAVITY_GEOMS and bool(alone.geom_conaffinity[h]) \
                and np.allclose(alone.geom_pos[h], CONST_CAVITY_GEOMS[theirs.name], rtol=0.0, atol=1e-12)
            cavity += in_cavity
            assert scene.geom_conaffinity[g] == (0 if in_cavity else alone.geom_conaffinity[h]), (theirs.name, 'geom_conaffinity')
    assert cavity == 2
    for index in range(alone.njnt):
        mine, theirs = scene.joint('ur10e/' + alone.joint(index).name), alone.joint(index)
        for field in ('range', 'axis', 'pos', 'damping', 'armature', 'frictionloss', 'stiffness'):
            assert np.array_equal(getattr(mine, field), getattr(theirs, field)), (theirs.name, field)
    for index in range(alone.nu):
        mine, theirs = scene.actuator('ur10e/' + alone.actuator(index).name), alone.actuator(index)
        for field in ('gainprm', 'biasprm', 'ctrlrange', 'forcerange', 'gear'):
            assert np.array_equal(getattr(mine, field), getattr(theirs, field)), (theirs.name, field)


def test_the_options_are_the_vendored_robots_and_gravity_is_the_flight_models(scene, configuration, root):
    """
    Description:
        The scene integrates as the vendored model does (implicitfast, its time step), and its gravity
        is the flight model's 9.81 m/s^2 along -z (Flight.Gravity).
    """

    alone = mujoco.MjModel.from_xml_path(os.path.join(root, CONST_VENDORED))
    assert alone.opt.integrator == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    assert scene.opt.integrator == alone.opt.integrator and scene.opt.timestep == alone.opt.timestep == 0.002
    assert scene.opt.gravity.tolist() == [0.0, 0.0, -configuration.Flight.Gravity]


def test_the_home_key_is_the_vendored_one_and_nothing_touches_there(scene):
    """
    Description:
        The vendored home position comes with the robot (the ball's start added before it), and at it
        no two colliding geoms are in contact: the cup on the flange and the stand under the base are
        placed without an overlap that the contact filter does not already exclude.
    """

    assert scene.nkey == 1 and scene.key(0).name == 'ur10e/home'
    assert np.array_equal(scene.key(0).qpos[7:], CONST_HOME) and np.array_equal(scene.key(0).ctrl, CONST_HOME)
    data = mujoco.MjData(scene)
    mujoco.mj_resetDataKeyframe(scene, data, 0)
    mujoco.mj_forward(scene, data)
    assert data.ncon == 0


def test_the_cup_is_rigid_on_the_flange_with_its_mouth_on_the_axis_of_joint_6(scene):
    """
    Description:
        The cup has no joint and hangs on wrist 3; the centre of its mouth is 0.190 m from the flange
        along the flange's axis, which is joint 6's: turning joint 6 moves neither the mouth's centre
        nor the cup's axis. That symmetry is what leaves five joints for a catch pose (F4 Part A).
    """

    cup = scene.body('ur10e/cup')
    assert cup.jntnum[0] == 0 and scene.body(int(cup.parentid[0])).name == 'ur10e/wrist_3_link'
    robot = Scene_Robot(scene)
    data = mujoco.MjData(scene)
    q = np.array([0.4, -1.1, 1.3, -0.6, 0.9, 0.0])
    mouth, axes = Forward_Kinematics(scene, data, q, Robot_Str=robot)
    flange = data.site_xpos[robot.Flange_Site].copy()
    assert np.allclose(mouth - flange, CONST_CUP_LENGTH * axes[:, 2], atol=1e-12)
    assert np.allclose(axes, data.site_xmat[robot.Flange_Site].reshape(3, 3), atol=1e-12)
    turned, turned_axes = Forward_Kinematics(scene, data, q + [0, 0, 0, 0, 0, 1.0], Robot_Str=robot)
    assert np.allclose(turned, mouth, atol=1e-12) and np.allclose(turned_axes[:, 2], axes[:, 2], atol=1e-12)
    assert not np.allclose(turned_axes[:, 0], axes[:, 0], atol=1e-3)


def test_the_cups_collision_cylinder_is_the_tight_hull_of_its_mesh(scene):
    """
    Description:
        In the cup's frame its mesh runs from the flange (z = 0) to the mouth (z = 0.190 m) within the
        outer radius 76.5 mm, and the collision cylinder is exactly that hull: every vertex is inside it
        (within 0.01 mm of the mesh's single precision), and the mesh reaches its ends and its radius.
    """

    vertices = _Vertices(scene, 'ur10e/cup_visual')
    radial = np.hypot(vertices[:, 0], vertices[:, 1])
    assert vertices[:, 2].min() >= -1e-5 and vertices[:, 2].max() <= CONST_CUP_LENGTH + 1e-5 and radial.max() <= CONST_CUP_RADIUS + 1e-5
    assert vertices[:, 2].min() <= 1e-5 and vertices[:, 2].max() >= CONST_CUP_LENGTH - 1e-5 and radial.max() >= CONST_CUP_RADIUS - 1e-5
    hull = scene.geom('ur10e/cup_collision')
    assert hull.type[0] == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    assert np.allclose(hull.size[:2], [CONST_CUP_RADIUS, CONST_CUP_LENGTH / 2.0], atol=1e-12)
    assert np.allclose(hull.pos, [0.0, 0.0, CONST_CUP_LENGTH / 2.0], atol=1e-12)
    assert np.allclose(scene.site('ur10e/cup_mouth').pos, [0.0, 0.0, CONST_CUP_LENGTH], atol=1e-12)


def test_the_cups_mass_is_its_mesh_volume_at_the_assumed_density(scene):
    """
    Description:
        The cup weighs its mesh's volume, 383.6 cm^3 (measured in F4 Part A by another script), times
        the assumed 1 240 kg/m^3, 0.476 kg, with its centre of mass on its axis between its ends.
    """

    cup = scene.body('ur10e/cup')
    assert abs(cup.mass[0] - CONST_CUP_DENSITY * CONST_CUP_VOLUME) < 5e-4
    assert np.allclose(cup.ipos[:2], 0.0, atol=1e-5) and 0.05 < cup.ipos[2] < 0.15
    assert np.all(cup.inertia > 0.0)


def test_the_ball_is_a_free_sphere_of_its_radius_and_mass_centred_on_its_mesh(scene):
    """
    Description:
        The ball is a free body of 0.122 kg whose only colliding geom is the sphere of radius 30 mm,
        with a homogeneous sphere's inertia, 2/5 m r^2. Its CAD mesh's origin is not the sphere's
        centre, so the mesh is shifted: in the body's frame at least a tenth of its vertices lie on the
        30 mm sphere about the origin within a micrometre, and the markers reach 45.9 mm.
    """

    ball = scene.body('ball')
    assert ball.jntnum[0] == 1 and scene.joint('ball').type[0] == int(mujoco.mjtJoint.mjJNT_FREE)
    assert ball.mass[0] == CONST_BALL_MASS
    assert np.allclose(ball.inertia, 0.4 * CONST_BALL_MASS * CONST_BALL_RADIUS ** 2, rtol=1e-9)
    sphere = scene.geom('ball_collision')
    assert sphere.type[0] == int(mujoco.mjtGeom.mjGEOM_SPHERE) and sphere.size[0] == CONST_BALL_RADIUS and np.all(sphere.pos == 0.0)
    distance = np.linalg.norm(_Vertices(scene, 'ball_visual'), axis=1)
    assert np.mean(np.abs(distance - CONST_BALL_RADIUS) < 1e-6) > 0.1
    assert abs(distance.max() - CONST_MARKER_REACH) < 1e-4


def test_only_the_declared_geometry_collides(scene):
    """
    Description:
        The colliding geoms are exactly the floor, the stand's four boxes, the ball's sphere, the cup's
        cylinder, the cup's shape for the ball (the bottom, 24 staves, 24 lip boxes; F11) and the vendored
        robot's nine; no mesh, no rail, no camera and no area collides. Except for the floor they are in the
        collision group, 3.
    """

    colliding = [index for index in range(scene.ngeom) if scene.geom_contype[index] or scene.geom_conaffinity[index]]
    named = {scene.geom(index).name for index in colliding if scene.geom(index).name}
    assert named == CONST_COLLIDING | CONST_CUP_SHAPE and len(CONST_CUP_SHAPE) == 49
    unnamed = {}
    for index in colliding:
        if not scene.geom(index).name:
            body = scene.body(int(scene.geom_bodyid[index])).name
            unnamed[body] = unnamed.get(body, 0) + 1
    assert unnamed == CONST_ROBOT_COLLIDING
    assert all(scene.geom_type[index] != int(mujoco.mjtGeom.mjGEOM_MESH) for index in colliding)
    assert all(scene.geom_group[index] == 3 for index in colliding if scene.geom(index).name != 'floor')
    assert all(scene.geom_group[index] in (1, 2) for index in range(scene.ngeom) if index not in colliding)


def _Cup_Probe(scene):
    """
    Description:
        The scene at the vendored home position with a function that puts the ball's centre at a point of
        the cup's frame (the flange face at z = 0, z out of the mouth) and returns the model's data, and
        the geoms of the cup's shape for the ball.
    """

    data = mujoco.MjData(scene)
    mujoco.mj_resetDataKeyframe(scene, data, scene.key('ur10e/home').id)
    mujoco.mj_kinematics(scene, data)
    cup = scene.body('ur10e/cup').id
    origin, axes = data.xpos[cup].copy(), data.xmat[cup].reshape(3, 3).copy()
    address = scene.jnt_qposadr[scene.body_jntadr[scene.body('ball').id]]
    shape = [scene.geom(name).id for name in sorted(CONST_CUP_SHAPE)]

    def Put(point):
        data.qpos[address:address + 3] = origin + axes @ np.asarray(point, dtype=np.float64)
        mujoco.mj_kinematics(scene, data)
        return data

    return Put, shape


def _Free_Radius(scene, put, shape, height, angle):
    """
    Description:
        How far the ball's centre can move from the cup's axis at one height and in one direction before
        the ball touches the cup's shape [m], by bisection on the exact distance.
    """

    ball = scene.geom('ball_collision').id
    inside, outside = 0.0, 0.08
    for _ in range(30):
        middle = 0.5 * (inside + outside)
        data = put([middle * np.cos(angle), middle * np.sin(angle), height])
        if min(mujoco.mj_geomDistance(scene, data, ball, geom, 0.2, None) for geom in shape) > 0.0:
            inside = middle
        else:
            outside = middle
    return inside


def test_the_cups_shape_for_the_ball_follows_the_cad_cone_without_a_gap(scene):
    """
    Description:
        The cup's shape for the ball is the CAD's (F11): the ball rests on the bottom's face with its centre
        35 mm above the flange; along the cone its centre is free up to the CAD cone's radius less the
        ball's radius over the cosine of the cone's angle, exactly opposite a stave's centre line and at
        most 0.6 mm more between two staves (a tangent polygon of 24 sides), in 13 directions at three
        heights; at the lip it is free up to 62.5 - 30 = 32.5 mm. Opposite the centre line of each of the 24
        staves and of each of the 24 lip boxes the free radius is the exact one, so every one of them is in
        its place and no direction is open (a missing stave lets the ball sink between its neighbours).
    """

    put, shape = _Cup_Probe(scene)
    ball = scene.geom('ball_collision').id
    data = put([0.0, 0.0, CONST_CUP_BOTTOM + CONST_BALL_RADIUS])
    distances = {scene.geom(geom).name: mujoco.mj_geomDistance(scene, data, ball, geom, 0.2, None) for geom in shape}
    assert abs(distances['ur10e/cup_bottom']) < 1e-9 and min(value for name, value in distances.items() if name != 'ur10e/cup_bottom') > 0.005
    slope = (CONST_CUP_CONE_RADIUS - CONST_CUP_BOTTOM_RADIUS) / (CONST_CUP_LIP - CONST_CUP_BOTTOM)
    step = 2.0 * np.pi / 24
    angles = [0.0, 0.11 * step, 0.25 * step, 0.5 * step, 0.75 * step, step] + [index * step + 0.3 * step for index in (3, 7, 11, 14, 17, 20, 23)]
    for height in (0.060, 0.100, 0.140):
        cone = CONST_CUP_BOTTOM_RADIUS + slope * (height - CONST_CUP_BOTTOM) - CONST_BALL_RADIUS * np.sqrt(1.0 + slope * slope)
        free = np.array([_Free_Radius(scene, put, shape, height, angle) for angle in angles])
        assert abs(free[0] - cone) < 2e-6 and abs(free[5] - cone) < 2e-6, (height, free[0], cone)
        assert np.all(free - cone > -2e-6) and np.all(free - cone < 0.0006) and free[3] - cone > 0.0001, (height, free - cone)
    lip = np.array([_Free_Radius(scene, put, shape, 0.185, angle) for angle in angles])
    assert abs(lip[0] - (CONST_CUP_MOUTH_RADIUS - CONST_BALL_RADIUS)) < 2e-6 and np.all(lip >= lip[0] - 2e-6) and np.all(lip < lip[0] + 0.0006)
    cone = CONST_CUP_BOTTOM_RADIUS + slope * (0.100 - CONST_CUP_BOTTOM) - CONST_BALL_RADIUS * np.sqrt(1.0 + slope * slope)
    for index in range(24):
        assert abs(_Free_Radius(scene, put, shape, 0.100, index * step) - cone) < 2e-6, index
        assert abs(_Free_Radius(scene, put, shape, 0.185, index * step) - (CONST_CUP_MOUTH_RADIUS - CONST_BALL_RADIUS)) < 2e-6, index


def _Would_Collide(scene, first, second):
    """
    Description:
        MuJoCo's rule for two geoms: a contact is generated when the contype of one shares a bit with the
        conaffinity of the other.
    """

    return bool((scene.geom_contype[first] & scene.geom_conaffinity[second]) or (scene.geom_contype[second] & scene.geom_conaffinity[first]))


def test_the_ball_lands_on_the_cups_shape_and_passes_the_shapes_kept_for_the_clearance(scene):
    """
    Description:
        The contact bits (F11). A ball pressed 1 mm into the cup's bottom overlaps the cup's cylinder and,
        by 12 mm each, exactly the wrist's two shapes named for the cavity; MuJoCo generates its one
        contact with the cup's bottom. A ball pressed into the floor or into the forearm still meets them. By
        the rule, the cup's shape meets the ball and nothing else, and the cylinder and the two wrist
        shapes meet the floor, the stand and the other links as before, never the ball.
    """

    put, shape = _Cup_Probe(scene)
    ball = scene.geom('ball_collision').id
    name = (lambda geom: scene.geom(geom).name or scene.body(int(scene.geom_bodyid[geom])).name)
    robot = scene.body_rootid[scene.body('ur10e/cup').id]
    kept = [geom for geom in range(scene.ngeom)
            if scene.geom_contype[geom] and geom not in shape and scene.body_rootid[scene.geom_bodyid[geom]] == robot]
    data = put([0.0, 0.0, CONST_CUP_BOTTOM + CONST_BALL_RADIUS - 0.001])
    overlapping = {name(geom): mujoco.mj_geomDistance(scene, data, ball, geom, 0.2, None) for geom in kept
                   if mujoco.mj_geomDistance(scene, data, ball, geom, 0.2, None) < 0.0}
    assert set(overlapping) == {'ur10e/cup_collision', 'ur10e/wrist_2_link', 'ur10e/wrist_3_link'}
    assert abs(overlapping['ur10e/wrist_2_link'] + 0.013) < 0.0005 and abs(overlapping['ur10e/wrist_3_link'] + 0.013) < 0.0005
    mujoco.mj_collision(scene, data)
    touched = [name(int(g)) for contact in data.contact[:data.ncon] for g in contact.geom if ball in contact.geom and int(g) != ball]
    assert touched == ['ur10e/cup_bottom']
    for point, expected in (([0.0, -5.0, CONST_BALL_RADIUS - 0.001], 'floor'),):
        data.qpos[scene.jnt_qposadr[scene.body_jntadr[scene.body('ball').id]]:][:3] = point
        mujoco.mj_kinematics(scene, data)
        mujoco.mj_collision(scene, data)
        assert [name(int(g)) for contact in data.contact[:data.ncon] for g in contact.geom if ball in contact.geom and int(g) != ball] == [expected]
    world = [geom for geom in range(scene.ngeom) if scene.geom_bodyid[geom] == 0 and scene.geom_contype[geom]]
    links = [geom for geom in kept if scene.geom_conaffinity[geom]]
    clearance_only = [geom for geom in kept if not scene.geom_conaffinity[geom]]
    assert len(world) == 5 and len(links) == 7 and len(clearance_only) == 3 and len(shape) == 49
    assert all(_Would_Collide(scene, ball, geom) for geom in shape + world + links)
    assert not any(_Would_Collide(scene, ball, geom) for geom in clearance_only)
    assert not any(_Would_Collide(scene, geom, other) for geom in shape for other in world + links + clearance_only + shape)
    assert all(_Would_Collide(scene, geom, other) for geom in clearance_only for other in world + links)


def test_the_stands_boxes_fill_its_cad_bounding_box_up_to_the_mounting_face(scene):
    """
    Description:
        The stand's four collision boxes span exactly the stand's bounding box of SPEC §4.4 (800 x 1478 x
        609 mm, standing on the floor); the pallet is 800 x 1200 x 144 mm; the mounting plate, 220 x 220 x
        20 mm, has its top in the mounting face at 0.180 m, and the stand's visual mesh hangs at that
        height.
    """

    boxes = [_Box(scene, name) for name in ('stand_pallet', 'stand_plate', 'stand_mount', 'stand_controller')]
    assert np.allclose(np.min([lower for lower, _ in boxes], axis=0), CONST_STAND_LOWER, atol=1e-12)
    assert np.allclose(np.max([upper for _, upper in boxes], axis=0), CONST_STAND_UPPER, atol=1e-12)
    assert np.allclose(boxes[0][1] - boxes[0][0], [0.800, 1.200, 0.144], atol=1e-12)
    assert abs(boxes[2][1][2] - CONST_MOUNT_HEIGHT) < 1e-12 and np.allclose(boxes[2][1] - boxes[2][0], [0.220, 0.220, 0.020], atol=1e-12)
    vertices = _Vertices(scene, 'stand_visual')
    assert np.allclose(vertices.min(axis=0), CONST_STAND_LOWER, atol=1e-5)
    assert np.all(vertices.max(axis=0) <= np.array(CONST_STAND_UPPER) + 1e-5) and vertices[:, 2].max() > CONST_STAND_UPPER[2] - 0.002
    assert np.allclose(scene.site('robot_mount').pos, [0.0, 0.0, CONST_MOUNT_HEIGHT], atol=1e-12)


def test_the_rail_and_the_twelve_cameras_are_drawn_at_their_assumed_places(scene):
    """
    Description:
        The rail is four boxes of a 100 mm profile whose outline is 5 x 11 m, centred above the flight
        area's centre (0, -3) at 3.0 m; twelve cameras hang under it at twelve different places on its
        centre line, each turned towards the flight area's centre.
    """

    rail = [_Box(scene, name) for name in ('rail_long_0', 'rail_long_1', 'rail_short_0', 'rail_short_1')]
    assert np.allclose(np.min([lower for lower, _ in rail], axis=0), [*CONST_RAIL_LOWER, CONST_RAIL_HEIGHT - 0.05], atol=1e-12)
    assert np.allclose(np.max([upper for _, upper in rail], axis=0), [*CONST_RAIL_UPPER, CONST_RAIL_HEIGHT + 0.05], atol=1e-12)
    cameras = [scene.geom('camera_%02d' % number) for number in range(1, 13)]
    places = {tuple(np.round(camera.pos, 6)) for camera in cameras}
    assert len(places) == 12 and sum(scene.geom(index).name.startswith('camera_') for index in range(scene.ngeom)) == 12
    for camera in cameras:
        x, y, z = camera.pos
        assert abs(abs(x) - 2.45) < 1e-9 or abs(abs(y + 3.0) - 5.45) < 1e-9
        assert abs(z + camera.size[2] - (CONST_RAIL_HEIGHT - 0.05)) < 1e-9
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, camera.quat)
        towards = np.array([0.0 - x, -3.0 - y, 0.0]) / np.hypot(x, y + 3.0)
        assert rotation.reshape(3, 3)[:, 0] @ towards > 1.0 - 1e-9
    assert sorted(np.round(cameras[0].size * 2000.0)) == [130.0, 325.0, 432.0]


def test_the_areas_are_drawn_from_the_configuration_and_never_collide(scene, configuration):
    """
    Description:
        The four areas of the configuration are in the scene, visual only: the three cuboids as boxes of
        their centre and half their size, the robot area as a hull whose vertices all lie in the area
        (as Geometry.Core decides it, allowing the mesh's single precision) and reach its radius and its
        flat face. A changed configuration changes the scene: the areas are read, not copied.
    """

    for name, area in (('area_throw', configuration.Areas.Throw), ('area_flight', configuration.Areas.Flight),
                       ('area_catch', configuration.Areas.Catch)):
        geom = scene.geom(name)
        assert np.allclose(geom.pos, area.Centre, atol=1e-12) and np.allclose(geom.size, np.array(area.Size) / 2.0, atol=1e-12)
    for name in ('area_throw', 'area_flight', 'area_catch', 'area_robot'):
        geom = scene.geom(name)
        assert geom.contype[0] == 0 and geom.conaffinity[0] == 0 and geom.group[0] == 1 and 0.0 < geom.rgba[3] < 0.5

    changed = copy.deepcopy(configuration.Areas)
    changed.Catch.Centre, changed.Catch.Size = [0.0, -0.75, 0.85], [1.4, 0.3, 0.7]
    changed.Robot.Radius = 0.5
    model = Build_Scene(Areas_Str=changed)
    assert np.allclose(model.geom('area_catch').pos, [0.0, -0.75, 0.85], atol=1e-12)
    assert np.allclose(model.geom('area_catch').size, [0.7, 0.15, 0.35], atol=1e-12)
    assert np.allclose(model.geom('area_throw').size, scene.geom('area_throw').size, atol=1e-12)
    for built, radius in ((scene, configuration.Areas.Robot.Radius), (model, 0.5)):
        robot_area = configuration.Areas.Robot
        vertices = _Vertices(built, 'area_robot')
        allowed = Spherical_Area_Cls(robot_area.Centre, robot_area.Orientation, radius + 1e-5, robot_area.Shape)
        lowered = np.array([0.0, 0.0, 1e-5])
        assert all(allowed.Is_Point_Inside(vertex + lowered) for vertex in vertices)
        distance = np.linalg.norm(vertices - np.array(robot_area.Centre), axis=1)
        assert abs(distance.max() - radius) < 1e-5 and abs(vertices[:, 2].min() - robot_area.Centre[2]) < 1e-5


@pytest.mark.parametrize('shape, y_range, z_range', [('sphere', (-1.0, 1.0), (-1.0, 1.0)), ('hemisphere', (-1.0, 1.0), (0.0, 1.0)),
                                                     ('quarter_sphere', (-1.0, 0.0), (0.0, 1.0))])
def test_a_spherical_area_is_drawn_as_its_shape(configuration, shape, y_range, z_range):
    """
    Description:
        With the shipped orientation (a quarter turn about z: the area's x axis points at the thrower,
        -y) a sphere is drawn whole, a hemisphere as its upper half and a quarter sphere as the upper
        half's part on the thrower's side, each reaching its radius in x.
    """

    areas = copy.deepcopy(configuration.Areas)
    areas.Robot.Shape = shape
    vertices = _Vertices(Build_Scene(Areas_Str=areas), 'area_robot') - np.array(areas.Robot.Centre)
    radius = areas.Robot.Radius
    assert np.allclose([vertices[:, 1].min(), vertices[:, 1].max()], np.array(y_range) * radius, atol=1e-5)
    assert np.allclose([vertices[:, 2].min(), vertices[:, 2].max()], np.array(z_range) * radius, atol=1e-5)
    assert np.allclose([vertices[:, 0].min(), vertices[:, 0].max()], [-radius, radius], atol=1e-5)
    assert np.linalg.norm(vertices, axis=1).max() <= radius + 1e-5


def test_the_compiled_scene_matches_its_stored_fingerprint(scene, root):
    """
    Description:
        The shipped scene compiles to the model whose fingerprint is committed: no size, option, body,
        joint, geom, site, mesh, material, actuator or key differs. A change of the scene, of the
        vendored robot, of the areas or of MuJoCo shows here as named fields.
    """

    stored = Read_Fingerprint(os.path.join(root, 'models', 'scene_fingerprint.json'))
    assert Compare_Fingerprints(stored, Fingerprint(scene)) == []
    assert set(stored) == {'sizes', 'option', 'body', 'joint', 'geom', 'site', 'mesh', 'material', 'actuator', 'key', 'exclude'}
    assert stored['sizes']['nq'] == 13 and stored['sizes']['nu'] == 6 and len(stored['geom']) == scene.ngeom


def test_the_specification_compiled_as_it_is_is_the_built_scene(root, configuration, tmp_path):
    """
    Description:
        The two steps of the build: the scene's specification compiled as it
        is has the stored fingerprint, as Build_Scene's model; a light added to the specification before it is compiled
        shows as the number of lights and nothing else, so the specification is what the view's theme extends. The two
        steps refuse what Build_Scene refuses: a models folder without the cup's file (the specification); a scene
        file MuJoCo warns about when the robot is attached and a specification MuJoCo cannot compile (the compilation:
        MuJoCo 3.16.0 prints the attach warning at once and raises it to Python when the specification is compiled).
    """

    stored = Read_Fingerprint(os.path.join(root, 'models', 'scene_fingerprint.json'))
    assert Compare_Fingerprints(stored, Fingerprint(Compile_Scene(Scene_Spec(Areas_Str=configuration.Areas)))) == []
    lit = Scene_Spec(Areas_Str=configuration.Areas)
    lit.worldbody.add_light(name='added', pos=[0.0, 0.0, 3.0])
    differences = Compare_Fingerprints(stored, Fingerprint(Compile_Scene(lit)))
    assert len(differences) == 1 and differences[0].startswith('sizes/nlight: stored 0, the model has 1')
    with pytest.raises(Scene_Error_Cls) as error:
        Scene_Spec(Areas_Str=configuration.Areas, models=_Models_Copy(root, tmp_path / 'missing', without=('cup.xml',)))
    assert 'cup.xml' in str(error.value) and 'git checkout' in str(error.value)
    changed = _Models_Copy(root, tmp_path / 'changed', scene_text=lambda text: text.replace('  <option integrator="implicitfast"/>\n', ''))
    warned = Scene_Spec(Areas_Str=configuration.Areas, models=changed)
    with pytest.raises(Scene_Error_Cls) as error:
        Compile_Scene(warned, changed)
    assert 'MuJoCo warned' in str(error.value) and 'integrator' in str(error.value) and changed in str(error.value)
    broken = Scene_Spec(Areas_Str=configuration.Areas)
    broken.geom('floor').material = 'absent'
    with pytest.raises(Scene_Error_Cls, match='MuJoCo could not build the scene'):
        Compile_Scene(broken)


def test_a_changed_field_is_named_by_its_section_element_and_field(scene):
    """
    Description:
        One changed size, mass, option, joint range, gain or mesh vertex gives exactly one line each,
        naming the section, the element and the field with both values; a geom without a name is named
        by its body and its place among the body's geoms.
    """

    stored = Fingerprint(scene)
    model = copy.copy(scene)
    model.geom('stand_pallet').size[0] = 0.5
    model.body('ball').mass[0] = 0.130
    model.opt.timestep = 0.001
    model.joint('ur10e/elbow_joint').range[1] = 3.0
    model.actuator('ur10e/wrist_3').gainprm[0] = 4000.0
    forearm = [index for index in range(model.ngeom) if model.body(int(model.geom_bodyid[index])).name == 'ur10e/forearm_link']
    model.geom_friction[forearm[5], 0] = 0.5
    stand = model.mesh('stand')
    model.mesh_vert[stand.vertadr[0], 0] += 0.001
    lines = Compare_Fingerprints(stored, Fingerprint(model))
    assert [line.split(':')[0] for line in lines] == ['actuator/ur10e/wrist_3/gainprm', 'body/ball/mass', 'geom/stand_pallet/size',
                                                      'geom/ur10e/forearm_link#5/friction', 'joint/ur10e/elbow_joint/range',
                                                      'mesh/stand/vertex_hash', 'option/timestep']
    assert lines[1] == 'body/ball/mass: stored 0.122, the model has 0.13'
    assert lines[6] == 'option/timestep: stored 0.002, the model has 0.001'
    assert Compare_Fingerprints(stored, Fingerprint(scene)) == []


def test_an_element_or_a_field_that_only_one_side_holds_is_named(scene):
    """
    Description:
        An element the stored fingerprint lacks is "not stored", one the model lacks is "not in the
        model", field by field, so that an added or a removed geom cannot pass as equal.
    """

    stored, current = Fingerprint(scene), Fingerprint(scene)
    del stored['geom']['floor']
    del current['site']['robot_mount']
    del current['sizes']['nq']
    lines = Compare_Fingerprints(stored, current)
    assert 'geom/floor/type: not stored, the model has 0' in lines
    assert 'site/robot_mount/pos: stored [0.0, 0.0, 0.18], not in the model' in lines
    assert 'sizes/nq: stored 13, not in the model' in lines
    assert all(line.startswith(('geom/floor/', 'site/robot_mount/', 'sizes/nq')) for line in lines)


def test_the_fingerprint_keeps_nine_digits(scene):
    """
    Description:
        A number is held to nine significant digits: a change in the tenth does not show, a change in the
        eighth does, on both sides of the rounding.
    """

    stored = Fingerprint(scene)
    model = copy.copy(scene)
    model.body('ball').mass[0] = 0.122 * (1.0 + 1e-11)
    assert Compare_Fingerprints(stored, Fingerprint(model)) == []
    model.body('ball').mass[0] = 0.122 * (1.0 + 1e-7)
    assert [line.split(':')[0] for line in Compare_Fingerprints(stored, Fingerprint(model))] == ['body/ball/mass']


def test_the_fingerprints_file_round_trips_and_leaves_its_environment_out(scene, tmp_path):
    """
    Description:
        A written fingerprint reads back equal, twice written it has the same bytes, and its file names
        the format's version and the environment (MuJoCo 3.16.0), which a comparison never sees: a file
        with another environment compares equal.
    """

    path, again = str(tmp_path / 'fingerprint.json'), str(tmp_path / 'again.json')
    Write_Fingerprint(path, Fingerprint(scene))
    Write_Fingerprint(again, Fingerprint(scene))
    with open(path, 'rb') as first, open(again, 'rb') as second:
        assert first.read() == second.read()
    assert Read_Fingerprint(path) == Fingerprint(scene)
    with open(path, 'r', encoding='utf-8') as file:
        document = json.load(file)
    assert document['format_version'] == 1 and document['environment']['mujoco'] == '3.16.0'
    assert set(document['environment']) == {'mujoco', 'numpy', 'python', 'platform', 'machine'}
    document['environment'] = {'mujoco': '0.0.0'}
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(document, file)
    assert Compare_Fingerprints(Read_Fingerprint(path), Fingerprint(scene)) == []


def test_a_fingerprint_file_that_cannot_be_used_is_refused_naming_why(scene, tmp_path):
    """
    Description:
        A missing file, text that is not JSON, another format version and a file without its model are
        each refused with the file's name and the reason, never read as an empty fingerprint.
    """

    path = str(tmp_path / 'fingerprint.json')
    with pytest.raises(Scene_Error_Cls, match='cannot be read'):
        Read_Fingerprint(path)
    with open(path, 'w', encoding='utf-8') as file:
        file.write('not json')
    with pytest.raises(Scene_Error_Cls, match='cannot be read'):
        Read_Fingerprint(path)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump({'format_version': 2, 'model': Fingerprint(scene)}, file)
    with pytest.raises(Scene_Error_Cls, match='format version 2; this code reads version 1'):
        Read_Fingerprint(path)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump({'format_version': 1, 'environment': {}}, file)
    with pytest.raises(Scene_Error_Cls, match='holds no "model"'):
        Read_Fingerprint(path)


def test_the_summary_states_the_scene_as_its_sources_have_it(scene):
    """
    Description:
        The summary's lines give the mounting height 0.180 m, the vendored ranges and home position, the
        masses (the robot's seven bodies 32.700 kg, the cup 0.476 kg, the ball 0.122 kg), the tool centre
        point at home 0.190 m below the flange, and the four areas as drawn and visual only.
    """

    lines = Summary_Lines(scene)
    assert 'Options: integrator implicitfast, time step 0.0020 s, gravity (0.00, 0.00, -9.81) m/s^2' in lines
    assert 'Robot: mounting face at z 0.180 m; joints, range [deg]:' in lines
    assert '  elbow_joint            -180.0    180.0' in lines and '  wrist_3_joint          -360.0    360.0' in lines
    assert 'Home [deg]: -90.0 -90.0 90.0 -90.0 -90.0 0.0' in lines
    assert 'Masses: robot %.3f kg, cup 0.476 kg, ball 0.122 kg' % CONST_ROBOT_MASS in lines
    tcp = [line for line in lines if line.startswith('Tool centre point at home')]
    assert len(tcp) == 1 and tcp[0].endswith('(-0.1740, 0.6910, 0.6840) m, the cup\'s axis (0.000, 0.000, -1.000)')
    assert [line for line in lines if line.startswith('Area ')] == ['Area Throw:  drawn, visual only', 'Area Flight: drawn, visual only',
                                                                    'Area Catch:  drawn, visual only', 'Area Robot:  drawn, visual only']
    assert lines == Summary_Lines(scene)
