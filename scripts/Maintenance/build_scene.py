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
# File Name: ../scripts/Maintenance/build_scene.py
## =========================================================================== ##

# System (Default)
import sys
# OS (Operating system interfaces)
import os
# The repository root, built from this file's location, so the script runs from any working
# directory.
#   Unit [string]
CONST_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
#   Add access if it is not in the system path.
if os.path.join(CONST_REPO_ROOT, 'src') not in sys.path:
    sys.path.append(os.path.join(CONST_REPO_ROOT, 'src'))
# Argparse (Command-line parsing)
import argparse
# Hashlib (Secure hashes)
import hashlib
# JSON (The build record)
import json
# Math (The cameras' headings)
import math
# Subprocess (Blender, for the stand's decimation only)
import subprocess
# Tempfile (Blender's script and its output)
import tempfile
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the areas drawn in the scene
from Configuration.Core import Load_Configuration, Configuration_Error_Cls
#   ../Scene/Core - the scene's builder and its fingerprint: the check goes through the project's own code
from Scene.Core import (Build_Scene, Fingerprint, Compare_Fingerprints, Read_Fingerprint, Write_Fingerprint, Scene_Error_Cls,
                        CONST_MODELS_DIR, CONST_SCENE_FILE, CONST_CUP_FILE, CONST_FINGERPRINT_FILE, CONST_MOUNT_SITE,
                        CONST_BALL_BODY, CONST_CUP_BODY, CONST_CUP_MOUTH_SITE, CONST_GROUP_VISUAL, CONST_GROUP_COLLISION,
                        CONST_BIT_WORLD, CONST_BIT_BALL, CONST_CUP_BOTTOM_GEOM, CONST_CUP_WALL_GEOM, CONST_CUP_LIP_GEOM)

"""
Description:
    Initialization of constants.

    Builds models/ from the CAD (SPEC §2.4, §4.4, §9.10): the scene's MJCF, the cup's MJCF, the meshes,
    the SHA-256 list of the vendored UR10e, the build record and the fingerprint of the compiled scene.
    The scene is built once and then not regenerated: without --write the script only CHECKS that
    models/ is what it would write. The vendored folder models/universal_robots_ur10e/ is read and
    hashed, never written. The lengths of the CAD are in millimetres and are converted to metres here,
    once (SPEC §9.1).

    Frame of the scene: the origin is on the floor under the robot's base
    axis, z up, the thrower at -y, the controller box behind the robot at +y. The stand's CAD frame has
    its origin in the centre of the mounting face, z up, the controller box at +y: it is the scene's
    frame raised by the height of the mounting face.
"""
# The shipped configuration, whose areas the fingerprinted scene shows (SPEC §8.1).
#   Unit [string]
CONST_CONFIG_FILE = os.path.join(CONST_REPO_ROOT, 'Config', 'ball_catching.yaml')
# The CAD sources (SPEC §4.4), read only.
#   Unit [string]
CONST_CAD_STAND = os.path.join(CONST_REPO_ROOT, 'CAD', 'Stand', 'STL', 'Assembly_Robot_Stand.stl')
CONST_CAD_CUP = os.path.join(CONST_REPO_ROOT, 'CAD', 'Gripper', 'STL', 'Assembly_Conical_Cylinder.stl')
CONST_CAD_BALL = os.path.join(CONST_REPO_ROOT, 'CAD', 'Ball', 'STL', 'Orange_Ball_R30_Spherical_Markers.stl')
# The files written into the models folder, relative to it.
#   Unit [string]
CONST_MESH_STAND = os.path.join('meshes', 'stand.stl')
CONST_MESH_CUP = os.path.join('meshes', 'cup.stl')
CONST_MESH_BALL = os.path.join('meshes', 'ball.stl')
CONST_SUMS_FILE = 'SHA256SUMS'
CONST_RECORD_FILE = 'build.json'
# The vendored robot: MuJoCo Menagerie's folder universal_robots_ur10e at the pinned commit (SPEC §5),
# and the id of that folder's git tree there (git rev-parse fb36886b:universal_robots_ur10e, read
# 2026-11-20). A tree id is a hash of the folder's content, so the folder in models/ has this id exactly
# when no file of it was edited, added or removed.
#   Unit [string]
CONST_VENDORED = 'universal_robots_ur10e'
CONST_MENAGERIE_COMMIT = 'fb36886b3023db08c198264e1e9956ab202c3e3b'
CONST_MENAGERIE_TREE = '466f0ad824356ae94da5f514e55d630ca67071c7'
# The height of the robot's mounting face above the floor: the stand's CAD has its origin on the mounting
# face and its lowest point, the pallet's underside, at z = -180.0 mm (pallet 144 mm, plate 16 mm,
# mounting plate 20 mm; measured on CAD_STAND, 2026-11-20). The 2022 hemisphere's centre, 0.1807 + 0.18 m,
# is the shoulder axis at this height.
#   Unit [mm]
CONST_MOUNT_HEIGHT = 180.0
# The stand's collision geometry: four boxes, each the bounding box of one part of CAD_STAND in the
# stand's frame (lower corner, upper corner), read from the mesh's horizontal faces on 2026-11-20: the
# pallet (800 x 1200 x 144), the plate on it, the mounting plate under the robot (220 x 220 x 20, SPEC
# §4.4) and the controller box with its brackets. Every vertex of the CAD mesh lies in their union
# (test/scripts/Maintenance/test_build_scene.py).
#   Unit [Dict<string, Tuple<Vector<float> 1x3, Vector<float> 1x3>>, mm]
CONST_STAND_BOXES = {'stand_pallet': ((-395.0, -330.0, -180.0), (405.0, 870.0, -36.0)),
                     'stand_plate': ((-175.0, -330.0, -36.0), (175.0, 980.0, -20.0)),
                     'stand_mount': ((-110.0, -110.0, -20.0), (110.0, 110.0, 0.0)),
                     'stand_controller': ((-231.0, 850.0, -20.0), (235.0, 1149.0, 429.0))}
# The stand's visual mesh: CAD_STAND has 608 435 triangles and MuJoCo 3.16.0 reads an STL of at most
# 200 000 faces (measured 2026-11-20: "number of faces should be between 1 and 200000"), so it is
# decimated once with Blender (a tool used once, not a dependency): vertices
# closer than the merge distance are merged (600 741 faces remain), then Blender's collapse decimation
# runs with the ratio target / remaining = 20 000 / 600 741 = 0.0333. The version is the one the shipped
# mesh was made with; another version is refused, because its result is not known to be the same.
#   Unit [string], [int], [mm], [string]
CONST_BLENDER = '/Applications/Blender.app/Contents/MacOS/blender'
CONST_BLENDER_VERSION = '4.2.8 LTS'
CONST_STAND_FACES = 20000
CONST_STAND_MERGE = 1e-4
# Run inside Blender: import, merge, decimate to the target number of faces, export.
#   Unit [string]
CONST_BLENDER_SCRIPT = '''
import bpy, sys
source, target, faces, merge = sys.argv[-4], sys.argv[-3], int(sys.argv[-2]), float(sys.argv[-1])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.stl_import(filepath=source)
mesh = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = mesh
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=merge)
bpy.ops.object.mode_set(mode='OBJECT')
merged = len(mesh.data.polygons)
modifier = mesh.modifiers.new('decimate', 'DECIMATE')
modifier.ratio = faces / merged
bpy.ops.object.modifier_apply(modifier='decimate')
bpy.ops.wm.stl_export(filepath=target, export_selected_objects=True)
print('DECIMATED %d %d' % (merged, len(mesh.data.polygons)))
'''
# The camera rail (SPEC §4.4, CAD/Mocap/STEP/Holder.stp): a rectangular frame of 11 000 x 5 000 mm
# outside, of a 100 mm profile. Drawn as four boxes (no STEP reader, the rail
# is visual only). ASSUMED (SPEC §14.1; KNOWN_ISSUES): its
# height, 3.0 m to the profile's centre, and its place, centred above the centre of the flight area (0, -3)
# with its long side along y, so that it spans the throw area, the flight and the robot.
#   Unit [mm], [mm], [mm], [m], [Vector<float> 1x2, m]
CONST_RAIL_LENGTH = 11000.0
CONST_RAIL_WIDTH = 5000.0
CONST_RAIL_PROFILE = 100.0
CONST_RAIL_HEIGHT = 3.0
CONST_RAIL_CENTRE = (0.0, -3.0)
# The 12 camera mounts along the rail's centre line, in the rail's own frame (x along its length, y along
# its width, the origin in its centre): the four corners, two on each long side at x = +-2.725 m and two
# on each short side at y = +-1.225 m, where Holder.stp carries features (SPEC §4.4; ASSUMED to be the
# mounts, SPEC §14.1).
#   Unit [List<Vector<float> 1x2>, mm]
CONST_CAMERA_MOUNTS = [(sx * 5450.0, sy * 2450.0) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)] + \
                      [(sx * 2725.0, sy * 2450.0) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)] + \
                      [(sx * 5450.0, sy * 1225.0) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
# A camera: the bounding box of the camera body with its mount (CAD/Mocap/STEP/Camera.stp, SPEC §4.4),
# drawn as one box (F4 Q10), visual only. ASSUMED (KNOWN_ISSUES):
# it hangs under the rail with its 432 mm side
# vertical and its 325 mm side towards the point it looks at, the centre of the flight area at the
# height of the flight cuboid's centre (SPEC §4.4: "the cameras pointing at the flight area's centre").
#   Unit [Vector<float> 1x3, mm], [Vector<float> 1x3, m]
CONST_CAMERA_BOX = (325.0, 130.0, 432.0)
CONST_CAMERA_TARGET = (0.0, -3.0, 2.25)
# The ball (SPEC §7.4): a sphere of radius 30 mm (CAD) with 6 markers; mass 0.122 kg, ASSUMED (not
# weighed); its inertia that of a homogeneous sphere of that mass and radius, ASSUMED
# (KNOWN_ISSUES). Only the sphere
# collides: the markers are visual (KNOWN_ISSUES). It starts at the centre of the shipped throw cuboid
# (SPEC §8.1); whoever throws it sets its state.
#   Unit [mm], [kg], [Vector<float> 1x3, m]
CONST_BALL_RADIUS = 30.0
CONST_BALL_MASS = 0.122
CONST_BALL_START = (-0.125, -5.25, 1.275)
# The sphere's centre in CAD_BALL is fitted to the vertices near the sphere of CONST_BALL_RADIUS about the
# last estimate, in rounds of a narrowing band: the first one takes in the whole sphere when the start is
# off by a marker's size, the last one only the sphere's own vertices.
#   Unit [Tuple<float>, mm]
CONST_BALL_FIT_BANDS = (10.0, 3.0, 1.0, 0.3, 0.05)
# The cup (CAD_CUP, SPEC §4.4): its axis is the CAD's y axis, the flange face at y = 0 and the mouth at
# y = 190 mm, the outer radius 76.5 mm. In the scene its frame is the flange's site, z along the axis.
# Its collision geometry for the clearance of the robot's motion is the cylinder around it, which the ball
# does not touch; the ball touches the cup's own shape below (F11). Its mass is the mesh's volume times the density of PLA
# printed solid, 1 240 kg/m^3, 0.476 kg: ASSUMED (the material and the infill are not recorded; accepted as
# assumed; KNOWN_ISSUES). It matters for F11 (the cup's inertia on the flange during the
# catch); a weighed value replaces it when one exists, as a decided change of this script.
#   Unit [mm], [mm], [kg/m^3]
CONST_CUP_LENGTH = 190.0
CONST_CUP_RADIUS = 76.5
CONST_CUP_DENSITY = 1240.0
# The cup's shape for the contact with the ball (F11), read from CAD_CUP (Level_Radii,
# pinned by the script's tests): a flat bottom whose face lies 5 mm above the flange and reaches the radius
# 40.75 mm (the plate under it 45 mm); the inner cone from there to the radius 67.0 mm at 180 mm; a lip ring
# from 180 mm to the mouth that narrows the opening to 62.5 mm. The wall is 4.8 mm thick, so the cone's outside
# reaches 71.6 mm under the lip, as the CAD's. The cone is CONST_CUP_STAVES boxes whose inner faces touch
# the cone along their centre lines (16, 24 and 32 gave the same catches within the sampling noise, F11 Part A),
# the lip as many boxes whose inner faces touch the opening, the bottom one cylinder.
#   Unit [mm], [int]
CONST_CUP_BOTTOM = 5.0
CONST_CUP_BOTTOM_RADIUS = 40.75
CONST_CUP_PLATE_RADIUS = 45.0
CONST_CUP_LIP = 180.0
CONST_CUP_CONE_RADIUS = 67.0
CONST_CUP_MOUTH_RADIUS = 62.5
CONST_CUP_WALL = 4.8
CONST_CUP_STAVES = 24
# The colours of the scene's own parts (red, green, blue, alpha), chosen for the look only.
#   Unit [Dict<string, string>]
CONST_COLOURS = {'floor': '0.55 0.57 0.6 1', 'stand': '0.62 0.62 0.64 1', 'collision': '0.9 0.3 0.3 0.4',
                 'rail': '0.75 0.76 0.78 1', 'camera': '0.15 0.15 0.17 1', 'ball': '1 0.45 0.05 1', 'cup': '0.85 0.85 0.87 1'}


def _Log(level: str, message: str) -> None:
    """
    Description:
        Prints one [INFO] / [ERROR] line on stderr, so stdout holds only the check's report.

    Args:
        (1 - 2) level, message [string]: The level and the sentence.
    """

    print('[%s] %s' % (level, message), file=sys.stderr, flush=True)


def _Number(value: float) -> str:
    """
    Description:
        A length or an angle as the MJCF files hold it: at most nine decimals, no trailing zeros, never
        "-0".

    Args:
        (1) value [float]: The number.

    Returns:
        (1) parameter [string]: Its text.
    """

    text = ('%.9f' % value).rstrip('0').rstrip('.')
    return '0' if text in ('-0', '') else text


def _Numbers(values: tp.Iterable[float]) -> str:
    """
    Description:
        Several numbers separated by spaces, as an MJCF attribute holds a vector.

    Args:
        (1) values [Vector<float> 1xN]: The numbers.

    Returns:
        (1) parameter [string]: Their text.
    """

    return ' '.join(_Number(float(value)) for value in values)


def Sha256_File(path: str) -> str:
    """
    Description:
        The SHA-256 hex digest of a file.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [string]: The digest.
    """

    with open(path, 'rb') as file:
        return hashlib.sha256(file.read()).hexdigest()


def Read_Stl(path: str) -> np.ndarray:
    """
    Description:
        The triangles of a binary STL file (an 80-byte header, the number of triangles, then per
        triangle a normal, three vertices and two attribute bytes, little-endian). A file whose size does
        not match its number of triangles is refused.

    Args:
        (1) path [string]: The STL file.

    Returns:
        (1) parameter [Array<float> Nx3x3]: The vertices of each triangle, in the file's unit.
    """

    with open(path, 'rb') as file:
        data = file.read()
    count = int(np.frombuffer(data[80:84], dtype='<u4')[0]) if len(data) >= 84 else -1
    if count < 1 or len(data) != 84 + 50 * count:
        raise ValueError('%s is not a binary STL file: %d bytes do not hold the %d triangles its header names.'
                         % (path, len(data), count))
    record = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    return np.frombuffer(data, dtype=record, offset=84, count=count)['vertices'].astype(np.float64)


def Box_Excess(points: np.ndarray, boxes: tp.Iterable[tp.Tuple[tp.Sequence[float], tp.Sequence[float]]]) -> np.ndarray:
    """
    Description:
        How far each point lies outside the union of axis-aligned boxes: its distance to the nearest box,
        0 inside one.

    Args:
        (1) points [Matrix<float> Nx3]: The points.
        (2) boxes [List<Tuple<Vector<float> 1x3, Vector<float> 1x3>>]: The lower and the upper corner of
                                                                      each box, in the points' unit.

    Returns:
        (1) parameter [Vector<float> 1xN]: The distances, in the points' unit.
    """

    excess = np.full(len(points), np.inf)
    for lower, upper in boxes:
        outside = np.maximum(np.maximum(np.asarray(lower, dtype=np.float64) - points, points - np.asarray(upper, dtype=np.float64)), 0.0)
        excess = np.minimum(excess, np.linalg.norm(outside, axis=1))
    return excess


def Ball_Centre(triangles: np.ndarray) -> np.ndarray:
    """
    Description:
        The centre of the ball's sphere in its CAD mesh, whose origin is not the centre: the
        least-squares sphere through the vertices near the sphere of CONST_BALL_RADIUS, found from the
        mesh's bounding-box centre in the rounds of CONST_BALL_FIT_BANDS. A fit whose radius is not the
        ball's is refused.

    Args:
        (1) triangles [Array<float> Nx3x3]: The mesh [mm].

    Returns:
        (1) parameter [Vector<float> 1x3]: The centre [mm].
    """

    points = np.unique(triangles.reshape(-1, 3), axis=0)
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    for band in CONST_BALL_FIT_BANDS:
        on_sphere = points[np.abs(np.linalg.norm(points - centre, axis=1) - CONST_BALL_RADIUS) < band]
        # |p - c|^2 = r^2 is linear in (c, r^2 - |c|^2): 2 p . c + (r^2 - |c|^2) = |p|^2.
        solution = np.linalg.lstsq(np.hstack([2.0 * on_sphere, np.ones((len(on_sphere), 1))]), (on_sphere ** 2).sum(axis=1),
                                   rcond=None)[0]
        centre = solution[:3]
    radius = float(np.sqrt(solution[3] + centre @ centre))
    if abs(radius - CONST_BALL_RADIUS) > CONST_BALL_FIT_BANDS[-1]:
        raise ValueError('The sphere fitted to the ball\'s mesh has the radius %.3f mm, not %.1f mm.' % (radius, CONST_BALL_RADIUS))
    return centre


def Mass_Properties(triangles: np.ndarray) -> tp.Tuple[float, np.ndarray, np.ndarray]:
    """
    Description:
        The volume, the centroid and the inertia per unit density of the solid a closed triangle mesh
        encloses, summed over the tetrahedra each triangle forms with the origin (the sign of a
        tetrahedron's volume follows the triangle's winding, so the parts outside cancel).

        Reference:
            The second moments of a tetrahedron (0, a, b, c) of volume V: the integral of x x^T is
            V / 20 * (a a^T + b b^T + c c^T + s s^T) with s = a + b + c.

    Args:
        (1) triangles [Array<float> Nx3x3]: The mesh, wound outwards [m].

    Returns:
        (1) parameter [float]: The volume [m^3].
        (2) parameter [Vector<float> 1x3]: The centroid [m].
        (3) parameter [Matrix<float> 3x3]: The inertia about the centroid for a density of 1 [m^5].
    """

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    volumes = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0
    volume = float(volumes.sum())
    s = a + b + c
    centroid = (volumes[:, None] * s).sum(axis=0) / (4.0 * volume)
    second = sum(np.einsum('i,ij,ik->jk', volumes, x, x) for x in (a, b, c, s)) / 20.0
    second = second - volume * np.outer(centroid, centroid)
    return volume, centroid, np.trace(second) * np.eye(3) - second


def Git_Tree_Id(folder: str) -> str:
    """
    Description:
        The id git gives the tree of a folder: the SHA-1 of its entries (mode, name, id), each file's id
        the SHA-1 of "blob <size>" and its bytes, the entries in git's order (a folder sorts as its name
        followed by "/"). Every file is taken as a regular one (mode 100644), as all files of the
        vendored folder are. macOS's .DS_Store, which .gitignore keeps out of the repository, is passed
        over.

    Args:
        (1) folder [string]: The folder.

    Returns:
        (1) parameter [string]: The tree's id, 40 hex digits.
    """

    entries = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if name == '.DS_Store':
            continue
        if os.path.isdir(path):
            entries.append((name + '/', b'40000', name, Git_Tree_Id(path)))
        else:
            with open(path, 'rb') as file:
                data = file.read()
            entries.append((name, b'100644', name, hashlib.sha1(b'blob %d\x00' % len(data) + data).hexdigest()))
    body = b''.join(mode + b' ' + name.encode('utf-8') + b'\x00' + bytes.fromhex(sha)
                    for _, mode, name, sha in sorted(entries, key=lambda entry: entry[0].encode('utf-8')))
    return hashlib.sha1(b'tree %d\x00' % len(body) + body).hexdigest()


def Vendored_Sums(models: str) -> str:
    """
    Description:
        The SHA-256 list of the vendored folder, one line per file in the form "shasum -a 256 -c" reads
        (digest, two spaces, the path relative to the models folder), sorted by path. install.sh checks
        the folder against it (SPEC §11).

    Args:
        (1) models [string]: The models folder.

    Returns:
        (1) parameter [string]: The list's text.
    """

    paths = []
    for directory, _, files in os.walk(os.path.join(models, CONST_VENDORED)):
        paths += [os.path.relpath(os.path.join(directory, name), models) for name in files if name != '.DS_Store']
    return ''.join('%s  %s\n' % (Sha256_File(os.path.join(models, path)), path.replace(os.sep, '/')) for path in sorted(paths))


def _Box(name: str, lower: tp.Sequence[float], upper: tp.Sequence[float], attributes: str) -> str:
    """
    Description:
        One box geom of the scene from its two corners.

    Args:
        (1) name [string]: The geom's name.
        (2 - 3) lower, upper [Vector<float> 1x3]: The corners [m].
        (4) attributes [string]: Its further attributes.

    Returns:
        (1) parameter [string]: The MJCF line.
    """

    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    return '    <geom name="%s" type="box" pos="%s" size="%s" %s/>' % (name, _Numbers(0.5 * (lower + upper)),
                                                                     _Numbers(0.5 * (upper - lower)), attributes)


def Scene_Xml(ball_centre: np.ndarray) -> str:
    """
    Description:
        The scene's MJCF (models/scene.xml): the floor, the stand, the site the robot is mounted on, the
        camera rail, the 12 cameras and the ball. The robot and the cup are attached in code
        (Scene.Core.Build_Scene); the areas are drawn from the configuration there. The integrator is the
        vendored robot's (implicitfast), so that attaching it changes nothing and warns of nothing.

    Args:
        (1) ball_centre [Vector<float> 1x3]: The sphere's centre in the ball's mesh [mm] (Ball_Centre).

    Returns:
        (1) parameter [string]: The file's text.
    """

    visual = 'contype="0" conaffinity="0" group="%d"' % CONST_GROUP_VISUAL
    mount = np.array([0.0, 0.0, CONST_MOUNT_HEIGHT]) / 1000.0
    lines = ['<!--',
             '  The scene of the ball-catching simulation (SPEC §2.4, §9.10).',
             '  Written by scripts/Maintenance/build_scene.py from the CAD; never edited by hand.',
             '  Frame: the origin on the floor under the robot\'s base axis, z up, the thrower at -y. Lengths in m.',
             '  The robot (models/universal_robots_ur10e/ur10e.xml) and the cup (models/cup.xml) are attached in',
             '  code at the site "%s"; the areas are drawn from the configuration (src/Scene/Core.py).' % CONST_MOUNT_SITE,
             '-->',
             '<mujoco model="ball_catching_scene">',
             '  <compiler angle="radian" meshdir="meshes"/>',
             '  <option integrator="implicitfast"/>',
             '  <asset>',
             '    <mesh name="stand" file="stand.stl" scale="0.001 0.001 0.001"/>',
             '    <mesh name="ball" file="ball.stl" scale="0.001 0.001 0.001"/>',
             '  </asset>',
             '  <worldbody>',
             '    <geom name="floor" type="plane" size="0 0 0.05" rgba="%s"/>' % CONST_COLOURS['floor'],
             '    <site name="%s" pos="%s" size="0.01"/>' % (CONST_MOUNT_SITE, _Numbers(mount)),
             '    <geom name="stand_visual" type="mesh" mesh="stand" pos="%s" rgba="%s" %s/>' % (_Numbers(mount), CONST_COLOURS['stand'],
                                                                                              visual)]
    for name, (lower, upper) in CONST_STAND_BOXES.items():
        lines.append(_Box(name, np.array(lower) / 1000.0 + mount, np.array(upper) / 1000.0 + mount,
                          'rgba="%s" group="%d"' % (CONST_COLOURS['collision'], CONST_GROUP_COLLISION)))

    # The rail: two long profiles along y and two short ones between them.
    half_length, half_width, half_profile = CONST_RAIL_LENGTH / 2000.0, CONST_RAIL_WIDTH / 2000.0, CONST_RAIL_PROFILE / 2000.0
    c_x, c_y = CONST_RAIL_CENTRE
    for index, side in enumerate((-1.0, 1.0)):
        x = c_x + side * (half_width - half_profile)
        lines.append(_Box('rail_long_%d' % index, [x - half_profile, c_y - half_length, CONST_RAIL_HEIGHT - half_profile],
                          [x + half_profile, c_y + half_length, CONST_RAIL_HEIGHT + half_profile],
                          'rgba="%s" %s' % (CONST_COLOURS['rail'], visual)))
    for index, side in enumerate((-1.0, 1.0)):
        y = c_y + side * (half_length - half_profile)
        lines.append(_Box('rail_short_%d' % index, [c_x - half_width + 2.0 * half_profile, y - half_profile, CONST_RAIL_HEIGHT - half_profile],
                          [c_x + half_width - 2.0 * half_profile, y + half_profile, CONST_RAIL_HEIGHT + half_profile],
                          'rgba="%s" %s' % (CONST_COLOURS['rail'], visual)))

    # The cameras: each hangs under the rail's profile at its mount and is turned about z towards the target.
    half_box = np.array(CONST_CAMERA_BOX) / 2000.0
    for index, (along, across) in enumerate(CONST_CAMERA_MOUNTS):
        position = np.array([c_x + across / 1000.0, c_y + along / 1000.0, CONST_RAIL_HEIGHT - half_profile - half_box[2]])
        heading = math.atan2(CONST_CAMERA_TARGET[1] - position[1], CONST_CAMERA_TARGET[0] - position[0])
        lines.append('    <geom name="camera_%02d" type="box" pos="%s" quat="%s" size="%s" rgba="%s" %s/>'
                     % (index + 1, _Numbers(position), _Numbers([math.cos(heading / 2.0), 0.0, 0.0, math.sin(heading / 2.0)]),
                        _Numbers(half_box), CONST_COLOURS['camera'], visual))

    radius = CONST_BALL_RADIUS / 1000.0
    inertia = 0.4 * CONST_BALL_MASS * radius ** 2
    lines += ['    <body name="%s" pos="%s">' % (CONST_BALL_BODY, _Numbers(CONST_BALL_START)),
              '      <freejoint name="%s"/>' % CONST_BALL_BODY,
              '      <inertial pos="0 0 0" mass="%s" diaginertia="%s"/>' % (_Number(CONST_BALL_MASS), _Numbers([inertia] * 3)),
              '      <geom name="ball_visual" type="mesh" mesh="ball" pos="%s" rgba="%s" %s/>' % (_Numbers(-ball_centre / 1000.0),
                                                                                                CONST_COLOURS['ball'], visual),
              '      <geom name="ball_collision" type="sphere" size="%s" rgba="%s" contype="%d" conaffinity="%d" group="%d"/>'
              % (_Number(radius), CONST_COLOURS['collision'], CONST_BIT_WORLD | CONST_BIT_BALL, CONST_BIT_BALL, CONST_GROUP_COLLISION),
              '    </body>',
              '  </worldbody>',
              '</mujoco>']
    return '\n'.join(lines) + '\n'


def Level_Radii(triangles: np.ndarray, level: float, up: bool) -> tp.Tuple[float, float]:
    """
    Description:
        The smallest and the largest distance from the cup's axis (the CAD's y axis) of the faces that lie
        across the axis at one height and face one way: the bottom's face, the lip's underside, the
        mouth's face. The cup's contact shape is read from them.

    Args:
        (1) triangles [Array<float> Nx3x3]: The cup's CAD mesh [mm], its axis along y.
        (2) level [float]: The height along the axis [mm]; a face within 0.01 mm of it is taken.
        (3) up [bool]: The faces looking out of the mouth (True) or towards the flange (False).

    Returns:
        (1) parameter [Tuple<float, float>]: The smallest and the largest radius of those faces' vertices [mm].
    """

    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(normal, axis=1)
    facing = normal[:, 1] / np.maximum(length, 1e-12) * (1.0 if up else -1.0)
    chosen = (length > 0.0) & (facing > 0.999) & (np.abs(triangles[:, :, 1] - level).max(axis=1) < 0.01)
    if not chosen.any():
        raise ValueError('The cup\'s mesh has no face across its axis at %.2f mm looking %s.' % (level, 'up' if up else 'down'))
    radius = np.hypot(triangles[chosen][:, :, 0], triangles[chosen][:, :, 2])
    return float(radius.min()), float(radius.max())


def Cup_Contact_Geoms() -> tp.List[str]:
    """
    Description:
        The cup's contact geometry for the ball (F11): the bottom's cylinder, CONST_CUP_STAVES boxes along the
        inner cone and as many around the lip, in the cup's frame (the flange face at z = 0, z out of the
        mouth). They take contacts from the ball only (contype 0, conaffinity CONST_BIT_BALL): the clearance
        of the robot's motion keeps the cylinder around the cup.

    Returns:
        (1) parameter [List<string>]: The MJCF lines.
    """

    attributes = 'rgba="%s" contype="0" conaffinity="%d" group="%d"' % (CONST_COLOURS['collision'], CONST_BIT_BALL, CONST_GROUP_COLLISION)
    bottom, lip, mouth = CONST_CUP_BOTTOM / 1000.0, CONST_CUP_LIP / 1000.0, CONST_CUP_LENGTH / 1000.0
    r_bottom, r_cone, r_mouth, r_outer = (CONST_CUP_BOTTOM_RADIUS / 1000.0, CONST_CUP_CONE_RADIUS / 1000.0, CONST_CUP_MOUTH_RADIUS / 1000.0,
                                          CONST_CUP_RADIUS / 1000.0)
    wall = CONST_CUP_WALL / 1000.0
    lines = ['      <geom name="%s" type="cylinder" pos="%s" size="%s" %s/>'
             % (CONST_CUP_BOTTOM_GEOM, _Numbers([0.0, 0.0, bottom / 2.0]), _Numbers([CONST_CUP_PLATE_RADIUS / 1000.0, bottom / 2.0]), attributes)]
    # 1\ The cone's staves. In the plane of the axis and a radius: g along the cone from the bottom to the lip, o
    #    its outward normal; a stave's frame is (the tangent, g, o), its inner face on the cone.
    slant = math.hypot(r_cone - r_bottom, lip - bottom)
    g_r, g_z = (r_cone - r_bottom) / slant, (lip - bottom) / slant
    o_r, o_z = g_z, -g_r
    centre_r = 0.5 * (r_bottom + r_cone) + 0.5 * wall * o_r
    centre_z = 0.5 * (bottom + lip) + 0.5 * wall * o_z
    half_width = (r_cone + wall) * math.tan(math.pi / CONST_CUP_STAVES)
    for index in range(CONST_CUP_STAVES):
        phi = 2.0 * math.pi * index / CONST_CUP_STAVES
        c, s = math.cos(phi), math.sin(phi)
        # The rotation whose columns are the stave's axes, as a quaternion (w, x, y, z): about z by phi + 90 deg,
        # then about the stave's own x axis by the cone's tilt, so that its y axis is g and its z axis is o.
        tilt = math.atan2(g_z, -g_r)
        half_turn, half_tilt = 0.5 * (phi + 0.5 * math.pi), 0.5 * tilt
        quaternion = [math.cos(half_turn) * math.cos(half_tilt), math.cos(half_turn) * math.sin(half_tilt),
                      math.sin(half_turn) * math.sin(half_tilt), math.sin(half_turn) * math.cos(half_tilt)]
        lines.append('      <geom name="%s%02d" type="box" pos="%s" quat="%s" size="%s" %s/>'
                     % (CONST_CUP_WALL_GEOM, index, _Numbers([centre_r * c, centre_r * s, centre_z]), _Numbers(quaternion),
                        _Numbers([half_width, slant / 2.0, wall / 2.0]), attributes))
    # 2\ The lip's boxes: from the opening's radius to the cup's outer radius, from the cone's top to the mouth.
    centre_r, half_depth = 0.5 * (r_mouth + r_outer), 0.5 * (r_outer - r_mouth)
    half_width = r_outer * math.tan(math.pi / CONST_CUP_STAVES)
    for index in range(CONST_CUP_STAVES):
        phi = 2.0 * math.pi * index / CONST_CUP_STAVES
        lines.append('      <geom name="%s%02d" type="box" pos="%s" quat="%s" size="%s" %s/>'
                     % (CONST_CUP_LIP_GEOM, index, _Numbers([centre_r * math.cos(phi), centre_r * math.sin(phi), 0.5 * (lip + mouth)]),
                        _Numbers([math.cos(phi / 2.0), 0.0, 0.0, math.sin(phi / 2.0)]), _Numbers([half_depth, half_width, 0.5 * (mouth - lip)]),
                        attributes))
    return lines


def Cup_Xml(triangles: np.ndarray) -> str:
    """
    Description:
        The cup's MJCF (models/cup.xml): one body in the frame of the robot's flange site, z along the
        cup's axis, with the cup's mesh (visual), the cylinder around it (the clearance of the robot's
        motion: it meets the world and the robot, never the ball), the cup's own shape for the ball
        (Cup_Contact_Geoms) and the site in the centre of its mouth, the tool centre point. Its mass and
        inertia are the mesh's at CONST_CUP_DENSITY.

    Args:
        (1) triangles [Array<float> Nx3x3]: The cup's CAD mesh [mm], its axis along y.

    Returns:
        (1) parameter [string]: The file's text.
    """

    # The CAD's y axis becomes the body's z axis: a quarter turn about x, (x, y, z) -> (x, -z, y).
    turned = np.stack([triangles[..., 0], -triangles[..., 2], triangles[..., 1]], axis=-1) / 1000.0
    volume, centroid, inertia = Mass_Properties(turned)
    inertia = CONST_CUP_DENSITY * inertia
    length, radius = CONST_CUP_LENGTH / 1000.0, CONST_CUP_RADIUS / 1000.0
    full = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    lines = ['<!--',
             '  The cup of the ball-catching simulation (SPEC §2.4, §9.10): the passive conical gripper.',
             '  Written by scripts/Maintenance/build_scene.py from the CAD; never edited by hand.',
             '  Frame: the robot\'s flange site, z along the cup\'s axis out of its mouth. Lengths in m.',
             '-->',
             '<mujoco model="cup">',
             '  <compiler angle="radian" meshdir="meshes"/>',
             '  <option integrator="implicitfast"/>',
             '  <asset>',
             '    <mesh name="cup" file="cup.stl" scale="0.001 0.001 0.001"/>',
             '  </asset>',
             '  <worldbody>',
             '    <body name="%s">' % CONST_CUP_BODY,
             '      <inertial pos="%s" mass="%s" fullinertia="%s"/>' % (_Numbers(centroid), _Number(CONST_CUP_DENSITY * volume), _Numbers(full)),
             '      <geom name="cup_visual" type="mesh" mesh="cup" quat="%s" rgba="%s" contype="0" conaffinity="0" group="%d"/>'
             % (_Numbers([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]), CONST_COLOURS['cup'], CONST_GROUP_VISUAL),
             '      <geom name="cup_collision" type="cylinder" pos="%s" size="%s" rgba="%s" contype="%d" conaffinity="0" group="%d"/>'
             % (_Numbers([0.0, 0.0, length / 2.0]), _Numbers([radius, length / 2.0]), CONST_COLOURS['collision'], CONST_BIT_WORLD,
                CONST_GROUP_COLLISION)]
    lines += Cup_Contact_Geoms()
    lines += ['      <site name="%s" pos="%s" size="0.005"/>' % (CONST_CUP_MOUTH_SITE, _Numbers([0.0, 0.0, length])),
             '    </body>',
             '  </worldbody>',
             '</mujoco>']
    return '\n'.join(lines) + '\n'


def Build_Record(models: str) -> str:
    """
    Description:
        The build record (models/build.json): the SHA-256 of every CAD file the models were made from
        and of every mesh of the models folder, the vendored folder's commit and tree id, and the tool
        and the settings of the stand's decimation. Sorted keys, no time and no path of this machine.

    Args:
        (1) models [string]: The models folder.

    Returns:
        (1) parameter [string]: The file's text.
    """

    record = {'format_version': 1,
              'inputs': {os.path.relpath(path, CONST_REPO_ROOT).replace(os.sep, '/'): Sha256_File(path)
                         for path in (CONST_CAD_STAND, CONST_CAD_CUP, CONST_CAD_BALL)},
              # The cup's and the ball's meshes are copies of their CAD files; the stand's is Blender's.
              'meshes': {CONST_MESH_STAND.replace(os.sep, '/'): Sha256_File(os.path.join(models, CONST_MESH_STAND)),
                         CONST_MESH_CUP.replace(os.sep, '/'): Sha256_File(CONST_CAD_CUP),
                         CONST_MESH_BALL.replace(os.sep, '/'): Sha256_File(CONST_CAD_BALL)},
              'stand': {'tool': 'Blender ' + CONST_BLENDER_VERSION, 'target_faces': CONST_STAND_FACES,
                        'merge_distance_mm': CONST_STAND_MERGE, 'faces': int(len(Read_Stl(os.path.join(models, CONST_MESH_STAND))))},
              'vendored': {'folder': CONST_VENDORED, 'source': 'google-deepmind/mujoco_menagerie', 'commit': CONST_MENAGERIE_COMMIT,
                           'tree': CONST_MENAGERIE_TREE}}
    return json.dumps(record, indent=1, sort_keys=True) + '\n'


def Expected_Files(models: str) -> tp.Dict[str, bytes]:
    """
    Description:
        Every file this script writes besides the stand's mesh and the fingerprint, with the bytes it
        must hold: the cup's and the ball's meshes (copies of the CAD files), the two MJCF files, the
        SHA-256 list of the vendored folder and the build record.

    Args:
        (1) models [string]: The models folder; its stand mesh and its vendored folder are read.

    Returns:
        (1) parameter [Dict<string, bytes>]: The bytes by path relative to the models folder.
    """

    with open(CONST_CAD_CUP, 'rb') as file:
        cup = file.read()
    with open(CONST_CAD_BALL, 'rb') as file:
        ball = file.read()
    return {CONST_MESH_CUP: cup, CONST_MESH_BALL: ball,
            CONST_SCENE_FILE: Scene_Xml(Ball_Centre(Read_Stl(CONST_CAD_BALL))).encode('utf-8'),
            CONST_CUP_FILE: Cup_Xml(Read_Stl(CONST_CAD_CUP)).encode('utf-8'),
            CONST_SUMS_FILE: Vendored_Sums(models).encode('utf-8'),
            CONST_RECORD_FILE: Build_Record(models).encode('utf-8')}


def Decimate_Stand(blender: str, target: str) -> tp.Tuple[int, int]:
    """
    Description:
        Decimate the stand's CAD mesh with Blender into `target` (CONST_BLENDER_SCRIPT). A Blender of
        another version than CONST_BLENDER_VERSION, and a run that does not report its face counts, are
        refused.

    Args:
        (1) blender [string]: The Blender executable.
        (2) target [string]: The STL file to write.

    Returns:
        (1) parameter [int]: The faces left after merging.
        (2) parameter [int]: The faces written.
    """

    if not os.path.isfile(blender):
        raise FileNotFoundError('Blender was not found at %s. Install Blender %s, or name it with --blender.'
                                % (blender, CONST_BLENDER_VERSION))
    version = subprocess.run([blender, '--version'], capture_output=True, text=True, timeout=120).stdout.splitlines()[:1]
    if version != ['Blender ' + CONST_BLENDER_VERSION]:
        raise FileNotFoundError('%s is %s, not Blender %s, which made the shipped mesh. Use that version.'
                                % (blender, version[0] if version else 'not Blender', CONST_BLENDER_VERSION))
    with tempfile.TemporaryDirectory() as folder:
        script = os.path.join(folder, 'decimate.py')
        with open(script, 'w', encoding='utf-8') as file:
            file.write(CONST_BLENDER_SCRIPT)
        result = subprocess.run([blender, '-b', '--factory-startup', '--python', script, '--', CONST_CAD_STAND, target,
                                 str(CONST_STAND_FACES), repr(CONST_STAND_MERGE)], capture_output=True, text=True, timeout=1800)
    counts = [line.split()[1:] for line in result.stdout.splitlines() if line.startswith('DECIMATED ')]
    if result.returncode != 0 or len(counts) != 1 or not os.path.isfile(target):
        raise RuntimeError('Blender did not decimate the stand (exit %d): %s' % (result.returncode, result.stderr.strip()[-400:]))
    return int(counts[0][0]), int(counts[0][1])


def main():
    """
    Description:
        Check, or with --write build, the models folder:

            .venv/bin/python scripts/Maintenance/build_scene.py [--models <dir>] [--config Config/ball_catching.yaml]
            .venv/bin/python scripts/Maintenance/build_scene.py --write [--stand [--blender <blender>]] [--models <dir>]

        Without --write nothing is written: the vendored folder is compared with MuJoCo Menagerie's tree
        id, every file the script writes with the bytes it would write, and the fingerprint of the scene
        built from the folder with the stored one. --write writes those files and the fingerprint;
        --stand also makes the stand's mesh again with Blender (minutes), which a normal run never does.
        The vendored folder is never written.

        Exit codes: 0 every check passed (--write: written); 1 a check failed, each difference named
        (--write: the vendored folder is not Menagerie's, nothing written); 2 could not check or write (a
        missing input, no Blender, an invalid configuration).
        The report goes to stdout, the [INFO] / [ERROR] lines to stderr.
    """

    parser = argparse.ArgumentParser(description='Check or build the models folder of the ball-catching scene.')
    parser.add_argument('--models', default=CONST_MODELS_DIR, help='the models folder (default: models/)')
    parser.add_argument('--config', default=CONST_CONFIG_FILE, help='the configuration whose areas the scene shows '
                                                                   '(default: Config/ball_catching.yaml)')
    parser.add_argument('--write', action='store_true', help='write the files instead of checking them')
    parser.add_argument('--stand', action='store_true', help='with --write: decimate the stand\'s mesh again with Blender')
    parser.add_argument('--blender', default=CONST_BLENDER, help='the Blender executable (default: %s)' % CONST_BLENDER)
    args = parser.parse_args()
    if args.stand and not args.write:
        parser.error('--stand makes the stand\'s mesh again and needs --write')

    # 1\ The inputs: the CAD files, the vendored folder, the configuration.
    vendored = os.path.join(args.models, CONST_VENDORED)
    missing = [path for path in (CONST_CAD_STAND, CONST_CAD_CUP, CONST_CAD_BALL) if not os.path.isfile(path)]
    missing += [vendored] if not os.path.isdir(vendored) else []
    if missing:
        _Log('ERROR', 'Missing: %s. Restore them from the repository (git checkout) and run again.' % ', '.join(missing))
        return 2
    try:
        configuration = Load_Configuration(args.config)
    except Configuration_Error_Cls as error:
        _Log('ERROR', str(error))
        return 2
    tree = Git_Tree_Id(vendored)
    if args.write and tree != CONST_MENAGERIE_TREE:
        _Log('ERROR', 'The vendored folder %s has the tree id %s, not %s of MuJoCo Menagerie at %s: a file of it was changed. '
                      'Restore it (git checkout -- %s); nothing was written.' % (vendored, tree, CONST_MENAGERIE_TREE, CONST_MENAGERIE_COMMIT,
                                                                                 vendored))
        return 1

    # 2\ The stand's mesh, on request only.
    stand = os.path.join(args.models, CONST_MESH_STAND)
    if args.stand:
        os.makedirs(os.path.dirname(stand), exist_ok=True)
        try:
            merged, faces = Decimate_Stand(args.blender, stand)
        except (FileNotFoundError, RuntimeError) as error:
            _Log('ERROR', str(error))
            return 2
        _Log('INFO', 'Decimated the stand with Blender %s: %d faces after merging, %d written to %s.'
             % (CONST_BLENDER_VERSION, merged, faces, stand))
    if not os.path.isfile(stand):
        _Log('ERROR', 'The stand\'s mesh %s is missing. Restore it from the repository, or make it again with --write --stand.' % stand)
        return 2

    # 3\ The files with known bytes: written, or compared.
    expected = Expected_Files(args.models)
    if args.write:
        for name, data in expected.items():
            path = os.path.join(args.models, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as file:
                file.write(data)
            _Log('INFO', 'Wrote %s.' % path)
    failures = 0
    report = ['%s: %s' % (CONST_VENDORED, 'the tree id of MuJoCo Menagerie at %s' % CONST_MENAGERIE_COMMIT[:8] if tree == CONST_MENAGERIE_TREE
                          else 'DIFFERS: its tree id is %s, not %s' % (tree, CONST_MENAGERIE_TREE))]
    failures += tree != CONST_MENAGERIE_TREE
    for name, data in expected.items():
        path = os.path.join(args.models, name)
        if not os.path.isfile(path):
            _Log('ERROR', '%s is missing. Restore it from the repository, or write it with --write.' % path)
            return 2
        with open(path, 'rb') as file:
            same = file.read() == data
        report.append('%s: %s' % (name.replace(os.sep, '/'), 'identical' if same else 'DIFFERS from what this script writes'))
        failures += not same

    # 4\ The fingerprint of the scene built from the folder, through the project's own builder.
    fingerprint_file = os.path.join(args.models, CONST_FINGERPRINT_FILE)
    try:
        current = Fingerprint(Build_Scene(Areas_Str=configuration.Areas, models=args.models))
        if args.write:
            Write_Fingerprint(fingerprint_file, current)
            _Log('INFO', 'Wrote %s.' % fingerprint_file)
        differences = Compare_Fingerprints(Read_Fingerprint(fingerprint_file), current)
    except Scene_Error_Cls as error:
        _Log('ERROR', str(error))
        return 2
    report.append('%s: %s' % (CONST_FINGERPRINT_FILE, 'identical' if not differences else 'DIFFERS in %d fields' % len(differences)))
    report += ['    ' + difference for difference in differences]
    failures += bool(differences)

    for line in report:
        print(line)
    if failures:
        _Log('ERROR', '%d of %d checks failed. The scene is built once: a change is a decided edit of this script, then --write.'
             % (failures, len(expected) + 2))
        return 1
    _Log('INFO', 'The models folder %s is what this script writes.' % args.models)
    return 0


if __name__ == '__main__':
    sys.exit(main())
