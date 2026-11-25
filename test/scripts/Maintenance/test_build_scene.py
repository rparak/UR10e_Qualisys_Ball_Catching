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
# File Name: ../test/scripts/Maintenance/test_build_scene.py
## =========================================================================== ##

# System (Default)
import os
import sys
# Importlib (The implementation of import)
import importlib.util
# Shutil (High-level file operations; a copy of the models folder)
import shutil
# Subprocess (Subprocess management)
import subprocess
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest

"""
Description:
    Initialization of constants.

    scripts/Maintenance/build_scene.py: its check of the shipped models folder, run the way a person
    runs it, and its parts. The stand's decimation needs Blender and is not run here; what it wrote is
    checked through the build record's hash and against the CAD mesh. Expectations come from the CAD
    files, MuJoCo Menagerie's commit and constructed meshes with known answers.
"""
# The script under test, relative to the repository root.
#   Unit [string]
CONST_SCRIPT = os.path.join('scripts', 'Maintenance', 'build_scene.py')
# The id of the tree of the folder universal_robots_ur10e in MuJoCo Menagerie at the pinned commit
# fb36886b3023db08c198264e1e9956ab202c3e3b (git rev-parse fb36886b:universal_robots_ur10e in a clone,
# 2026-11-20), and its number of files there (git ls-tree -r).
#   Unit [string], [int]
CONST_MENAGERIE_TREE = '466f0ad824356ae94da5f514e55d630ca67071c7'
CONST_MENAGERIE_FILES = 26
# The lines of a passed check, in the script's order.
#   Unit [List<string>]
CONST_REPORT = ['universal_robots_ur10e: the tree id of MuJoCo Menagerie at fb36886b', 'meshes/cup.stl: identical', 'meshes/ball.stl: identical',
                'scene.xml: identical', 'cup.xml: identical', 'SHA256SUMS: identical', 'build.json: identical',
                'scene_fingerprint.json: identical']
# The stand's CAD mesh and MuJoCo 3.16.0's limit for an STL file (SPEC §4.4), the decimation's target,
# and the allowance for a vertex on a box's face: the STL's single precision, 0.5 um at 1 m.
#   Unit [int], [int], [int], [mm]
CONST_STAND_TRIANGLES = 608435
CONST_MUJOCO_FACE_LIMIT = 200000
CONST_STAND_FACES = 20000
CONST_BOX_EXCESS = 0.001
# How far the decimation moved a vertex of the shipped stand mesh out of the four boxes at most (measured
# 2026-11-20 on the shipped mesh, which the build record pins by its hash): a regression pin. The mesh is
# visual only.
#   Unit [mm]
CONST_DECIMATION_EXCESS = 1.191
# The centre of the ball's sphere in its CAD mesh, measured 2026-11-20 by a separate fit (F4 Part B).
#   Unit [Vector<float> 1x3, mm]
CONST_BALL_CENTRE = (-4.278, 0.729, 0.0)


def _Load(root):
    spec = importlib.util.spec_from_file_location('build_scene', os.path.join(root, CONST_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _Run(root, cwd, *args):
    return subprocess.run([sys.executable, os.path.join(root, CONST_SCRIPT), *args], cwd=str(cwd), capture_output=True, text=True,
                          timeout=300)


def _Cube(lower, upper):
    """
    Description:
        The twelve outward-wound triangles of an axis-aligned box.
    """

    x0, y0, z0 = lower
    x1, y1, z1 = upper
    corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return corners[np.array(faces)]


def _Write_Stl(path, triangles):
    record = np.zeros(len(triangles), dtype=np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]))
    record['vertices'] = triangles
    with open(path, 'wb') as file:
        file.write(b'\x00' * 80 + np.array([len(triangles)], dtype='<u4').tobytes() + record.tobytes())


def test_the_shipped_models_folder_is_what_the_script_writes(root, tmp_path):
    """
    Description:
        Without --write the script checks and writes nothing: for the shipped models folder every file it
        writes is byte for byte what it would write, the vendored folder has MuJoCo Menagerie's tree id
        and the scene built from the folder has the stored fingerprint. Exit 0, the report on stdout.
    """

    result = _Run(root, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == CONST_REPORT
    assert all(line.startswith('[INFO] ') for line in result.stderr.splitlines())
    assert os.listdir(str(tmp_path)) == []


def test_the_check_names_a_changed_file_a_changed_vendored_file_and_a_changed_scene(root, tmp_path):
    """
    Description:
        In a copy of the models folder, a scene file with another mounting height fails its own line and
        the fingerprint's, with the moved fields named; an edited vendored file fails the tree id and the
        hash list. Exit 1 both times, and --write refuses to write over an edited vendored folder.
    """

    models = tmp_path / 'models'
    shutil.copytree(os.path.join(root, 'models'), str(models))
    assert _Run(root, tmp_path, '--models', str(models)).returncode == 0
    scene = models / 'scene.xml'
    text = scene.read_text(encoding='utf-8')
    scene.write_text(text.replace('<site name="robot_mount" pos="0 0 0.18"', '<site name="robot_mount" pos="0 0 0.2"'), encoding='utf-8')
    result = _Run(root, tmp_path, '--models', str(models))
    lines = result.stdout.splitlines()
    assert result.returncode == 1 and 'scene.xml: DIFFERS from what this script writes' in lines
    assert 'scene_fingerprint.json: DIFFERS in 2 fields' in lines
    assert '    body/ur10e/base/pos: stored [0.0, 0.0, 0.18], the model has [0.0, 0.0, 0.2]' in lines
    assert '    site/robot_mount/pos: stored [0.0, 0.0, 0.18], the model has [0.0, 0.0, 0.2]' in lines
    assert [line for line in lines if line.endswith('identical')] == CONST_REPORT[1:3] + CONST_REPORT[4:7]
    assert '[ERROR] 2 of 8 checks failed.' in result.stderr

    scene.write_text(text, encoding='utf-8')
    readme = models / 'universal_robots_ur10e' / 'README.md'
    readme.write_text(readme.read_text(encoding='utf-8') + '\nedited\n', encoding='utf-8')
    result = _Run(root, tmp_path, '--models', str(models))
    lines = result.stdout.splitlines()
    assert result.returncode == 1 and lines[0].startswith('universal_robots_ur10e: DIFFERS: its tree id is ')
    assert 'SHA256SUMS: DIFFERS from what this script writes' in lines and 'scene_fingerprint.json: identical' in lines
    before = sorted((path.name, path.stat().st_mtime_ns) for path in models.iterdir())
    result = _Run(root, tmp_path, '--models', str(models), '--write')
    assert result.returncode == 1 and 'a file of it was changed' in result.stderr and 'nothing was written' in result.stderr
    assert sorted((path.name, path.stat().st_mtime_ns) for path in models.iterdir()) == before


def test_the_stand_is_not_made_again_without_blender_and_a_missing_mesh_is_named(root, tmp_path):
    """
    Description:
        --stand without the Blender it needs is exit 2 naming Blender and its version, and --stand
        without --write is a usage error; a models folder without the stand's mesh cannot be checked
        (exit 2), with the command that makes it again. Nothing is written into the folder.
    """

    models = tmp_path / 'models'
    models.mkdir()
    os.symlink(os.path.join(root, 'models', 'universal_robots_ur10e'), str(models / 'universal_robots_ur10e'))
    result = _Run(root, tmp_path, '--models', str(models), '--write', '--stand', '--blender', str(tmp_path / 'no_blender'))
    assert result.returncode == 2 and 'Blender was not found' in result.stderr and 'Blender 4.2.8 LTS' in result.stderr
    assert _Run(root, tmp_path, '--models', str(models), '--stand').returncode == 2
    result = _Run(root, tmp_path, '--models', str(models))
    assert result.returncode == 2 and 'stand.stl is missing' in result.stderr and '--write --stand' in result.stderr
    assert [str(path.relative_to(models)) for path in models.rglob('*') if path.is_file() and 'universal_robots_ur10e' not in path.parts] == []


def test_the_vendored_folder_has_the_tree_id_of_menagerie_at_the_pinned_commit(root):
    """
    Description:
        The vendored folder is MuJoCo Menagerie's at the pinned commit: its 26 files hash to the id git
        gives that folder's tree there, which no edited, added or removed file would leave. The script
        pins the same id and commit, and the SHA-256 list names the 26 files with their digests.
    """

    module = _Load(root)
    folder = os.path.join(root, 'models', 'universal_robots_ur10e')
    assert module.Git_Tree_Id(folder) == CONST_MENAGERIE_TREE
    assert module.CONST_MENAGERIE_TREE == CONST_MENAGERIE_TREE and module.CONST_MENAGERIE_COMMIT.startswith('fb36886b')
    with open(os.path.join(root, 'models', 'SHA256SUMS'), 'r', encoding='utf-8') as file:
        sums = [line.rstrip('\n').split('  ') for line in file]
    assert len(sums) == CONST_MENAGERIE_FILES and [path for _, path in sums] == sorted(path for _, path in sums)
    for digest, path in sums:
        assert path.startswith('universal_robots_ur10e/') and digest == module.Sha256_File(os.path.join(root, 'models', path)), path
    assert sum(len(files) for _, _, files in os.walk(folder)) == CONST_MENAGERIE_FILES


def test_the_tree_id_is_the_one_git_computes(root, tmp_path):
    """
    Description:
        Git_Tree_Id of a small folder with a sub-folder, a name that sorts between the sub-folder's name
        with and without its slash, and a file changed afterwards equals what git itself writes for it
        (git add, git write-tree), and changes when a byte changes.
    """

    module = _Load(root)
    folder = tmp_path / 'tree'
    (folder / 'assets').mkdir(parents=True)
    for name, text in (('ur10e.xml', '<mujoco/>\n'), ('assets/base.obj', 'v 0 0 0\n'), ('assets.txt', 'between\n'), ('LICENSE', 'text')):
        (folder / name).write_text(text, encoding='utf-8')
    environment = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)

    def Git(*args):
        return subprocess.run(['git', *args], cwd=str(folder), capture_output=True, text=True, timeout=60, check=True, env=environment).stdout.strip()

    Git('init', '-q')
    Git('add', '.')
    assert _Tree_Without_Git_Folder(module, folder) == Git('write-tree')
    (folder / 'LICENSE').write_text('texT', encoding='utf-8')
    assert _Tree_Without_Git_Folder(module, folder) != Git('write-tree')


def _Tree_Without_Git_Folder(module, folder):
    """
    Description:
        The tree id of the test's folder computed on a copy without its .git folder.
    """

    copy = folder.parent / 'copy'
    if copy.exists():
        shutil.rmtree(str(copy))
    shutil.copytree(str(folder), str(copy), ignore=shutil.ignore_patterns('.git'))
    return module.Git_Tree_Id(str(copy))


def test_the_stands_mesh_is_decimated_below_mujocos_limit_and_stays_inside_its_boxes(root):
    """
    Description:
        The stand's CAD mesh has 608 435 triangles, above the 200 000 faces MuJoCo 3.16.0 reads from an
        STL file; the shipped mesh has the 20 000 the decimation aims at. Every vertex of the CAD mesh
        lies in one of the four collision boxes, and the boxes' union has the CAD mesh's bounding box, so
        the boxes stand for the stand; the decimation moved the shipped visual mesh's vertices out of them
        by at most 1.19 mm. The mounting plate's box is the 220 x 220 x 20 mm of SPEC §4.4.
    """

    module = _Load(root)
    cad = module.Read_Stl(os.path.join(root, 'CAD', 'Stand', 'STL', 'Assembly_Robot_Stand.stl'))
    shipped = module.Read_Stl(os.path.join(root, 'models', 'meshes', 'stand.stl'))
    assert len(cad) == CONST_STAND_TRIANGLES > CONST_MUJOCO_FACE_LIMIT and len(shipped) == CONST_STAND_FACES <= CONST_MUJOCO_FACE_LIMIT
    boxes = list(module.CONST_STAND_BOXES.values())
    assert module.Box_Excess(cad.reshape(-1, 3), boxes).max() <= CONST_BOX_EXCESS
    assert abs(module.Box_Excess(shipped.reshape(-1, 3), boxes).max() - CONST_DECIMATION_EXCESS) < 0.001
    mount = np.array(module.CONST_STAND_BOXES['stand_mount'])
    assert (mount[1] - mount[0]).tolist() == [220.0, 220.0, 20.0] and mount[1, 2] == 0.0
    points = cad.reshape(-1, 3)
    assert np.allclose(np.min([lower for lower, _ in boxes], axis=0), points.min(axis=0), atol=0.5)
    assert np.allclose(np.max([upper for _, upper in boxes], axis=0), points.max(axis=0), atol=1.0)
    assert points[:, 2].min() == -module.CONST_MOUNT_HEIGHT == -180.0


def test_the_distance_outside_the_boxes_is_zero_inside_and_the_gap_outside(root):
    """
    Description:
        Box_Excess is 0 for a point inside a box or on its face, the distance to the nearest box
        otherwise (along one axis, and across a corner).
    """

    module = _Load(root)
    boxes = [((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), ((2.0, 0.0, 0.0), (3.0, 1.0, 1.0))]
    points = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.2, 0.5, 0.5], [1.9, 0.5, 0.5], [4.0, 2.0, 0.5], [-3.0, -4.0, 0.5]])
    assert np.allclose(module.Box_Excess(points, boxes), [0.0, 0.0, 0.2, 0.1, np.sqrt(2.0), 5.0], atol=1e-12)


def test_the_balls_centre_is_fitted_in_its_cad_mesh_and_in_a_constructed_one(root):
    """
    Description:
        The sphere's centre in the ball's CAD mesh is (-4.278, 0.729, 0.0) mm, as a separate fit measured
        it, not the mesh's origin; and for a constructed mesh, a sphere of radius 30 mm about a known
        centre with a marker's bump on one side, the fit returns that centre within a micrometre.
    """

    module = _Load(root)
    centre = module.Ball_Centre(module.Read_Stl(os.path.join(root, 'CAD', 'Ball', 'STL', 'Orange_Ball_R30_Spherical_Markers.stl')))
    assert np.allclose(centre, CONST_BALL_CENTRE, atol=0.001)
    polar, around = np.meshgrid(np.linspace(0.05, np.pi - 0.05, 40), np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False), indexing='ij')
    sphere = 30.0 * np.stack([np.sin(polar) * np.cos(around), np.sin(polar) * np.sin(around), np.cos(polar)], axis=-1).reshape(-1, 3)
    marker = np.array([38.0, 0.0, 0.0]) + 8.0 * sphere[::7] / 30.0
    known = np.array([3.5, -1.25, 0.75])
    points = np.vstack([sphere, marker]) + known
    triangles = np.stack([points[:-2], points[1:-1], points[2:]], axis=1)
    assert np.allclose(module.Ball_Centre(triangles), known, atol=1e-3)


def test_the_cups_contact_shape_is_read_from_its_cad_mesh(root, tmp_path):
    """
    Description:
        The faces across the cup's axis in its CAD mesh give the shape the ball touches (F11): the bottom's
        face at 5 mm reaches the radius 40.75 mm, the plate's underside at 0 reaches 45 mm, the lip's underside
        at 180 mm spans 62.5 to 67.0 mm (the opening and the cone's top), the mouth's face at 190 mm begins
        at 62.5 mm; the script's constants are these values, and its cup has one bottom, 24 staves and 24
        lip boxes that take contacts from the ball's bit only. A height without such a face is refused,
        and so a constructed box gives its own half-diagonal.
    """

    module = _Load(root)
    cup = module.Read_Stl(os.path.join(root, 'CAD', 'Gripper', 'STL', 'Assembly_Conical_Cylinder.stl'))
    assert abs(module.Level_Radii(cup, 5.0, True)[1] - 40.75) < 0.005
    assert abs(module.Level_Radii(cup, 0.0, False)[1] - 45.0) < 0.005
    low, high = module.Level_Radii(cup, 180.0, False)
    assert abs(low - 62.5) < 0.005 and abs(high - 67.0) < 0.005
    assert abs(module.Level_Radii(cup, 190.0, True)[0] - 62.5) < 0.005
    assert (module.CONST_CUP_BOTTOM, module.CONST_CUP_BOTTOM_RADIUS, module.CONST_CUP_PLATE_RADIUS, module.CONST_CUP_LIP,
            module.CONST_CUP_CONE_RADIUS, module.CONST_CUP_MOUTH_RADIUS, module.CONST_CUP_LENGTH) == (5.0, 40.75, 45.0, 180.0, 67.0, 62.5, 190.0)
    with pytest.raises(ValueError, match='no face across its axis at 100.00 mm looking up'):
        module.Level_Radii(cup, 100.0, True)
    box = _Cube((-3.0, 0.0, -4.0), (3.0, 7.0, 4.0))
    assert np.allclose(module.Level_Radii(box, 7.0, True), (5.0, 5.0)) and np.allclose(module.Level_Radii(box, 0.0, False), (5.0, 5.0))
    lines = module.Cup_Contact_Geoms()
    assert len(lines) == 49 and sum('type="cylinder"' in line for line in lines) == 1 and sum('type="box"' in line for line in lines) == 48
    assert all('contype="0" conaffinity="2"' in line for line in lines)


def test_the_mass_properties_of_a_box_are_the_textbook_ones(root):
    """
    Description:
        For a box of 2 x 3 x 4 m away from the origin the volume is 24 m^3, the centroid its centre and
        the inertia about the centroid, per unit density, V / 12 (b^2 + c^2) and so on: the cup's mass and
        inertia come from this sum over its mesh.
    """

    module = _Load(root)
    volume, centroid, inertia = module.Mass_Properties(_Cube((1.0, -1.0, 2.0), (3.0, 2.0, 6.0)))
    assert abs(volume - 24.0) < 1e-12 and np.allclose(centroid, [2.0, 0.5, 4.0], atol=1e-12)
    assert np.allclose(inertia, np.diag([24.0 / 12.0 * (9.0 + 16.0), 24.0 / 12.0 * (4.0 + 16.0), 24.0 / 12.0 * (4.0 + 9.0)]), atol=1e-9)


def test_the_stl_reader_returns_the_triangles_and_refuses_a_truncated_file(root, tmp_path):
    """
    Description:
        A binary STL written by the test reads back triangle for triangle; a file cut short, whose size
        no longer matches the count in its header, is refused naming the file.
    """

    module = _Load(root)
    path = str(tmp_path / 'box.stl')
    triangles = _Cube((0.0, 0.0, 0.0), (10.0, 20.0, 30.0))
    _Write_Stl(path, triangles)
    assert np.array_equal(module.Read_Stl(path), triangles)
    with open(path, 'rb') as file:
        data = file.read()
    with open(path, 'wb') as file:
        file.write(data[:-10])
    with pytest.raises(ValueError, match='is not a binary STL file'):
        module.Read_Stl(path)


def test_the_cups_and_the_balls_meshes_are_copies_of_their_cad_files(root):
    """
    Description:
        The cup's and the ball's meshes in the models folder are byte for byte their CAD files, and the
        build record names the CAD files' and the three meshes' digests as they are on disk.
    """

    import json
    module = _Load(root)
    with open(os.path.join(root, 'models', 'build.json'), 'r', encoding='utf-8') as file:
        record = json.load(file)
    pairs = (('meshes/cup.stl', 'CAD/Gripper/STL/Assembly_Conical_Cylinder.stl'),
             ('meshes/ball.stl', 'CAD/Ball/STL/Orange_Ball_R30_Spherical_Markers.stl'))
    for mesh, cad in pairs:
        digest = module.Sha256_File(os.path.join(root, *cad.split('/')))
        assert module.Sha256_File(os.path.join(root, 'models', *mesh.split('/'))) == digest == record['meshes'][mesh] == record['inputs'][cad]
    assert record['meshes']['meshes/stand.stl'] == module.Sha256_File(os.path.join(root, 'models', 'meshes', 'stand.stl'))
    assert record['stand'] == {'tool': 'Blender 4.2.8 LTS', 'target_faces': 20000, 'merge_distance_mm': 0.0001, 'faces': 20000}
    assert record['vendored']['commit'] == 'fb36886b3023db08c198264e1e9956ab202c3e3b' and record['vendored']['tree'] == CONST_MENAGERIE_TREE
