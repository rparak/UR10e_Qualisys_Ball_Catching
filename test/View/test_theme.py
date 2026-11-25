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
# File Name: ../test/View/test_theme.py
## =========================================================================== ##

# System (Default)
import os
# Hashlib (The models folder's files before and after)
import hashlib
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation) [pip3 install mujoco]
import mujoco
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Scene/Core - the shipped scene and its fingerprint
from Scene.Core import Build_Scene, Fingerprint, Compare_Fingerprints, Read_Fingerprint
#   ../View/Core - the themes
from View.Core import CONST_THEMES
#   ../View/Theme
from View.Theme import Themed_Scene, Visual_Only, CONST_VISUAL_FIELDS, CONST_GROUP_RAIL, CONST_TILE

"""
Description:
    Initialization of constants.

    The view's theme (SPEC §7.13; goal C, Q7 and the addition on the dark theme and the checkerboard
    floor): applied in memory, it changes what the scene shows and nothing it computes, and leaves models/ and the
    stored fingerprint as they are. Expectations come from the stored fingerprint, the scene's own files and the
    owner's decisions.
"""
# The fields a theme may change, written out (never read from the module under test): what a geom and a material
# show, and the numbers of lights, materials and textures.
#   Unit [Set<Tuple<string, string>>]
CONST_ALLOWED = {('geom', 'rgba'), ('geom', 'group'), ('geom', 'material'), ('material', 'rgba'), ('material', 'specular'),
                 ('material', 'shininess'), ('material', 'reflectance'), ('material', 'emission'), ('sizes', 'nlight'), ('sizes', 'nmat'),
                 ('sizes', 'ntex')}


def _Models_Digest(root):
    """
    Description:
        The SHA-256 of every file of models/ outside the vendored robot's folder, by name.
    """

    folder = os.path.join(root, 'models')
    out = {}
    for directory, names, files in os.walk(folder):
        names[:] = [name for name in names if name != 'universal_robots_ur10e']
        for name in files:
            with open(os.path.join(directory, name), 'rb') as file:
                out[os.path.relpath(os.path.join(directory, name), folder)] = hashlib.sha256(file.read()).hexdigest()
    return out


@pytest.mark.parametrize('name', ['dark', 'light'])
def test_a_theme_changes_only_what_the_scene_shows(root, configuration, name):
    """
    Description:
        The themed model against the stored fingerprint of the shipped scene: it differs (the theme does something:
        two lights, one material and two textures more, the floor's material, colours and groups), and every
        difference lies in the visual fields: no body, joint, option, actuator, mesh or key, no geom's size, position,
        contact bits, friction or solver field, no count of bodies, joints or geoms. The shipped scene built afterwards
        still has the stored fingerprint, and no file of models/ changed.
    """

    before = _Models_Digest(root)
    stored = Read_Fingerprint(os.path.join(root, 'models', 'scene_fingerprint.json'))
    themed = Themed_Scene(Areas_Str=configuration.Areas, Theme_Str=CONST_THEMES[name])
    differences = Compare_Fingerprints(stored, Fingerprint(themed))
    found = {(line.split(':', 1)[0].split('/')[0], line.split(':', 1)[0].split('/')[-1]) for line in differences}
    assert found and found <= CONST_ALLOWED and Visual_Only(differences) == []
    assert {('sizes', 'nlight'), ('sizes', 'nmat'), ('sizes', 'ntex'), ('geom', 'material'), ('geom', 'rgba'), ('geom', 'group')} <= found
    assert (themed.nlight, themed.nmat - stored['sizes']['nmat'], themed.ntex - stored['sizes']['ntex']) == (2, 1, 2)
    assert (themed.nq, themed.nu, themed.nbody, themed.ngeom) == tuple(stored['sizes'][key] for key in ('nq', 'nu', 'nbody', 'ngeom'))
    assert Compare_Fingerprints(stored, Fingerprint(Build_Scene(Areas_Str=configuration.Areas))) == []
    assert _Models_Digest(root) == before and CONST_VISUAL_FIELDS == CONST_ALLOWED


def test_a_change_of_what_the_scene_computes_is_not_a_visual_one(scene, configuration, root):
    """
    Description:
        The check behind the theme's test, on forced cases: a themed model whose floor friction, ball mass or time step
        is changed, or whose cup has another contact bit, shows that field as a difference outside the visual ones,
        named with its element; differences of colour, group and material alone are all visual.
    """

    stored = Read_Fingerprint(os.path.join(root, 'models', 'scene_fingerprint.json'))
    themed = Themed_Scene(Areas_Str=configuration.Areas, Theme_Str=CONST_THEMES['dark'])
    floor = mujoco.mj_name2id(themed, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    themed.geom_friction[floor, 0] = 0.9
    themed.body_mass[mujoco.mj_name2id(themed, mujoco.mjtObj.mjOBJ_BODY, 'ball')] = 0.2
    themed.opt.timestep = 0.001
    themed.geom_conaffinity[mujoco.mj_name2id(themed, mujoco.mjtObj.mjOBJ_GEOM, 'ur10e/cup_bottom')] = 1
    left = Visual_Only(Compare_Fingerprints(stored, Fingerprint(themed)))
    assert sorted(line.split(':', 1)[0] for line in left) == ['body/ball/mass', 'geom/floor/friction', 'geom/ur10e/cup_bottom/conaffinity',
                                                              'option/timestep']
    assert Visual_Only(['geom/floor/rgba: stored 1, the model has 2', 'sizes/nlight: stored 0, the model has 2',
                        'material/view_floor/reflectance: not stored']) == []
    assert Visual_Only(['geom/floor/size: stored 1, the model has 2', 'sizes/ngeom: stored 108, the model has 109']) == \
        ['geom/floor/size: stored 1, the model has 2', 'sizes/ngeom: stored 108, the model has 109']


def test_the_themed_scene_has_a_checkerboard_floor_matte_materials_and_the_frames_buffer(scene, configuration):
    """
    Description:
        What the theme sets: the floor takes a material with MuJoCo's builtin checker in the theme's
        two tones, a tile 0.5 m wide (one repeat of the two-by-two texture per metre), without reflection; a background
        texture; a key light that casts shadows and a fill light that does not, both directional; no material shines
        (specular at most 0.15, shininess at most 0.1, no reflectance); the model's own area boxes are invisible and
        its collision shapes keep their groups; the rail and the 12 cameras are in their own group; the offscreen buffer
        has the size asked, with 8 samples per pixel. The dark and the light theme differ in the floor's tones only by
        their colours: the same geometry.
    """

    theme = CONST_THEMES['dark']
    model = Themed_Scene(Areas_Str=configuration.Areas, Theme_Str=theme, width=3840, height=2160)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    material = int(model.geom_matid[floor])
    assert material >= 0 and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material) == 'view_floor'
    assert CONST_TILE == 0.5 and np.allclose(model.mat_texrepeat[material], 1.0) and model.mat_texuniform[material] == 1
    assert model.mat_reflectance[material] == 0.0 and model.mat_specular[material] == 0.0
    texture = int(model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB])
    assert texture >= 0 and model.tex_type[texture] == mujoco.mjtTexture.mjTEXTURE_2D
    start, width, height = int(model.tex_adr[texture]), int(model.tex_width[texture]), int(model.tex_height[texture])
    pixels = np.array(model.tex_data[start:start + 3 * width * height]).reshape(height, width, 3)
    quarters = ((height // 4, width // 4), (height // 4, 3 * width // 4), (3 * height // 4, width // 4), (3 * height // 4, 3 * width // 4))
    tones = {tuple(int(value) for value in pixels[y, x]) for y, x in quarters}
    expected = {tuple(int(round(255 * value)) for value in tone) for tone in (theme.Floor, theme.Floor_Second)}
    assert len(tones) == 2 and all(min(max(abs(a - b) for a, b in zip(tone, other)) for other in expected) <= 1 for tone in tones)
    assert tuple(pixels[height // 4, width // 4]) == tuple(pixels[3 * height // 4, 3 * width // 4])
    assert sum(model.tex_type[k] == mujoco.mjtTexture.mjTEXTURE_SKYBOX for k in range(model.ntex)) == 1
    assert model.nlight == 2 and sorted(int(value) for value in model.light_castshadow) == [0, 1]
    assert all(int(value) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL) for value in model.light_type)
    assert model.mat_specular.max() <= 0.15 + 1e-6 and model.mat_shininess.max() <= 0.1 + 1e-6 and model.mat_reflectance.max() == 0.0
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, k) or '' for k in range(model.ngeom)]
    areas = [k for k, name in enumerate(names) if name.startswith('area_')]
    assert len(areas) == 4 and all(model.geom_rgba[k, 3] == 0.0 for k in areas)
    rail = [k for k, name in enumerate(names) if name.startswith(('rail', 'camera'))]
    assert len(rail) == 16 and all(model.geom_group[k] == CONST_GROUP_RAIL == 5 for k in rail)
    assert np.allclose(model.geom_rgba[rail[0], :3], theme.Rail)
    sphere = names.index('ball_collision')
    assert np.array_equal(model.geom_group[[k for k in range(model.ngeom) if k not in rail + [sphere]]],
                          scene.geom_group[[k for k in range(scene.ngeom) if k not in rail + [sphere]]])
    assert (model.vis.global_.offwidth, model.vis.global_.offheight, model.vis.quality.offsamples) == (3840, 2160, 8)
    light = Themed_Scene(Areas_Str=configuration.Areas, Theme_Str=CONST_THEMES['light'])
    assert (light.vis.global_.offwidth, light.vis.global_.offheight) == (1920, 1080)
    assert not np.array_equal(light.tex_data, model.tex_data) and np.array_equal(light.geom_group, model.geom_group)
    assert np.array_equal(light.geom_size, model.geom_size) and np.array_equal(light.geom_pos, model.geom_pos)


def test_the_ball_is_drawn_as_its_collision_sphere_unless_the_markers_are_asked_for(scene, configuration, root):
    """
    Description:
        The ball in the view: by default its collision sphere, R 30 mm, is drawn opaque among the
        visual geoms in the ball's own colour (1, 0.45, 0.05) and the marked mesh is invisible, because the sphere is
        what the contact simulates and the mesh's markers, which reach 45.9 mm from the centre, show through the cup's
        wall. With the markers asked for the mesh is drawn and the sphere stays a hidden collision shape, as in the
        shipped scene. Either way the sphere's size, contact bits and the ball's mass are the shipped scene's, and the
        themed model differs from the stored fingerprint in visual fields only.
    """

    stored = Read_Fingerprint(os.path.join(root, 'models', 'scene_fingerprint.json'))
    plain = Themed_Scene(Areas_Str=configuration.Areas, Theme_Str=CONST_THEMES['dark'])
    marked = Themed_Scene(Areas_Str=configuration.Areas, Theme_Str=CONST_THEMES['dark'], markers=True)
    sphere = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_GEOM, 'ball_collision')
    mesh = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_GEOM, 'ball_visual')
    assert np.allclose(plain.geom_rgba[sphere], (1.0, 0.45, 0.05, 1.0)) and plain.geom_group[sphere] == 2 and plain.geom_rgba[mesh, 3] == 0.0
    assert np.allclose(marked.geom_rgba[mesh], (1.0, 0.45, 0.05, 1.0)) and marked.geom_group[mesh] == 2
    assert marked.geom_group[sphere] == 3 == scene.geom_group[sphere] and np.allclose(marked.geom_rgba[sphere], scene.geom_rgba[sphere])
    vertices = marked.mesh_vert[marked.mesh_vertadr[marked.geom_dataid[mesh]]:][:marked.mesh_vertnum[marked.geom_dataid[mesh]]]
    assert 0.0455 < np.linalg.norm(vertices + marked.geom_pos[mesh], axis=1).max() < 0.0465
    ball = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_BODY, 'ball')
    for model in (plain, marked):
        assert model.geom_size[sphere, 0] == 0.030 and (model.geom_contype[sphere], model.geom_conaffinity[sphere]) == (3, 2)
        assert model.body_mass[ball] == scene.body_mass[ball] == 0.122
        differences = Compare_Fingerprints(stored, Fingerprint(model))
        assert differences and Visual_Only(differences) == []
