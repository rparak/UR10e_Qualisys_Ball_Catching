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
# File Name: ../View/Theme.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation; here the scene's specification and the compiled model's visual fields) [pip3 install mujoco]
import mujoco
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the areas' structures
from Configuration.Core import Areas_Str
#   ../Scene/Core - the scene's specification, its compilation, its area geoms and its models folder
from Scene.Core import Scene_Spec, Compile_Scene, CONST_AREAS, CONST_MODELS_DIR, CONST_GROUP_VISUAL, CONST_BALL_BODY
#   ../View/Core - a theme's colours
from View.Core import Theme_Str

"""
Description:
    Initialization of constants.

    The look of the scene's model in the view (SPEC §7.13): a
    theme applied in memory to the scene's specification (a plain gradient background, a checkerboard floor in MuJoCo's
    style, two lights)
    and to the compiled model (matte materials, the stand's and the rail's colours, the lighting and the offscreen
    buffer), so that models/ and the stored fingerprint keep the shipped scene. A theme changes what the model shows
    and nothing it computes: the themed model's fingerprint differs from the stored one in CONST_VISUAL_FIELDS only (a
    test). The values of the lighting were chosen by looking at rendered frames (Part A, 2026-11-22) and are the
    owner's to approve, not measured. MuJoCo's classic renderer draws shadows with hard edges; they are lightened by
    the ambient light, not softened.
"""
# The fields of the scene's fingerprint (Scene.Core.Fingerprint: section, field) a theme may change: what a geom and a
# material show, and the numbers of lights, materials and textures. Every other field is the shipped scene's.
#   Unit [Set<Tuple<string, string>>]
CONST_VISUAL_FIELDS = {('geom', 'rgba'), ('geom', 'group'), ('geom', 'material'), ('material', 'rgba'), ('material', 'specular'),
                       ('material', 'shininess'), ('material', 'reflectance'), ('material', 'emission'), ('sizes', 'nlight'),
                       ('sizes', 'nmat'), ('sizes', 'ntex')}
# The geom group of the camera rail and its cameras in a themed model, so that a camera preset shows or hides them
# (View.Core.Set_Camera); the shipped scene has them among the visual geoms (group 2).
#   Unit [int]
CONST_GROUP_RAIL = 5
# The names the rail's and the cameras' geoms begin with (scripts/Maintenance/build_scene.py), and the stand's visual geom.
#   Unit [Tuple<string>], [string]
CONST_RAIL_PREFIXES = ('rail', 'camera')
CONST_STAND_GEOM = 'stand_visual'
# The key light (from above the flight area, casting the shadows) and the fill light (from the other side, without):
# position [m], direction, diffuse intensity. The headlight's ambient and diffuse intensity; the most a material
# reflects (specular, shininess): matte.
#   Unit [Tuple<Vector, Vector, float>], [Tuple<Vector, Vector, float>], [float], [float], [float], [float]
CONST_KEY_LIGHT = ((-1.0, -3.0, 8.0), (0.12, 0.18, -1.0), 0.46)
CONST_FILL_LIGHT = ((3.0, -2.0, 4.0), (-0.5, 0.2, -1.0), 0.22)
CONST_AMBIENT = 0.46
CONST_HEADLIGHT = 0.14
CONST_SPECULAR = 0.15
CONST_SHININESS = 0.1
# The shadow map's size and the offscreen buffer's samples per pixel (anti-aliasing), and the near and far clipping
# planes as fractions of the model's extent.
#   Unit [int], [int], [float], [float]
CONST_SHADOW_SIZE = 8192
CONST_SAMPLES = 8
CONST_NEAR = 0.02
CONST_FAR = 60.0
# The size of the background's and of the floor's texture.
#   Unit [int, px]
CONST_TEXTURE_SIZE = 512
# The edge of a tile of the floor's checkerboard (a size that reads in the overview; the side preset
# sees the floor edge-on).
#   Unit [float, m]
CONST_TILE = 0.5


def Themed_Scene(*, Areas_Str: Areas_Str, Theme_Str: Theme_Str, width: int = 1920, height: int = 1080, markers: bool = False,
                 models: str = CONST_MODELS_DIR) -> mujoco.MjModel:
    """
    Description:
        The scene with a theme, compiled for the view. In the specification (Scene.Core.Scene_Spec): a gradient
        background, a matte checkerboard floor (MuJoCo's builtin checker texture in the theme's two tones, tiles of
        CONST_TILE), a key light with shadows and a fill light. In the compiled
        model: the lighting, the offscreen buffer of the size given with anti-aliasing, matte materials, the stand's
        and the rail's colours, the rail and its cameras in their own geom group (CONST_GROUP_RAIL), and the model's
        own area boxes invisible (the view draws the zones as decorations, which cast no shadow). The ball is drawn
        as its collision sphere, in the ball's colour: the sphere is what the contact simulates;
        the mesh's six markers, 15.9 mm proud of it, are visual only and would show through the cup's 5 mm wall when
        the ball lies in the cup. With `markers` the marked mesh is drawn instead, as in the shipped scene. Nothing
        the model computes is changed.

    Args:
        (1) Areas_Str [Areas_Str(object)]: REQUIRED, keyword-only: the areas of the configuration (SPEC §8.1).
        (2) Theme_Str [Theme_Str(object)]: REQUIRED, keyword-only: the theme (View.Core.CONST_THEMES).
        (3 - 4) width, height [int]: The offscreen buffer's size [px]: the largest frame rendered.
        (5) markers [bool]: Draw the ball's marked mesh instead of its collision sphere.
        (6) models [string]: The models folder (the shipped one by default).

    Returns:
        (1) parameter [mujoco.MjModel]: The themed model.
    """

    scene = Scene_Spec(Areas_Str=Areas_Str, models=models)
    # 1\ What only the specification takes: the background, the floor's material, the lights.
    scene.add_texture(name='view_sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                      rgb1=list(Theme_Str.Sky_Top), rgb2=list(Theme_Str.Sky_Horizon), width=CONST_TEXTURE_SIZE, height=6 * CONST_TEXTURE_SIZE)
    scene.add_texture(name='view_floor', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                      rgb1=list(Theme_Str.Floor), rgb2=list(Theme_Str.Floor_Second), mark=mujoco.mjtMark.mjMARK_EDGE,
                      markrgb=list(Theme_Str.Grid), width=CONST_TEXTURE_SIZE, height=CONST_TEXTURE_SIZE)
    # The builtin checker is two tiles by two; with a uniform texture a repeat is counted per metre.
    repeat = 1.0 / (2.0 * CONST_TILE)
    material = scene.add_material(name='view_floor', texrepeat=[repeat, repeat], texuniform=True, reflectance=0.0, specular=0.0, shininess=0.0)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'view_floor'
    scene.geom('floor').material = 'view_floor'
    for name, (position, direction, diffuse), shadow in (('view_key', CONST_KEY_LIGHT, True), ('view_fill', CONST_FILL_LIGHT, False)):
        scene.worldbody.add_light(name=name, pos=list(position), dir=list(direction), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                  castshadow=shadow, diffuse=[diffuse] * 3, specular=[0.0] * 3, ambient=[0.0] * 3)
    model = Compile_Scene(scene, models)
    # 2\ The compiled model's visual fields.
    visual = model.vis
    visual.headlight.ambient[:], visual.headlight.diffuse[:], visual.headlight.specular[:] = CONST_AMBIENT, CONST_HEADLIGHT, 0.0
    visual.global_.offwidth, visual.global_.offheight = int(width), int(height)
    visual.quality.shadowsize, visual.quality.offsamples = CONST_SHADOW_SIZE, CONST_SAMPLES
    visual.map.znear, visual.map.zfar = CONST_NEAR, CONST_FAR
    visual.rgba.haze[:] = (*Theme_Str.Sky_Horizon, 1.0)
    model.mat_specular[:] = np.minimum(model.mat_specular, CONST_SPECULAR)
    model.mat_shininess[:] = np.minimum(model.mat_shininess, CONST_SHININESS)
    model.mat_reflectance[:] = 0.0
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ''
        if name in CONST_AREAS:
            model.geom_rgba[geom, 3] = 0.0
        elif name == CONST_STAND_GEOM:
            model.geom_rgba[geom] = (*Theme_Str.Stand, 1.0)
        elif name.startswith(CONST_RAIL_PREFIXES) and model.geom_group[geom] == CONST_GROUP_VISUAL:
            model.geom_rgba[geom] = (*Theme_Str.Rail, 1.0)
            model.geom_group[geom] = CONST_GROUP_RAIL
    # 3\ The ball: its collision sphere drawn in the mesh's colour and the mesh invisible, unless the markers are asked for.
    if not markers:
        ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONST_BALL_BODY)
        of_ball = [geom for geom in range(model.ngeom) if model.geom_bodyid[geom] == ball]
        meshes = [geom for geom in of_ball if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH]
        spheres = [geom for geom in of_ball if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE]
        for sphere in spheres:
            model.geom_rgba[sphere] = (*model.geom_rgba[meshes[0], :3], 1.0)
            model.geom_group[sphere] = CONST_GROUP_VISUAL
        for mesh in meshes:
            model.geom_rgba[mesh, 3] = 0.0
    return model


def Visual_Only(differences: tp.Sequence[str]) -> tp.List[str]:
    """
    Description:
        Of the differences between two fingerprints of the scene (Scene.Core.Compare_Fingerprints: lines
        'section/element/field: ...', a size 'sizes/field: ...'), those a theme may not make: every line whose
        section and field are not in CONST_VISUAL_FIELDS.

    Args:
        (1) differences [Sequence<string>]: The differences.

    Returns:
        (1) parameter [List<string>]: The differences outside the visual fields.
    """

    out = []
    for line in differences:
        parts = line.split(':', 1)[0].split('/')
        if (parts[0], parts[-1]) not in CONST_VISUAL_FIELDS:
            out.append(line)
    return out
