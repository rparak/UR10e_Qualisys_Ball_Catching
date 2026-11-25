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
# File Name: ../test/Geometry/test_core.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the areas under test
#   ../Geometry/Core
from Geometry.Core import Cuboid_Cls, Euclidean_Norm, Geometry_Error_Cls, Rotation_Matrix_X, Rotation_Matrix_Y, \
    Rotation_Matrix_Z, Spherical_Area_Cls

"""
Description:
    Initialization of constants.

    The values are the legacy areas of SPEC §8 (the flight cuboid; the catch hemisphere about
    z = 0.1807 + 0.18 m, turned by -90 deg about z), the rule of SPEC §7.2 and §10 (a point on a
    boundary is inside, one 1e-9 m beyond is outside) and the owner's allowance of 1e-12 m
    (2026-11-09). Nothing is read back from the code under test.
"""
# The flight cuboid of the earlier code: centre and edge lengths.
#   Unit [m]
CONST_FLIGHT_CENTRE = np.array([0.0, 3.0, 2.25])
CONST_FLIGHT_SIZE = np.array([4.0, 4.0, 3.5])
# The catch hemisphere of the earlier code: centre, orientation (-90 deg about z) and radius.
#   Unit [m], [rad], [m]
CONST_ROBOT_CENTRE = np.array([0.0, 0.0, 0.1807 + 0.18])
CONST_ROBOT_ORIENTATION = np.array([0.0, 0.0, -90.0]) * (np.pi / 180.0)
CONST_ROBOT_RADIUS = 1.0
# The distance beyond a boundary that must be outside (SPEC §10).
#   Unit [m]
CONST_BEYOND = 1e-9
# The rounding allowance, written out here, not read from Geometry.
#   Unit [m]
CONST_ALLOWANCE = 1e-12


def _Robot_Area(shape):
    return Spherical_Area_Cls(CONST_ROBOT_CENTRE, CONST_ROBOT_ORIENTATION, CONST_ROBOT_RADIUS, shape)


@pytest.mark.parametrize('axis', [0, 1, 2], ids=['x', 'y', 'z'])
@pytest.mark.parametrize('sign', [1.0, -1.0], ids=['plus', 'minus'])
def test_a_point_on_a_cuboid_face_is_inside_and_one_1e9_m_beyond_is_outside(axis, sign):
    """
    Description:
        Both sides of each face of the flight cuboid (SPEC §7.2, §10). The np.sum change of 2fb4273
        ... bd12f7e shrank the cuboid's axes, so points beyond a face were judged inside; this test
        fails on that Cuboid.
    """

    cuboid = Cuboid_Cls(CONST_FLIGHT_CENTRE, CONST_FLIGHT_SIZE)
    p_on = CONST_FLIGHT_CENTRE.copy()
    p_on[axis] += sign * CONST_FLIGHT_SIZE[axis] / 2.0
    p_beyond = p_on.copy()
    p_beyond[axis] += sign * CONST_BEYOND
    assert cuboid.Is_Point_Inside(p_on), 'on the face: inside'
    assert not cuboid.Is_Point_Inside(p_beyond), '1e-9 m beyond the face: outside'


def test_the_release_point_of_legacy_throw_0_is_outside_the_flight_area():
    """
    Description:
        The first sample of throw 0 of Data/Legacy/Experiment_1_Midpoint_DRAG_False.txt is 0.87 m
        beyond the flight cuboid (y = 6.13 m > 5 m). With the np.sum Cuboid it was inside, so throw
        detection fired on a single sample and every 2022 benchmark run failed (SPEC §4.1).
    """

    assert not Cuboid_Cls(CONST_FLIGHT_CENTRE, CONST_FLIGHT_SIZE).Is_Point_Inside([-1.6610495, 6.1329746, 0.8721319])


@pytest.mark.parametrize('shape, direction', [('sphere', [0.0, 0.0, -1.0]),
                                              ('hemisphere', [1.0, 0.0, 0.0]),
                                              ('quarter_sphere', [0.0, -1.0, 0.0])],
                         ids=['sphere', 'hemisphere', 'quarter_sphere'])
def test_a_point_on_the_curved_surface_is_inside_and_one_1e9_m_beyond_is_outside(shape, direction):
    """
    Description:
        Both sides of the curved surface of each shape, at a point inside the shape's flat faces
        (the quarter sphere, turned by -90 deg about z, covers y <= 0).
    """

    area = _Robot_Area(shape)
    u = np.array(direction)
    assert area.Is_Point_Inside(CONST_ROBOT_CENTRE + CONST_ROBOT_RADIUS * u), 'on the surface: inside'
    assert not area.Is_Point_Inside(CONST_ROBOT_CENTRE + (CONST_ROBOT_RADIUS + CONST_BEYOND) * u), '1e-9 m beyond: outside'


def test_the_hemisphere_holds_every_point_of_its_surface_and_none_1e9_m_beyond():
    """
    Description:
        20 000 points on the hemisphere's surface up to rounding (their float distance from the
        centre is within 3.3e-16 m of the radius; seed 0) are inside, and the same points 1e-9 m
        further out are outside. Without the allowance, 532 of them were outside (measured
        2026-11-09, on the 0e59db2 code and on this one): the rotation moves them by rounding.
    """

    area = _Robot_Area('hemisphere')
    rng = np.random.default_rng(0)
    u = rng.normal(size=(20000, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    u[:, 2] = np.abs(u[:, 2])
    outside = [i for i, u_i in enumerate(u) if not area.Is_Point_Inside(CONST_ROBOT_CENTRE + u_i)]
    assert outside == [], '%d surface points outside, first %s' % (len(outside), outside[:5])
    inside = [i for i, u_i in enumerate(u) if area.Is_Point_Inside(CONST_ROBOT_CENTRE + (1.0 + CONST_BEYOND) * u_i)]
    assert inside == [], '%d points 1e-9 m beyond inside, first %s' % (len(inside), inside[:5])


@pytest.mark.parametrize('shape, p_on, step', [('hemisphere', [0.3, 0.2, 0.0], [0.0, 0.0, -1.0]),
                                               ('quarter_sphere', [0.0, -0.3, 0.0], [0.0, 0.0, -1.0]),
                                               ('quarter_sphere', [-0.5, 0.0, 0.2], [0.0, 1.0, 0.0]),
                                               ('quarter_sphere', [0.5, 0.0, 0.2], [0.0, 1.0, 0.0])],
                         ids=['hemisphere-z', 'quarter-z', 'quarter-face-minus-x', 'quarter-face-plus-x'])
def test_a_point_on_a_flat_face_is_inside_and_one_1e9_m_beyond_is_outside(shape, p_on, step):
    """
    Description:
        Both sides of each flat face, relative to the centre. The quarter sphere's face y = 0 is
        its rotated x = 0 plane: there cos(pi/2) leaves a residual of -3.1e-17 m at (-0.5, 0, z),
        which the 0e59db2 code judged outside (measured 2026-11-09); the allowance keeps it inside.
    """

    area = _Robot_Area(shape)
    p_on = CONST_ROBOT_CENTRE + np.array(p_on)
    assert area.Is_Point_Inside(p_on), 'on the face: inside'
    assert not area.Is_Point_Inside(p_on + CONST_BEYOND * np.array(step)), '1e-9 m beyond: outside'


def test_the_orientation_turns_the_quarter_sphere():
    """
    Description:
        The quarter sphere covers x >= 0 in its own frame: without rotation that is the world's
        x >= 0, turned by -90 deg about z it is the world's y <= 0.
    """

    p_plus_x, p_minus_x = CONST_ROBOT_CENTRE + [0.5, 0.0, 0.2], CONST_ROBOT_CENTRE + [-0.5, 0.0, 0.2]
    p_minus_y, p_plus_y = CONST_ROBOT_CENTRE + [0.0, -0.5, 0.2], CONST_ROBOT_CENTRE + [0.0, 0.5, 0.2]
    straight = Spherical_Area_Cls(CONST_ROBOT_CENTRE, [0.0, 0.0, 0.0], CONST_ROBOT_RADIUS, 'quarter_sphere')
    turned = _Robot_Area('quarter_sphere')
    assert straight.Is_Point_Inside(p_plus_x) and not straight.Is_Point_Inside(p_minus_x)
    assert turned.Is_Point_Inside(p_minus_y) and not turned.Is_Point_Inside(p_plus_y)


@pytest.mark.parametrize('name', ['cuboid', 'sphere', 'hemisphere', 'quarter_sphere'])
def test_every_random_point_is_inside_its_area_and_they_fill_it(name):
    """
    Description:
        1000 random points (seed 1) are inside the area and reach within 10 % of its extent on each
        axis: a sampler that returned the centre, or left the area, fails.
    """

    if name == 'cuboid':
        area = Cuboid_Cls(CONST_FLIGHT_CENTRE, CONST_FLIGHT_SIZE)
        low, high = CONST_FLIGHT_CENTRE - CONST_FLIGHT_SIZE / 2.0, CONST_FLIGHT_CENTRE + CONST_FLIGHT_SIZE / 2.0
    else:
        area = _Robot_Area(name)
        low = CONST_ROBOT_CENTRE - CONST_ROBOT_RADIUS * np.array([1.0, 1.0, 0.0 if name != 'sphere' else 1.0])
        high = CONST_ROBOT_CENTRE + CONST_ROBOT_RADIUS * np.array([1.0, 0.0 if name == 'quarter_sphere' else 1.0, 1.0])
    rng = np.random.default_rng(1)
    points = np.array([area.Random_Point_Inside(rng) for _ in range(1000)])
    assert all(area.Is_Point_Inside(p) for p in points)
    reach = 0.1 * (high - low)
    assert np.all(points.min(axis=0) < low + reach) and np.all(points.max(axis=0) > high - reach)


def test_the_same_seed_gives_the_same_random_points():
    """
    Description:
        The points come from the generator passed in, so a seed reproduces them (SPEC §2.6).
    """

    area = _Robot_Area('hemisphere')
    first = [area.Random_Point_Inside(np.random.default_rng(7)) for _ in range(3)]
    second = [area.Random_Point_Inside(np.random.default_rng(7)) for _ in range(3)]
    assert np.array_equal(first, second)


@pytest.mark.parametrize('matrix, axis_in, axis_out', [(Rotation_Matrix_X, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
                                                       (Rotation_Matrix_Y, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
                                                       (Rotation_Matrix_Z, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])],
                         ids=['x', 'y', 'z'])
def test_each_rotation_matrix_turns_the_next_axis_by_90_degrees(matrix, axis_in, axis_out):
    """
    Description:
        The right-hand rule: +90 deg about x turns y into z, about y turns z into x, about z turns x
        into y.
    """

    assert np.allclose(matrix(np.pi / 2.0) @ np.array(axis_in), axis_out, atol=1e-15)


def test_the_euclidean_norm_of_3_4_12_is_13():
    """
    Description:
        sqrt(9 + 16 + 144) = 13, exactly in floating point.
    """

    assert Euclidean_Norm(np.array([3.0, 4.0, 12.0])) == 13.0


def test_an_unknown_shape_is_refused_naming_the_shapes():
    """
    Description:
        A misspelt shape is an error naming it and the legal shapes, never a silent default.
    """

    with pytest.raises(Geometry_Error_Cls, match='half_sphere.*sphere, hemisphere, quarter_sphere'):
        Spherical_Area_Cls(CONST_ROBOT_CENTRE, CONST_ROBOT_ORIENTATION, CONST_ROBOT_RADIUS, 'half_sphere')


@pytest.mark.parametrize('name', ['cuboid', 'sphere'])
def test_the_rounding_allowance_is_1e12_m_on_both_sides(name):
    """
    Description:
        A point half the allowance beyond the boundary is inside, one twice the allowance beyond is
        outside: the allowance is 1e-12 m, not 0 and not larger.
    """

    if name == 'cuboid':
        area, p, u = Cuboid_Cls(CONST_FLIGHT_CENTRE, CONST_FLIGHT_SIZE), CONST_FLIGHT_CENTRE + [2.0, 0.0, 0.0], np.array([1.0, 0.0, 0.0])
    else:
        area, p, u = _Robot_Area('sphere'), CONST_ROBOT_CENTRE + [0.0, 0.0, 1.0], np.array([0.0, 0.0, 1.0])
    assert area.Is_Point_Inside(p + 0.5 * CONST_ALLOWANCE * u)
    assert not area.Is_Point_Inside(p + 2.0 * CONST_ALLOWANCE * u)
