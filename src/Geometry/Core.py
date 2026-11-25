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
# File Name: ../Geometry/Core.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp

"""
Description:
    Initialization of constants.

    The areas of the experiment (SPEC §7.2): an axis-aligned cuboid and a sphere, hemisphere or
    quarter sphere with an orientation; whether a point is inside, and a random point inside drawn
    from a generator the caller passes in. Ported from Utilities/Geometry.py, Transformation.py and
    Mathematics.py of commit 0e59db2. Pure computation: no file, no configuration, no printing.
"""
# The rounding allowance of every boundary test: a point at most this far outside an area's
# boundary is inside. A point exactly on a boundary is inside (SPEC §7.2), but the inside test of
# a rotated area rotates the point first, and that rounding (cos(pi/2) = 6.1e-17, not 0) moves a
# point on a flat face or on the sphere by about 1e-16 m to either side (measured 2026-11-09 on the
# 0e59db2 areas: points on the quarter sphere's rotated face and 532 of 20 000 points on the
# hemisphere's surface fell outside). 1e-12 m is four orders above that rounding and three below
# the 1e-9 m that SPEC §10 requires to be outside (SPEC §7.2).
#   Unit [m]
CONST_BOUNDARY_EPSILON = 1e-12
# The shapes of a spherical area (SPEC §8, Areas.Robot.Shape).
#   Unit [Tuple<string>]
CONST_SPHERICAL_SHAPES = ('sphere', 'hemisphere', 'quarter_sphere')


class Geometry_Error_Cls(ValueError):
    """
    Description:
        An area cannot be built from the values given. The message names the value and what it must
        be; a script prints it verbatim as an [ERROR] line.
    """


def Euclidean_Norm(x: np.ndarray) -> float:
    """
    Description:
        The Euclidean norm ||x||_2 = sqrt(x_1^2 + ... + x_n^2) of a vector.

        Note:
            The squares are added one by one from the first component and the root is taken as a
            power of 1/2, as in Utilities/Mathematics.py of 0e59db2: a different summation order
            or numpy.linalg.norm can differ in the last bit, and the 2022 baseline is reproduced
            bit for bit (SPEC §2.1).

    Args:
        (1) x [Vector<float> 1xN]: The vector.

    Returns:
        (1) parameter [float]: Its Euclidean norm, in the unit of x.
    """

    return sum([x_i ** 2 for x_i in np.asarray(x)]) ** 0.5


def Rotation_Matrix_X(theta: float) -> np.ndarray:
    """
    Description:
        The rotation by theta about the x axis (right-hand rule).

    Args:
        (1) theta [float]: The angle [rad].

    Returns:
        (1) parameter [Matrix<float> 3x3]: The rotation matrix.
    """

    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(theta), (-1.0) * np.sin(theta)],
                     [0.0, np.sin(theta), np.cos(theta)]])


def Rotation_Matrix_Y(theta: float) -> np.ndarray:
    """
    Description:
        The rotation by theta about the y axis (right-hand rule).

    Args:
        (1) theta [float]: The angle [rad].

    Returns:
        (1) parameter [Matrix<float> 3x3]: The rotation matrix.
    """

    return np.array([[np.cos(theta), 0.0, np.sin(theta)],
                     [0.0, 1.0, 0.0],
                     [(-1.0) * np.sin(theta), 0.0, np.cos(theta)]])


def Rotation_Matrix_Z(theta: float) -> np.ndarray:
    """
    Description:
        The rotation by theta about the z axis (right-hand rule).

    Args:
        (1) theta [float]: The angle [rad].

    Returns:
        (1) parameter [Matrix<float> 3x3]: The rotation matrix.
    """

    return np.array([[np.cos(theta), (-1.0) * np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


class Cuboid_Cls(object):
    """
    Description:
        An axis-aligned cuboid area. A point on a face is inside; the allowance is
        CONST_BOUNDARY_EPSILON.

    Initialization of the Class:
        Args:
            (1) centre [Vector<float> 1x3]: Centre of the cuboid [m].
            (2) size [Vector<float> 1x3]: Edge lengths along x, y, z [m].

        Example:
            Initialization:
                # Initialization of the class.
                Cls = Cuboid_Cls([0.0, 3.0, 2.25], [4.0, 4.0, 3.5])

            Features:
                # Properties of the class.
                Cls.Centre; Cls.Size

                # Functions of the class.
                Cls.Is_Point_Inside([0.0, 3.0, 2.25])
                Cls.Random_Point_Inside(np.random.default_rng(0))
    """

    def __init__(self, centre: tp.Sequence[float], size: tp.Sequence[float]) -> None:
        # << PRIVATE >> #
        self.__centre = np.array(centre, dtype=np.float64)
        self.__size = np.array(size, dtype=np.float64)

    @property
    def Centre(self) -> np.ndarray:
        """
        Description:
            The centre of the cuboid.

        Returns:
            (1) parameter [Vector<float> 1x3]: The centre [m].
        """

        return self.__centre.copy()

    @property
    def Size(self) -> np.ndarray:
        """
        Description:
            The edge lengths of the cuboid.

        Returns:
            (1) parameter [Vector<float> 1x3]: The edge lengths along x, y, z [m].
        """

        return self.__size.copy()

    def Is_Point_Inside(self, p: tp.Sequence[float]) -> bool:
        """
        Description:
            Whether a point is inside the cuboid: on every axis its distance from the centre is at
            most half the edge plus CONST_BOUNDARY_EPSILON.

        Args:
            (1) p [Vector<float> 1x3]: The point [m].

        Returns:
            (1) parameter [bool]: True inside or on a face, False outside.
        """

        d = np.asarray(p, dtype=np.float64) - self.__centre
        return bool(np.all(np.absolute(d) - self.__size / 2.0 <= CONST_BOUNDARY_EPSILON))

    def Random_Point_Inside(self, rng: np.random.Generator) -> np.ndarray:
        """
        Description:
            A point drawn uniformly inside the cuboid.

        Args:
            (1) rng [np.random.Generator]: The random generator; the caller seeds it.

        Returns:
            (1) parameter [Vector<float> 1x3]: The point [m].
        """

        return rng.uniform(self.__centre - self.__size / 2.0, self.__centre + self.__size / 2.0)


class Spherical_Area_Cls(object):
    """
    Description:
        A sphere, a hemisphere (z >= 0 in the area's frame) or a quarter sphere (x >= 0 and z >= 0
        in the area's frame) about a centre, turned by an orientation. A point on the surface or on
        a flat face is inside; the allowance is CONST_BOUNDARY_EPSILON.

        Note:
            The area's frame is turned by R = R_x(alpha) @ R_y(beta) @ R_z(gamma) from the frame of
            the points, as in Utilities/Geometry.Modular_Sphere of 0e59db2; a point p is tested at
            R^T (p - centre), computed as R_z(-gamma) @ R_y(-beta) @ R_x(-alpha) @ (p - centre).

    Initialization of the Class:
        Args:
            (1) centre [Vector<float> 1x3]: Centre of the sphere [m].
            (2) orientation [Vector<float> 1x3]: The angles alpha, beta, gamma [rad].
            (3) radius [float]: Radius of the sphere [m].
            (4) shape [string]: One of CONST_SPHERICAL_SHAPES.

        Example:
            Initialization:
                # Initialization of the class.
                Cls = Spherical_Area_Cls([0.0, 0.0, 0.3607], [0.0, 0.0, -np.pi / 2.0], 1.0, 'hemisphere')

            Features:
                # Properties of the class.
                Cls.Centre; Cls.Orientation; Cls.Radius; Cls.Shape

                # Functions of the class.
                Cls.Is_Point_Inside([0.0, -0.5, 0.8607])
                Cls.Random_Point_Inside(np.random.default_rng(0))
    """

    def __init__(self, centre: tp.Sequence[float], orientation: tp.Sequence[float], radius: float, shape: str) -> None:
        if shape not in CONST_SPHERICAL_SHAPES:
            raise Geometry_Error_Cls('The shape "%s" of a spherical area is not one of %s. Use one of them.'
                                     % (shape, ', '.join(CONST_SPHERICAL_SHAPES)))
        # << PRIVATE >> #
        self.__centre = np.array(centre, dtype=np.float64)
        self.__orientation = np.array(orientation, dtype=np.float64)
        self.__radius = float(radius)
        self.__shape = shape
        alpha, beta, gamma = self.__orientation
        self.__R_inv = Rotation_Matrix_Z(-gamma) @ Rotation_Matrix_Y(-beta) @ Rotation_Matrix_X(-alpha)

    @property
    def Centre(self) -> np.ndarray:
        """
        Description:
            The centre of the sphere.

        Returns:
            (1) parameter [Vector<float> 1x3]: The centre [m].
        """

        return self.__centre.copy()

    @property
    def Orientation(self) -> np.ndarray:
        """
        Description:
            The orientation of the area.

        Returns:
            (1) parameter [Vector<float> 1x3]: The angles alpha, beta, gamma [rad].
        """

        return self.__orientation.copy()

    @property
    def Radius(self) -> float:
        """
        Description:
            The radius of the sphere.

        Returns:
            (1) parameter [float]: The radius [m].
        """

        return self.__radius

    @property
    def Shape(self) -> str:
        """
        Description:
            The shape of the area.

        Returns:
            (1) parameter [string]: One of CONST_SPHERICAL_SHAPES.
        """

        return self.__shape

    def Is_Point_Inside(self, p: tp.Sequence[float]) -> bool:
        """
        Description:
            Whether a point is inside the area: in the area's frame, its distance from the centre is
            at most the radius, and for a hemisphere z >= 0, for a quarter sphere also x >= 0, each
            with the allowance CONST_BOUNDARY_EPSILON.

        Args:
            (1) p [Vector<float> 1x3]: The point [m].

        Returns:
            (1) parameter [bool]: True inside or on the boundary, False outside.
        """

        q = self.__R_inv @ (np.asarray(p, dtype=np.float64) - self.__centre)
        inside = Euclidean_Norm(q) - self.__radius <= CONST_BOUNDARY_EPSILON
        if self.__shape in ('hemisphere', 'quarter_sphere'):
            inside = inside and q[2] >= -CONST_BOUNDARY_EPSILON
        if self.__shape == 'quarter_sphere':
            inside = inside and q[0] >= -CONST_BOUNDARY_EPSILON
        return bool(inside)

    def Random_Point_Inside(self, rng: np.random.Generator) -> np.ndarray:
        """
        Description:
            A point drawn uniformly inside the area: drawn in the cube about the centre with edges
            of twice the radius until one is inside.

        Args:
            (1) rng [np.random.Generator]: The random generator; the caller seeds it.

        Returns:
            (1) parameter [Vector<float> 1x3]: The point [m].
        """

        while True:
            p = rng.uniform(self.__centre - self.__radius, self.__centre + self.__radius)
            if self.Is_Point_Inside(p):
                return p
