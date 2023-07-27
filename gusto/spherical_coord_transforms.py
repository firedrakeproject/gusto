"""
Stores some common routines to transform coordinates between spherical and
Cartesian systems.
"""

"""
Some common coordinate transforms.
"""
from firedrake import (pi, sin, cos, sqrt, Max, Min, asin, atan_2, as_vector,
                       dot, cross, Constant, grad, CellNormal, curl, inner)
import numpy as np

__all__ = ["xyz_from_lonlatr", "lonlatr_from_xyz", "xyz_vector_from_lonlatr",
           "lonlatr_vector_from_xyz", "rodrigues_rotation", "rotate_pole",
           "rotated_lonlatr_coords_and_vectors"]

def magnitude(u):
    """
    Returns the pointwise magnitude of a vector field
    """

    if len(u) == 1:
        return sqrt(u[0]**2)
    elif len(u) == 2:
        return sqrt(u[0]**2+u[1]**2)
    elif len(u) == 3:
        return sqrt(u[0]**2+u[1]**2+u[2]**2)
    else:
        raise NotImplementedError('magnitude not implemented for your function')


def xyz_from_lonlatr(lon, lat, r):
    """
    Returns the global Cartesian coordinates x, y, z from
    spherical lon, lat and r coordinates.

    Result is returned in metres.

    :arg lon: longitude in radians.
    :arg lat: latitude in radians.
    :arg r: radius in metres.
    """

    x = r * cos(lon) * cos(lat)
    y = r * sin(lon) * cos(lat)
    z = r * sin(lat)

    return x, y, z


def lonlatr_from_xyz(x, y, z):
    """
    Returns the spherical lon, lat and r coordinates from
    the global Cartesian x, y, z coordinates.

    Result is returned in metres and radians.

    :arg x: x-coordinate in metres.
    :arg y: y-coordinate in metres.
    :arg z: z-coordinate in metres.
    """

    lon = atan_2(y, x)
    r = sqrt(x**2 + y**2 + z**2)
    l = sqrt(x**2 + y**2)
    lat = atan_2(z, l)

    return lon, lat, r


def xyz_vector_from_lonlatr(lonlatr_vector, position_vector):
    """
    Returns the Cartesian x, y and z components of a vector from a
    vector whose components are in lon, lat and r spherical coordinates.
    Needs a position vector, whose components are also assumed to be in
    spherical coordinates.

    :arg lonlatr_vector: a vector whose components are spherical lon-lat
    components.
    :arg position_vector: the position vector in spherical lon-lat coordinates,
    i.e. the longitude and latitude (in radians), radius (in metres)
    """

    lon = position_vector[0]
    lat = position_vector[1]
    r = position_vector[2]

    xyz_vector = [0.0, 0.0, 0.0]

    # f is our vector, e_i is the ith unit basis vector
    # f = f_r * e_r + f_lon * e_lon + f_lat * e_lat
    # We want f = f_x * e_x + f_y * e_y + f_z * e_z

    # f_x = dot(f, e_x)
    # e_x = cos(lon)*cos(lat) * e_r - sin(lon) * e_lon - cos(lon)*sin(lat) * e_lat
    xyz_vector[0] = (cos(lon)*cos(lat) * lonlatr_vector[2]
                     - sin(lon) * lonlatr_vector[0]
                     - cos(lon)*sin(lat) * lonlatr_vector[1])

    # f_y = dot(f, e_y)
    # e_y = sin(lon)*cos(lat) * e_r + cos(lon) * e_lon - sin(lon)*sin(lat) * e_lat
    xyz_vector[1] = (sin(lon)*cos(lat) * lonlatr_vector[2]
                     + cos(lon) * lonlatr_vector[0]
                     - sin(lon)*sin(lat) * lonlatr_vector[1])

    # f_z = dot(f, e_z)
    # e_z = sin(lat) * e_r + cos(lat) * e_lat
    xyz_vector[2] = (sin(lat) * lonlatr_vector[2]
                     + cos(lat) * lonlatr_vector[1])


    return xyz_vector


def lonlatr_vector_from_xyz(xyz_vector, position_vector):
    """
    Returns the spherical lon, lat and r components of a vector from a
    vector whose components are in x, y, z Cartesian coordinates.
    Needs a position vector, whose components are also assumed to be in
    Cartesian coordinates.

    :arg xyz_vector: a vector whose components are the Cartesian x, y and z
    components.
    :arg position_vector: the position vector in Cartesian x, y and z components,
    i.e. the x, y and z values of the position (in metres)
    """

    x = position_vector[0]
    y = position_vector[1]
    z = position_vector[2]

    lon, lat, r = lonlatr_from_xyz(x, y, z)

    lonlatr_vector = [0.0, 0.0, 0.0]

    # f is our vector, e_i is the ith unit basis vector
    # f = f_x * e_x + f_y * e_y + f_z * e_z
    # We want f = f_r * e_r + f_lon * e_lon + f_lat * e_lat

    # f_lon = dot(f, e_lon)
    # e_lon = -y/l * e_x + x/l * e_y
    lonlatr_vector[0] = (-sin(lon) * xyz_vector[0]
                         + cos(lon) * xyz_vector[1])

    # f_lat = dot(f, e_lat)
    # e_lat = -x*z/(r*l) * e_x - y*z/(r*l) * e_y + l/r * e_z
    lonlatr_vector[1] = (-cos(lon) * sin(lat) * xyz_vector[0]
                         -sin(lon) * sin(lat) * xyz_vector[1]
                         + cos(lat) * xyz_vector[2])

    # f_r = dot(f, e_r)
    # e_r = x/r * e_x + y/r * e_y + z/r * e_z
    lonlatr_vector[2] = (cos(lon) * cos(lat) * xyz_vector[0] +
                         sin(lon) * cos(lat) * xyz_vector[1] +
                         sin(lat) * xyz_vector[2])

    return lonlatr_vector


def rodrigues_rotation(old_vector, rot_axis, rot_angle):
    """
    Performs the rotation of a vector about some axis by some angle, as given
    by Rodrigues' rotation formula:

    v_rot = v * \cos(\theta) + (k \cross v) \sin(\theta) + k (k \dot v)*(1 - \cos(\theta))

    Returns a new vector. All components are assumed to be (X,Y,Z) components.

    :arg old_vector: The original vector to be rotated.
    :arg rot_axis:   A vector describing the axis about which to rotate. Should
                     be of magnitude 1.
    :arg rot_angle:  The angle to rotate by.
    """

    # Handle numpy routines separately
    if isinstance(old_vector, tuple) and isinstance(rot_axis, tuple):
        old_vector = np.array(old_vector)
        rot_axis = np.array(rot_axis)
        new_vector = (old_vector * np.cos(rot_angle)
                      + np.cross(rot_axis, old_vector) * np.sin(rot_angle)
                      + rot_axis * np.dot(rot_axis, old_vector) * (1 - np.cos(rot_angle)))

    # otherwise we use the Firedrake routines
    else:
        new_vector = (old_vector * cos(rot_angle)
                      + cross(rot_axis, old_vector) * sin(rot_angle)
                      + rot_axis * dot(rot_axis, old_vector) * (1 - cos(rot_angle)))

    return new_vector


def rotate_pole(new_pole):
    """
    Computes the rotation axis and angle associated with rotating the pole.
    We assume that the old pole is at lon = 0 and lat = pi / 2.
    Returns the rotation axis and angle for use in the rodrigues_rotation.

    :arg new_pole:   A tuple giving the longitude and latitude of the new pole.
    """

    # We assume that the old pole is old_lon_p = 0 and old_lat_p = pi / 2.
    old_lon_p = 0.0
    old_lat_p = pi / 2

    # Then moving the pole to new_lon_p, new_lat_p is akin to rotating the pole
    # about lon_rot = new_lon + pi / 2, lat_rot = 0
    new_lon_p, new_lat_p = new_pole
    lon_rot = new_lon_p + pi / 2
    lat_rot = 0.0

    # The rotation angle is only in the latitudinal direction
    rot_angle = old_lat_p - new_lat_p

    # Turn rotation axis into a vector
    # it points in the radial direction and has a length of one
    rot_axis = as_vector(xyz_vector_from_lonlatr((0, 0, 1), (lon_rot, lat_rot, 1)))

    return rot_axis, rot_angle

def rotated_lonlatr_coords_and_vectors(old_xyz, old_e_lonlatr, new_pole):
    """
    Returns the rotated (lon,lat,r) coordinates and unit basis vectors given
    a rotation axis and the (X,Y,Z) coordinates and the old (lon,lat,r)
    unit basis vectors.

    :arg old_xyz:        A tuple of the old (X,Y,Z) coordinates.
    :arg old_e_lonlatr:  A tuple of the old (lon,lat,r) unit vectors.
    :arg new_pole:       A tuple giving the longitude and latitude of the new pole.
    """

    rot_axis, rot_angle = rotate_pole(new_pole)

    old_e_lon, old_e_lat, old_e_r = old_e_lonlatr

    # Do rotations to get new coordinates
    new_xyz = rodrigues_rotation(old_xyz, rot_axis, rot_angle)
    new_lonlatr = lonlatr_from_xyz(new_xyz[0], new_xyz[1], new_xyz[2])

    new_e_lon = grad(new_lonlatr[0]) / magnitude(grad(new_lonlatr[0]))
    new_e_lat = grad(new_lonlatr[1]) / magnitude(grad(new_lonlatr[1]))
    new_e_r = old_e_r

    new_e_lonlatr = (new_e_lon, new_e_lat, new_e_r)

    return new_lonlatr, new_e_lonlatr