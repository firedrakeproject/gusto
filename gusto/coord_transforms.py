"""
Stores some common routines to transform coordinates between spherical and
Cartesian systems.
"""

import importlib
import numpy as np
import ufl

__all__ = ["xyz_from_lonlatr", "lonlatr_from_xyz", "xyz_vector_from_lonlatr",
           "lonlatr_vector_from_xyz", "rodrigues_rotation", "pole_rotation",
           "rotated_lonlatr_coords_and_vectors", "periodic_distance",
           "great_arc_length"]


def firedrake_or_numpy(variable):
    """
    A function internal to this module, used to determine whether to import
    other routines from firedrake or numpy.

    Args:
        variable (:class:`np.ndarray` or :class:`ufl.Expr`): a variable to be
            used in the coordinate transform routines.

    Returns:
        tuple (:class:`module`, str): either the firedrake or numpy module, with
            its name.
    """

    if isinstance(variable, ufl.core.expr.Expr):
        module_name = 'firedrake'
    else:
        module_name = 'numpy'

    module = importlib.import_module(module_name)

    return module, module_name


def magnitude(u):
    """
    A function internal to this module for returning the pointwise magnitude of
    a vector-valued field.

    Args:
        u (:class:`np.ndarray` or :class:`ufl.Expr`): the vector-valued field to
            take the magnitude of.

    Returns:
        :class:`np.ndarray` or :class:`ufl.Expr`: |u|, the pointwise magntiude
            of the vector field.
    """

    # Determine whether to use firedrake or numpy functions
    module = firedrake_or_numpy(u)
    sqrt = module.sqrt
    dot = module.dot

    return sqrt(dot(u, u))


def xyz_from_lonlatr(lon, lat, r, angle_units='rad'):
    """
    Returns the geocentric Cartesian coordinates x, y, z from spherical lon, lat
    and r coordinates.

    Args:
        lon (:class:`np.ndarray` or :class:`ufl.Expr`): longitude coordinate.
        lat (:class:`np.ndarray` or :class:`ufl.Expr`): latitude coordinate.
        r (:class:`np.ndarray` or :class:`ufl.Expr`): radial coordinate.
        angle_units (str, optional): the units used for the angle. Valid
            options are 'rad' (radians) or 'deg' (degrees). Defaults to 'rad'.

    Returns:
        tuple of :class`np.ndarray` or tuple of :class:`ufl.Expr`: the tuple
            of (x, y, z) coordinates in the appropriate form given the
            provided arguments.
    """

    if angle_units not in ['rad', 'deg']:
        raise ValueError(f'angle_units arg {angle_units} not valid')

    # Import routines
    module, _ = firedrake_or_numpy(lon)
    cos = module.cos
    sin = module.sin
    pi = module.pi

    if angle_units == 'deg':
        unit_factor = pi/180.0
    if angle_units == 'rad':
        unit_factor = 1.0

    lon = lon*unit_factor
    lat = lat*unit_factor

    x = r * cos(lon) * cos(lat)
    y = r * sin(lon) * cos(lat)
    z = r * sin(lat)

    return x, y, z


def lonlatr_from_xyz(x, y, z, angle_units='rad'):
    """
    Returns the spherical lon, lat and r coordinates from the global geocentric
    Cartesian x, y, z coordinates.

    Args:
        x (:class:`np.ndarray` or :class:`ufl.Expr`): x-coordinate.
        y (:class:`np.ndarray` or :class:`ufl.Expr`): y-coordinate.
        z (:class:`np.ndarray` or :class:`ufl.Expr`): z-coordinate.
        angle_units (str, optional): the units to use for the angle. Valid
            options are 'rad' (radians) or 'deg' (degrees). Defaults to 'rad'.

    Returns:
        tuple of :class`np.ndarray` or tuple of :class:`ufl.Expr`: the tuple
            of (lon, lat, r) coordinates in the appropriate form given the
            provided arguments.
    """

    if angle_units not in ['rad', 'deg']:
        raise ValueError(f'angle_units arg {angle_units} not valid')

    # Determine whether to use firedrake or numpy functions
    module, _ = firedrake_or_numpy(x)
    atan_2 = module.atan_2 if hasattr(module, "atan_2") else module.arctan2
    sqrt = module.sqrt
    pi = module.pi

    if angle_units == 'deg':
        unit_factor = 180./pi
    if angle_units == 'rad':
        unit_factor = 1.0

    lon = atan_2(y, x)
    r = sqrt(x**2 + y**2 + z**2)
    l = sqrt(x**2 + y**2)
    lat = atan_2(z, l)

    return lon*unit_factor, lat*unit_factor, r


def xyz_vector_from_lonlatr(lonlatr_vector, position_vector,
                            position_units="xyz"):
    """
    Returns the Cartesian geocentric x, y and z components of a vector from a
    vector whose components are in lon, lat and r spherical coordinates.

    Args:
        lonlatr_vector (:class:`np.ndarray` or :class:`ufl.Expr`): the input
            vector in (zonal, meridional, radial) components.
        position_vector (:class:`np.ndarray` or :class:`ufl.Expr`): the position
            vector, either as (x, y, z) or (lon, lat, radius) coordinates,
            subject to the `position_units` argument. Should match the shape of
            the input vector.
        position_units (str, optional): in which units the provided position
            vector is. Valid options are ["xyz", "lonlatr_rad", "lonlatr_deg"].
            Defaults to "xyz".

    Returns:
        :class:`np.ndarray` or :class:`ufl.Expr`: (x, y, z) components of the
            input vector.
    """

    # Check position units argument is valid
    if position_units not in ["xyz", "lonlatr_rad", "lonlatr_deg"]:
        raise ValueError('xyz_vector_from_lonlatr: the `position_units` arg '
                         + 'must be one of "xyz", "lonlatr_rad", "lonlatr_deg "'
                         + f'but {position_units} was provided')

    # Determine whether to use firedrake or numpy functions
    module, _ = firedrake_or_numpy(position_vector[0])
    pi = module.pi
    cos = module.cos
    sin = module.sin

    # Convert position to lonlatr_rad
    if position_units == 'xyz':
        lon, lat, _ = lonlatr_from_xyz(position_vector[0], position_vector[1],
                                       position_vector[2])
    elif position_units == 'lonlatr_rad':
        lon, lat, _ = position_vector
    elif position_units == 'lonlatr_deg':
        lon, lat = position_vector[0]*pi/180., position_vector[1]*pi/180.

    # f is our vector, e_i is the ith unit basis vector
    # f = f_r * e_r + f_lon * e_lon + f_lat * e_lat
    # We want f = f_x * e_x + f_y * e_y + f_z * e_z

    # f_x = dot(f, e_x)
    # e_x = cos(lon)*cos(lat) * e_r - sin(lon) * e_lon - cos(lon)*sin(lat) * e_lat
    x_component = (cos(lon)*cos(lat) * lonlatr_vector[2]
                   - sin(lon) * lonlatr_vector[0]
                   - cos(lon)*sin(lat) * lonlatr_vector[1])

    # f_y = dot(f, e_y)
    # e_y = sin(lon)*cos(lat) * e_r + cos(lon) * e_lon - sin(lon)*sin(lat) * e_lat
    y_component = (sin(lon)*cos(lat) * lonlatr_vector[2]
                   + cos(lon) * lonlatr_vector[0]
                   - sin(lon)*sin(lat) * lonlatr_vector[1])

    # f_z = dot(f, e_z)
    # e_z = sin(lat) * e_r + cos(lat) * e_lat
    z_component = (sin(lat) * lonlatr_vector[2]
                   + cos(lat) * lonlatr_vector[1])


    return (x_component, y_component, z_component)


def lonlatr_vector_from_xyz(xyz_vector, position_vector, position_units='xyz'):
    """
    Returns the spherical (zonal, meridional, radial) components of a vector-
    valued field from a vector which is expressed in geocentric Cartesian
    (x, y, z) components.

    Args:
        xyz_vector (:class:`np.ndarray` or :class:`ufl.Expr`): the input vector
            in geocentric Cartesian (x, y, z) components.
        position_vector (:class:`np.ndarray` or :class:`ufl.Expr`): the position
            vector, either as (x, y, z) or (lon, lat, radius) coordinates,
            subject to the `position_units` argument. Should match the shape of
            the input vector.
        position_units (str, optional): in which units the provided position
            vector is. Valid options are ["xyz", "lonlatr_rad", "lonlatr_deg"].
            Defaults to "xyz".

    Returns:
        :class:`np.ndarray` or :class:`ufl.Expr`: (zonal, meridional, radial)
            components of the input vector.
    """

    # Check position units argument is valid
    if position_units not in ["xyz", "lonlatr_rad", "lonlatr_deg"]:
        raise ValueError('xyz_vector_from_lonlatr: the `position_units` arg '
                         + 'must be one of "xyz", "lonlatr_rad", "lonlatr_deg "'
                         + f'but {position_units} was provided')

    # Determine whether to use firedrake or numpy functions
    module, _ = firedrake_or_numpy(position_vector[0])
    pi = module.pi
    cos = module.cos
    sin = module.sin

    # Convert position to lonlatr_rad
    if position_units == 'xyz':
        lon, lat, _ = lonlatr_from_xyz(position_vector[0], position_vector[1],
                                       position_vector[2])
    elif position_units == 'lonlatr_rad':
        lon, lat, _ = position_vector
    elif position_units == 'lonlatr_deg':
        lon, lat = position_vector[0]*pi/180., position_vector[1]*pi/180.

    # f is our vector, e_i is the ith unit basis vector
    # f = f_x * e_x + f_y * e_y + f_z * e_z
    # We want f = f_r * e_r + f_lon * e_lon + f_lat * e_lat

    # f_lon = dot(f, e_lon)
    # e_lon = -y/l * e_x + x/l * e_y
    zonal_component = (-sin(lon) * xyz_vector[0]
                       + cos(lon) * xyz_vector[1])

    # f_lat = dot(f, e_lat)
    # e_lat = -x*z/(r*l) * e_x - y*z/(r*l) * e_y + l/r * e_z
    meridional_component = (-cos(lon) * sin(lat) * xyz_vector[0]
                            -sin(lon) * sin(lat) * xyz_vector[1]
                            + cos(lat) * xyz_vector[2])

    # f_r = dot(f, e_r)
    # e_r = x/r * e_x + y/r * e_y + z/r * e_z
    radial_component = (cos(lon) * cos(lat) * xyz_vector[0] +
                        sin(lon) * cos(lat) * xyz_vector[1] +
                        sin(lat) * xyz_vector[2])

    return (zonal_component, meridional_component, radial_component)


def rodrigues_rotation(old_vector, rot_axis, rot_angle):
    u"""
    Performs the rotation of a vector v about some axis k by some angle ϕ, as
    given by Rodrigues' rotation formula:                                     \n

    v_rot = v * cos(ϕ) + (k \cross v) sin(ϕ) + k (k \dot v)*(1 - cos(ϕ))      \n

    Returns a new vector. All components must be (x,y,z) components.

    Args:
        old_vector (:class:`np.ndarray` or :class:`ufl.Expr`): the original
            vector or vector-valued field to be rotated, to be expressed in
            geocentric Cartesian (x,y,z) components in the original coordinate
            system.
        rot_axis (tuple or :class:`ufl.as_vector`): the vector representing the
            axis to rotate around, expressed in geocentric Cartesian (x,y,z)
            components (in the frame before the rotation).
        rot_angle (float): the angle to rotate by.

    Returns:
        :class:`np.ndarray` or :class:`ufl.Expr`: the rotated vector or
            vector-valued field.
    """

    # Determine whether to use firedrake or numpy functions
    module, module_name = firedrake_or_numpy(old_vector)
    cos = module.cos
    sin = module.sin
    cross = module.cross
    dot = module.dot

    # Numpy vector may need reshaping
    if module_name == 'numpy' and np.shape(rot_axis) != np.shape(old_vector):
        # Construct shape for tiling vector
        tile_shape = [dim for dim in np.shape(old_vector)[:-1]]+[1]
        # Tile rot_axis vector to create an ndarray
        rot_axis = np.tile(rot_axis, tile_shape)
        # Replace dot product routine with something that does elementwise dot
        def dot(a, b):
            dotted_vectors = np.einsum('ij,ij->i', a, b)
            # Add new axis to allow further multiplication by a vector
            return dotted_vectors[:, np.newaxis]

    new_vector = (old_vector * cos(rot_angle)
                  + cross(rot_axis, old_vector) * sin(rot_angle)
                  + rot_axis * dot(rot_axis, old_vector) * (1 - cos(rot_angle)))

    return new_vector


def pole_rotation(new_pole):
    """
    Computes the rotation axis and angle associated with rotating the pole from
    lon = 0 and lat = pi / 2 to a new longitude and latitude. Returns the
    rotation axis and angle for use in the Rodrigues rotation.

    Args:
        new_pole (tuple): a tuple of floats (lon, lat) of the new pole. The
            longitude and latitude must be expressed in radians.

    Returns:
        tuple: (rot_axis, rot_angle). This describes the rotation axis (a tuple
            of (x, y, z) components of the rotation axis, and a float describing
            the rotation angle.
    """

    import numpy as np

    # We assume that the old pole is old_lon_p = 0 and old_lat_p = pi / 2.
    old_lat_p = np.pi / 2

    # Then moving the pole to new_lon_p, new_lat_p is akin to rotating the pole
    # about lon_rot = new_lon + pi / 2, lat_rot = 0
    new_lon_p, new_lat_p = new_pole
    lon_rot = new_lon_p + np.pi / 2
    lat_rot = 0.0

    # The rotation angle is only in the latitudinal direction
    rot_angle = old_lat_p - new_lat_p

    # Turn rotation axis into a vector
    # it points in the radial direction and has a length of one
    rot_axis = xyz_vector_from_lonlatr((0, 0, 1), (lon_rot, lat_rot, 1),
                                       position_units='lonlatr_rad')

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

    rot_axis, rot_angle = pole_rotation(new_pole)

    old_e_lon, old_e_lat, old_e_r = old_e_lonlatr

    # Do rotations to get new coordinates
    new_xyz = rodrigues_rotation(old_xyz, rot_axis, rot_angle)
    new_lonlatr = lonlatr_from_xyz(new_xyz[0], new_xyz[1], new_xyz[2])

    new_e_lon = grad(new_lonlatr[0]) / magnitude(grad(new_lonlatr[0]))
    new_e_lat = grad(new_lonlatr[1]) / magnitude(grad(new_lonlatr[1]))
    new_e_r = old_e_r

    new_e_lonlatr = (new_e_lon, new_e_lat, new_e_r)

    return new_lonlatr, new_e_lonlatr


def periodic_distance():
    pass


def great_arc_length():
    pass