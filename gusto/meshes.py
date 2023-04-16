"""
This file provides some specialised meshes not provided by Firedrake
"""

from firedrake import (FiniteElement, par_loop, READ, WRITE,
                       VectorFunctionSpace, dx, interval, TensorProductElement,
                       functionspace, function, mesh, Constant,
                       Function, op2, Mesh, PeriodicRectangleMesh)
from firedrake.petsc import PETSc
from firedrake.utils import RealType
import numpy as np
import ufl
from pyop2.mpi import COMM_WORLD

__all__ = ["GeneralIcosahedralSphereMesh", "GeneralCubedSphereMesh",
           "get_flat_latlon_mesh"]


@PETSc.Log.EventDecorator()
def GeneralIcosahedralSphereMesh(radius, num_cells_per_edge_of_panel,
                                 degree=1, reorder=None,
                                 distribution_parameters=None, comm=COMM_WORLD,
                                 name=mesh.DEFAULT_MESH_NAME):
    """
    Generate an icosahedral approximation to the surface of the sphere.

    Args:
        radius (float): The radius of the sphere to approximate. For a radius R
            the edge length of the underlying icosahedron will be             \n
            a = \\frac{R}{\\sin(2 \\pi / 5)}
        num_cells_per_edge_of_panel (int): number of cells per edge of each of
            the 20 panels of the icosahedron (1 gives an icosahedron).
        degree (int, optional): polynomial degree of coordinate space used to
            approximate the sphere. Defaults to 1, describing flat triangles.
        reorder: (bool, optional): optional flag indicating whether to reorder
           meshes for better cache locality. Defaults to False.
        comm (communicator, optional): optional communicator to build the mesh
            on. Defaults to COMM_WORLD.
        name (str, optional): optional name to give to the mesh. Defaults to
            Firedrake's default mesh name.
    """
    if num_cells_per_edge_of_panel < 1 or num_cells_per_edge_of_panel % 1:
        raise RuntimeError("Number of cells per edge must be a positive integer")

    if degree < 1:
        raise ValueError("Mesh coordinate degree must be at least 1")

    big_N = num_cells_per_edge_of_panel

    # Strategy:
    # A. Create lists for an actual icosahedron:
    #     1. List corners of the icosahedron
    #     2. Create a list of icosahedron edges: this list returns indices of
    #        the vertices associated with this edge
    #     3. Create a list of icosahedron panels: this list returns indices of
    #        the edges associated with this panel
    #     4. There may also be a use for a list of icosahedron vertices
    #        associated with each panel
    # B. Begin refinement. Create lists of refined vertices:
    #     1. Make an array of coordinates of new vertices along each edge
    #        (Work from lowest indexed vertex to higher one)
    #     2. Make an array of coordinates of new vertices along each panel
    #        (Work from near lowest indexed vertex, parallel to lowest indexed
    #         edge and across)
    #     3. Aggregate all the coordinates into a single array
    # C. Find all the new faces -- list the indices of coordinates in the
    #    coordinate array for the vertices of these new faces
    #     1. Try to think in terms of faces. Work firstly in order of
    #        icosahedral panels. Then start at the lowest indexed vertex and
    #        move parallel to lowest indexed edge.

    from math import sqrt
    phi = (1 + sqrt(5)) / 2
    # vertices of an icosahedron with an edge length of 2
    base_vertices = np.array([[-1, phi, 0],
                              [1, phi, 0],
                              [-1, -phi, 0],
                              [1, -phi, 0],
                              [0, -1, phi],
                              [0, 1, phi],
                              [0, -1, -phi],
                              [0, 1, -phi],
                              [phi, 0, -1],
                              [phi, 0, 1],
                              [-phi, 0, -1],
                              [-phi, 0, 1]],
                             dtype=np.double)

    # edges of the base icosahedron
    panel_edges = np.array([[0, 11],
                            [5, 11],
                            [0, 5],
                            [0, 1],
                            [1, 5],
                            [1, 7],
                            [0, 7],
                            [0, 10],
                            [10, 11],
                            [1, 9],
                            [5, 9],
                            [4, 5],
                            [4, 11],
                            [2, 10],
                            [2, 11],
                            [6, 7],
                            [6, 10],
                            [7, 10],
                            [1, 8],
                            [7, 8],
                            [3, 4],
                            [3, 9],
                            [4, 9],
                            [2, 3],
                            [2, 4],
                            [2, 6],
                            [3, 6],
                            [6, 8],
                            [3, 8],
                            [8, 9]], dtype=np.int32)

    # edges of the base icosahedron
    panel_edges = np.array([[0, 1],
                            [0, 5],
                            [0, 7],
                            [0, 10],
                            [0, 11],
                            [1, 5],
                            [1, 7],
                            [1, 8],
                            [1, 9],
                            [2, 3],
                            [2, 4],
                            [2, 6],
                            [2, 10],
                            [2, 11],
                            [3, 4],
                            [3, 6],
                            [3, 8],
                            [3, 9],
                            [4, 5],
                            [4, 9],
                            [4, 11],
                            [5, 9],
                            [5, 11],
                            [6, 7],
                            [6, 8],
                            [6, 10],
                            [7, 8],
                            [7, 10],
                            [8, 9],
                            [10, 11]], dtype=np.int32)

    # faces of the base icosahedron
    panels = np.array([[0, 5, 11],
                       [0, 1, 5],
                       [0, 1, 7],
                       [0, 7, 10],
                       [0, 10, 11],
                       [1, 5, 9],
                       [4, 5, 11],
                       [2, 10, 11],
                       [6, 7, 10],
                       [1, 7, 8],
                       [3, 4, 9],
                       [2, 3, 4],
                       [2, 3, 6],
                       [3, 6, 8],
                       [3, 8, 9],
                       [4, 5, 9],
                       [2, 4, 11],
                       [2, 6, 10],
                       [6, 7, 8],
                       [1, 8, 9]], dtype=np.int32)

    panels_to_edges = np.array([[1, 4, 22],
                                [0, 1, 5],
                                [0, 2, 6],
                                [2, 3, 27],
                                [3, 4, 29],
                                [5, 8, 21],
                                [18, 20, 22],
                                [12, 13, 29],
                                [23, 25, 27],
                                [6, 7, 26],
                                [14, 17, 19],
                                [9, 10, 14],
                                [9, 11, 15],
                                [15, 16, 24],
                                [16, 17, 28],
                                [18, 19, 21],
                                [10, 13, 20],
                                [11, 12, 25],
                                [23, 24, 26],
                                [7, 8, 28]], dtype=np.int32)

    for i, (face_edge, face) in enumerate(zip(panels_to_edges, panels)):
        edges_for_this_face = panel_edges[face_edge]
        for edge in edges_for_this_face:
            if edge[0] not in face:
                raise ValueError('Something has gone wrong with the faces')
            if edge[1] not in face:
                raise ValueError('Something has gone wrong with the faces')

    # 12 base vertices of icosahedron
    num_base_vertices = 12
    num_edge_vertices = 30 * (big_N - 1)
    num_face_vertices = int(20 * (big_N - 1) * (big_N - 2) / 2)
    total_num_vertices = num_base_vertices + num_edge_vertices + num_face_vertices

    # We can find number of faces per panel by adding together two triangle numbers
    num_faces_per_panel = int(big_N * (big_N + 1) / 2
                              + big_N * (big_N - 1) / 2)

    # This is the number of panels * triangle number of faces per icosahedral panel
    total_num_faces = 20 * num_faces_per_panel

    # ------------------------------------------------------------------------ #
    # Fill array with all vertex values
    # ------------------------------------------------------------------------ #

    # First find vertices on edges ------------------------------------------- #
    edge_vertices = np.zeros((num_edge_vertices, 3))

    edge_weights = np.linspace(0, 1, num=(big_N+1))

    # Loop through panel edges and add new vertices
    if big_N > 1:
        edge_vertex_counter = 0
        for i, edge in enumerate(panel_edges):
            # edge is a list of indices of its vertices
            x0 = base_vertices[edge[0]]
            x1 = base_vertices[edge[1]]

            # Loop through vertices associated with panel edge
            for j in range(big_N - 1):
                w = edge_weights[j+1]
                edge_vertices[edge_vertex_counter, :] = x0 + w * (x1 - x0)
                edge_vertex_counter += 1

    # Second find vertices on faces ------------------------------------------ #
    face_vertices = np.zeros((num_face_vertices, 3))

    if big_N > 2:
        face_vertex_counter = 0

        # Loop through panels
        for i, this_panel_edges in enumerate(panels_to_edges):

            # Work along lines parallel to lowest indexed edge
            for j in range(big_N - 2):

                # Therefore the points we want are on the 1st and 2nd indexed edges
                x0 = edge_vertices[this_panel_edges[1]*(big_N-1) + j]
                x1 = edge_vertices[this_panel_edges[2]*(big_N-1) + j]

                face_row_weights = np.linspace(0, 1, num=(big_N-j))

                for k in range(big_N - j - 2):
                    w = face_row_weights[k+1]
                    face_vertices[face_vertex_counter, :] = x0 + w * (x1 - x0)
                    face_vertex_counter += 1

    # Put all vertices together into array ----------------------------------- #
    all_vertices = np.zeros((total_num_vertices, 3))

    all_vertices[0:num_base_vertices] = base_vertices[:]
    all_vertices[num_base_vertices:num_base_vertices+num_edge_vertices] = edge_vertices[:]
    all_vertices[num_base_vertices+num_edge_vertices:] = face_vertices[:]

    # ------------------------------------------------------------------------ #
    # Fill array with all vertex values
    # ------------------------------------------------------------------------ #
    all_faces = np.zeros((total_num_faces, 3), dtype=np.int32)
    vertices_per_edge = int(num_edge_vertices / len(panel_edges))
    vertices_per_face = int(num_face_vertices / len(panels))

    if big_N > 1:

        # Loop over panels
        for i in range(len(panels)):

            # Find indices of vertices --------------------------------------- #

            # We need to find the indices in all_vertices of the vertices for this panel
            # We break this down into those associated with faces, edges and base vertices

            # The vertices of the base icosahedron are the original panel indices
            indices_of_base_vertices = panels[i]

            # The vertices on edges. Store these indices as a list of indices (for each edge)
            edge_indices = panels_to_edges[i]
            indices_of_edge_vertices = np.empty((3, vertices_per_edge), dtype=np.int32)

            for j, edge in enumerate(edge_indices):
                indices_of_edge_vertices[j, :] = range(num_base_vertices + edge * vertices_per_edge,
                                                       num_base_vertices + (edge+1) * vertices_per_edge)

            # Get indices of vertices associated with faces
            face_offset = num_base_vertices + num_edge_vertices
            indices_of_face_vertices = range(face_offset + i * vertices_per_face,
                                             face_offset + (i+1) * vertices_per_face)

            # Iterate through to obtain the vertices for each new face ------- #
            # If we are here then we have at least two cells per panel edge
            this_panel_faces = np.empty((num_faces_per_panel, 3), dtype=np.int32)

            # Start in the corner of the panel with the lowest indexed vertex
            # Loop in rows from lowest indexed edge towards highest indexed vertex
            # e.g. this would be the numbering of faces on a panel
            # ---------------------------- #
            # \       / \      / \       /
            #  \  0  /   \  1 /   \  2  /
            #   \   /  3  \  /  4  \   /
            #    \ /       \/       \ /
            #     \------- /\--------/
            #      \  5   /  \  6   /
            #       \    /    \    /
            #        \  /  7   \  /
            #         \/--------\/
            #          \       /
            #           \  8  /
            #            \   /
            #             \ /

            face_counter = 0
            for j in range(big_N):
                # Loop across row from edge[1] to edge[2]
                # First we do triangles oriented with panel
                # 0-----1
                #  \   /
                #   \ /
                #    2
                for k in range(big_N - j):

                    # First focus on vertex[0] and vertex[1]
                    if j == 0:
                        if k == 0:
                            this_panel_faces[face_counter, 0] = indices_of_base_vertices[0]
                        else:
                            this_panel_faces[face_counter, 0] = indices_of_edge_vertices[0, k-1]

                        if k == (big_N - j - 1):
                            this_panel_faces[face_counter, 1] = indices_of_base_vertices[1]
                        else:
                            this_panel_faces[face_counter, 1] = indices_of_edge_vertices[0, k]

                    else:
                        if k == 0:
                            this_panel_faces[face_counter, 0] = indices_of_edge_vertices[1, j-1]
                        else:
                            # Total number per face, subtract triangle number below row
                            face_index = vertices_per_face - int((big_N-j-1)*(big_N-j) / 2) + k - 1
                            this_panel_faces[face_counter, 0] = indices_of_face_vertices[face_index]

                        if k == (big_N - j - 1):
                            this_panel_faces[face_counter, 1] = indices_of_edge_vertices[2, j-1]
                        else:
                            # Total number per face, subtract triangle number below row
                            face_index = vertices_per_face - int((big_N-j-1)*(big_N-j) / 2) + k
                            this_panel_faces[face_counter, 1] = indices_of_face_vertices[face_index]

                    # Now do vertex[2]
                    if j == (big_N - 1):
                        this_panel_faces[face_counter, 2] = indices_of_base_vertices[2]

                    elif k == 0:
                        this_panel_faces[face_counter, 2] = indices_of_edge_vertices[1, j]
                    elif k == (big_N - j - 1):
                        this_panel_faces[face_counter, 2] = indices_of_edge_vertices[2, j]
                    else:
                        # Total number per face, subtract triangle number below row
                        face_index = vertices_per_face - int((big_N-j-2)*(big_N-j-1) / 2) + k - 1
                        this_panel_faces[face_counter, 2] = indices_of_face_vertices[face_index]

                    face_counter += 1

                if j < big_N - 1:
                    # Now do triangles oriented against panel
                    #    0
                    #   / \
                    #  /   \
                    # 2 ----1
                    for k in range(big_N - j - 1):

                        # vertex[0]
                        if j == 0:
                            this_panel_faces[face_counter, 0] = indices_of_edge_vertices[0, k]
                        else:
                            # Total number per face, subtract triangle number below row
                            face_index = vertices_per_face - int((big_N-j-1)*(big_N-j) / 2) + k
                            this_panel_faces[face_counter, 0] = indices_of_face_vertices[face_index]

                        # vertex[1]
                        if k == (big_N - j - 2):
                            this_panel_faces[face_counter, 1] = indices_of_edge_vertices[2, j]
                        else:
                            # Total number per face, subtract triangle number below row
                            face_index = vertices_per_face - int((big_N-j-2)*(big_N-j-1) / 2) + k
                            this_panel_faces[face_counter, 1] = indices_of_face_vertices[face_index]

                        # vertex[2]
                        if k == 0:
                            this_panel_faces[face_counter, 2] = indices_of_edge_vertices[1, j]
                        else:
                            # Total number per face, subtract triangle number below row
                            face_index = vertices_per_face - int((big_N-j-2)*(big_N-j-1) / 2) + k - 1
                            this_panel_faces[face_counter, 2] = indices_of_face_vertices[face_index]

                        face_counter += 1

            all_faces[i*num_faces_per_panel:(i+1)*num_faces_per_panel, :] = this_panel_faces[:, :]
    else:
        # If there is no refinement then we just use the original panels
        all_faces = panels

    num_occurrences = []
    for i in range(len(all_vertices)):
        num = np.count_nonzero(all_faces == i)
        num_occurrences.append(num)
        if num not in [5, 6]:
            raise ValueError('Num of times vertex %i is called in all_faces is %i' % (i, num))

    plex = mesh.plex_from_cell_list(2, all_faces, all_vertices, comm)

    coords = plex.getCoordinatesLocal().array.reshape(-1, 3)
    scale = (radius / np.linalg.norm(coords, axis=1)).reshape(-1, 1)
    coords *= scale
    m = mesh.Mesh(plex, dim=3, reorder=reorder, name=name, comm=comm,
                  distribution_parameters=distribution_parameters)
    if degree > 1:
        new_coords = function.Function(functionspace.VectorFunctionSpace(m, "CG", degree))
        new_coords.interpolate(ufl.SpatialCoordinate(m))
        # "push out" to sphere
        new_coords.dat.data[:] *= (radius / np.linalg.norm(new_coords.dat.data, axis=1)).reshape(-1, 1)
        m = mesh.Mesh(new_coords, name=name, comm=comm)
    m._radius = radius

    return m


def _cubedsphere_cells_and_coords(radius, cells_per_cube_edge):
    """Generate vertex and face lists for cubed sphere """
    # We build the mesh out of 6 panels of the cube
    # this allows to build the gnonomic cube transformation
    # which is defined separately for each panel

    # Start by making a grid of local coordinates which we use
    # to map to each panel of the cubed sphere under the gnonomic
    # transformation
    dtheta = 2*np.arctan(1.0) / cells_per_cube_edge
    a = 3.0**(-0.5)*radius
    theta = np.arange(np.arctan(-1.0), np.arctan(1.0)+dtheta, dtheta, dtype=np.double)
    x = a*np.tan(theta)
    Nx = x.size

    # Compute panel numberings for each panel
    # We use the following "flatpack" arrangement of panels
    #   3
    #  102
    #   4
    #   5

    # 0 is the bottom of the cube, 5 is the top.
    # All panels are numbered from left to right, top to bottom
    # according to this diagram.

    panel_numbering = np.zeros((6, Nx, Nx), dtype=np.int32)

    # Numbering for panel 0
    panel_numbering[0, :, :] = np.arange(Nx**2, dtype=np.int32).reshape(Nx, Nx)
    count = panel_numbering.max()+1

    # Numbering for panel 5
    panel_numbering[5, :, :] = count + np.arange(Nx**2, dtype=np.int32).reshape(Nx, Nx)
    count = panel_numbering.max()+1

    # Numbering for panel 4 - shares top edge with 0 and bottom edge
    #                         with 5
    # interior numbering
    panel_numbering[4, 1:-1, :] = count + np.arange(Nx*(Nx-2),
                                                    dtype=np.int32).reshape(Nx-2, Nx)

    # bottom edge
    panel_numbering[4, 0, :] = panel_numbering[5, -1, :]
    # top edge
    panel_numbering[4, -1, :] = panel_numbering[0, 0, :]
    count = panel_numbering.max()+1

    # Numbering for panel 3 - shares top edge with 5 and bottom edge
    #                         with 0
    # interior numbering
    panel_numbering[3, 1:-1, :] = count + np.arange(Nx*(Nx-2),
                                                    dtype=np.int32).reshape(Nx-2, Nx)
    # bottom edge
    panel_numbering[3, 0, :] = panel_numbering[0, -1, :]
    # top edge
    panel_numbering[3, -1, :] = panel_numbering[5, 0, :]
    count = panel_numbering.max()+1

    # Numbering for panel 1
    # interior numbering
    panel_numbering[1, 1:-1, 1:-1] = count + np.arange((Nx-2)**2,
                                                       dtype=np.int32).reshape(Nx-2, Nx-2)
    # left edge of 1 is left edge of 5 (inverted)
    panel_numbering[1, :, 0] = panel_numbering[5, ::-1, 0]
    # right edge of 1 is left edge of 0
    panel_numbering[1, :, -1] = panel_numbering[0, :, 0]
    # top edge (excluding vertices) of 1 is left edge of 3 (downwards)
    panel_numbering[1, -1, 1:-1] = panel_numbering[3, -2:0:-1, 0]
    # bottom edge (excluding vertices) of 1 is left edge of 4
    panel_numbering[1, 0, 1:-1] = panel_numbering[4, 1:-1, 0]
    count = panel_numbering.max()+1

    # Numbering for panel 2
    # interior numbering
    panel_numbering[2, 1:-1, 1:-1] = count + np.arange((Nx-2)**2,
                                                       dtype=np.int32).reshape(Nx-2, Nx-2)
    # left edge of 2 is right edge of 0
    panel_numbering[2, :, 0] = panel_numbering[0, :, -1]
    # right edge of 2 is right edge of 5 (inverted)
    panel_numbering[2, :, -1] = panel_numbering[5, ::-1, -1]
    # bottom edge (excluding vertices) of 2 is right edge of 4 (downwards)
    panel_numbering[2, 0, 1:-1] = panel_numbering[4, -2:0:-1, -1]
    # top edge (excluding vertices) of 2 is right edge of 3
    panel_numbering[2, -1, 1:-1] = panel_numbering[3, 1:-1, -1]
    count = panel_numbering.max()+1

    # That's the numbering done.

    # Set up an array for all of the mesh coordinates
    Npoints = panel_numbering.max()+1
    coords = np.zeros((Npoints, 3), dtype=np.double)
    lX, lY = np.meshgrid(x, x)
    lX.shape = (Nx**2,)
    lY.shape = (Nx**2,)
    r = (a**2 + lX**2 + lY**2)**0.5

    # Now we need to compute the gnonomic transformation
    # for each of the panels
    panel_numbering.shape = (6, Nx**2)

    def coordinates_on_panel(panel_num, X, Y, Z):
        I = panel_numbering[panel_num, :]
        coords[I, 0] = radius / r * X
        coords[I, 1] = radius / r * Y
        coords[I, 2] = radius / r * Z

    coordinates_on_panel(0, lX, lY, -a)
    coordinates_on_panel(1, -a, lY, -lX)
    coordinates_on_panel(2, a, lY, lX)
    coordinates_on_panel(3, lX, a, lY)
    coordinates_on_panel(4, lX, -a, -lY)
    coordinates_on_panel(5, lX, -lY, a)

    # Now we need to build the face numbering
    # in local coordinates
    vertex_numbers = np.arange(Nx**2, dtype=np.int32).reshape(Nx, Nx)
    local_faces = np.zeros(((Nx-1)**2, 4), dtype=np.int32)
    local_faces[:, 0] = vertex_numbers[:-1, :-1].reshape(-1)
    local_faces[:, 1] = vertex_numbers[1:, :-1].reshape(-1)
    local_faces[:, 2] = vertex_numbers[1:, 1:].reshape(-1)
    local_faces[:, 3] = vertex_numbers[:-1, 1:].reshape(-1)

    cells = panel_numbering[:, local_faces].reshape(-1, 4)
    return cells, coords


@PETSc.Log.EventDecorator()
def GeneralCubedSphereMesh(radius, num_cells_per_edge_of_panel, degree=1,
                           reorder=None, distribution_parameters=None,
                           comm=COMM_WORLD, name=mesh.DEFAULT_MESH_NAME):
    """
    Generate an cubed approximation to the surface of the sphere.

    Args:
        radius (float): The radius of the sphere to approximate.
    num_cells_per_edge_of_panel (int): number of cells per edge of each of
        the 6 panels of the cubed sphere (1 gives a cube).
    degree (int, optional): polynomial degree of coordinate space used to
        approximate the sphere. Defaults to 1, describing flat quadrilaterals.
    reorder: (bool, optional): optional flag indicating whether to reorder
        meshes for better cache locality. Defaults to False.
    distribution_parameters (dict, optional): a dictionary of options for
           parallel mesh distribution. Defaults to None.
    comm (communicator, optional): optional communicator to build the mesh
        on. Defaults to COMM_WORLD.
    name (str, optional): optional name to give to the mesh. Defaults to
        Firedrake's default mesh name.
    """
    if num_cells_per_edge_of_panel < 1 or num_cells_per_edge_of_panel % 1:
        raise RuntimeError("Number of cells per edge must be a positive integer")

    if degree < 1:
        raise ValueError("Mesh coordinate degree must be at least 1")

    cells, coords = _cubedsphere_cells_and_coords(radius, num_cells_per_edge_of_panel)
    plex = mesh.plex_from_cell_list(2, cells, coords, comm, mesh._generate_default_mesh_topology_name(name))

    m = mesh.Mesh(plex, dim=3, reorder=reorder, name=name, comm=comm,
                  distribution_parameters=distribution_parameters)

    if degree > 1:
        new_coords = function.Function(functionspace.VectorFunctionSpace(m, "Q", degree))
        new_coords.interpolate(ufl.SpatialCoordinate(m))
        # "push out" to sphere
        new_coords.dat.data[:] *= (radius / np.linalg.norm(new_coords.dat.data, axis=1)).reshape(-1, 1)
        m = mesh.Mesh(new_coords, name=name, comm=comm)
    m._radius = radius
    return m


def get_flat_latlon_mesh(mesh):
    """
    Construct a planar latitude-longitude mesh from a spherical mesh.

    Args:
        mesh (:class:`Mesh`): the mesh on which the simulation is performed.
    """
    coords_orig = mesh.coordinates
    coords_fs = coords_orig.function_space()

    if coords_fs.extruded:
        cell = mesh._base_mesh.ufl_cell().cellname()
        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_elt = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
    else:
        cell = mesh.ufl_cell().cellname()
        DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)
    coords_dg = Function(vec_DG1).interpolate(coords_orig)
    coords_latlon = Function(vec_DG1)
    shapes = {"nDOFs": vec_DG1.finat_element.space_dimension(), 'dim': 3}

    radius = np.min(np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2))
    # lat-lon 'x' = atan2(y, x)
    coords_latlon.dat.data[:, 0] = np.arctan2(coords_dg.dat.data[:, 1], coords_dg.dat.data[:, 0])
    # lat-lon 'y' = asin(z/sqrt(x^2 + y^2 + z^2))
    coords_latlon.dat.data[:, 1] = np.arcsin(coords_dg.dat.data[:, 2]/np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2))
    # our vertical coordinate is radius - the minimum radius
    coords_latlon.dat.data[:, 2] = np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2) - radius

# We need to ensure that all points in a cell are on the same side of the branch cut in longitude coords
# This kernel amends the longitude coords so that all longitudes in one cell are close together
    kernel = op2.Kernel("""
#define PI 3.141592653589793
#define TWO_PI 6.283185307179586
void splat_coords(double *coords) {{
    double max_diff = 0.0;
    double diff = 0.0;

    for (int i=0; i<{nDOFs}; i++) {{
        for (int j=0; j<{nDOFs}; j++) {{
            diff = coords[i*{dim}] - coords[j*{dim}];
            if (fabs(diff) > max_diff) {{
                max_diff = diff;
            }}
        }}
    }}

    if (max_diff > PI) {{
        for (int i=0; i<{nDOFs}; i++) {{
            if (coords[i*{dim}] < 0) {{
                coords[i*{dim}] += TWO_PI;
            }}
        }}
    }}
}}
""".format(**shapes), "splat_coords")

    op2.par_loop(kernel, coords_latlon.cell_set,
                 coords_latlon.dat(op2.RW, coords_latlon.cell_node_map()))
    return Mesh(coords_latlon)
