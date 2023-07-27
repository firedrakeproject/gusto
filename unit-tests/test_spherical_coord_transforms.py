"""
Test the formulae in spherical_py

The strategy is to check that some obvious coordinates go to what we expect.
"""
import numpy as np
from gusto.spherical_coord_transforms import *

tol = 1e-12


def test_xyz_and_lonlatr():

    # Consider the following sets of coordinates:
    # (r, lon, lat)  <--> (x, y, z)
    # (2,0,pi/2)     <--> (0, 0, 2)
    # (0.5,pi,0)     <--> (-0.5, 0, 0)
    # (10,-pi/2,0)   <--> (0,-10, 0)
    # (0,0,0)        <--> (0, 0, 0)

    llr_coords = [[0.0, np.pi/2, 2.0],
                  [np.pi, 0.0, 0.5],
                  [-np.pi/2, 0.0, 10],
                  [0.0, 0.0, 0.0]]

    xyz_coords = [[0.0, 0.0, 2.0],
                  [-0.5, 0.0, 0.0],
                  [0.0, -10.0, 0.0],
                  [0.0, 0.0, 0.0]]

    for i, (llr, xyz) in enumerate(zip(llr_coords, xyz_coords)):
        new_llr = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
        new_xyz = xyz_from_lonlatr(llr[0], llr[1], llr[2])

        llr_correct = ((abs(new_llr[0] - llr[0]) < tol)
                        and (abs(new_llr[1] - llr[1]) < tol)
                        and (abs(new_llr[2] - llr[2]) < tol))

        assert llr_correct, ("""
                             phir coordinates not correct, got (%.2e %.2e %.2e)
                             when expecting (%.2e %.2e %.2e)
                             """ % (new_llr[0], new_llr[1], new_llr[2],
                                    llr[0], llr[1], llr[2]))

        xyz_correct = ((abs(new_xyz[0] - xyz[0]) < tol)
                       and (abs(new_xyz[1] - xyz[1]) < tol)
                       and (abs(new_xyz[2] - xyz[2]) < tol))

        assert xyz_correct, ("""
                            xyz coordinates not correct, got (%.2e %.2e %.2e)
                            when expecting (%.2e %.2e %.2e)
                            """ % (new_xyz[0], new_xyz[1], new_xyz[2],
                                   xyz[0], xyz[1], xyz[2]))


def test_xyz_and_lonlatr_vectors():

    # Consider the following vectors:
    # (r,lon,lat) components  <--> (x,y,z) components at (x,y,z) or (r,lon,lat)
    # (10,-6,0.5)             <--> (10,-6,0.5)        at (5,0,0) or (5,0,0)
    # (0.7,3,1.2)             <--> (3,-0.7,1.2)       at (0,-0.5,0) or (0.5,-pi/2,0)
    # (2,0,5)                 <--> (5,0,-2)           at (0,0,-15) or (15,0,-pi/2)

    llr_coords = [[0.0, 0.0, 5.0],
                  [-np.pi/2, 0.0, 0.5],
                  [0.0, -np.pi/2, 15.0]]

    xyz_coords = [[5.0, 0.0, 0.0],
                  [0.0, -0.5, 0.0],
                  [0.0, 0.0, -15.0]]

    llr_vectors = [[-6.0, 0.5, 10.0],
                   [3.0, 1.2, 0.7],
                   [0.0, 5.0, 2.0]]

    xyz_vectors = [[10.0, -6.0, 0.5],
                   [3.0, -0.7, 1.2],
                   [5.0, 0.0, -2.0]]

    for i, (llr, xyz, llr_comp, xyz_comp) in enumerate(zip(llr_coords, xyz_coords,
                                                           llr_vectors, xyz_vectors)):

        new_llr_comp = lonlatr_vector_from_xyz(xyz_comp, xyz)
        new_xyz_comp = xyz_vector_from_lonlatr(llr_comp, llr)

        llr_correct = ((abs(new_llr_comp[0] - llr_comp[0]) < tol)
                        and (abs(new_llr_comp[1] - llr_comp[1]) < tol)
                        and (abs(new_llr_comp[2] - llr_comp[2]) < tol))

        assert llr_correct, ("""
                             lonlatr components not correct, got (%.2e %.2e %.2e)
                             when expecting (%.2e %.2e %.2e)
                             """ % (new_llr_comp[0], new_llr_comp[1], new_llr_comp[2],
                                     llr_comp[0], llr_comp[1], llr_comp[2]))

        xyz_correct = ((abs(new_xyz_comp[0] - xyz_comp[0]) < tol)
                       and (abs(new_xyz_comp[1] - xyz_comp[1]) < tol)
                       and (abs(new_xyz_comp[2] - xyz_comp[2]) < tol))

        assert xyz_correct, ("""
                             xyz components not correct, got (%.2e %.2e %.2e)
                             when expecting (%.2e %.2e %.2e)
                             """ % (new_xyz_comp[0], new_xyz_comp[1], new_xyz_comp[2],
                                    xyz_comp[0], xyz_comp[1], xyz_comp[2]))