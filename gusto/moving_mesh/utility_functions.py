from __future__ import absolute_import
import numpy as np


def spherical_logarithm(X0, X1, v, R):
    """
    Find vector function v such that X1 = exp(v)X0 on
    a sphere of radius R, centre the origin.
    """

    v.assign(X1 - X0)

    # v <- v - X0(v.X0/R^2); make v orthogonal to X0
    v.dat.data[:] -= X0.dat.data_ro*np.einsum('ij,ij->i', v.dat.data_ro, X0.dat.data_ro).reshape(-1, 1)/R**2

    # v <- theta*R*v-hat, where theta is the angle between X0 and X1
    v.dat.data[:] = np.arccos(np.einsum('ij,ij->i', X0.dat.data_ro, X1.dat.data_ro)/R**2).reshape(-1, 1)*R*v.dat.data_ro[:]/np.linalg.norm(v.dat.data_ro, axis=1).reshape(-1, 1)
