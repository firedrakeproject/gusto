import numpy as np


__all__ = ["spherical_logarithm"]


def spherical_logarithm(X0, X1, v, R):
    """
    Find vector function v such that X1 = exp(v)X0 on
    a sphere of radius R, centre the origin.
    """

    v.assign(X1 - X0)

    # v <- v - X0(v.X0/R^2); make v orthogonal to X0
    v.dat.data[:] -= X0.dat.data_ro*np.einsum('ij,ij->i', v.dat.data_ro, X0.dat.data_ro).reshape(-1, 1)/R**2

    # v <- theta*R*v-hat, where theta is the angle between X0 and X1
    # fmin(theta, 1.0) is used to avoid silly floating point errors
    # fmax(|v|, 1e-16*R) is used to avoid division by zero
    v.dat.data[:] = np.arccos(np.fmin(np.einsum('ij,ij->i', X0.dat.data_ro, X1.dat.data_ro)/R**2, 1.0)).reshape(-1, 1)*R*v.dat.data_ro[:]/np.fmax(np.linalg.norm(v.dat.data_ro, axis=1), R*1e-16).reshape(-1, 1)
