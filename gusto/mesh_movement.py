from __future__ import absolute_import
from firedrake import FunctionSpace

def spherical_logarithm(X0,X1,v,R,dt)
    """
    Find vector function v such that X1 = exp(dt*v)X0 on
    a sphere of radius R, centre the origin.
    """

    v.assign(X1-X0)
    alpha = -inner(v, X0)/inner(X0, X0)
    v.interpolate( v - inner(v,X0)/inner(X0,X0)*X0 )
    normX0 = sqrt(inner(X0, X0))
    normX1 = sqrt(inner(X1, X1))
    normv = sqrt(inner(v, v))
    theta = acos( inner(X0, X1)/normX0/normX1 )
    v.interpolate( theta*R*v/dt/normv )
