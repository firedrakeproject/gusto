from __future__ import absolute_import
from firedrake import FunctionSpace

def spherical_logarithm(X0,X1,v,dt)
    """
    Find vector function v such that X1 = exp(dt*v)X0 on
    a sphere of radius R, centre the origin.
    """
    
    v.assign(X1-X0)
    alpha = -inner(v, X0)/inner(X0, X0)
    v.interpolate( v - inner(v,X0)/inner(X0,X0)*X0 )
    R = sqrt(inner(X0, X0)
    normv = sqrt(inner(v, v))
    theta = acos( inner(X0, X1)/R**2 )
    v.interpolate( theta*R*v/dt/normv )
