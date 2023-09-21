from firedrake import conditional, atan, SpatialCoordinate, le, ge, pi

def atan2_hack(mesh):
    x0, y0, _ = SpatialCoordinate(mesh)
    angle = atan(y0/x0)
    err_tol = 1e-12
    atan2 = conditional(ge(x0,err_tol), angle, 
                    conditional(le(y0,-err_tol), angle - pi, angle + pi))
    # Ensure vales at x== 0
    atan2_corrected = conditional(ge(x0,err_tol,), atan2, conditional(le(x0,-err_tol),
                            atan2,conditional(le(y0,-err_tol), -pi/2, pi/2) ) )  

    return atan2_corrected
