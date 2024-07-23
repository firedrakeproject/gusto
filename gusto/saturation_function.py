from firedrake import exp
_all_ = ["compute_saturation"]

def compute_saturation(q0, H, g, D, b, B=None):
    if B is None:
        sat_expr = q0*H/(D) * exp(20*(1-b/g))
    else:
        sat_expr = q0*H/(D+B) * exp(20*(1-b/g))
    return sat_expr
