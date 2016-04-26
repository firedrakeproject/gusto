from firedrake import assemble, dot, dx
from math import sqrt


class Diagnostics(object):
    pass


class ShallowWaterDiagnostics(Diagnostics):

    def l2(self, f):
        return sqrt(assemble(dot(f, f)*dx))
