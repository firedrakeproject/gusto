from firedrake import assemble, dot, dx
from math import sqrt


class Diagnostics(object):

    def __init__(self):
        self.fields = []

    def register(self, fieldlist):
        self.fields += fieldlist


class ShallowWaterDiagnostics(Diagnostics):

    @staticmethod
    def l2(f):
        return sqrt(assemble(dot(f, f)*dx))
