from firedrake import assemble, dot, dx
from math import sqrt


class Diagnostics(object):

    def __init__(self, *fields):

        self.fields = list(fields)

    def register(self, *fields):

        fset = set(self.fields)
        for f in fields:
            if f not in fset:
                self.fields.append(f)

    @staticmethod
    def l2(f):
        return sqrt(assemble(dot(f, f)*dx))
