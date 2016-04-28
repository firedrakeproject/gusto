from firedrake import assemble, dot, dx
from math import sqrt


class Diagnostics(object):

    def __init__(self, fieldlist=None):

        if fieldlist is not None:
            self.fields = fieldlist
        else:
            self.fields = []

    def register(self, fieldlist):
        self.fields += fieldlist

    @staticmethod
    def l2(f):
        return sqrt(assemble(dot(f, f)*dx))
