from firedrake import assemble, dot, dx, FunctionSpace, Function, TestFunction, sqrt


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


class DiagnosticFields(object):

    def __init__(self, state, diagnostic_field_dict):

        self.diagnostic_field_dict = diagnostic_field_dict
        if 'Courant' in diagnostic_field_dict.keys():
            DG0 = FunctionSpace(state.mesh, "DG", 0)
            self.diagnostic_field_dict['Courant'] = Function(DG0, name='Courant')
            phiC = TestFunction(DG0)
            self.Area = assemble(phiC*dx)

    def Courant(self, state):

        u = state.field_dict['u']
        dt = state.timestepping.dt
        self.diagnostic_field_dict['Courant'].project(sqrt(dot(u,u))/sqrt(self.Area)*dt)
