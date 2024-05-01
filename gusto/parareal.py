from firedrake import Function
from gusto.fields import Fields
from firedrake.petsc import PETSc


class PararealFields(object):

    def __init__(self, equation, nlevels):
        levels = [str(n) for n in range(nlevels+1)]
        self.add_fields(equation, levels)

    def add_fields(self, equation, levels):
        if levels is None:
            levels = self.levels
        for level in levels:
            try:
                x = getattr(self, level)
                x.add_field(equation.field_name, equation.function_space)
            except AttributeError:
                setattr(self, level, Fields(equation))

    def __call__(self, n):
        return getattr(self, str(n))


class Parareal(object):

    def __init__(self, domain, coarse_scheme, fine_scheme, nG, nF,
                 n_intervals, max_its):

        assert coarse_scheme.nlevels == 1
        assert fine_scheme.nlevels == 1
        self.nlevels = 1

        self.coarse_scheme = coarse_scheme
        self.coarse_scheme.dt.assign(domain.dt/n_intervals/nG)
        self.fine_scheme = fine_scheme
        self.fine_scheme.dt.assign(domain.dt/n_intervals/nG)
        self.nG = nG
        self.nF = nF
        self.n_intervals = n_intervals
        self.max_its = max_its

    def setup(self, equation, apply_bcs=True, ensemble=None, *active_labels):
        self.coarse_scheme.fixed_subcycles = self.nG
        self.coarse_scheme.setup(equation, apply_bcs, *active_labels)
        self.fine_scheme.fixed_subcycles = self.nF
        self.fine_scheme.setup(equation, apply_bcs, *active_labels)
        self.x = PararealFields(equation, self.n_intervals)
        self.xF = PararealFields(equation, self.n_intervals)
        self.xn = Function(equation.function_space)
        self.xGk = PararealFields(equation, self.n_intervals)
        self.xGkm1 = PararealFields(equation, self.n_intervals)
        self.xFn = Function(equation.function_space)
        self.xFnp1 = Function(equation.function_space)
        self.name = equation.field_name
        self.ensemble = ensemble
        if ensemble is not None:
            assert ensemble.ensemble_comm.size == self.n_intervals + 1

    def setup_transporting_velocity(self, uadv):
        self.coarse_scheme.setup_transporting_velocity(uadv)
        self.fine_scheme.setup_transporting_velocity(uadv)

    def apply(self, x_out, x_in):

        if self.ensemble.ensemble_comm.rank == 0:
            self.xn.assign(x_in)
            x0 = self.x(0)(self.name)
            x0.assign(x_in)
            xF0 = self.xF(0)(self.name)
            xF0.assign(x_in)

            # compute first guess from coarse scheme
            for n in range(self.n_intervals):
                PETSc.Sys.Print("computing first coarse guess for interval: ", n)
                # apply coarse scheme and save data as initial conditions for fine
                xGnp1 = self.xGkm1(n+1)(self.name)
                self.coarse_scheme.apply(xGnp1, self.xn)
                xnp1 = self.x(n+1)(self.name)
                xnp1.assign(xGnp1)
                self.xn.assign(xnp1)

            for k in range(self.max_its):

                # apply fine scheme in each interval using previously
                # calculated coarse data
                for n in range(self.n_intervals):
                    xn = self.x(n)(self.name)
                    xFnp1 = self.xF(n+1)(self.name)
                    self.ensemble.send(xn, n+1)
                    self.ensemble.recv(xFnp1, n+1)

                # compute correction
                for n in range(k, self.n_intervals):
                    xn = self.x(n)(self.name)
                    xGk = self.xGk(n+1)(self.name)
                    # compute new coarse guess
                    self.coarse_scheme.apply(xGk, xn)
                    xnp1 = self.x(n+1)(self.name)
                    xGkm1 = self.xGkm1(n+1)(self.name)
                    xFnp1 = self.xF(n+1)(self.name)
                    xnp1.assign(xGk - xGkm1 + xFnp1)
                    xGkm1.assign(xGk)

        else:
            for k in range(self.max_its):
                self.ensemble.recv(self.xFn, 0)
                self.fine_scheme.apply(self.xFnp1, self.xFn)
                self.ensemble.send(self.xFnp1, 0)

        x_out.assign(xnp1)
