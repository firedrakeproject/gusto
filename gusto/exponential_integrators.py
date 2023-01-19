class ExponentialEuler(object):

    def __init__(self, state, method):
        self.nlevels = 1
        self.dt = state.dt
        self.compute_exponential = method

    def setup(self, eqn, ubar):
        pass

    def apply(self, x_out, x_in):
        x_out.assign(self.compute_exponential.solve(x_in, self.dt))
