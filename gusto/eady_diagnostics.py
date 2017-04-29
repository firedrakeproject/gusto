from firedrake import FunctionSpace, Function, Expression, \
    SpatialCoordinate
from gusto.diagnostics import DiagnosticField, KineticEnergy, \
    CompressibleKineticEnergy
from gusto.forcing import exner


class KineticEnergyV(DiagnosticField):
    name = "KineticEnergyV"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

    def compute(self, state):
        u = state.fields("u")
        kineticv = 0.5*u[1]*u[1]
        return self.field(state.mesh).interpolate(kineticv)


class CompressibleKineticEnergyV(DiagnosticField):
    name = "CompressibleKineticEnergyV"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

    def compute(self, state):
        u = state.fields("u")
        rho = state.fields("rho")
        rhobar = state.fields("rhobar")
        kineticv = 0.5*(rho-rhobar)*u[1]*u[1]
        return self.field(state.mesh).interpolate(kineticv)


class EadyPotentialEnergy(DiagnosticField):
    name = "EadyPotentialEnergy"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

    def compute(self, state):
        x, y, z = SpatialCoordinate(state.mesh)
        b = state.fields("b")
        bbar = state.fields("bbar")
        H = state.parameters.H
        potential = -(z-H/2)*(b-bbar)
        return self.field(state.mesh).interpolate(potential)


class CompressibleEadyPotentialEnergy(DiagnosticField):
    name = "CompressibleEadyPotentialEnergy"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

    def compute(self, state):
        x, y, z = SpatialCoordinate(state.mesh)
        g = state.parameters.g
        cp = state.parameters.cp
        cv = state.parameters.cv
        pi0 = state.parameters.pi0

        rho = state.fields("rho")
        rhobar = state.fields("rhobar")
        rho_p = rho-rhobar

        theta = state.fields("theta")
        thetabar = state.fields("thetabar")
        theta_p = theta-thetabar

        pi = exner(theta, rho, state)
        pibar = exner(thetabar, rhobar, state)
        pi_p = pi-pibar

        potential = rho_p*(- g*z + cv*pi_p*theta_p
                           - cp*pi0*theta_p)
        return self.field(state.mesh).interpolate(potential)


class EadyTotalEnergy(DiagnosticField):
    name = "EadyTotalEnergy"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

    def compute(self, state):
        kinetic = KineticEnergy()
        potential = EadyPotentialEnergy()
        total = kinetic.compute(state) + potential.compute(state)
        return self.field(state.mesh).interpolate(total)


class CompressibleEadyTotalEnergy(DiagnosticField):
    name = "CompressibleEadyTotalEnergy"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

    def compute(self, state):
        kinetic = CompressibleKineticEnergy()
        potential = CompressibleEadyPotentialEnergy()
        total = kinetic.compute(state) + potential.compute(state)
        return self.field(state.mesh).interpolate(total)
