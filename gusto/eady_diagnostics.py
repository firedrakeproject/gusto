from firedrake import SpatialCoordinate
from gusto.diagnostics import Energy
from gusto.forcing import exner


class KineticEnergyY(Energy):
    name = "KineticEnergyY"

    def compute(self, state):
        """
        Computes the kinetic energy of the y component
        """
        u = state.fields("u")
        energy = self.kinetic(u[1])
        return self.field(state.mesh).interpolate(energy)


class CompressibleKineticEnergyY(Energy):
    name = "CompressibleKineticEnergyY"

    def compute(self, state):
        """
        Computes the kinetic energy of the y component
        """
        u = state.fields("u")
        rho = state.fields("rho")
        energy = self.kinetic(u[1], rho)
        return self.field(state.mesh).interpolate(energy)


class EadyPotentialEnergy(Energy):
    name = "EadyPotentialEnergy"

    def compute(self, state):
        x, y, z = SpatialCoordinate(state.mesh)
        b = state.fields("b")
        bbar = state.fields("bbar")
        H = state.parameters.H
        potential = -(z-H/2)*(b-bbar)
        return self.field(state.mesh).interpolate(potential)


class CompressibleEadyPotentialEnergy(Energy):
    name = "CompressibleEadyPotentialEnergy"

    def compute(self, state):
        x, y, z = SpatialCoordinate(state.mesh)
        g = state.parameters.g
        cp = state.parameters.cp
        cv = state.parameters.cv
        Pi0 = state.parameters.Pi0

        rho = state.fields("rho")
        theta = state.fields("theta")
        Pi = exner(theta, rho, state)

        potential = rho*(g*z + cv*Pi*theta - cp*Pi0*theta)
        return self.field(state.mesh).interpolate(potential)
