from firedrake import (Constant, Function, dx, TestFunction, TrialFunction, as_vector,
                       inner, dot, cross, div, jump, LinearVariationalProblem,
                       LinearVariationalSolver, SpatialCoordinate, tan, FacetNormal,
                       sqrt, avg, dS_v)

from gusto.core.coord_transforms import lonlatr_from_xyz
from gusto.diagnostics.diagnostics import DiagnosticField
from gusto.equations.compressible_euler_equations import CompressibleEulerEquations
from gusto.equations.thermodynamics import exner_pressure

class GeostrophicImbalance(DiagnosticField):
    """Geostrophic imbalance diagnostic field."""
    name = "GeostrophicImbalance"

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        # Work out required fields
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['rho', 'theta', 'u']
            self.equations = equations
            self.parameters = equations.parameters
        else:
            raise NotImplementedError(f'Geostrophic Imbalance not implemented for {type(equations)}')
        super().__init__(space=space, method=method, required_fields=required_fields)

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        Vcurl = domain.spaces("HCurl")
        u = state_fields('u')
        rho = state_fields('rho')
        theta = state_fields('theta')
 
        exner = exner_pressure(self.parameters, rho, theta)
        cp = Constant(self.parameters.cp)
        n = FacetNormal(domain.mesh)
        # TODO: Generilise this for cases that aren't solid body rotation case 
        omega = Constant(7.292e-5)
        Omega = as_vector((0., 0., omega))
        k = domain.k

        F = TrialFunction(Vcurl)
        w = TestFunction(Vcurl)

        imbalance = Function(Vcurl)
        a = inner(w, F)*dx

        L = (- cp*div(theta*w)*exner*dx + cp*jump(theta*w, n)*avg(exner)*dS_v # exner pressure grad discretisation
             - inner(w, cross(2*Omega, u))*dx # coriolis
             + inner(w, dot(k, cross(2*Omega, u) )*k)*dx #vertical part of coriolis
             + cp*div((theta*k)*dot(k,w))*exner*dx  # removes vertical part of the pressure divergence
             - cp*jump((theta*k)*dot(k,w), n)*avg(exner)*dS_v) # removes vertical part of pressure jump condition
            

        bcs = self.equations.bcs['u']

        imbalanceproblem = LinearVariationalProblem(a, L, imbalance, bcs=bcs)
        self.imbalance_solver = LinearVariationalSolver(imbalanceproblem)
        self.expr = dot(imbalance, domain.k)
        super().setup(domain, state_fields)

    def compute(self):
        """Compute and return the diagnostic field from the current state.
        """
        self.imbalance_solver.solve()
        super().compute()


class SolidBodyImbalance(DiagnosticField):
    """Solid Body imbalance diagnostic field."""
    name = "SolidBodyImbalance"

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        # Work out required fields
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['rho', 'theta', 'u']
            self.equations = equations
            self.parameters = equations.parameters
        else:
            raise NotImplementedError(f'Geostrophic Imbalance not implemented for {type(equations)}')
        super().__init__(space=space, method=method, required_fields=required_fields)

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        Vu = domain.spaces("HDiv")
        u = state_fields('u')
        rho = state_fields('rho')
        theta = state_fields('theta')
 
        exner = tde.exner_pressure(self.parameters, rho, theta)
        cp = Constant(self.parameters.cp)
        n = FacetNormal(domain.mesh)
        # TODO: Generilise this for cases that aren't solid body rotation case 
        omega = Constant(7.292e-5)
        Omega = as_vector((0., 0., omega))

        # generating the spherical co-ords, and spherical components of velocity
        x, y, z = SpatialCoordinate(domain.mesh)
        x_hat = Constant(as_vector([1.0, 0.0, 0.0]))
        y_hat = Constant(as_vector([0.0, 1.0, 0.0]))
        z_hat = Constant(as_vector([0.0, 0.0, 1.0]))
        R = sqrt(x**2 + y**2)  # distance from z axis
        r = sqrt(x**2 + y**2 + z**2)  # distance from origin
        lambda_hat = (x * y_hat - y * x_hat) / R
        lon_dot = inner(u, lambda_hat)
        phi_hat = (-x*z/R * x_hat - y*z/R * y_hat + R * z_hat) / r
        lat_dot = inner(u, phi_hat)
        r_hat = (x * x_hat + y * y_hat + z * z_hat) / r 
        r_dot = inner(u, r_hat)     
        mesh = domain.mesh
        k=domain.k
        #HACK loweing the quadrature manually
        dx_low_quadrature = dx(degree=3)
        lat = latlon_coords(mesh)[0]
        F = TrialFunction(Vu)
        w = TestFunction(Vu)

        imbalance = Function(Vu)
        a = inner(w, F)*dx

        #TODO Segmentaion error somehwere here

        L = (- cp*div((theta)*w)*exner*dx + cp*jump((theta)*w, n)*avg(exner)*dS_v # exner pressure grad discretisation
             - inner(w, cross(2*Omega, u))*dx # coriolis
             + inner(w, dot(k, cross(2*Omega, u) )*k)*dx # vertical part of coriolis
             + cp*div((theta*k)*dot(k,w))*exner*dx  # vertical part of the pressure divergence
             - cp*jump((theta*k)*dot(k,w), n)*avg(exner)*dS_v # vertical part of pressure jump condition
            # BUG it is these terms arising from the non-linear that are the problem
             - (lat_dot * lon_dot * tan(lat) / r)*inner(w, lambda_hat)*dx_low_quadrature 
             + (lon_dot * r_dot / r)*dot(w, lambda_hat)*dx_low_quadrature # lambda component of non linear term            
             + (lat_dot**2 * tan(lat) / r)*inner(w, phi_hat)*dx_low_quadrature   # phi component 1
             + (lat_dot * r_dot / r)*inner(w, phi_hat)*dx_low_quadrature # phi component 1
            )
           

        bcs = self.equations.bcs['u']

        imbalanceproblem = LinearVariationalProblem(a, L, imbalance, bcs=bcs)
        self.imbalance_solver = LinearVariationalSolver(imbalanceproblem)
        self.expr = dot(imbalance, domain.k)
        super().setup(domain, state_fields)
        

    def compute(self):
        """Compute and return the diagnostic field from the current state.
        """
        self.imbalance_solver.solve()
        super().compute()