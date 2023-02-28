from firedrake import *
from gusto.rexi import *
from gusto import *

dt = 900
tmax = 900

# set up mesh and function spaces
a_ref = 6.37122e6               # Radius of the Earth (m)
X = 125.0                       # Reduced-size Earth reduction factor
a = a_ref/X                     # Scaled radius of planet (m)
refinements = 4
nlayers = 10
H = 1.0e4

m = CubedSphereMesh(radius=a, refinement_level=refinements, degree=2)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers,
                    extrusion_type="radial")

domain = Domain(mesh, dt, "RTCF", 1)

# set up equations
class CompressibleBoussinesqParameters(Configuration):
    """
    Physical parameters for the compressible Boussinesq equations
    """
    f = 1.e-4  # Coriolis paramater
    N = 1.e-2  # Brunt-Vaisala frequency
    c = 300  # sound speed

parameters = CompressibleBoussinesqParameters()

class CompressibleBoussinesqEquations(PrognosticEquationSet):

    def __init__(self, domain, parameters, Omega=None,
                 no_normal_flow_bc_ids=None):

        field_names = ['u', 'p', 'b']

        super().__init__(field_names, domain,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids)

        f = Constant(parameters.f)
        Nsq = Constant(parameters.N**2)
        csq = Constant(parameters.c**2)
        k = domain.k

        w, phi, gamma = self.tests
        u, p, b = split(self.X)

        mass_form = self.generate_mass_terms()

        pressure_gradient_form = subject(prognostic(-inner(p, div(w))*dx, "u"),
                                         self.X)

        gravity_form = subject(prognostic(-b*inner(k, w)*dx, "u"), self.X)

        coriolis_form = subject(prognostic(f*inner(cross(k, u), w)*dx, "u"),
                                self.X)

        self.residual = mass_form + pressure_gradient_form + gravity_form + coriolis_form

        self.residual += subject(prognostic(csq*inner(div(u), phi)*dx, "p"),
                                 self.X)

        self.residual += subject(prognostic(Nsq*inner(dot(u, k), gamma)*dx,
                                            "b"),
                                 self.X)

eqns = CompressibleBoussinesqEquations(domain, parameters)

# I/O
output = OutputParameters(dirname='compressible_boussinesq_sphere',
                          dumpfreq=1)
io = IO(domain, output)

# set up timestepper
stepper = Timestepper(eqns, ExponentialEuler(domain, Rexi(eqns, RexiParameters())), io)

# interpolate initial conditions
b0 = stepper.fields("b")
x = SpatialCoordinate(mesh)

# Create polar coordinates:
# Since we use a CG1 field, this is constant on layers
W_Q1 = FunctionSpace(mesh, "CG", 1)
z_expr = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - a
z = Function(W_Q1).interpolate(z_expr)
th, lm = latlon_coords(mesh)
#lat_expr = asin(real(x[2])/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))
#lat = Function(W_Q1).interpolate(lat_expr)
#lon = Function(W_Q1).interpolate(atan_2(real(x[1]), real(x[0])))
lat = Function(W_Q1).interpolate(th)
lon = Function(W_Q1).interpolate(lm)

d = 5000.0                      # Width parameter for Theta'
lamda_c = 1.05*pi/3.0            # Longitudinal centerpoint of Theta'
phi_c = 0.0                     # Latitudinal centerpoint of Theta' (equator)
deltaTheta = 1.0                # Maximum amplitude of Theta' (K)
L_z = 20000.0                   # Vertical wave length of the Theta' perturb.

sin_tmp = sin(lat) * sin(phi_c)
cos_tmp = cos(lat) * cos(phi_c)
rr = a*acos(sin_tmp + cos_tmp*cos(lon-lamda_c))
r = conditional(rr>0, rr, 0)
s = (d**2)/(d**2 + r**2)
theta_pert = deltaTheta*s*sin(2*pi*z/L_z)
b0.interpolate(conditional(r<4.5*d, theta_pert, 0))

stepper.run(t=0, tmax=tmax)
