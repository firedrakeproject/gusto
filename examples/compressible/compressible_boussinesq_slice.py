from firedrake import *
from gusto.rexi import *
from gusto import *

dt = 900
tmax = 3000.

# set up mesh and function spaces

L = 3.0e5  # Domain length
H = 1.0e4  # Height position of the model top
nlayers = 10
columns = 150

m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
domain = Domain(mesh, dt, "CG", 1)

# set up equations
parameters = CompressibleBoussinesqParameters(cs=300)

class MyCompressibleBoussinesqEquations(PrognosticEquationSet):

    def __init__(self, domain, parameters, f=None,
                 no_normal_flow_bc_ids=None):

        field_names = ['u', 'p', 'b']

        super().__init__(field_names, domain,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids)

        Nsq = Constant(parameters.N**2)
        csq = Constant(parameters.c**2)
        k = domain.k

        w, phi, gamma = self.tests
        u, p, b = split(self.X)

        mass_form = self.generate_mass_terms()

        pressure_gradient_form = subject(prognostic(-inner(p, div(w))*dx, "u"),
                                         self.X)

        gravity_form = subject(prognostic(-b*inner(k, w)*dx, "u"), self.X)

        self.residual = mass_form + pressure_gradient_form + gravity_form

        if f is not None:
            coriolis_form = subject(prognostic(f*inner(cross(k, u), w)*dx, "u"),
                                    self.X)
            self.residual += coriolis_form

        self.residual += subject(prognostic(csq*inner(div(u), phi)*dx, "p"),
                                 self.X)

        self.residual += subject(prognostic(Nsq*inner(dot(u, k), gamma)*dx,
                                            "b"),
                                 self.X)

eqns = CompressibleBoussinesqEquations(domain, parameters)

# I/O
diagnostic_fields = [CourantNumber(), Perturbation('b'), Perturbation('p')]
output = OutputParameters(dirname='boussinesq_slice', dumpfreq=1)
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# set up timestepper
lines_parameters = {
    "snes_converged_reason": None,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm",
    "assembled_pc_star_sub_sub_pc_factor_reuse_ordering": None,
    "assembled_pc_star_sub_sub_pc_factor_reuse_fill": None,
    "assembled_pc_star_sub_sub_pc_factor_fill": 1.2,
}
# rexi = Rexi(eqns, RexiParameters(), solver_parameters=lines_parameters)

# stepper = Timestepper(eqns, ImplicitMidpoint(domain), io)

transported_fields = [ImplicitMidpoint(domain, "u"),
                      SSPRK3(domain, "p"),
                      SSPRK3(domain, "b", options=SUPGOptions())]

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields)

# interpolate initial conditions
b0 = stepper.fields("b")
x, z = SpatialCoordinate(mesh)

b_0 = Constant(0.01)
A = 5000
xc = 0.5 * L
bref = Function(b0.function_space()).interpolate(-parameters.N**2 * z)
bpert = b_0 * sin(pi*z/H) / (1 + (x-xc)**2/A**2)
b0.interpolate(bref + bpert)

# initialise hydrostatic pressure
p0 = stepper.fields("p")
pref = Function(p0.function_space())
incompressible_hydrostatic_balance(eqns, bref, pref)
p0.assign(pref)

stepper.set_reference_profiles([('p', pref),
                                ('b', bref)])
                                   
stepper.run(t=0, tmax=tmax)
# rexi_output = File("results/compressible_rexi.pvd")

# U_in = Function(eqns.function_space)
# Uexpl = Function(eqns.function_space)
# u, p, b = U_in.split()
# b.interpolate(bexpr)
# rexi_output.write(u, p, b)

# rexi = Rexi(eqns, RexiParameters(M=128), solver_parameters=lines_parameters)
# Uexpl.assign(rexi.solve(U_in, tmax))

# uexpl, pexpl, bexpl = Uexpl.split()
# u.assign(uexpl)
# p.assign(pexpl)
# b.assign(bexpl)
# rexi_output.write(u, p, b)
