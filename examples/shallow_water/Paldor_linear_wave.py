
# set up mesh

# set up output parameters

# set up physical parameters for the shallow water equations

# set up state

# set up equations
eqns = LinearShallowWaterEquations(state, "BDM", 1, fexpr=fexpr)

# interpolate initial conditions

# set up timestepper
stepper = Timestepper(state, ((eqns, RK4(state)),))

stepper.run(t=0, tmax=tmax)
