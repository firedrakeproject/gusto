"""
The moist rising bubble test from Bryan & Fritsch (2002), in a cloudy
atmosphere.

The rise of the thermal is fueled by latent heating from condensation.
"""

from gusto import *
from gusto import thermodynamics
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, pi, sqrt, exp,
                       NonlinearVariationalProblem, FunctionSpace,
                       NonlinearVariationalSolver, TestFunction, dx,
                       TrialFunction, Function, VectorFunctionSpace,
                       LinearVariationalProblem, LinearVariationalSolver,
                       errornorm, norm, plot)
import sys
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #
dts = [  2.0, 1.5, 1.2, 1.0, 0.8,  0.5, 0.25, 0.125, 0.125/2.0]
nts=[ 60, 80, 100, 120, 150, 240, 480, 960, 1920 ]
dts = [  1.0, 0.8,  0.5, 0.25, 0.125, 0.125/2.0]
nts=[  120, 150, 240, 480, 960, 1920 ]
#tmaxs=[30, 30, 30, 30]
L = 120.0
deltax = 2.0
u_max = 1.0
cfl = 0.2

schemes = [ "SSPRK3", "RK4", "BE", "tr_bdf2","bdf2"]
#schemes = ["tr_bdf2"]

g1=(2.0 -sqrt(2.0))
# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #


error_tr_bdf2=[]
error_bdf2=[]
error_fe=[]
error_be=[]
error_rk4=[]
error_ssprk3=[]
error_heun=[]
for scheme in schemes:
    for i in range(0,len(dts)):
        deltax=dts[i]/cfl
        #cfl = dts[i]/deltax
        tmax = nts[i]*dts[i]
        print('cfl',cfl)
        print('dx',deltax)
        # Domain
        nx = int(L/deltax)
        print('nx', nx)
        mesh = PeriodicIntervalMesh(nx, L)
        degree = 2
        x =SpatialCoordinate(mesh)[0]
        domain = Domain(mesh, dts[i], "CG", degree)

        Vf = domain.spaces("DG")
        Vint = FunctionSpace(mesh, "CG", 5)
        Vu = VectorFunctionSpace(mesh, "CG", degree)

        # Equation
        eqn = AdvectionEquation(domain, Vf, field_name="f", Vu=Vu)

        # I/O
        ndumps = 20
        dirname = "%s_linear_transport_dt%s" % (scheme,dts[i])
        dumpfreq = int(tmax / (ndumps*dts[i]))
        output = OutputParameters(dirname=dirname,
                                dumpfreq=dumpfreq)
        io = IO(domain, output)
        if (scheme =="tr_bdf2"):
            transport_scheme = TR_BDF2(domain,gamma=g1)
        elif (scheme =="bdf2"):
            transport_scheme = BDF2(domain)
        elif(scheme =="FE"):
            transport_scheme = ForwardEuler(domain)
        elif(scheme =="BE"):
            transport_scheme = BackwardEuler(domain)
        elif(scheme =="RK4"):
            transport_scheme = RK4(domain)
        elif(scheme =="SSPRK3"):
            transport_scheme = SSPRK3(domain)
        elif(scheme =="Heun"):
            transport_scheme = Heun(domain)
        timestepper = PrescribedTransport(eqn, transport_scheme, io)

        # define f0
        xc = L / 2.0
        ftemp= Function(Vint).interpolate(
            exp(-(x-xc)**2/20.0))
        f0 =  Function(Vf).project(ftemp)

        # Initial conditions
        timestepper.fields("f").interpolate(f0)
        timestepper.fields("u").project(as_vector([u_max]))
        # ---------------------------------------------------------------------------- #
        # Run
        # ---------------------------------------------------------------------------- #
        timestepper.run(t=0, tmax=tmax)
        x_true=xc+dts[i]*nts[i]
        if (x_true>L):
            x_true = x_true -L

        print("x_true", x_true)
        print("xc", xc)
        true_temp= Function(Vint).interpolate(
            exp(-(x-x_true)**2/20.0))
        true_sol =  Function(Vf).project(true_temp)
        error_norm = errornorm(f0, timestepper.fields("f"),mesh=mesh)
        norm_f = norm(f0, mesh=mesh)
        print('error_norm', error_norm)
        print('error', error_norm/norm_f)

        error=error_norm/norm_f
        if (scheme =="tr_bdf2"):
            error_tr_bdf2.append(error)
        elif (scheme =="bdf2"):
            error_bdf2.append(error)
        elif(scheme =="FE"):
            error_fe.append(error)
        elif(scheme =="BE"):
            error_be.append(error)
        elif(scheme =="RK4"):
            error_rk4.append(error)
        elif(scheme =="SSPRK3"):
            error_ssprk3.append(error)
        elif(scheme =="Heun"):
            error_heun.append(error)
plt.loglog(dts,error_tr_bdf2,'b',label='TR_BDF2')
plt.loglog(dts,error_bdf2,'r',label='BDF2')
plt.loglog(dts,error_be,'g',label='Backward Euler')
#plt.loglog(dts,error_fe,'c',label='Forward Euler')
plt.loglog(dts,error_rk4,'m',label='RK4')
plt.loglog(dts,error_ssprk3,'y',label='SSPRK3')
#plt.loglog(dts,error_heun,'k',label='Heun')
plt.loglog(dts[0:3],np.power(dts[0:3],4),linestyle='--',label=r'$\mathcal{O}(\Delta t^{cats})$'.replace('cats',str(4)))
plt.loglog(dts[0:3],np.power(dts[0:3],3),linestyle='--',label=r'$\mathcal{O}(\Delta t^{cats})$'.replace('cats',str(3)))
plt.loglog(dts[0:3],np.power(dts[0:3],2),linestyle='--',label=r'$\mathcal{O}(\Delta t^{cats})$'.replace('cats',str(2)))
plt.loglog(dts[0:3],np.power(dts[0:3],1),linestyle='--',label=r'$\mathcal{O}(\Delta t^{cats})$'.replace('cats',str(1)))
plt.loglog(dts[0:3],np.power(dts[0:3],0),linestyle='--',label=r'$\mathcal{O}(\Delta t^{cats})$'.replace('cats',str(0)))
plt.legend()
plt.title("Linear Advection Convergece")
figname = "short_convergence_cfl_%s_degree_%s.png" % (cfl, degree)
plt.savefig(figname)
plt.show()