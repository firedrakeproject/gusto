"""
Plots for the Gusto demonstrating convergence of different time discretisation
"""
import matplotlib.pyplot as plt
import netCDF4 as nc
from numpy import log

def findtime():
    data = nc.Dataset('examples/Convergence/Data/Heun_scheme_dt=60/diagnostics.nc')     
    return data.variables['time'][:] 

def finderror(scheme, dt, degree=2):
    error = []
    for step in dt:
        if degree == 3:
            filep = f'examples/Convergence/Data/{scheme}_scheme_dt={step}_degree=3/diagnostics.nc'
        else:
            filep = f'examples/Convergence/Data/{scheme}_scheme_dt={step}/diagnostics.nc'
        
        data = nc.Dataset(filep)
        normalised_U2_error = data.groups['u_error']['l2'][:] / data.groups['u']['l2'][0]
        error.append(normalised_U2_error[-1])
    return error


time = findtime()
dt = [120, 110, 100, 90, 80, 70]
fig, ax = plt.subplots()
plt.xlabel('TimeStep')
plt.ylabel('Normalise L2 velocity Error')
for schemes in ['SSPRK3', 'RK4', 'Heun']: 
    plt.title(f'{schemes} Convergence Plot')
    plt.plot(dt, log(finderror(schemes, dt)), label=f'{schemes}')
    plt.show()
ax.set_xlim(max(dt), min(dt))

# Implicit midpoint
dt = [200, 250, 300, 350, 400, 450, 500]
fig, ax = plt.subplots()
plt.xlabel('TimeStep')
plt.ylabel('Normalise L2 velocity Error')
fig.suptitle('Convergence plot for Implicity Midpoint')
plt.plot(dt, log(finderror('ImplicitMidpoint', dt, 3)))
plt.show()
ax.set_xlim(max(dt), min(dt))