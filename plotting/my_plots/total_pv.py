import functions as fcs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

filepath = 'new_single_fplane_Bu2b1p5Rop2_l500dt250df30'

plot_dir = f'/data/home/sh1293/results/jupiter_sw/{filepath}/Plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

But = filepath.split('Bu')[1].split('b')[0]
try:
    Bui = float(But.split('p')[0])
    Bud = float(But.split('p')[1])*10**-len(But.split('p')[1])
except ValueError:
    Bu = float(But)
try:
    bi = float(filepath.split('b')[1].split('p')[0])
    bd = float(filepath.split('b')[1].split('Ro')[0].split('p')[1])*10**-len(filepath.split('b')[1].split('Ro')[0].split('p')[1])
    b = bi+bd
except ValueError:
    b = float(filepath.split('b')[1].split('Ro')[0])
Ro = float(filepath.split('Rop')[1].split('_')[0])*10**-len(filepath.split('Rop')[1].split('_')[0])

g = 24.79
Omega = 1.74e-4
R = 71.4e6
f0 = 2 * Omega        # Planetary vorticity
rm = 1e6              # Radius of vortex (m)
vm = Ro * f0 * rm     # Calculate speed with Ro
phi0 = Bu * (f0*rm)**2
H = phi0/g
t_day = 2*np.pi/Omega

pv, times = fcs.make_structured(filepath, 'PotentialVorticity')
total_pv = (pv-f0/H).integrate('x').integrate('y')

fig, ax = plt.subplots(1,1, figsize=(10,10/1.666))
total_plot = total_pv.plot(ax=ax, label='Total domain PV - planetary PV')
plt.legend()

plt.savefig(f'{plot_dir}/totalpv_evolution.pdf')
print(f'Plot made:\n{plot_dir}/totalpv_evolution.pdf')