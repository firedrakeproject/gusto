import functions as fcs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import pdb
import scipy.special as scpspc

filepath = 'new_single_fplane_Bu2b1p5Rop2_l500dt250df30'

accurate = False

But = filepath.split('Bu')[1].split('b')[0]
try:
    Bui = float(But.split('p')[0])
    Bud = float(But.split('p')[1])*10**-len(But.split('p')[1])
except IndexError:
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

plot_dir = f'/data/home/sh1293/results/jupiter_sw/{filepath}/Plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

pv, times = fcs.make_structured(filepath, 'PotentialVorticity')
H_model, _ = fcs.make_structured(filepath, 'D')
spacing = pv.x[-1]/len(pv.x)
ypole = np.median(pv.y)
pvcs = pv.where(pv.y==ypole, drop=True).squeeze()
xpole = np.median(pv.x)
pvcs_centre = pvcs.where(pv.x>=xpole-40*spacing, drop=True).where(pv.x<=xpole+40*spacing, drop=True)
initial = pvcs_centre[:,0]
final = pvcs_centre[:,-1]

dx = 7e7/(1e3*256)
initialgrid = initial.where(initial.x%dx==0, drop=True)
finalgrid = final.where(final.x%dx==0, drop=True)

def analytic_pv(x):
    r = abs(x-xpole)*1e3
    num = 1+Ro*(2-(r/rm)**b)*np.exp(1/b*(1-(r/rm)**b))
    dem = 1-Ro/Bu*np.exp(1/b)*b**(2/b-1)*scpspc.gammaincc(2/b, 1/b*(r/rm)**b)*scpspc.gamma(2/b)
    pv = f0/H*num/dem
    return pv

x_centre = pvcs_centre.x
analytic = analytic_pv(x_centre)
analyticgrid = analytic.where(analytic.x%dx==0, drop=True)

if accurate:
    accurate_end_pv = xr.open_dataarray(f'/data/home/sh1293/results/jupiter_sw/{filepath}/end_pv.nc')
    accurate_end_pv['x'] = accurate_end_pv.x/1e3

fig, ax = plt.subplots(1,1, figsize=(10,10/1.666))
initial_plot = (initialgrid-f0/H).plot(ax=ax, label='Initial', linestyle='--', marker='.')
final_plot = (finalgrid-f0/H).plot(ax=ax, label=f'Final, {(times[-1]/t_day):.2f} days', linestyle=':', marker='.')
analytic_plot = (analyticgrid-f0/H).plot(ax=ax, label='Analytic', linestyle='-')
if accurate:
    accurate_final_plot = accurate_end_pv.plot(ax=ax, label=f'Accurate final, {(times[-1]/t_day):.2f} days', color='fuchsia', marker='.', linestyle='-')
plt.legend()
plt.title(f'{filepath}')


plt.savefig(f'{plot_dir}/pv_profiles.pdf')
print(f'Plot made:\n{plot_dir}/pv_profiles.pdf')