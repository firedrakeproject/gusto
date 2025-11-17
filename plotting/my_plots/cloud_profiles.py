import functions as fcs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import pdb
import scipy.special as scpspc
import sys

def extract_centre(filepath, field_name):
    field, times = fcs.make_structured(filepath, field_name)
    ypole = np.median(field.y)
    xpole = np.median(field.x)
    field_centre = field.interp(x=xpole, y=ypole)
    return field_centre, times


if len(sys.argv) != 2:
    print('Wrong number of arguments')
    print(len(sys.argv))
    sys.exit(1)

filepath = sys.argv[1]
# filepath = 'beta3900q01em2xi1em1_Bu1b1p5Rop2_l1dt250df1'

Bu = int(filepath.split('Bu')[1].split('b')[0])
print(f'Bu={Bu}')

q0 = 1e-2
# Bu = 10
Omega = 1.74e-4
f0 = 2 * Omega
rm = 1e6
phi0 = Bu * (f0*rm)**2
g = 24.79
H = phi0/g

plot_dir = f'/data/home/sh1293/results/jupiter_sw/{filepath}/Plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

pv, times = extract_centre(filepath, 'PotentialVorticity')
rv, _ = extract_centre(filepath, 'RelativeVorticity')
D, _ = extract_centre(filepath, 'D')
cloud, _ = extract_centre(filepath, 'cloud_water')
relhum, _ = extract_centre(filepath, 'RelativeHumidity')
wvap, _ = extract_centre(filepath, 'water_vapour')

qsat = q0*H/D

fig, ax = plt.subplots(4,2, figsize=(20,15), sharex=True)
pvplot = (pv).plot(label='PV', ax=ax[0,0])
# rvplot = (rv).plot(label='RV', ax=ax[0,0])
ax[0,0].legend()
Dplot = (D/H).plot(label='D/H', ax=ax[0,1])
ax[0,1].legend()
rvplot = (rv).plot(label='RV', ax=ax[1,0])
ax[1,0].legend()
cloudplot = (cloud).plot(label='cloud', ax=ax[2,0])
ax[2,0].legend()
relhumpolt = (relhum/100).plot(label='Relative humidity/100', ax=ax[2,1])
ax[2,1].legend()
wvapplot = (wvap).plot(label='Water vapour', ax=ax[3,0])
ax[3,0].legend()
qsatplot = (qsat).plot(label='qsat', ax=ax[3,1])
ax[3,1].legend()
# fakeplot = (wvap/qsat).plot(label='relhum?', ax=ax[1,1], linestyle='--')
plt.ylabel('')
plt.legend()
# plt.title(f'{filepath}')


plt.savefig(f'{plot_dir}/cloud_profiles.pdf')
print(f'Plot made:\n{plot_dir}/cloud_profiles.pdf')