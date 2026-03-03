import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import functions as fcs

# file1 = 'single-new_gamma_plane-shift_fns_Bu1b1p5Rop2_l1dt250df1'
file1 = 'check_gamma_plane_step_trap'
# file2 = 'single-new_gamma_plane-default-no_trap_Bu1b1p5Rop2_l1dt250df1'
file2 = 'check_gamma_plane_no_trap'

Bu = 1
f0 = 2 * 1.74e-4
rm = 1e6
g = 24.79
phi0 = Bu * (f0*rm)**2
H = phi0/g

basepath = '/data/home/sh1293/results/jupiter_sw'

d1, times = fcs.make_structured(file1, 'PotentialVorticity')
d2, _ = fcs.make_structured(file2, 'PotentialVorticity')

# d2 += 2*1.74e-4/9770.391286809197

Lx = 7e7
Ly = Lx

ylen = len(d1.y)

d2splice = d2[155,:,0]
d1splice = d1[155,:,0]

# d2splice = d2[:,155,0]
# d1splice = d1[:,155,0]

# d1splice = d1[:,:,0].mean(dim='y')
# d2splice = d2[:,:,0].mean(dim='y')

# breakpoint()

x = d1splice.x

fig, ax = plt.subplots(1, 1, figsize = (10, 10/1.666))

ax.plot(d1splice.x, d1splice.values, color='red', label='file1-good')
ax.plot(d2splice.x, d2splice.values, color='blue', label='file2')
ax.hlines(f0/H, xmin=0, xmax=70000, linestyle='--')
plt.legend()

if not os.path.exists(f'{basepath}/{file1}/Plots'):
    os.makedirs(f'{basepath}/{file1}/Plots')                    

plt.savefig(f'{basepath}/{file1}/Plots/pv_comparison.pdf')
print(f'Plot made:\n{basepath}/{file1}/Plots/pv_comparison.pdf')

# breakpoint()
