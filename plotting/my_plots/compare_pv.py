import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import functions as fcs

file1 = 'single-new_gamma_plane-shift_fns_Bu1b1p5Rop2_l1dt250df1'
file2 = 'single_Bu2b1p5Rop2_l4dt250df1'

basepath = '/data/home/sh1293/results/jupiter_sw'

d1, times = fcs.make_structured(file1, 'PotentialVorticity')
d2, _ = fcs.make_structured(file2, 'PotentialVorticity')

d2 += 2*1.74e-4/9770.391286809197

Lx = 7e7
Ly = Lx

ylen = len(d1.y)

# d1splice = d1[int(ylen/2),:,0]
# d2splice = d2[int(ylen/2),:,0]

d1splice = d1[:,:,0].mean(dim='y')
d2splice = d2[:,:,0].mean(dim='y')

fig, ax = plt.subplots(1, 1, figsize = (10, 10/1.666))

ax.plot(d1splice.x, d1splice.values, color='red', label='new')
ax.plot(d2splice.x, d2splice.values, color='blue', label='old')
plt.legend()

if not os.path.exists(f'{basepath}/{file1}/Plots'):
    os.makedirs(f'{basepath}/{file1}/Plots')                    

plt.savefig(f'{basepath}/{file1}/Plots/pv_comparison.pdf')
print(f'Plot made:\n{basepath}/{file1}/Plots/pv_comparison.pdf')

# breakpoint()
