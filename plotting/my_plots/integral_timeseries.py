import matplotlib.pyplot as plt
import numpy as np
import functions as fcs
import os
import xarray as xr
from gusto import (
    rtheta_from_lonlat
)

def radial_crop(field_structured, rlim, L):
    X, Y = xr.broadcast(field_structured.x, field_structured.y)
    mask = (X-L/2)**2 + (Y-L/2)**2 <= rlim**2
    mask = mask.broadcast_like(field_structured)
    field_cut = field_structured.where(mask, drop=True)
    # breakpoint()
    return field_cut

def field_domain_integral(file, field_name, rlim, L):
    field_structured, times = fcs.make_structured(file, field_name, folder=folder)
    field_cut = radial_crop(field_structured, rlim, L).fillna(0)
    field_integral = field_cut.integrate(coord='x').integrate(coord='y')*1e6
    # breakpoint()
    return field_integral, times

def field_domain_L2_error_integral(file, field_name, rlim, L):
    field_structured, times = fcs.make_structured(file, field_name, folder=folder)
    field_cut = radial_crop(field_structured, rlim, L)
    field_mean = field_cut.mean(dim='x').mean(dim='y')
    field_error = (field_cut - field_mean).fillna(0)
    field_error_sq = field_error**2
    L2e = np.sqrt(field_error_sq.integrate(coord='x').integrate(coord='y')*1e6)
    return L2e, times

def field_spatial_mean(file, field_name, rlim, L):
    field_structured, times = fcs.make_structured(file, field_name, folder=folder)
    field_cut = radial_crop(field_structured, rlim, L)
    field_mean = field_cut.mean(dim='x').mean(dim='y')
    return field_mean, times

def field_spatial_max(file, field_name, rlim, L):
    field_structured, times = fcs.make_structured(file, field_name, folder=folder)
    field_cut = radial_crop(field_structured, rlim, L)
    field_max = field_cut.max(dim='x').max(dim='y')
    # breakpoint()
    return field_max, times

folder = 'jupiter_sw'

limited_area = False

file = 'single-step_trap_radt5beta390000q01em2xi1em1_Bu10b1p5Rop2_l100dt250df10'

lat_lim = 70*np.pi/180.
lat_lim_full = 10*np.pi/180.
R = 71.4e6
rlim, _ = rtheta_from_lonlat(0, lat_lim, R=R)
rlim /= 1e3
L = 7e4
rlim_full, _ = rtheta_from_lonlat(0, lat_lim_full, R=R)
rlim_full /= 1e3

if not limited_area:
    rlim = rlim_full


path = f'/data/home/sh1293/results/{folder}'

fig, axs = plt.subplots(2,2, figsize=(12,12))

# pv, times = field_domain_integral(file, 'PotentialVorticity', rlim, L)
# avlPE, _ = field_domain_integral(file, 'ShallowWaterAvailablePotentialEnergy', rlim, L)
# PE, _ = field_domain_integral(file, 'ShallowWaterPotentialEnergy', rlim, L)
q, _ = field_domain_integral(file, 'water_vapour', rlim, L)
# qE = q * 2.4e6
RH, _ = field_domain_integral(file, 'RelativeHumidity', rlim, L)
RHmax, _ = field_spatial_max(file, 'RelativeHumidity', rlim, L)
if folder == 'vp20_moist_jupiter':
    rain, _ = field_domain_integral(file, 'rain', rlim, L)
    rain_diff = xr.concat([rain.isel(time=0), rain.diff(dim='time')], dim='time')
else:
    cloud, _ = field_domain_integral(file, 'cloud_water', rlim, L)
D_L2, _ = field_domain_L2_error_integral(file, 'D', rlim, L)

# pv.plot(ax=axs[0,0], color=colour, label=f'{beta}', alpha=alpha, linestyle='-' if i==0 else '--' if i==1 else ':')
# (avlPE+qE).plot(ax=axs[0,1], color=colour, label=f'{beta} wet', alpha=alpha)
# (avlPE).plot(ax=axs[0,1], color=colour, label=f'{beta} dry', linestyle='--', alpha=alpha)
q.plot(ax=axs[0,0])
(RHmax).plot(ax=axs[0,1])
if folder == 'vp20_moist_jupiter':
    rain_diff.plot(ax=axs[1,0])
else:
    cloud.plot(ax=axs[1,0])
D_L2.plot(ax=axs[1,1])




axs[0,0].legend()
axs[0,1].legend()
axs[1,0].legend()
axs[1,1].legend()

axs[0,1].set_yscale('log')
axs[1,1].set_yscale('log')

# axs[0,0].set_title('Potential Vorticity integral')

# breakpoint()
# axs[2,0].set_title('Available PE')

if not limited_area:
    extra_name = f'_full'
else:
    extra_name = ''

if not os.path.exists(f'{path}/{file}/Plots'):
    os.makedirs(f'{path}/{file}/Plots')

plt.savefig(f'{path}/{file}/Plots/integral_timeseries{extra_name}.pdf')
print(f'Plot made:\n{path}/{file}/Plots/integral_timeseries{extra_name}.pdf')