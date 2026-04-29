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

short = True

rad = True

Bu10 = False

if folder == 'vp20_moist_jupiter':
    files_short = ['single-step_trap-qg_tophat_cD1em3gamma3900q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10',
                   'single-step_trap-qg_tophat_cD1em3gamma39000q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10',
                   'single-step_trap-qg_tophat_cD1em3gamma390000q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10']
    files_rad = ['single-step_trap-qg_tophat_radt5cD1em3gamma3900q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10',
                 'single-step_trap-qg_tophat_radt5cD1em3gamma39000q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10',
                 'single-step_trap-qg_tophat_radt5cD1em3gamma390000q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10']
    files = ['single-step_trap-qg_tophat_radt5cD1em3gamma3900q05em1xi1em1_Bu1b1p5Rop2_l200dt250df30',
             'single-step_trap-qg_tophat_radt5cD1em3gamma39000q05em1xi1em1_Bu1b1p5Rop2_l200dt250df30', 
            'single-step_trap-qg_tophat_radt5cD1em3gamma390000q05em1xi1em1_Bu1b1p5Rop2_l200dt250df30']
elif folder == 'jupiter_sw':
    files_short = ['single-step_trap_beta3900q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10',
             'single-step_trap_beta39000q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10',
             'single-step_trap_beta390000q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10']
    files_rad = ['single-step_trap_radt5beta3900q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10',
                 'single-step_trap_radt5beta39000q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10',
                 'single-step_trap_radt5beta390000q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10']
    files = ['single-step_trap_beta3900q01em2xi1em1_Bu1b1p5Rop2_l200dt250df30',
             'single-step_trap_beta39000q01em2xi1em1_Bu1b1p5Rop2_l200dt250df30',
             'single-step_trap_beta390000q01em2xi1em1_Bu1b1p5Rop2_l200dt250df30']
    files_Bu10 = ['single-step_trap_radt5beta3900q01em2xi1em1_Bu10b1p5Rop2_l10dt250df10',
                  'single-step_trap_radt5beta39000q01em2xi1em1_Bu10b1p5Rop2_l10dt250df10',
                  'single-step_trap_radt5beta390000q01em2xi1em1_Bu10b1p5Rop2_l10dt250df10']

if short:
    files = files_short
if rad:
    files = files_rad
if Bu10:
    files = files_Bu10
if rad and not short:
    raise ValueError("Incorrect combination: 'rad' must be used with 'short'")
if Bu10 and not rad:
    raise ValueError("Bu10 must be with rad")
if Bu10 and not short:
    raise ValueError("Bu10 must be with short")

if len(files)==3:
    betas = [3900, 39000, 390000]
    colours = ['aqua', 'dodgerblue', 'midnightblue']
    alphas = [1, 0.7, 0.55]
else:
    betas = [3900, 39000]#, 390000]
    colours = ['aqua', 'dodgerblue']#, 'midnightblue']
    alphas = [1, 0.7]#, 0.55] 

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
for i in range(len(files)):
    file = files[i]
    beta = betas[i]
    colour = colours[i]
    alpha = alphas[i]
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
    q.plot(ax=axs[0,0], color=colour, label=f'{beta}', alpha=alpha)
    (RHmax).plot(ax=axs[0,1], color=colour, label=f'{beta}', alpha=alpha)
    if folder == 'vp20_moist_jupiter':
        rain_diff.plot(ax=axs[1,0], color=colour, label=f'{beta}', alpha=alpha)
    else:
        cloud.plot(ax=axs[1,0], color=colour, label=f'{beta}', alpha=alpha)
    D_L2.plot(ax=axs[1,1], color=colour, label=f'{beta}', alpha=alpha)




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
if short:
    extra_name = f'{extra_name}_short'
if rad:
    extra_name = f'{extra_name}_rad'
if Bu10:
    extra_name = f'{extra_name}_Bu10'

plt.savefig(f'{path}/integral_timeseries{extra_name}.pdf')
print(f'Plot made:\n{path}/integral_timeseries{extra_name}.pdf')