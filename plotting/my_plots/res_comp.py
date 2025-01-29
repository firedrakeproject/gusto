import xarray as xr
import functions as fcs
import matplotlib.pyplot as plt
import os
import numpy as np

files = {4:'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-300sols_tracer_tophat-80_ref-4',
        5:'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-300sols_tracer_tophat-80_ref-5'}

path = f'/data/home/sh1293/results'

fig, axs = plt.subplots(5,1, figsize=(24,20), sharex=True)

ls = ['--', '.']
cs = ['red', 'blue']
i=0

for lev in files:
    dt = (0.5)**(lev-4) * 450.
    ds = xr.open_dataset(f'{path}/{files[lev]}/regrid_output.nc')
    ds['sol'] = ds.time/88774
    tracer = ds.tracer
    q = ds.PotentialVorticity
    max_zonal_mean = fcs.max_zonal_mean(q)
    lat_thresh = max_zonal_mean
    lat_thresh_mean = lat_thresh.where(ds.sol>=100, drop=True).mean(dim='time')['max_lat']
    max_grad = fcs.max_merid_grad(q)
    # print(f'{lev}')
    # print(max_grad)
    lat_thresh_grad = max_grad.where(ds.sol>=100, drop=True).mean(dim='time')['max_grad_lat']
    pole_tracer, _ = fcs.tracer_integral(tracer, lat_thresh_mean, 'pole')
    total_tracer = fcs.total_tracer_integral(tracer)
    pole_tracer_frac = pole_tracer/total_tracer
    pole_tracer_grad, _ = fcs.tracer_integral(tracer, lat_thresh_grad, 'pole')
    pole_tracer_grad_frac = pole_tracer_grad/total_tracer
    edd_ens = fcs.scaled2_eddy_enstrophy(ds)
    q_edd_ens = edd_ens.PotentialVorticity_eddens
    condensing_fraction = fcs.condensing_area(ds)
    delta_q_inst = max_zonal_mean.max_val
    # if i==0:
    #     pole_tracer_0 = pole_tracer[0]
    #     q_edd_ens_0 = q_edd_ens[0]
    #     condensing_fraction_0 = condensing_fraction[0]
    #     delta_q_0 = delta_q[0]
    (pole_tracer_frac).plot(ax=axs[0], color=cs[i], label=f'Refinement level {lev}')
    (pole_tracer_grad_frac).plot(ax=axs[0], color=cs[i], linestyle='--')
    axs[0].set_ylabel('Poleward tracer fraction')
    axs[0].axvline(8877400, 0, 1, alpha=0.5, linestyle='--')
    axs[0].legend()
    (q_edd_ens).plot(ax=axs[1], color=cs[i])
    axs[1].set_ylabel('Eddy enstrophy')
    axs[1].axvline(8877400, 0, 1, alpha=0.5, linestyle='--')
    (condensing_fraction).plot(ax=axs[2], color=cs[i])
    axs[2].set_ylabel('Condensing fraction')
    axs[2].axvline(8877400, 0, 1, alpha=0.5, linestyle='--')
    (lat_thresh.max_lat).plot(ax=axs[3], color=cs[i])
    axs[3].set_ylabel('Lat of max zonal mean PV')
    axs[3].axvline(8877400, 0, 1, alpha=0.5, linestyle='--')
    (delta_q_inst).plot(ax=axs[4], color=cs[i])
    axs[4].set_ylabel('Delta q')
    i+=1
    # plt.legend()

plt.savefig(f'{path}/{files[4]}/Plots/resolution_comparison.pdf')
print(f'Plot made:\n{path}/{files[4]}/Plots/resolution_comparison.pdf')