import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import functions as fcs
import pdb
import os

# 'strip' or 'hat'
tracer = 'strip'

if tracer == 'hat':
    file1 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4'
    file4 = 'Relax_to_pole_and_CO2/annular_vortex_mars_55-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4'
    file5 = 'Relax_to_pole_and_CO2/annular_vortex_mars_65-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4'
    file6 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4_pvmax-1-4'
    file7 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4_pvmax-1-8'
    file8 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--2-0_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4'
    file9 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--3-0_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4'
    file10 = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4'

elif tracer == 'strip':
    file1 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file4 = 'Relax_to_pole_and_CO2/annular_vortex_mars_55-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file5 = 'Relax_to_pole_and_CO2/annular_vortex_mars_65-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file6 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4_pvmax-1-4'
    file7 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4_pvmax-1-8'
    file8 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--2-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file9 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--3-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file10 = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'

fulldss = {1:file1, 4:file4, 5:file5, 6:file6, 7:file7, 8:file8, 9:file9}
anndss = {10:file10}

fullcolours = {1:'black', 2:'orange', 4:'tomato', 5:'red', 6:'deepskyblue', 7:'blue', 8:'violet', 9:'magenta'}
anncolours = {10:'grey', 11:'gold', 13:'darksalmon', 14:'maroon', 15:'darkturquoise', 16:'midnightblue'}

fullnames = {1:r'$C_F$', 4:r'$\phi_F^-$', 5:r'$\phi_F^+$', 6:r'$PV_F^-$', 7:r'$PV_F^+$'}
annnames = {10:r'$C_A$'}

path = '/data/home/sh1293/results'

fig, axs = plt.subplots(3,3, figsize=(30,20))
axs111 = axs[1,1].twinx()

def plotting(i, relax_type, path=path):
    if relax_type == 'full':
        ds = xr.open_dataset(f'{path}/{fulldss[i]}/regrid_output.nc')
        m = 'o'
        l = '-'
        l1 = ':'
        colour = fullcolours[i]
        contour_path = f'{path}/{fulldss[i]}/contours.nc'
        name = fullnames[i]
    elif relax_type == 'ann':
        ds = xr.open_dataset(f'{path}/{anndss[i]}/regrid_output.nc')
        m = 'x'
        l = '--'
        l1 = '-.'
        colour = anncolours[i]
        contour_path = f'{path}/{anndss[i]}/contours.nc'
        name = annnames[i]
    else:
        print('Invalid relaxation type')
        exit
    q = ds.PotentialVorticity
    tracer = ds.tracer_rs
    q_late_mean = q.where(q.time>=100*88774., drop=True).mean(dim='lon').mean(dim='time')
    t_late = tracer.where(tracer.time>=270*88774., drop=True).mean(dim='time').mean(dim='lon')
    ds_eddens = fcs.scaled2_eddy_enstrophy(ds)
    q_eddens = ds_eddens.PotentialVorticity_eddens
    max_zonal_mean = fcs.max_zonal_mean(q)
    max_merid_grad = fcs.max_merid_grad(q)
    dq = max_zonal_mean.max_val-max_zonal_mean.pole_val
    dq_mean = dq.where(dq.time>=100*88774., drop=True).mean(dim='time')
    dphi = 90.-max_zonal_mean.max_lat
    dphi_mean = dphi.where(dphi.time>=100*88774., drop=True).mean(dim='time')
    q_eddens_mean = q_eddens.where(q_eddens.time>=100*88774., drop=True).mean(dim='time')
    lat_thresh = max_zonal_mean.where(ds.time>=100*88774, drop=True).mean(dim='time')['max_lat']
    lat_thresh_grad = max_merid_grad.where(ds.time>=100*88774, drop=True).mean(dim='time')['max_grad_lat']
    pole_tracer, _ = fcs.tracer_integral(tracer, lat_thresh, 'pole')
    pole_tracer_grad, _ = fcs.tracer_integral(tracer, lat_thresh_grad, 'pole')
    total_tracer = fcs.total_tracer_integral(tracer)
    pole_tracer_frac = pole_tracer/total_tracer
    pole_tracer_grad_frac = pole_tracer_grad/total_tracer
    pole_tracer_weighted = pole_tracer/(1-np.cos(dphi_mean*np.pi/180))
    # if os.path.exists(contour_path):
    #     print('Contours netcdf4 exists')
    #     contour_length = xr.open_dataarray(contour_path)
    # else:
    #     print('Contours netcdf4 does not exist')
    #     contour_length = fcs.contour_length(tracer, 15e-3)
    #     contour_length.to_netcdf(contour_path)
    # contour_length_mean = contour_length.mean(dim='time')
    fft = fcs.fft(q, lat_thresh, 10, 100)
    fft_ave = fft.where(fft.time>=100*88774., drop=True).mean(dim='time')
    fft_ave_cut = fft_ave.where(fft_ave.freq_lon>0, drop=True).where(fft_ave.freq_lon<=10, drop=True)
    condensing_fraction = fcs.condensing_area(ds)
    condensing_fraction_mean = condensing_fraction.where(condensing_fraction.time>=100*88774., drop=True).mean(dim='time')
    plot0 = axs[0,0].scatter(x=dq_mean, y=q_eddens_mean, label=name, marker=m, color=colour)
    axs[0,0].text(dq_mean+0.03e-9, q_eddens_mean, name)
    axs[0,0].set_ylabel('Eddy enstrophy')
    axs[0,0].set_xlabel('dq')
    plot1 = axs[1,0].scatter(x=dphi_mean, y=q_eddens_mean, label=name, marker=m, color=colour)
    axs[1,0].text(dphi_mean+0.15, q_eddens_mean, name)
    axs[1,0].set_ylabel('Eddy enstrophy')
    axs[1,0].set_xlabel('dphi')
    # plot2 = axs[2,0].scatter(x=dq_mean/dphi_mean, y=q_eddens_mean, label=i, marker=m, color=colour)
    # axs[2,0].text(dq_mean/dphi_mean+0.02e-10, q_eddens_mean, i)
    # axs[2,0].set_ylabel('Eddy enstrophy')
    # axs[2,0].set_xlabel('dq/dphi')
    plot2 = axs[2,0].scatter(x=dphi_mean, y=pole_tracer_weighted[-1], color=colour, marker=m)
    axs[2,0].text(dphi_mean+0.15, pole_tracer_weighted[-1], name)
    axs[2,0].set_ylabel('Poleward area weighted tracer')
    axs[2,0].set_xlabel('dphi')
    tracer_plot = pole_tracer_weighted.where(pole_tracer_weighted.time>=100*88774., drop=True).plot(ax=axs[0,2], color=colour, label=f'{name} max, {lat_thresh.values:.2f}')
    # tracer_grad_plot = pole_tracer_grad_frac.plot(ax=axs[0,2], linestyle='--', color=colour, label=f'{name} grad, {lat_thresh_grad.values:.2f}')
    axs[0,2].set_ylabel('Poleward area weighted tracer')
    axs[0,2].set_xlabel('time')
    axs[0,2].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    # axs[0,1].scatter(x=q_eddens_mean, y=pole_tracer_frac[-1], color=colour, marker=m, label=f'{i} max')
    # axs[0,1].scatter(x=q_eddens_mean, y=pole_tracer_grad_frac[-1], color=colour, marker=m, s=10, label=f'{i} grad')
    axs[0,1].scatter(x=q_eddens_mean, y=pole_tracer_weighted[-1], color=colour, marker=m)
    axs[0,1].set_ylabel('Final tracer fraction area weighted')
    axs[0,1].set_xlabel('Eddy enstrophy')
    axs[0,1].text(q_eddens_mean+0.01, pole_tracer_weighted[-1], f'{name}')
    legend_elements = [Line2D([0], [0], color='w', markerfacecolor='k', marker='o', markersize=5, label='max merid grad'),
                        Line2D([0], [0], color='w', markerfacecolor='k', marker='o', markersize=10, label='max zonal mean')]
    axs[0,1].legend(handles=legend_elements)
    fft_plot = fft_ave_cut.plot(ax=axs[1,2], color=colour, marker=m, label=name)
    axs[1,2].set_ylabel('Amplitude')
    axs[1,2].set_xlabel('Zonal wavenumber')
    axs[1,2].legend()
    # contour_plot = contour_length.plot(ax=axs[2,2], color=colour, label=i)
    # axs[2,2].set_ylabel('Contour length of tracer=1.5e-3')
    # axs[2,2].set_xlabel('time')
    # axs[2,2].legend()
    # plot6 = axs[1,1].scatter(x=contour_length_mean, y=pole_tracer_frac[-1], s=q_eddens_mean*130, color=colour, marker=m)
    # axs[1,1].text(contour_length_mean+10, pole_tracer_frac[-1], f'{i}, {q_eddens_mean:.1f}')
    # axs[1,1].set_ylabel('Final tracer fraction')
    # axs[1,1].set_xlabel('Mean length of t=15e-3 contour')
    # axs[1,1].legend()
    plot7 = axs[2,1].scatter(x=condensing_fraction_mean, y=pole_tracer_weighted[-1], s=q_eddens_mean*130, color=colour, marker=m)
    axs[2,1].text(condensing_fraction_mean+0.1, pole_tracer_weighted[-1], f'{name}, {q_eddens_mean:.1f}')
    axs[2,1].set_ylabel('Final tracer weighted')
    axs[2,1].set_xlabel('Mean condensing area fraction')
    q_late_mean.plot(ax=axs[1,1], color=colour, linestyle=l, label=name)
    axs[1,1].set_ylabel('PV')
    axs[1,1].set_xlabel('Latitude')
    t_late.plot(ax=axs111, color=colour, alpha=0.75, linestyle=l1)
    axs111.axvline(lat_thresh.values, ymin=0, ymax=0.1, color=colour, alpha=0.5)
    axs111.set_ylabel('Zonal mean tracer value, 270-300sol')
    lines11, labels11 = axs[1,1].get_legend_handles_labels()
    lines111, labels111 = axs111.get_legend_handles_labels()
    axs111.legend(lines11 + lines111, labels11 + labels111, loc='upper left')
    plot8 = axs[2,2].scatter(x=dq_mean/dphi_mean, y=pole_tracer_weighted[-1], color=colour, marker=m)
    axs[2,2].text(dq_mean/dphi_mean+0.02e-10, pole_tracer_weighted[-1], name)
    axs[2,2].set_ylabel('Final tracer weighted')
    axs[2,2].set_xlabel('dq/dphi')




for i in [1, 4, 5, 6, 7]:
    print(i)
    plotting(i, 'full')


for i in [10]:
    print(i)
    plotting(i, 'ann')
    

plt.savefig(f'{path}/Plots/multirun_scatter.pdf')
print(f'Plot made:\n {path}/Plots/multirun_scatter.pdf')