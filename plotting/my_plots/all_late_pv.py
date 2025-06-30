import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import functions as fcs
import pdb
import os

res = False

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
    file2 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-5'
    file4 = 'Relax_to_pole_and_CO2/annular_vortex_mars_55-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file5 = 'Relax_to_pole_and_CO2/annular_vortex_mars_65-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file6 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4_pvmax-1-4'
    file7 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4_pvmax-1-8'
    file8 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--2-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file9 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--3-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file10 = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file11 = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-5'
    file13 = 'Relax_to_annulus/annular_vortex_mars_62-67_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file14 = 'Relax_to_annulus/annular_vortex_mars_52-57_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file15 = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--1-8_PVpole--1-05_tau_r--2sol_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'
    file16 = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-4_PVpole--1-05_tau_r--2sol_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'

fulldss = {1:file1, 2:file2, 4:file4, 5:file5, 6:file6, 7:file7, 8:file8, 9:file9}
anndss = {10:file10, 11:file11, 14:file14, 13:file13, 15:file15, 16:file16}

fullcolours = {1:'black', 2:'orange', 4:'tomato', 5:'red', 6:'deepskyblue', 7:'blue', 8:'violet', 9:'magenta'}
anncolours = {10:'grey', 11:'gold', 14:'darksalmon', 13:'maroon', 15:'darkturquoise', 16:'midnightblue'}

fullnames = {1:r'$C_F$', 2:r'$r_F^+$', 4:r'$\phi_F^-$', 5:r'$\phi_F^+$', 6:r'$PV_F^-$', 7:r'$PV_F^+$'}
annnames = {10:r'$C_A$', 11:r'$r_A^+$', 14:r'$\phi_A^-$', 13:r'$\phi_A^+$', 15:r'$PV_A^-$', 16:r'$PV_A^+$'}

path = '/data/home/sh1293/results'

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(1,1, figsize=(10,10/1.666))

R = 3396000.
H = 17000.
Omega = 2*np.pi/88774.
g = 3.71
twomh = 2*Omega/H

def plotting(i, relax_type, path=path, **kwargs):
    alpha = kwargs.pop('alpha', 1)
    text_colour = kwargs.pop('text_colour', 'black')
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
    q_late_mean = q.where(q.time>=100*88774., drop=True).mean(dim='lon').mean(dim='time')
    q_crop = q_late_mean#.where(q_late_mean.lat>=0, drop=True)
    # q_crop.plot(ax=ax, color=colour, linestyle=l, label=name)
    q_crop.plot(ax=ax, color=colour, linestyle=l, label='Spatially variable' if i==1 else 'Zonally symmetric' if i==10 else '')
    ax.set_ylabel('PV')
    ax.set_xlabel(r'Latitude $\left(\degree\right)$')
    # ax.set_xticks([0,10,20,30,40,50,60,70,80,90])
    # ax.set_yticks(ticks=[0, 0.25*twomh, 0.5*twomh, 0.75*twomh, twomh, 1.25*twomh, 1.5*twomh], labels=['0', '', r'$\dfrac{\Omega}{H}$', '', r'$\dfrac{2\Omega}{H}$', '', r'$\dfrac{3\Omega}{H}$'])
    ax.set_xticks(ticks=[-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90], labels=[-90, '', '', -60, '', '', -30, '', '', 0,  '', '', 30, '', '', 60, '', '', 90])
    ax.set_yticks(ticks=[-twomh, -0.75*twomh, -0.5*twomh, -0.25*twomh, 0, 0.25*twomh, 0.5*twomh, 0.75*twomh, twomh, 1.25*twomh], labels=[r'$-\dfrac{2\Omega}{H}$', '', '', '', '0', '', '', '', r'$\dfrac{2\Omega}{H}$', ''])
    # lg = plt.legend(ncol=2)

for i in [1]:#, 4, 5, 6, 7]:
    print(i)
    plotting(i, 'full')
if res:
    for i in [2]:
        print(i)
        plotting(i, 'full', alpha=0.6)

for i in [10]:#, 14, 13, 15, 16]:
    print(i)
    plotting(i, 'ann')

if res:
    for i in [11]:
        print(i)
        plotting(i, 'ann', alpha=0.6)
    extra_name = '_res'
else:
    extra_name = ''

# lg = ax.legend(ncol=2)
lg = ax.legend(loc='upper left')
# c=0
# for text in lg.get_texts():
#     if c==1 or c==3:
#         text.set_color('grey')
#     c+=1
    

plt.savefig(f'{path}/Plots/control_late_pv_relabel{extra_name}.pdf')
print(f'Plot made:\n {path}/Plots/control_late_pv_relabel{extra_name}.pdf')