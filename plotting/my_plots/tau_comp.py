import xarray as xr
import functions as fcs
import matplotlib.pyplot as plt

path = '/data/home/sh1293/results'

d001 = xr.open_dataset(f'{path}/Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-300sols_tracer_tophat/regrid_output.nc')
d0005 = xr.open_dataset(f'{path}/Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.005sol_beta--1_A0-0-norel_len-300sols/regrid_output.nc')
d002 = xr.open_dataset(f'{path}/Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.02sol_beta--1-0_A0-0.0-norel_len-300sols_tracer_tophat-80/regrid_output.nc')


datasets = {1:d001, 2:d0005, 3:d002}
colours = {1:'red', 2:'blue', 3:'green'}

fig, ax = plt.subplots(4,1, figsize=(8,20))

i=0
for beta in [3, 1, 2]:
    print(beta)
    # name = f'{int(10*beta):02d}'
    ds = datasets[beta]
    colour = colours[beta]
    print('calculating eddy enstrophy')
    ds_eddens = fcs.scaled2_eddy_enstrophy(ds)
    print('merging')
    ds_eddens_renamed = ds_eddens.rename({var: f'{var}_eddens' for var in ds_eddens.variables if var !='time'})
    ds_merged = xr.merge([ds, ds_eddens_renamed])
    ds_merged['condensing_fraction'] = fcs.condensing_area(ds)
    ds_merged['delta_q'] = fcs.delta_q_inst(ds_merged.PotentialVorticity)

    pveddens = ds_merged.PotentialVorticity_eddens.plot(ax=ax[0], color=colour, label='tau_c=0.01sol' if beta==1 else 'tau_c=0.005sol' if beta==2 else 'tau_c=0.02sol')
    fr = ds_merged.condensing_fraction.plot(ax=ax[1], color=colour)
    freddens = ds_merged.D_minus_H_rel_flag_less_eddens.plot(ax=ax[2], color=colour)
    dq = ds_merged.delta_q.plot(ax=ax[3], color=colour)

    ax[0].set_title('PV eddy enstrophy')
    ax[1].set_title('Fraction of cap condensing')
    ax[2].set_title('Eddy enstrophy of cap condensing')
    ax[3].set_title('dq')

    ax[0].legend()

plt.savefig(f'{path}/Relax_to_pole_and_CO2/tau_comp.pdf')