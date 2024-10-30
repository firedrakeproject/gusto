import xarray as xr
import functions as fcs
import matplotlib.pyplot as plt

path = '/data/home/sh1293/results'

dfull = xr.open_dataset(f'{path}/Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-300sols_tracer_tophat/regrid_output.nc')
dann = xr.open_dataset(f'{path}/Relax_to_annulus/annular_vortex_mars_55-60_PVmax--2-4_PVpole--1-0_tau_r--2sol_A0-0.0-norel_len-300sols_tracer_tophat-80/regrid_output.nc')



datasets = {1:dfull, 2:dann}
colours = {1:'red', 2:'blue'}
labels = {1:'Full relaxation', 2:'Relax to annulus'}

fig, ax = plt.subplots(3,1, figsize=(8,15))

i=0
for beta in [1, 2]:
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
    ds_merged['delta_q'], ds_merged['delta_phi'] = fcs.delta_q_inst(ds_merged.PotentialVorticity)

    pveddens = ds_merged.PotentialVorticity_eddens.plot(ax=ax[0], color=colour, label=labels[beta])
    dq = ds_merged.delta_q.plot(ax=ax[1], color=colour)
    dphi = ds_merged.delta_phi.plot(ax=ax[2], color=colour)

    ax[0].set_title('PV eddy enstrophy')
    ax[1].set_title('dq')
    ax[2].set_title('dphi')

    ax[0].legend()

plt.savefig(f'{path}/Relax_to_pole_and_CO2/full_vs_ann_comp.pdf')