import xarray as xr
import functions as fcs
import matplotlib.pyplot as plt

path = '/data/home/sh1293/results/Relax_to_pole_and_CO2/'

b10 = xr.open_dataset(f'{path}/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_alpha--1_working_long/regrid_output.nc')
b20 = xr.open_dataset(f'{path}/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--2_len-50sols/regrid_output.nc')
b30 = xr.open_dataset(f'{path}/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--3_len-50sols/regrid_output.nc')
b40 = xr.open_dataset(f'{path}/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--4_len-50sols/regrid_output.nc')
b05 = xr.open_dataset(f'{path}/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--0-5_A0-0-norel_len-50sols/regrid_output.nc')
b15 = xr.open_dataset(f'{path}/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-5_A0-0-norel_len-50sols/regrid_output.nc')
b25 = xr.open_dataset(f'{path}/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--2-5_A0-0-norel_len-50sols/regrid_output.nc')
b35 = xr.open_dataset(f'{path}/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--3-5_A0-0-norel_len-50sols/regrid_output.nc')


datasets = {0.5:b05, 1:b10, 1.5:b15, 2:b20, 2.5:b25, 3:b30, 3.5:b35, 4:b40}

fig, ax = plt.subplots(1,1, figsize=(8,8))

i=0
for beta in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
    print(beta)
    # name = f'{int(10*beta):02d}'
    ds = datasets[beta]
    print('calculating eddy enstrophy')
    ds_eddens = fcs.scaled2_eddy_enstrophy(ds)
    print('merging')
    ds_eddens_renamed = ds_eddens.rename({var: f'{var}_eddens' for var in ds_eddens.variables if var !='time'})
    ds_merged = xr.merge([ds, ds_eddens_renamed])
    ds_merged['condensing_fraction'] = fcs.condensing_area(ds)
    ds_merged['beta'] = beta
    ds_merged['delta_q'] = fcs.delta_q(ds_merged.PotentialVorticity, tmin=30*88774, tmax=50*88774)
    print('time averaging')
    ds_time = ds_merged.where(ds_merged.time>=30*88774, drop=True).where(ds_merged.time<=50*88774).mean(dim='time')
    if i==0:
        pveddens_0 = ds_time.PotentialVorticity_eddens
        fr_0 = ds_time.condensing_fraction
        freddens_0 = ds_time.D_minus_H_rel_flag_less_eddens
        dq_0 = ds_merged.delta_q
    pveddens = plt.scatter(x=beta, y=ds_time.PotentialVorticity_eddens/pveddens_0,
                           label=f'Potential Vorticity eddy enstrophy/{pveddens_0.values.item():.3g}' if i==0 else '', color='red')
    fr = plt.scatter(x=beta, y=ds_time.condensing_fraction/fr_0,
                     label=f'fraction of polar cap condensing/{fr_0.values.item():.3g}' if i==0 else '', color='blue')
    freddens = plt.scatter(x=beta, y=ds_time.D_minus_H_rel_flag_less_eddens/freddens_0,
                           label=f'Condensing locations eddy enstrophy/{freddens_0.values.item():.3g}' if i==0 else '', color='green')
    dq = plt.scatter(x=beta, y=ds_merged.delta_q/dq_0,
                     label=f'PV_max - PV_pole/{dq_0.values.item():.3g}' if i==0 else '', color='orange')
    i+=1


plt.legend()

plt.savefig(f'{path}/beta_comparison.pdf')


# b1_eddens = fcs.scaled2_eddy_enstrophy(b1)
# b2_eddens = fcs.scaled2_eddy_enstrophy(b2)
# b3_eddens = fcs.scaled2_eddy_enstrophy(b3)
# b4_eddens = fcs.scaled2_eddy_enstrophy(b4)

# b1_eddens_timeave = b1_eddens.where(b1_eddens.time>=30*88774, drop=True).where(b1_eddens.time<=50*88774, drop=True).mean(dim='time')
# b2_eddens_timeave = b2_eddens.where(b2_eddens.time>=30*88774, drop=True).where(b2_eddens.time<=50*88774, drop=True).mean(dim='time')
# b3_eddens_timeave = b3_eddens.where(b3_eddens.time>=30*88774, drop=True).where(b3_eddens.time<=50*88774, drop=True).mean(dim='time')
# b4_eddens_timeave = b4_eddens.where(b4_eddens.time>=30*88774, drop=True).where(b4_eddens.time<=50*88774, drop=True).mean(dim='time')

# frac1 = fcs.condensing_area(b1)
# frac2 = fcs.condensing_area(b2)
# frac3 = fcs.condensing_area(b3)
# frac4 = fcs.condensing_area(b4)

# frac1_timeave = frac1.where(frac1.time>=30*88774, drop=True).where(frac1.time<=50*88774, drop=True).mean(dim='time')
# frac2_timeave = frac2.where(frac2.time>=30*88774, drop=True).where(frac2.time<=50*88774, drop=True).mean(dim='time')
# frac3_timeave = frac3.where(frac3.time>=30*88774, drop=True).where(frac3.time<=50*88774, drop=True).mean(dim='time')
# frac4_timeave = frac4.where(frac4.time>=30*88774, drop=True).where(frac4.time<=50*88774, drop=True).mean(dim='time')

# fix, ax = plt.subplots(1, 1, figsize=(8,8))
# pv1 = plt.scatter(x=1, y=b1_eddens_timeave.PotentialVorticity, label='Potential Vorticity Eddy enstrophy', color='red')
# pv2 = plt.scatter(x=2, y=b2_eddens_timeave.PotentialVorticity, color='red')
# pv3 = plt.scatter(x=3, y=b3_eddens_timeave.PotentialVorticity, color='red')
# pv4 = plt.scatter(x=4, y=b4_eddens_timeave.PotentialVorticity, color='red')

# fr1 = plt.scatter(x=1, y=frac1_timeave, label = 'fraction of polar cap condensing', color='blue')
# fr2 = plt.scatter(x=2, y=frac2_timeave, color='blue')
# fr3 = plt.scatter(x=3, y=frac3_timeave, color='blue')
# fr4 = plt.scatter(x=4, y=frac4_timeave, color='blue')

