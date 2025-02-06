import numpy as np
import xarray as xr
import os

file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4'

ref_lev = 4

path = f'/data/home/sh1293/results/{file}'

dt = (0.5)**(ref_lev-4) * 450. * 10.

ds = xr.open_dataset(f'{path}/field_output.nc')
ds_new = ds.where(ds.time%dt==0, drop=True)
if len(ds_new.time)==len(ds.time):
    print('Original dataset was fine')
elif len(ds_new.time)<len(ds.time):
    print('Original dataset too long')
    if len(ds_new.time)==5919:
        print('Yay as expected')
        ds.close()
        os.rename(f'{path}/field_output.nc', f'{path}/dodgy_field_output.nc')
        ds_new.to_netcdf(f'{path}/field_output.nc')
    else:
        print(f'Not the length expected, length {len(ds_new.time)}')
        # print(ds_new.time)
