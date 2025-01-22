import xarray as xr
import spharm
import numpy as np
import matplotlib.pyplot as plt
import functions as fcs

file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-300sols_tracer_tophat-80_ref-4'
path = f'/data/home/sh1293/results/{file}'

lat_val=64.5

ds = xr.open_dataset(f'{path}/regrid_output.nc')

# nlats = len(ds.lat)
nlons = len(ds.lon)

rad = 3396000.
ntrunc=6



ds_rearr = ds.assign_coords(lon=((ds.lon+360)%360))
q = ds_rearr.PotentialVorticity
results = []
numwave = int((ntrunc+1)*(ntrunc+2)/2)
q_late = q.where(q.time>=100*88774., drop=True)
lat_val = fcs.max_zonal_mean(q_late).max_lat.mean(dim='time').values
q_lat = q.where(q.lat<=lat_val+5, drop=True).where(q.lat>=lat_val-5, drop=True)
nlats = len(q_lat.lat)
spharm1 = spharm.Spharmt(nlons, nlats, rad, gridtype='regular')
for t in range(len(q.time)):
# for t in range(5):
    data = q_lat.values[t,:]
    spectral_coeffs = spharm1.grdtospec(data, ntrunc)
    zonal_numbers = np.abs(spectral_coeffs)
    results.append(zonal_numbers)

results_da = xr.DataArray(results,
                        dims=["time", "wavenumber"],
                        coords={"time":q.time,
                                "wavenumber":range(numwave)})

da_plot = results_da.where(results_da.wavenumber<=10, drop=True).where(results_da.wavenumber>=1, drop=True)

fig, ax = plt.subplots(1,1, figsize=(8,8))
for i in range(1, 11):
    da_plot.where(da_plot.wavenumber==i, drop=True).plot(ax=ax, x='time', label=f'{i}')

plt.legend()

plt.savefig(f'{path}/Plots/zonal_wavenumber_decomposition.pdf')
print(f'Plot made:\n {path}/Plots/zonal_wavenumber_decomposition.pdf')
