import xarray as xr
import functions as fcs
import matplotlib.pyplot as plt

path = '/data/home/sh1293/results'

file_full = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4'
file_ann = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4'

dfull = xr.open_dataset(f'{path}/{file_full}/regrid_output.nc')
dann = xr.open_dataset(f'{path}/{file_ann}/regrid_output.nc')



datasets = {1:dfull, 2:dann}
colours = {1:'red', 2:'blue'}
labels = {1:'Full relaxation', 2:'Relax to annulus'}

fig, ax = plt.subplots(3,1, figsize=(8,15))

i=0
for beta in [1, 2]:
    print(beta)
    # name = f'{int(10*beta):02d}'
    ds = datasets[beta]
    ds['sol'] = ds.time/88774
    colour = colours[beta]
    print('calculating eddy enstrophy')
    ds_eddens = fcs.scaled2_eddy_enstrophy(ds)
    cond_frac = fcs.condensing_area(ds)
    q = ds.PotentialVorticity
    tracer = ds.tracer
    max_zonal_mean = fcs.max_zonal_mean(q)
    max_merid_grad = fcs.max_merid_grad(q)
    lat_thresh = max_zonal_mean.where(ds.sol>=100, drop=True).mean(dim='time')['max_lat']
    lat_thresh_grad = max_merid_grad.where(ds.sol>=100, drop=True).mean(dim='time')['max_grad_lat']
    pole_tracer, _ = fcs.tracer_integral(tracer, lat_thresh, 'pole')
    pole_tracer_grad, _ = fcs.tracer_integral(tracer, lat_thresh_grad, 'pole')
    total_tracer = fcs.total_tracer_integral(tracer)
    pole_tracer_frac = pole_tracer/total_tracer
    pole_tracer_grad_frac = pole_tracer_grad/total_tracer
    fft = fcs.fft(q, lat_thresh, 10, 100)
    # zonal_power_zero = fft.where(fft.freq_lon==0, drop=True).sum(dim='freq_lon')/fft.where(fft.freq_lon!=0, drop=True).sum(dim='freq_lon')
    fft_ave = fft.where(fft.time>=100*88774., drop=True).mean(dim='time')
    fft_ave_cut = fft_ave.where(fft_ave.freq_lon>0, drop=True).where(fft_ave.freq_lon<=10, drop=True)


    pveddens = ds_eddens.PotentialVorticity_eddens.plot(ax=ax[0], color=colour, label=labels[beta])
    tracer_plot = pole_tracer_frac.plot(ax=ax[1], color=colour, label=labels[beta]+f', {lat_thresh.values:.2f}')
    tracer_grad_plot = pole_tracer_grad_frac.plot(ax=ax[1], color=colour, linestyle='--', label=labels[beta]+f', {lat_thresh_grad.values:.2f}')
    # power_plot = zonal_power_zero.plot(ax=ax[2], color=colour, label=labels[beta])
    fft_plot = fft_ave_cut.plot(ax=ax[2], color=colour, label=labels[beta], marker='*')

    ax[0].set_title('PV eddy enstrophy')
    ax[1].set_title('Poleward tracer integral')
    ax[2].set_title('Amplitude of zonal wavenumbers')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

plt.savefig(f'{path}/Relax_to_pole_and_CO2/full_vs_ann_comp.pdf')
print(f'Plot made:\n {path}/Relax_to_pole_and_CO2/full_vs_ann_comp.pdf')