import numpy as np
import xarray as xr
import xrft
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from shapely.geometry import LineString

def scaled2_eddy_enstrophy(q, **kwargs):
    '''
    Calculate the eddy enstrophy from dataarray q
    '''
    latmin = kwargs.pop('latmin', 50)

    q = q.where(q.lat >= latmin, drop = True)
    # q = q.where(q.lon < 179.5, drop = True)
    qbar = q.mean(dim = 'lon')
    qbar = qbar.expand_dims({'lon':q.lon})

    qprime = (q - qbar)

    cos = np.cos(np.deg2rad(q.lat))

    qp = qprime **2 * cos

    Z = qp.sum(dim = 'lat').sum(dim = 'lon') / (cos.sum(dim = 'lat') * 2 * np.pi * q.mean(dim='lon').mean(dim='lat')**2)

    Z_renamed = Z.rename({var: f'{var}_eddens' for var in Z.variables if var !='time'})
    
    return Z_renamed


# def condensing_area(q):
#     q = q.where(q.lat >= 50, drop=True)
#     qbar = q.mean(dim='lat').mean(dim='lon')
#     frac = qbar.D_minus_H_rel_flag_less
#     return frac


def condensing_area(q):
    q = q.where(q.lat >= 50, drop=True)
    cumulative = q.CO2cond_flag_cumulative
    cond = cumulative.diff(dim='time')
    cond = cond.reindex(time=cumulative.time, method='nearest', fill_value=0)
    condbar = cond.mean(dim='lat').mean(dim='lon')
    return condbar


def delta_q(da, tmin, tmax):
    da = da.where(da.lat >= 50, drop=True)
    dabar = da.mean(dim='lon')
    daave = dabar.where(dabar.time>=tmin, drop=True).where(dabar.time<=tmax, drop=True).mean(dim='time')
    PVmax = np.max(daave.values)
    PVmax_lat = daave.lat[daave.argmax(dim='lat')]
    latmax = np.max(daave.lat.values)
    PVpole = daave.where(daave.lat==latmax, drop=True).values.item()
    dq = PVmax - PVpole
    dphi = latmax - PVmax_lat
    return dq, dphi


def delta_q_inst(da):
    da = da.where(da.lat >= 50, drop=True)
    daave = da.mean(dim='lon')
    PVmax = daave.max(dim='lat')
    PVmax_lat = daave.lat[daave.argmax(dim='lat')]
    latmax = np.max(daave.lat.values)
    PVpole = daave.where(daave.lat==latmax, drop=True).mean(dim='lat')
    dq = PVmax - PVpole
    dphi = latmax - PVmax_lat
    return dq, dphi

# there's a load of functions I've deleted because I'm an idiot. They all need rewriting (at a time when I can't remember what they hell they are or what they're doing arghhhhhhhhhh)

def tracer_integral(da, lat_thresh, direction):
    lat_below = np.max(da.lat.where(da.lat <= lat_thresh, drop=True)).values.item()
    lat_above = np.min(da.lat.where(da.lat >= lat_thresh, drop=True)).values.item()
    if lat_below != lat_above:
        # do the interpolation stuff in here
        new_lats = np.linspace(lat_below, lat_above, num=200)
        interpolated_da = da.interp(lat=new_lats, method='quadratic')
        da = xr.concat([da, interpolated_da], dim='lat').sortby('lat')
    if direction == 'pole':
        da = da.where(da.lat >= lat_thresh, drop=True)
        # lat_th = np.min(da.lat).values.item()
    elif direction == 'equator':
        da = da.where(da.lat <= lat_thresh, drop=True)
        # lat_th = np.max(da.lat).values.item()
    else:
        raise ValueError("Direction must be either 'pole' or 'equator'.")
    da['coslat'] = np.cos(da.lat * np.pi/180.)
    integrand = da.coslat * da
    integrand['lat'] = integrand.lat * np.pi/180.
    integrand['lon'] = integrand.lon * np.pi/180.
    integral = integrand.integrate('lat').integrate('lon')
    return integral, lat_thresh


def total_tracer_integral(da):
    da['coslat'] = np.cos(da.lat * np.pi/180.)
    integrand = da.coslat * da
    integrand['lat'] = integrand.lat * np.pi/180.
    integrand['lon'] = integrand.lon * np.pi/180.
    integral = integrand.integrate('lat').integrate('lon')
    return integral


def max_zonal_mean(da):
    dabar = da.mean(dim='lon')
    latmax_ind = dabar.argmax(dim='lat')
    lat_len = xr.DataArray(len(dabar.lat), dims=latmax_ind.dims, coords=latmax_ind.coords)
    start_idx = latmax_ind-1
    start_idx = start_idx.where(start_idx>=0, 0)
    end_idx = latmax_ind+2
    end_idx = end_idx.where(end_idx<=lat_len, lat_len)
    latmin = dabar.lat[start_idx]
    latmax = dabar.lat[end_idx-1]
    danear = dabar.where((dabar.lat>=latmin) & (dabar.lat<=latmax), drop=True)
    coefs = danear.polyfit(dim='lat', deg=2, skipna=True)
    try:
        max_lat = -coefs.sel(degree=1)/(2*coefs.sel(degree=2))
        max_val = xr.polyval(max_lat, coefs)
    except ZeroDivisionError:
        print('Coefficient of x**2=0 so an error')
    pole_lat = dabar.lat.max().item()
    pole_val = dabar.sel(lat=pole_lat).reset_coords('lat', drop=True)
    result = xr.Dataset({'max_lat':max_lat['polyfit_coefficients'], 'max_val':max_val['polyfit_coefficients'], 'pole_val':pole_val})
    return result


def max_merid_grad(da):
    dabar = da.mean(dim='lon')
    grad = dabar.differentiate(coord='lat')
    latmax_ind = grad.argmax(dim='lat')
    start_idx = latmax_ind-1
    start_idx = start_idx.where(start_idx>=0, start_idx, 0)
    end_idx = latmax_ind+2
    end_idx = end_idx.where(end_idx<=len(grad.lat), end_idx, len(grad.lat))
    latmin = grad.lat[start_idx]
    latmax = grad.lat[end_idx-1]
    danear = grad.where((grad.lat>=latmin) & (grad.lat<=latmax), drop=True)
    coefs = danear.polyfit(dim='lat', deg=2, skipna=True)
    try:
        max_lat = -coefs.sel(degree=1)/(2*coefs.sel(degree=2))
        max_val = xr.polyval(max_lat, coefs)
    except ZeroDivisionError:
        print('Coefficient of x**2=0 so an error')
    result = xr.Dataset({'max_grad_lat':max_lat['polyfit_coefficients'], 'max_grad_val':max_val['polyfit_coefficients']})
    return result


def fft(da, lat_centre, lat_range, max_wav):
    da_lat = da.where(da.lat<=lat_centre+lat_range/2, drop=True).where(da.lat>=lat_centre-lat_range/2, drop=True)
    da_mean = da_lat.mean(dim='lat')
    fft = np.abs(xrft.fft(da_mean, dim='lon'))
    fft = fft.assign_coords({'freq_lon':fft.freq_lon*360.})
    fft_pos = fft.where(fft.freq_lon>=0, drop=True)
    fft_select = fft_pos.where(fft_pos.freq_lon<=max_wav, drop=True)
    return fft_select


def contour_length(da, contour_value, start_sol=100):
    # da = da.where(da.mean(dim='time')>0, drop=True)
    da = da.where(da.time>=start_sol*88774., drop=True)
    lons = da.lon.values
    lats = da.lat.values
    times = da.time.values
    contour_lengths = []
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.NorthPolarStereo()})
    for i in range(len(times)):
        if i%200==0:
            print(i)
        contour_set = ax.contour(lons, lats, da[i], transform=ccrs.PlateCarree(), levels=[contour_value])
        for collection in contour_set.collections:
            for path in collection.get_paths():
                coords = path.vertices
                line_string = LineString(coords)
                contour_lengths.append(line_string.length)
            collection.remove()
    plt.close(fig)
    contour_lengths = xr.DataArray(contour_lengths, coords={'time':times}, dims=['time'])
    return contour_lengths
