import numpy as np
import xarray as xr

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
    
    return Z


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


def max_zonal_mean(da):
    dabar = da.mean(dim='lon')
    latmax_ind = dabar.argmax(dim='lat')
    start_idx = latmax_ind-1
    start_idx = start_idx.where(start_idx>=0, start_idx, 0)
    end_idx = latmax_ind+2
    end_idx = end_idx.where(end_idx<=len(dabar.lat), end_idx, len(dabar.lat))
    latmin = dabar.lat[start_idx]
    latmax = dabar.lat[end_idx-1]
    danear = dabar.where((dabar.lat>=latmin) & (dabar.lat<=latmax), drop=True)
    coefs = danear.polyfit(dim='lat', deg=2, skipna=True)
    try:
        max_lat = -coefs.sel(degree=1)/(2*coefs.sel(degree=2))
        max_val = xr.polyval(max_lat, coefs)
    except ZeroDivisionError:
        print('Coefficient of x**2=0 so an error')
    result = xr.Dataset({'max_lat':max_lat['polyfit_coefficients'], 'max_val':max_val['polyfit_coefficients']})
    return result
