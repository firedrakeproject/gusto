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


def condensing_area(q):
    q = q.where(q.lat >= 50, drop=True)
    qbar = q.mean(dim='lat').mean(dim='lon')
    frac = qbar.D_minus_H_rel_flag_less
    return frac


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
    if direction == 'pole':
        da = da.where(da.lat >= lat_thresh, drop=True)
        lat_th = np.min(da.lat).values.item()
    elif direction == 'equator':
        da = da.where(da.lat <= lat_thresh, drop=True)
        lat_th = np.max(da.lat).values.item()
    da['coslat'] = np.cos(da.lat * np.pi/180.)
    integrand = da.coslat * da
    integrand['lat'] = integrand.lat * np.pi/180.
    integrand['lon'] = integrand.lon * np.pi/180.
    integral = integrand.integrate('lat').integrate('lon')
    return integral, lat_th