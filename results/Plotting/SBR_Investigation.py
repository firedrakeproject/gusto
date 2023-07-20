import netCDF4 as nc 
import matplotlib.pyplot as plt
from numpy import reshape
def PlotMeridonal(fp):
    data =nc.Dataset(fp)
    time = data.variables['time'][:]
    length=len(time)
    u_merid_min = data.groups['u_meridional']['min'][:]
    u_merid_max = data.groups['u_meridional']['max'][:]
    u_merid_total = data.groups['u_meridional']['total'][:]
    u_merid_total_norm = data.groups['u_meridional']['total'][:] / data.groups['u_meridional']['total'][0]
    u_merid_rms = data.groups['u_meridional']['rms'][:]
    u_merid_rms_norm = data.groups['u_meridional']['rms'][:] / data.groups['u_meridional']['rms'][0]
    u_merid_l2 = data.groups['u_meridional']['l2'][:]
    u_merid_l2_norm = data.groups['u_meridional']['l2'][:] / data.groups['u_meridional']['l2'][0]

    u_merid_measure = [[u_merid_min], [u_merid_max], [u_merid_l2], [u_merid_l2_norm], 
                       [u_merid_rms],[u_merid_rms_norm], [u_merid_total], [u_merid_total_norm]]
    measures = ['min', 'max', 'l2', 'l2 norm', 'rms', 'rms norm', 'total','total norm']
    rows , cols = 4, 2
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(rows, cols,
                            sharex = 'col')
    mes = 0
    for row in range(rows):
        for col in range(cols):
            ax[row,col].scatter(time, reshape(u_merid_measure[mes], (length,1)))
            ax[row,col].set(title=f"Graph of Meridonal velocity, {measures[mes]} measure",
                            ylabel = measures[mes])
            mes += 1
    plt.show()
    return fig

def PlotZonal(fp):
    data =nc.Dataset(fp)
    time = data.variables['time'][:]
    length=len(time)
    u_min = data.groups['u_zonal']['min'][:]
    u_max = data.groups['u_zonal']['max'][:]
    u_total = data.groups['u_zonal']['total'][:]
    u_total_norm = data.groups['u_zonal']['total'][:] / data.groups['u_zonal']['total'][0]
    u_rms = data.groups['u_zonal']['rms'][:]
    u_rms_norm = data.groups['u_zonal']['rms'][:] / data.groups['u_zonal']['rms'][0]
    u_l2 = data.groups['u_zonal']['l2'][:]
    u_l2_norm = data.groups['u_zonal']['l2'][:] / data.groups['u_zonal']['l2'][0]

    u_measure = [[u_min], [u_max], [u_l2], [u_l2_norm], 
                       [u_rms],[u_rms_norm], [u_total], [u_total_norm]]
    measures = ['min', 'max', 'l2', 'l2 norm', 'rms', 'rms norm', 'total','total norm']
    rows , cols = 4, 2
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(rows, cols,
                            sharex = 'col')
    mes = 0
    for row in range(rows):
        for col in range(cols):
            ax[row,col].scatter(time, reshape(u_measure[mes], (length,1)))
            ax[row,col].set(title=f"Graph of zonal velocity, {measures[mes]} measure",
                            ylabel = measures[mes])
            mes += 1
    plt.show()
    return fig

def PlotRadial(fp):
    data =nc.Dataset(fp)
    time = data.variables['time'][:]
    length=len(time)
    u_min = data.groups['u_radial']['min'][:]
    u_max = data.groups['u_radial']['max'][:]
    u_total = data.groups['u_zonal']['total'][:]
    u_total_norm = data.groups['u_radial']['total'][:] / data.groups['u_radial']['total'][0]
    u_rms = data.groups['u_radial']['rms'][:]
    u_rms_norm = data.groups['u_radial']['rms'][:] / data.groups['u_radial']['rms'][0]
    u_l2 = data.groups['u_radial']['l2'][:]
    u_l2_norm = data.groups['u_radial']['l2'][:] / data.groups['u_radial']['l2'][0]

    u_measure = [[u_min], [u_max], [u_l2], [u_l2_norm], 
                       [u_rms],[u_rms_norm], [u_total], [u_total_norm]]
    measures = ['min', 'max', 'l2', 'l2 norm', 'rms', 'rms norm', 'total','total norm']
    rows , cols = 4, 2
    plt.style.use('seaborn-whitegrid')
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(rows, cols,
                            sharex = 'col')
    mes = 0
    for row in range(rows):
        for col in range(cols):
            ax[row,col].scatter(time, reshape(u_measure[mes], (length,1)))
            ax[row,col].set(title=f"Graph of radial velocity, {measures[mes]} measure",
                            ylabel = measures[mes])
            mes += 1
    plt.show()
    return fig

def PlotGeostrophic(fp):
    data =nc.Dataset(fp)
    time = data.variables['time'][:]
    length=len(time)
    u_min = data.groups['GeostrophicImbalance']['min'][:]
    u_max = data.groups['GeostrophicImbalance']['max'][:]
    u_total = data.groups['GeostrophicImbalance']['total'][:]
    u_total_norm = data.groups['GeostrophicImbalance']['total'][:] / data.groups['GeostrophicImbalance']['total'][0]
    u_rms = data.groups['GeostrophicImbalance']['rms'][:]
    u_rms_norm = data.groups['GeostrophicImbalance']['rms'][:] / data.groups['GeostrophicImbalance']['rms'][0]
    u_l2 = data.groups['GeostrophicImbalance']['l2'][:]
    u_l2_norm = data.groups['GeostrophicImbalance']['l2'][:] / data.groups['GeostrophicImbalance']['l2'][0]

    u_measure = [[u_min], [u_max], [u_l2], [u_l2_norm], 
                       [u_rms],[u_rms_norm], [u_total], [u_total_norm]]
    measures = ['min', 'max', 'l2', 'l2 norm', 'rms', 'rms norm', 'total','total norm']
    rows , cols = 4, 2
    plt.style.use('seaborn-whitegrid')
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(rows, cols,
                            sharex = 'col')
    mes = 0
    for row in range(rows):
        for col in range(cols):
            ax[row,col].scatter(time, reshape(u_measure[mes], (length,1)))
            ax[row,col].set(title=f"Graph of Geostrophic Imbalance, {measures[mes]} measure",
                            ylabel = measures[mes])
            ax[row,col].ylim(10)
            mes += 1
    plt.show()
    return fig

fp = '/home/d-witt/firedrake/src/gusto/results/Plotting/SBR_longDiagnostics/diagnostics.nc'
PlotMeridonal(fp)
PlotZonal(fp)
PlotRadial(fp)
