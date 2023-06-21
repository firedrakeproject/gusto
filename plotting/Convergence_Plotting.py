"""
Script for convergence plotting for the williams 3 case

To Do:
Iterate through the Williams3 Test and generature figures showing convergence plots for different combos of TimeSteppers and Discretisation

"""
import matplotlib.pyplot as plt
import netCDF4 as nc
from numpy import log, linspace, poly1d, unique, polyfit

def convergenceplots(ref_levels,  fp):
    u2_5day_error = []
    D2_5day_error = []
    for ref in ref_levels:

        filep = 'results/archiveplot/ConvergenceData/Williamson3_ref=%s/diagnostics.nc' % ref
        data = nc.Dataset(filep)

        normalised_U2_error = data.groups['u_error']['l2'][:] / data.groups['u']['l2'][0]
        u2_5day_error.append(normalised_U2_error[-1])

        normalised_D2_error = data.groups['D_error']['l2'][:] / data.groups['D']['l2'][0]
        D2_5day_error.append(normalised_D2_error[-1])

        time = data.variables['time'][:]

        # Plottting
        fig, ax = plt.subplots(2, 1, sharex=True)
        plt.xlabel('Time')
        fig.suptitle('Graph of the normalised L2 error of the depth and velocity fields, ref level %s' % ref)
        ax[0].plot(time, normalised_U2_error)
        ax[0].set_ylabel('U Error')
        ax[1].plot(time, normalised_D2_error)
        ax[1].set_ylabel('D Error')
        #fig.savefig("%s/U-D-L2-norm_error-reflevel%d" % (fp, ref))

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.xlabel('Reference Levels')
    fig.suptitle('Convergence plots of depth and velocity fields')
    ax[0].scatter(ref_levels, u2_5day_error)
    ax[0].set_ylabel('$l_2(v)$')
    ax[0].set_xticks(linspace(2,6,5))
    ax[0].set_xticklabels(['2', '3', '4' ,'5', '6' ])
    ax[1].scatter(ref_levels, D2_5day_error)
    ax[1].set_ylabel('$l_2(h)$')
    ax[1].set_xticks(linspace(2,6,5))
    ax[1].set_xticklabels(['2', '3', '4' ,'5', '6' ])
   # fig.savefig("%s/Williams3-Convergence-plot" % fp)

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.xlabel('Reference Levels')
    fig.suptitle('Convergence log plots of depth and velocity fields')
    uslope, uintercept = polyfit(log(ref_levels), log(u2_5day_error), 1)
    dslope, dintercept = polyfit(log(ref_levels), log(D2_5day_error), 1)
    print(uslope)
    print(dslope)
    ax[0].scatter(log(ref_levels), log(u2_5day_error), color='k')
    ax[0].plot(unique(log(ref_levels)), poly1d(polyfit(log(ref_levels), log(u2_5day_error), 1))(unique(log(ref_levels))))
    ax[0].set_ylabel('$\log(l_2(v))$')
    ax[0].text(-5,60, uslope, fontsize = 22)
    ax[1].scatter(log(ref_levels), log(D2_5day_error), color='k')
    ax[1].plot(unique(log(ref_levels)), poly1d(polyfit(log(ref_levels), log(D2_5day_error), 1))(unique(log(ref_levels))))
    ax[1].set_ylabel('$\log(l_2(h))$')
    ax[1].set_xlabel('log(Reference Level)')
   # ax[1].set_xticks(linspace(2,6,5))
   # ax[1].set_xticklabels(['2', '3', '4' ,'5', '6' ])
    plt.show()
    return
    #fig.savefig("%s/Williams3-log_Convergence-plot" % fp)
convergenceplots([2,3,4,5,6],'/home/d-witt/firedrake/src/gusto/results/archiveplot/Convergenceplots' )

filep = '/home/d-witt/firedrake/src/gusto/results/Williamson3_longrun/diagnostics.nc'
data = nc.Dataset(filep)
time = data.variables['time'][:]
l2_velocity = data.groups['u_error']['l2'][:] / data.groups['u']['l2'][0]
max_velocity = data.groups['u_error']['max'][:] / data.groups['u']['max'][0]
l2_height = data.groups['D_error']['l2'][:] / data.groups['D']['l2'][0]
max_height = data.groups['D_error']['max'][:] /  data.groups['D']['max'][0]

fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle('normalised l2 and Max error for the velocity field')
ax[0].plot(time, l2_velocity)
ax[0].set_ylabel('l2 Error, velocity field')
ax[1].plot(time, max_velocity)
ax[1].set_ylabel('max Error, velocity field')
ax[1].set_xticks(linspace(0, time[-1], 5))
ax[1].set_xticklabels(['6', '12', '18' ,'24', '30' ])
ax[1].set_xlabel('Time (days)')
plt.show()

fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle('normalised l2 and Max error for the height field')
ax[0].plot(time, l2_height)
ax[0].set_ylabel('l2 Error, height field')
ax[1].plot(time, max_height)
ax[1].set_ylabel('max Error, height field')
ax[1].set_xticks(linspace(0, time[-1], 5))
ax[1].set_xticklabels(['6', '12', '18' ,'24', '30' ])
ax[1].set_xlabel('Time (days)')
plt.show()
print(len(time))