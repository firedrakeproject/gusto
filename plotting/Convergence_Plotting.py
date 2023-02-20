"""
Script for convergence plotting for the williams 3 case

To Do:
Iterate through the Williams3 Test and generature figures showing convergence plots for different combos of TimeSteppers and Discretisation

"""
import matplotlib.pyplot as plt
import netCDF4 as nc
from numpy import log

def convergenceplots(ref_levels, elapsed_time, fp):
    u2_5day_error = []
    D2_5day_error = []
    for ref in ref_levels:

        filep = 'results/Williams3convergence_ref%s/diagnostics.nc' % ref
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
        fig.savefig("%s/U-D-L2-norm_error-reflevel%d" % (fp, ref))

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.xlabel('Reference Levels')
    fig.suptitle('Convergence plots of depth and velocity fields')
    ax[0].plot(ref_levels, u2_5day_error)
    ax[0].set_ylabel('U Convergence')
    ax[1].plot(ref_levels, D2_5day_error)
    ax[1].set_ylabel('D Convergence')
    fig.savefig("%s/Williams3-Convergence-plot" % fp)

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.xlabel('Reference Levels')
    fig.suptitle('Convergence log plots of depth and velocity fields')
    ax[0].plot(ref_levels, log(u2_5day_error))
    ax[0].set_ylabel('log, U Convergence')
    ax[1].plot(ref_levels, log(D2_5day_error))
    ax[1].set_ylabel('log, D Convergence')
    fig.savefig("%s/Williams3-log_Convergence-plot" % fp)

    fig, ax = plt.subplots()
    plt.xlabel('Reference Levels')
    plt.ylabel('Wall Clock Time')
    fig.suptitle('Convergence plots of depth and velocity fields')
    plt.plot(ref_levels, elapsed_time)
    fig.savefig("%s/Williams3-wallclocktime_Convergence-plot" % fp)