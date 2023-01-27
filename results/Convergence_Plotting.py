"""
Script for convergence plotting for the williams 3 case

To Do:
Iterate through the Williams3 Test and generature figures showing convergence plots for different combos of TimeSteppers and Discretisation

"""
import matplotlib.pyplot as plt
import netCDF4 as nc

#-------------------------------------------------------------------#
# Data Import
#-------------------------------------------------------------------#
#Uses a for loop to loop through the different reference levels of equation run
ref_levels = [3]
for ref in ref_levels:
    fp='results/Williams3convergence_ref%s/diagnostics.nc' %ref
    data = nc.Dataset(fp)   

    l2_error = data.groups['u_error']['l2'][:]
    normalised_l2_error = l2_error / l2_error[1]
    l2_5day_error = l2_error[-1]
    D2_error = data.groups['D_error']['l2'][:]
    normalised_D2_error = D2_error / D2_error[1]
    D2_5day_error = D2_error[-1]
    time = data.variables['time'][:]
    
    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.xlabel('Time')
    fig.suptitle('Graph of the normalised L2 error of the depth and velocity fields, ref level %s') %ref
    ax[0, 0].plot(time, normalised_l2_error)
    ax[0, 0].set_ylabel('U Error')
    ax[0, 1].plot(time, normalised_D2_error)
    ax[0, 0].set_ylabel('D Error')








