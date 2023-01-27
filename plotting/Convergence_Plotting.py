"""
Script for convergence plotting for the williams 3 case

To Do:
Iterate through the Williams3 Test and generature figures showing convergence plots for different combos of TimeSteppers and Discretisation

"""
import matplotlib.pyplot as plt
import netCDF4 as nc

# -------------------------------------------------------------------#
# Data Import
# -------------------------------------------------------------------#
# Uses a for loop to loop through the different reference levels of equation run
ref_levels = [3, 4]
u2_5day_error = []
D2_5day_error = []
for index, ref in enumerate(ref_levels):

    fp = 'results/Williams3convergence_ref%s/diagnostics.nc' %ref
    data = nc.Dataset(fp)   

    normalised_U2_error = data.groups['u_error']['l2'][:] / data.groups['u']['l2'][0]
    u2_5day_error.append(normalised_U2_error[0]) 

    normalised_D2_error = data.groups['D_error']['l2'][:] / data.groups['D']['l2'][0]
    D2_5day_error.append(normalised_D2_error[0])
    
    time = data.variables['time'][:]
    
    # Plottting 
    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.xlabel('Time')
    fig.suptitle('Graph of the normalised L2 error of the depth and velocity fields, ref level %s' %ref ) 
    ax[0].plot(time, normalised_U2_error)
    ax[0].set_ylabel('U Error')
    ax[1].plot(time, normalised_D2_error)
    ax[1].set_ylabel('D Error')
    plt.show()









