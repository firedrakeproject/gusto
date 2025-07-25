import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from firedrake import (Function, dot, SpatialCoordinate)
from firedrake.pyplot import tricontourf

def extract_data (state, field_name, mixed_space=False):

    if not hasattr(state, field_name):
        raise ValueError(f"Field '{field_name}' not found in state.")

    fname = getattr(state, field_name)
    value = fname.dat.data
    return value

def create_function_space(state, equation, field_name, mixed_space=False):

    if field_name =='w':
        k = equation.domain.k
        wind = state.u
        V = equation.domain.spaces('theta')
        plotting_space = Function(V).interpolate(dot(k, wind))

    elif field_name == 'theta':
        V = state.theta.function_space()
        plotting_space = Function(V).interpolate(state.theta)

    elif field_name == 'rho':
        V = equation.domain.spaces('DG')
        plotting_space = Function(V).interpolate(state.rho)

    return plotting_space

def extract_mixed_space(mixed_space, equation, field_name):
    if field_name == 'w' :
        index = 0
        V = equation.domain.spaces('theta')
        wind = mixed_space.subfunctions[index]
        k = equation.domain.k
        plotting_space=Function(V).interpolate(dot(k, wind))

    elif field_name == 'rho':
        index = 1
        V = equation.domain.spaces('DG')
        plotting_space = Function(V)
        plotting_space.dat.data[:] = mixed_space.subfunctions[1].dat.data[:]
    elif field_name == 'theta':
        index = 2
        V = equation.domain.spaces('theta')
        plotting_space = Function(V)
        plotting_space.dat.data[:] = mixed_space.subfunctions[2].dat.data[:]

    return plotting_space


def plot_time_level_state(state, equation, field_name,file_name=None,
                          time_level_name=None, save=True, title=None,
                          mixed_space=False):
    """
    Plot the field at a specific time level.

    Args:
        state: The state object containing the field data.
        equation: The equation object to get the function space.
        field_name: The name of the field to plot.
        step: The current time step of the simulation.
        dt: The time step size.
        time_level_name: Optional name for the time level (default is None).
        save: Boolean indicating whether to save the plot (default is False).
    """

    return None

    if mixed_space:
        plotting_space = extract_mixed_space(state, equation, field_name)
    else:
        plotting_space = create_function_space(state, equation, field_name, mixed_space=False)

    x_coord, z_coord = SpatialCoordinate(equation.domain.mesh)
    x = Function(plotting_space.function_space()).interpolate(x_coord)
    z = Function(plotting_space.function_space()).interpolate(z_coord)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data using tricontourf
    cmap = plt.cm.coolwarm
    norm = colors.CenteredNorm()
    contours = ax.tricontourf(x.dat.data, z.dat.data,
                              plotting_space.dat.data, axes=ax, cmap=cmap, norm=norm)
    fig.colorbar(contours)

    # Set title and labels
    if title:
        ax.set_title(f'{title}')
    else:
        ax.set_title(f"{field_name} at time level {time_level_name if time_level_name else ''}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    if save:
        if file_name is None:
           raise ValueError(f"Please provide a file name to save the plot.")
        save_plot(fig, file_name)
        print(f'saving plot to {file_name}')

    else:
        plt.show()

def make_subplot(states, state_names, equation, field_name, file_name=None, save=True, dir=None):
    """
    Create a subplot for the field data.

    Args:
        stats: The statistics object containing the field data.
        field_name: The name of the field to plot.
        step: The current time step of the simulation.
        dt: The time step size.
        time_level_name: Optional name for the time level (default is None).
        save: Boolean indicating whether to save the plot (default is False).
    """
    rows = 1
    cols = len(states)
    fig, axarray = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.subplots_adjust(wspace=0.4)

    for _, (state, state_name, ax) in enumerate(zip(states, state_names, axarray.flatten())):

        plotting_space = create_function_space(state, equation, field_name)


        # Plot the data using tricontourf
        cmap = plt.cm.coolwarm
        norm = colors.CenteredNorm()
        contours = tricontourf(plotting_space, axes=ax, cmap=cmap, norm=norm)
        fig.colorbar(contours)

        # Set title and labels
        ax.set_title(f"TR-BDF2: {field_name} at time level {state_name}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")


    if save:

        if file_name is None:
           raise ValueError(f"Please provide a file name to save the plot.")
        if dir is None:
            raise ValueError(f"Please provide a directory to save the plot.")


        save_plot(fig, file_name)
        print(f'saving plot to {file_name}')
    else:
        plt.show()

def save_plot(fig, filename, dir='/home/thomas/venv-firedrake/gusto/results/tr_bdf2_sk_gw'):
    """
    Save the plot to a file.

    Args:
        fig: The figure object to save.
        filename: The name of the file to save the plot as.
        dir: The directory to save the plot in (default is current directory).
    """
    fig.savefig(f"{dir}/{filename}.png")
    print(f"Plot saved as {filename}.png in {dir}")
    plt.close(fig)  # Close the figure to free up memory
