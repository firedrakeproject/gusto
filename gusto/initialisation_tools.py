"""
A module containing some tools for computing initial conditions, such
as balanced initial conditions.
"""


def compressible_hydrostatic_balance(state, theta0, rho0,
                                     top=True, rho_boundary=0):
    """
    Compute a hydrostatically balanced density given a potential temperature
    profile.

    :arg state: The :class:`State` object.
    :arg theta0: :class:`.Function`containing the potential temperature.
    :arg rho0: :class:`.Function` to write the initial density into.
    :arg top: If True, set a boundary condition at the top. Otherwise, set
    it at the bottom.
    :arg rho_boundary: a field or expression to use as boundary data on
    the top or bottom as specified.
    """

    

