Obtaining Gusto
===============

Gusto requires installation of `Firedrake
<http://firedrakeproject.org>`_ (available for Ubuntu, Mac, and in
principle other Linux and Linux-like systems) and must be run from
within the Firedrake virtual environment.

Having installed Firedrake, check out the `Gusto
<http://github.com/firedrakeproject/gusto>`_ repository on Github under
the Firedrake project. Then, start the Firedrake virtualenv add the
`dcore` subdirectory to your Python path. Try executing files from
the `examples` directory using e.g., ::

  python examples/embedded_DG.py

More description of these examples will appear here soon.

Visualisation software
----------------------

Gusto can output data in VTK format, suitable for viewing in
Paraview.  On Ubuntu and similar systems, you can obtain Paraview by
installing the ``paraview`` package.  On Mac OS, the easiest approach
is to download a binary from the `Paraview website <http://www.paraview.org>`_.
