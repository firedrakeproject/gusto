Obtaining Gusto
===============

Gusto requires installation of `Firedrake
<http://firedrakeproject.org>`_ (available for Ubuntu, Mac, and in
principle other Linux and Linux-like systems) and must be run from
within the Firedrake virtual environment.

If you installed Firedrake yourself
-----------------------------------

You can directly install Gusto in your Firedrake installation by
activating the Firedrake virtualenv and running::

    firedrake-update --install gusto

The Gusto source will be installed in the ``src/gusto`` subdirectory
of your Firedrake install. Using this install method you should
**not** add add Gusto to your ``PYTHONPATH``. Instead, Gusto will
automatically be available to import whenever your Firedrake
virtualenv is active.


If you are using a shared, pre-installed Firedrake (such as on some clusters)
-----------------------------------------------------------------------------

Check out the `Gusto <http://github.com/firedrakeproject/gusto>`_
repository on Github. Then, start the Firedrake virtualenv and add the
``gusto`` subdirectory to your ``PYTHONPATH``.


Testing your installation
-------------------------

Try executing files from the ``examples`` directory using e.g., ::

  python examples/sk_nonlinear.py

More description of these examples will appear here soon.

Visualisation software
----------------------

Gusto can output data in VTK format, suitable for viewing in
Paraview.  On Ubuntu and similar systems, you can obtain Paraview by
installing the ``paraview`` package.  On Mac OS, the easiest approach
is to download a binary from the `Paraview website <http://www.paraview.org>`_.
