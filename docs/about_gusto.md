### About Gusto
Gusto is a library of finite element methods for geophysical fluid dynamics.
In particular, Gusto focuses on using compatible finite element discretisations, in which variables lie in function spaces that preserve the underlying geometric structure of the equations.
These compatible finite element methods underpin the Met Office's next-generation model, [LFRic](https://www.metoffice.gov.uk/research/approach/modelling-systems/lfric).

## Installing

Before installing Gusto you should first install Firedrake using the instructions found [here](https://firedrakeproject.org/install).
Once this is done Gusto can then be installed by running:
```
$ git clone https://github.com/firedrakeproject/gusto.git
$ pip install --editable ./gusto
```
or equivalently:
```
$ pip install --src . --editable git+https://github.com/firedrakeproject/gusto.git#egg=gusto
```

To test your Gusto installation you can run the test suite with:
```
$ cd gusto
$ make test
```

#### Parallel output with netCDF

By default the [`netCDF4`](https://pypi.org/project/netCDF4/) package installed by Gusto does not support parallel I/O.
This means that, when Gusto is run in parallel, distributed data structures must first be gathered onto rank 0 before they can be output.
This is *extremely inefficient* at high levels of parallelism.

To avoid this it is possible to build a parallel-aware version of `netCDF4`.
The steps to do this for an Ubuntu machine are as follows:

1. Uninstall the existing `netCDF4` package:
    ```
    $ pip uninstall netCDF4
    ```
2. Set necessary environment variables and install the build dependencies:
    ```
    $ export PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-firedrake-default
    $ export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
    $ pip install Cython
    ```
    On non-Ubuntu platform additional environment variables like `HDF5_DIR` and
    `NETCDF4_DIR` may need to be set (see [here](https://unidata.github.io/netcdf4-python/#developer-install) for more information).
3. Install the parallel version of `netCDF4`:
    ```
    $ pip install --no-binary netCDF4 --no-build-isolation netCDF4
    ```

### Getting Started

Once you have a working installation, the best way to get started with Gusto is to play with some of the examples in the `gusto/examples` directory.
Our documentation can be found [here](https://firedrakeproject.org/gusto/).

<!--
- comment about searching read-the-docs
- link to jupyter-notebooks
- other questions link to get in touch (below)
-->

<!--
### The Gusto Team

Here is the team
-->

### Getting in touch

If you're interested in using Gusto we'd love to hear from you! The best way to get in touch with the Gusto developers is through our [Github page](https://github.com/firedrakeproject/gusto) or the Gusto channel on the Firedrake project [Slack channel](https://firedrakeproject.slack.com/). Alternatively you can email [Jemma Shipton](https://mathematics.exeter.ac.uk/staff/js1075) or [Tom Bendall](https://www.metoffice.gov.uk/research/people/tom-bendall)

<!--
### Funding and Citation

Some details of our funders are below.

If you use Gusto as part of your research, please cite us! The best way to do this is ...

Publications that used Gusto include:
-->
