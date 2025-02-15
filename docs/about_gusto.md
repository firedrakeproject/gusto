### About Gusto
Gusto is a library of finite element methods for geophysical fluid dynamics.
In particular, Gusto focuses on using compatible finite element discretisations, in which variables lie in function spaces that preserve the underlying geometric structure of the equations.
These compatible finite element methods underpin the Met Office's next-generation model, [LFRic](https://www.metoffice.gov.uk/research/approach/modelling-systems/lfric).

### Download

The best way to install Gusto is as an additional package when installing [Firedrake](http://firedrakeproject.org). Usually, for a Mac with Homebrew or an Ubuntu installation this is done by downloading the Firedrake install script and executing it:
```
curl -0 https://raw.githubusercontent/com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --install gusto
```
For an up-to-date installation guide, see the [firedrake installation instructions](http://firedrakeproject.org/download.html). Once installed, Gusto must be run from within the Firedrake virtual environment, which is activated via
```
source firedrake/bin/activate
```
To test your Gusto installation, run the test-suites:
```
cd firedrake/src/gusto
make test
```

#### Parallel output with netCDF

The [`netCDF4`](https://pypi.org/project/netCDF4/) package installed by Gusto does not support parallel usage.
This means that, when Gusto is run in parallel, distributed data structures must first be gathered onto rank 0 before they can be output.
This is *extremely inefficient* at high levels of parallelism.

To avoid this it is possible to build a parallel-aware version of `netCDF4`.
The steps to do this are as follows:

1. Activate the Firedrake virtual environment.
2. Uninstall the existing `netCDF4` package:
    ```
    $ pip uninstall netCDF4
    ```
3. Set necessary environment variables (note that this assumes that PETSc was built by Firedrake):
    ```
    $ export HDF5_DIR=$VIRTUAL_ENV/src/petsc/default
    $ export NETCDF4_DIR=$HDF5_DIR
    ```
4. Install the parallel version of `netCDF4`:
    ```
    $ pip install --no-build-isolation --no-binary netCDF4 netCDF4
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

If you're interested in using Gusto we'd love to hear from you! The best way to get in touch with the Gusto developers is through our [Github page](https://github.com/firedrakeproject/gusto) or the Gusto channel on the Firedrake project [slack channel](https://firedrakeproject.slack.com/). Alternatively you can email [Jemma Shipton](https://mathematics.exeter.ac.uk/staff/js1075) or [Tom Bendall](https://www.metoffice.gov.uk/research/people/tom-bendall)

<!--
### Funding and Citation

Some details of our funders are below.

If you use Gusto as part of your research, please cite us! The best way to do this is ...

Publications that used Gusto include:
-->
