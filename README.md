# Gusto

Gusto is a Python code library, providing a toolkit of finite element methods for modelling geophysical fluids, such as the atmosphere and ocean.
The methods used by Gusto are underpinned by the [Firedrake](http://firedrakeproject.org) finite element code generation software.

Gusto is particularly targeted at the numerical methods used by the dynamical cores used in numerical weather prediction and climate models.
Gusto focuses on compatible finite element discretisations, in which variables lie in function spaces that preserve the underlying geometric structure of the equations.
These compatible methods underpin the Met Office's next-generation model, [LFRic](https://www.metoffice.gov.uk/research/approach/modelling-systems/lfric).

### Gusto is designed to provide:
- a testbed for **rapid prototyping** of novel numerical methods
- a **flexible framework** for exploring different modelling choices for geophysical fluid dynamics
- a **simple environment** for setting up and running test cases

## Download

The best way to install Gusto is as an additional package when installing [Firedrake](http://firedrakeproject.org). Usually, for a Mac with Homebrew or an Ubuntu installation this is done by downloading the Firedrake install script and executing it:
```
curl -0 https://raw.githubusercontent/com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --install gusto
```
For an up-to-date installation guide, see the [firedrake installation instructions](http://firedrakeproject.org/download.html). Once installed, Gusto must be run from within the Firedrake virtual environment, which is activated via
```
source firedrake/bin/activate
```
## Getting Started

To test your Gusto installation, run the test-suites:
```
cd firedrake/src/gusto
make test
```

The `examples` directory contains several test cases, which you can play with to get started with Gusto.
You can also see the [gusto case studies repository](https://github.com/firedrakeproject/gusto_case_studies), which contains a larger collection of test cases that use Gusto.

Gusto is documented [here](https://www.firedrakeproject.org/gusto-docs/), which is generated from the doc-strings in the codebase.

## Visualisation

Gusto can produce output in two formats:
- VTU files, which can be viewed with the [Paraview](https://www.paraview.org/) software
- netCDF files, which has data that can be plotted using standard python packages such as matplotlib. We suggest using the [tomplot](https://github.com/tommbendall/tomplot) Python library, which contains several routines to simplify the plotting of Gusto output.

## Website

For more information, please see our [website](https://www.firedrakeproject.org/gusto/), and please do get in touch via the Gusto channel on the Firedrake project [slack workspace](https://firedrakeproject.slack.com/).
