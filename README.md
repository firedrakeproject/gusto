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

## Installing

Before installing Gusto you should first install Firedrake using the instructions found [here](https://firedrakeproject.org/install).
Once this is done Gusto can then be installed by running:
```
$ git clone https://github.com/firedrakeproject/gusto.git
$ pip install --editable ./gusto
```
or equivalently:
```
$ pip install --src . --editable git+https://github.com/firedrakeproject/gusto.git
```

## Getting Started

To test your Gusto installation you can run the test suite with:
```
$ cd gusto
$ make test
```

The `examples` directory contains several test cases, which you can play with to get started with Gusto.
You can also see the [gusto case studies repository](https://github.com/firedrakeproject/gusto_case_studies), which contains a larger collection of test cases that use Gusto.

Gusto is documented [here](https://www.firedrakeproject.org/gusto-docs/), which is generated from the doc-strings in the codebase.

## Visualisation

Gusto can produce output in two formats:
- VTU files, which can be viewed with the [Paraview](https://www.paraview.org/) software
- netCDF files, which has data that can be plotted using standard python packages such as matplotlib. We suggest using the [tomplot](https://github.com/tommbendall/tomplot) Python library, which contains several routines to simplify the plotting of Gusto output.

## Website

For more information, please see our [website](https://www.firedrakeproject.org/gusto/), and please do get in touch via the Gusto channel on the Firedrake project [Slack workspace](https://firedrakeproject.slack.com/).
