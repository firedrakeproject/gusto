#!/usr/bin/env python

from distutils.core import setup

setup(name="gusto",
      version="1.0",
      description="Toolkit for compatible finite element dynamical cores",
      author="The Gusto Team",
      url="http://www.firedrakeproject.org/gusto/",
      packages=["gusto",
                "gusto.core",
                "gusto.diagnostics",
                "gusto.equations",
                "gusto.initialisation",
                "gusto.physics"
                "gusto.recovery",
                "gusto.solvers",
                "gusto.spatial_methods",
                "gusto.time_discretisation",
                "gusto.timeloop"])
