from gusto import *


def test_label():
    mass = Label("mass")
    assert(mass.mass)
    implicit = Label("time", "implicit")
    assert(implicit.time == "implicit")
