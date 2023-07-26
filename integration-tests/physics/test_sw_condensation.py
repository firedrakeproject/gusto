"""
# This tests the SW_AdjustableSaturation physics class. In the first scenario it
# creates a bubble of water vapour that is advected by a prescribed velocity and
# should be converted to cloud where it exceeds a saturation threshold. In the
# second test it creates a cloud in a subsaturated atmosphere that should
# evaporate. The first test passes if the cloud is non-zero, the vapour has
# decreased the buoyancy field is increased and the total moisture is conserved.
# The second test passes if cloud is zero, vapour has increased, buoyancy has
# decreased and total moisture is conserved.
"""

from os import path
from gusto import *
from firedrake import 
