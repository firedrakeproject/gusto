"""
Plotting the invariants of the William 6 test case
"""
import matplotlib.pyplot as plt
import netCDF4 as nc

data = nc.Dataset("results/Rossby-Haurwitx_Wave_Invariants/diagnostics.nc")
