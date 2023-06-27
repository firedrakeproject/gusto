import numpy as np
import matplotlib.pyplot as plt

def zeta(z, zt=1.5e4):
    return 1 - 3*(z / zt)**2 + 2*(z / zt)**3

z = np.linspace(0,1.5e4, 4)
a = 6.371229e6
d0 = a / 6
Vp = 1          
lon_c , lat_c = np.pi/9,  2*np.pi/9
d = np.linspace(0, d0, 100)
perturbation_magnitude = np.zeros(len(d))



for height in z:
    for i in range(len(d)):
        perturbation_magnitude[i] = 16*zeta(height)*Vp / (3*np.sqrt(3)) * np.cos(np.pi*d[i] / (2*d0))**3 * np.sin(np.pi*d[i] / (2*d0))
    plt.plot(d,perturbation_magnitude)
plt.show()
# TODO plot perturbation magnitude

# TODO plot the zonal perturbation

# TODO plot the meridional perturbation