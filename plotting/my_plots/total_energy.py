import functions as fcs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

from_diag = True

filepath = 'new_single_fplane_Bu2b1p5Rop2_l500dt250df30'

plot_dir = f'/data/home/sh1293/results/jupiter_sw/{filepath}/Plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


if from_diag:
    ke_density, times = fcs.make_structured(filepath, 'ShallowWaterKineticEnergy')
    KE_total = ke_density.integrate('x').integrate('y')

    pe_density, _ = fcs.make_structured(filepath, 'ShallowWaterPotentialEnergy')
    PE_total = pe_density.integrate('x').integrate('y')

    try:
        avlpe_density, _ = fcs.make_structured(filepath, 
                                            'ShallowWaterAvailablePotentialEnergy')
        avlPE_total = avlpe_density.integrate('x').integrate('y')
        print(f"Using 'Available PE' diagnostic")
    except IndexError as e:
        if "not found in" in str(e):
            print(f"No 'Available PE' diagnostic")
            g = 24.79
            Bu = float(filepath.split('Bu')[1].split('b')[0])
            Omega = 1.74e-4
            R = 71.4e6
            f0 = 2 * Omega
            rm = 1e6
            phi0 = Bu * (f0*rm)**2
            H = phi0/g
            avlpe_density = 1/2*g*(np.sqrt(2*pe_density/g)-H)**2
            avlPE_total = avlpe_density.integrate('x').integrate('y')
        else:
            raise

    e_density = ke_density + pe_density
    E_total = e_density.integrate('x').integrate('y')

    avle_density = ke_density + avlpe_density
    avlE_total = avle_density.integrate('x').integrate('y')

else:
    g = 24.79
    D = fcs.make_structured(filepath, 'D')
    pe_density = 1/2*g*D[0]**2
    PE_total = pe_density.integrate('x').integrate('y')
    KE_total = 0
    E_total = PE_total + KE_total

E_scaled = (E_total/E_total[0]-1)*100
KE_scaled = (KE_total/E_total[0]-1)*100
PE_scaled = (PE_total/E_total[0]-1)*100

fig, ax = plt.subplots(1,1, figsize=(10,10/1.666))
# e_plot = E_total.plot(ax=ax, label='Total')
avle_plot = avlE_total.plot(ax=ax, label='Total available')
ke_plot = KE_total.plot(ax=ax, label='Kinetic', linestyle='--')
# pe_plot = PE_total.plot(ax=ax, label='Potential', linestyle='--')
avlpe_plot = avlPE_total.plot(ax=ax, label='Potential available', linestyle='--')

# esc_plot = E_scaled.plot(ax=ax, label='Total scaled')
# kesc_plot = KE_scaled.plot(ax=ax, label='Kinetic scaled', linestyle='--')
# pesc_plot = PE_scaled.plot(ax=ax, label='Potential scaled', linestyle='--')
plt.legend()
# ax.set_ylim(np.array([-1,1])*1e-3)
# ax.set_ylabel('% deviation from initial total energy')

if from_diag:
    extra = ''
else:
    extra='_manual'

plt.savefig(f'{plot_dir}/energy_evolution{extra}.pdf')
print(f'Plot made:\n{plot_dir}/energy_evolution{extra}.pdf')