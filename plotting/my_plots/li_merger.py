import matplotlib.pyplot as plt
import numpy as np
import functions as fcs
import os


file = 'step_trap_Bu2bp6Rop2_l500dt250df30'

path = '/data/home/sh1293/results/jupiter_sw'

frames = [0, 600, 850, 950, 1100, -1]

field_name = 'RelativeVorticity'

plot_dir = f'{path}/{file}/Plots/{field_name}'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

frames_to_days = 250*30/(2*np.pi/1.74e-4)
Omega = 1.74e-4
f0 = 2*Omega

field_structured, times = fcs.make_structured(file, field_name)

# breakpoint()

fig, axs = plt.subplots(3,2, figsize=(8,12))
vmin = 0
# vmax = np.max(field_structured[:,:,0])
vmax = f0

for i in range(6):
    print(i, end="\r")
    # time = frames[i]*frames_to_days
    time = times[frames[i]]/(2*np.pi/1.74e-4)
    axs[i//2][i%2].set_box_aspect(1)
    pcolor = field_structured[:,:,frames[i]].plot.imshow(ax=axs[i//2][i%2], x='x', y='y', cmap='RdBu_r', extend='both',
                                                         add_colorbar=False, vmin=vmin, vmax=vmax)
    axs[i//2][i%2].set_xlabel('')
    axs[i//2][i%2].set_ylabel('')
    axs[i//2][i%2].set_xticks([])
    axs[i//2][i%2].set_yticks([])
    axs[i//2][i%2].set_title(f'{int(time)} days')


cbar_ax = fig.add_axes([
    0.25,
    0.05,
    0.5,
    0.03
])
cb = fig.colorbar(pcolor, cax=cbar_ax, orientation='horizontal', extend='both', ticks=([0,0.25*f0, 0.5*f0, 0.75*f0, f0]))
cb.set_ticklabels([r'$0$', r'$f_0/4$', r'$f_0/2$', r'$3f_0/4$', f'$f_0$'])
cb.set_label('Relative Vorticity')

plt.savefig(f'{plot_dir}/comic_strip.pdf', bbox_inches='tight')
plt.close()
print(f'Plot made:\n{plot_dir}/comic_strip.pdf')