import os
import ffmpeg

results_dir = f'/data/home/sh1293/results/Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_alpha--1_working_long'

path = f'{results_dir}/Plots/Stereo_ani/'
delay = 1
print('making gif')
os.system(f'convert -delay {delay} {path}/*.pdf {path}/animation.gif')

print('making mp4')
(ffmpeg.input(f'{path}/animation.gif')
    .output(f'{path}/animation.mp4')
    .run(overwrite_output=True))