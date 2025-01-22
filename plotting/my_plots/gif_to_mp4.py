import ffmpy

file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-300sols_tracer_tophat-80_ref-5'
type = 'tracer'

if type == 'PV':
    folder = 'Stereo_ani'
elif type == 'tracer':
    folder = 'Tracer_stereo'

path = f'/data/home/sh1293/results/{file}/Plots/{folder}/animation'

ff = ffmpy.FFmpeg(
    inputs={f'{path}.gif': None},
    outputs={f'{path}.mp4': '-y'}
)
ff.run()
