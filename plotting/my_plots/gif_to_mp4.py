import ffmpy
import sys
import os
import subprocess

if len(sys.argv) != 3:
    print('Wrong number of arguments')
    sys.exit(1)

path = sys.argv[1]
name = path.split('.')[0]
print(name)

def get_gif_fps_and_duration(path):
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output = result.stdout.strip().split('\n')
    fps = output[0]  # e.g. "10/1"
    duration = float(output[1])
    # Convert fps fraction to float
    num, denom = map(int, fps.split('/'))
    fps_float = num / denom
    return fps_float, duration

# def get_gif_fps_and_duration(path):
#     import imageio
#     gif = imageio.get_reader(f'{path}')
#     duration = 0
#     i=0
#     for frame in gif:
#         duration += gif.get_meta_data(index=frame)["duration"]
#         i += 1
#     fps = i/duration
#     return fps, duration

gif_fps, _ = get_gif_fps_and_duration(f'{name}.gif')

target_fps = float(sys.argv[2])
speed_up = gif_fps/target_fps

if os.path.isfile(f'{name}.mp4'):
    os.remove(f'{name}.mp4')

ff = ffmpy.FFmpeg(
    inputs={f'{name}.gif': None},
    outputs={f'{name}.mp4': (
                '-y '
                f'-filter_complex "[0:v] fps={target_fps},setpts={speed_up}*PTS,scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                '-c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p '
                '-c:a aac -movflags +faststart'
            )})
ff.run()

file = name.split('jupiter_sw/')[1]

print(f'Filename for moving file:\n{file}.mp4')

print(f'Animation made:\n {name}.mp4')