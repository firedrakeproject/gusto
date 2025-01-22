import os
import ffmpeg
from PIL import Image
import glob
from pdf2image import convert_from_path

file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-300sols_tracer_tophat-80_ref-5'
results_dir = f'/data/home/sh1293/results/{file}'

path = f'{results_dir}/Plots/Tracer_stereo'
delay = 1
print('making gif')
# os.system(f'convert -delay {delay} {path}/*.pdf {path}/animation.gif')

def create_gif_from_pdfs(path, delay):
    # Load all PDF images
    pdf_files = sorted(glob.glob(os.path.join(path, "*.pdf")))
    frames = []

    # Convert each PDF page to an image and add to frames list
    for pdf_file in pdf_files:
        # pdf_image = Image.open(pdf_file)
        # Convert PDF to RGB if not already in RGB mode
        # pdf_image = pdf_image.convert("RGB")
        # frames.append(pdf_image)
        image = convert_from_path(pdf_file)
        frames.extend(image)

    # Save as GIF
    if frames:
        frames[0].save(
            os.path.join(path, "animation.gif"),
            save_all=True,
            append_images=frames[1:],
            duration=delay,
            loop=0
        )
    else:
        print("No PDF files found.")

# Usage
create_gif_from_pdfs(f'{path}/', delay=2.5)



# print('making mp4')
# (ffmpeg.input(f'{path}/animation.gif')
#     .output(f'{path}/animation.mp4')
#     .run(overwrite_output=True))

# print(f'animation made for {file}')