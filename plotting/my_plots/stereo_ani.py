import os
import ffmpeg
from PIL import Image
import glob
from pdf2image import convert_from_path

file = 'Relax_to_pole_and_CO2/annular_vortex_mars_65-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_tophat-80_ref-4'

ref_lev = 4

start_frame = 2770
end_frame = 2820

dt = (0.5)**(ref_lev-4) * 2.5

results_dir = f'/data/home/sh1293/results/{file}'

path = f'{results_dir}/Plots/Stereo_ani'
print('making gif')
# os.system(f'convert -delay {delay} {path}/*.pdf {path}/animation.gif')

def create_gif_from_pdfs(path, delay):
    # Load all PDF images
    pdf_files = sorted(glob.glob(os.path.join(path, "*.pdf")))
    frames = []
    filtered_pdfs = [pdf for pdf in pdf_files if start_frame <= int(os.path.basename(pdf).split('.')[0]) <= end_frame]
    # Convert each PDF page to an image and add to frames list
    for pdf_file in filtered_pdfs:
        print(os.path.basename(pdf_file).split('.')[0])
        # pdf_image = Image.open(pdf_file)
        # Convert PDF to RGB if not already in RGB mode
        # pdf_image = pdf_image.convert("RGB")
        # frames.append(pdf_image)
        image = convert_from_path(pdf_file)
        frames.extend(image)

    # Save as GIF
    if frames:
        frames[0].save(
            os.path.join(path, f'animation_{start_frame}-{end_frame}.gif'),
            save_all=True,
            append_images=frames[1:],
            duration=delay,
            loop=0
        )
    else:
        print("No PDF files found.")

# Usage
create_gif_from_pdfs(f'{path}/', delay=dt)



# print('making mp4')
# (ffmpeg.input(f'{path}/animation.gif')
#     .output(f'{path}/animation.mp4')
#     .run(overwrite_output=True))

# print(f'animation made for {file}')