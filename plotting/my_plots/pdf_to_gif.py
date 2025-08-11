import os
import glob
from pdf2image import convert_from_path
import sys
import pdb
from tqdm import tqdm

# filepath = 'intruder_Bu10b1p5Rop23_l1dt250df1'
# path = f'/data/home/sh1293/results/jupiter_sw/{filepath}/Plots/PV_square'

if len(sys.argv) != 3 and len(sys.argv) != 4:
    print('Wrong number of arguments')
    print(len(sys.argv))
    sys.exit(1)

path = sys.argv[1]
delay = float(sys.argv[2])
if len(sys.argv) == 4:
    extra_name = sys.argv[3]
else:
    extra_name = ''

print(f'making gif: animation{extra_name}.gif')

# pdb.set_trace()

if os.path.isfile(f'{path}/animation{extra_name}.gif'):
    os.remove(f'{path}/animation{extra_name}.gif')

def create_gif_from_pdfs(path, delay):
    pdf_files = sorted(glob.glob(os.path.join(path, "*.pdf")))
    frames = []

    for pdf_file in tqdm(pdf_files, desc='Converting PDFs', unit='file'):
        image = convert_from_path(pdf_file)
        frames.extend(image)

    if frames:
        frames[0].save(
            os.path.join(path, f'animation{extra_name}.gif'),
            save_all=True,
            append_images=frames[1:],
            duration=delay,
            loop=0
        )
    else:
        print("No PDF files found.")

# Usage
create_gif_from_pdfs(f'{path}/', delay=delay)

print(f'gif made:\n {path}/animation{extra_name}.gif')