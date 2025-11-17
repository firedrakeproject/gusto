#!/bin/sh

set -e

# want to type ./animation.sh <filepath> <variable> <frame rate>
FILEPATH=$1
VARIABLE=$2
FRAME_RATE=$3

if [ $# -ne 3 ]; then
    echo "Usage: $0 <filepath> <variable> <frame_rate>"
    exit 1
fi

BASE_DIR="/data/home/sh1293/results/jupiter_sw/${FILEPATH}/Plots/${VARIABLE}"

python /data/home/sh1293/firedrake-real-opt_jun25/src/gusto/plotting/my_plots/PV_square.py "${FILEPATH}" "${VARIABLE}"
python /data/home/sh1293/firedrake-real-opt_jun25/src/gusto/plotting/my_plots/pdf_to_gif.py "${BASE_DIR}"
python /data/home/sh1293/firedrake-real-opt_jun25/src/gusto/plotting/my_plots/gif_to_mp4.py "${BASE_DIR}/animation.gif" "${FRAME_RATE}"