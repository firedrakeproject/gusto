#!/bin/sh

set -e

# want to type ./animation.sh <filepath> <variable> <frame rate>
FILEPATH=$1
VARIABLE=$2
FRAME_RATE=$3
FOLDER=$4

# echo "${FOLDER}"

if [ $# -ne 4 ]; then
    echo "Usage: $0 <filepath> <variable> <frame_rate> <folder>"
    exit 1
fi

BASE_DIR="/data/home/sh1293/results/${FOLDER}/${FILEPATH}/Plots/${VARIABLE}"

python /data/home/sh1293/firedrake-feb26/gusto/plotting/my_plots/PV_square.py "${FILEPATH}" "${VARIABLE}" "${FOLDER}"
python /data/home/sh1293/firedrake-feb26/gusto/plotting/my_plots/pdf_to_gif.py "${BASE_DIR}"
python /data/home/sh1293/firedrake-feb26/gusto/plotting/my_plots/gif_to_mp4.py "${BASE_DIR}/animation.gif" "${FRAME_RATE}"