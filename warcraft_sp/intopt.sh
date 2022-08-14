#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
thr=${3}
damping=${4}
lr=${5}
tag= ${6}
id=${7}
echo ${tag}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --thr "${lr}" --damping "${damping}" --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/DCOLrun_${id}.log
exit 0
