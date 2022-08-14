#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
mu=${3}
lr=${4}
tag= ${5}
id=${6}
echo ${tag}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --mu "${mu}" --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/DCOLrun_${id}.log
exit 0
