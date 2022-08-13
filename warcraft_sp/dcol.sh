#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
lr=${3}
tag= ${4}
id=${5}
echo ${tag}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/DCOLrun_${id}.log
exit 0
