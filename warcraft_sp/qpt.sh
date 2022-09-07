#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
seed=${3}
mu=${4}
lr=${5}
tag= ${6}
id=${7}
echo ${tag}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --seed ${seed} --mu "${mu}" --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/DCOLrun_${id}.log
exit 0
