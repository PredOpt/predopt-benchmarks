#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
lr=${3}
tag= ${3}
id=${5}


source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --lr ${lr} --output_tag ${tag} --index ${id}   > ./log/aug4evning_${id}.log
exit 0
