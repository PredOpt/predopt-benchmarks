#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
tau=${3}
growth=${4}
lr=${5}
tag= ${6}
id=${7}
echo ${tag}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --tau${tau} --growth ${growth} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/Rankingrun2_${id}.log
exit 0
