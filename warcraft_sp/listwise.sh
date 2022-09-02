#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
tau=${3}
growth=${4}
seed=${5}
lr=${6}
tag= ${7}
id=${8}
echo ${tag}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --growth ${growth} --lr "${lr}" --tau "${tau}" --seed ${seed} --output_tag "${tag}" --index ${id}   > ./log/Listwise_${id}.log
exit 0
