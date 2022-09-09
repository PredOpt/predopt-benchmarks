#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
growth=${3}
seed=${4}
lr=${5}
tag=${6}
id=${7}
echo ${tag}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --growth ${growth} --seed ${seed} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/Rankingrun1_${id}.log
exit 0
