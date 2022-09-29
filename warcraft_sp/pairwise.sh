#!/usr/bin/env bash

modelname=${1}
loss=${2}
imgsz=${3}
tau=${4}
growth=${5}
seed=${6}
lr=${7}
tag=${8}
id=${9}
echo ${tag}

source ~/.bashrc
source warcraft_venv/bin/activate
python TestWarcraft.py --model ${modelname} --loss ${loss} --img_size ${imgsz} --growth ${growth} --lr "${lr}" --tau "${tau}" --seed ${seed} --output_tag "${tag}" --index ${id}   > ./log/Pairwisewise_${id}.log
exit 0
