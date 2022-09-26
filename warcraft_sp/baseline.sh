#!/usr/bin/env bash

modelname=${1}
loss=${2}
imgsz=${3}
seed=${4}
lr=${5}
tag=${6}
id=${7}
echo ${tag}

source ~/.bashrc
source warcraft_venv/bin/activate
python TestWarcraft.py --model ${modelname} --loss ${loss} --img_size ${imgsz} --seed ${seed} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/baseline_${id}.log
exit 0
