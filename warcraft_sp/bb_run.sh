#!/usr/bin/env bash

modelname=${1}
loss=${2}
imgsz=${3}
lambda=${4}
seed=${5}
lr=${6}
tag=${7}
id=${8}
echo ${tag}

source ~/.bashrc
source warcraft_venv/bin/activate
python TestWarcraft.py --model ${modelname} --loss ${loss} --img_size ${imgsz} --seed ${seed} --lambda_val ${lambda} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/blackboxrun_${id}.log
exit 0
