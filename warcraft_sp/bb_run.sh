#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
seed=${3}
lambda=${4}
lr=${5}
tag= ${6}
id=${7}
echo ${tag}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --seed ${seed} --lambda_val ${lambda} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/blackboxrun_${id}.log
exit 0
