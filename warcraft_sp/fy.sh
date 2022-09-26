#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
sig=${3}
nsamp=${4}
seed=${5}
lr=${6}
tag=${7}
id=${8}

source ~/.bashrc
source warcraft_venv/bin/activate
python TestWarcraft.py --model ${modelname} --img_size ${imgsz} --sigma ${sig} --seed ${seed} --num_samples ${nsamp} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/FYrun_${id}.log
exit 0
