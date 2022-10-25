#!/usr/bin/env bash

modelname=${1}
loss=${2}
imgsz=${3}
sig=${4}
nsamp=${5}
seed=${6}
lr=${7}
tag=${8}
id=${9}

source ~/.bashrc
source warcraft_venv/bin/activate
python TestWarcraft.py --model ${modelname} --loss ${loss} --img_size ${imgsz} --sigma ${sig} --seed ${seed} --num_samples ${nsamp} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/DPOrun_${id}.log
exit 0
