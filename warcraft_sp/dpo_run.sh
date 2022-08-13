#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
sig=${3}
nsamp=${4}
lr=${5}
tag= ${6}
id=${7}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --sigma ${sig} --num_samples ${nsamp} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/dporun_${id}.log
exit 0
