#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
inptemp=${3}
trgttemp=${4}
nsamp=${5}
niter=${6}
k=${7}
lr=${8}
tag= ${9}
id=${10}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --input_noise_temp ${inptemp} --target_noise_temp ${trgttemp} --num_samples ${nsamp} --num_iter ${niter} --k {k} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/imlerun_${id}.log
exit 0
