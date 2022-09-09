#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
inptemp=${3}
trgttemp=${4}
nsamp=${5}
niter=${6}
k=${7}
seed=${8}
lr=${9}
tag=${10}
id=${11}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --input_noise_temp ${inptemp} --target_noise_temp ${trgttemp} --num_samples ${nsamp} --num_iter ${niter} --k ${k} --seed ${seed} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/imlerun_${id}.log
exit 0
