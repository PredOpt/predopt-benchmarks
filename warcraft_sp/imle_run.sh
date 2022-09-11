#!/usr/bin/env bash

modelname=${1}
imgsz=${2}
inptemp=${3}
trgttemp=${4}
beta=${5}
nsamp=${6}
niter=${7}
k=${8}
seed=${9}
lr=${10}
tag=${11}
id=${12}

source ~/.bashrc
source warcraft_venv/bin/activate
python Test${modelname}.py --img_size ${imgsz} --input_noise_temp ${inptemp} --target_noise_temp ${trgttemp} --beta "{beta}" --num_samples ${nsamp} --num_iter ${niter} --k ${k} --seed ${seed} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/imlerun_${id}.log
exit 0
