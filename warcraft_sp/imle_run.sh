#!/usr/bin/env bash

modelname=${1}
loss=${2}
imgsz=${3}
temp=${4}
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
python TestWarcraft.py --model ${modelname} --loss ${loss} --img_size ${imgsz} --temperature ${temp} --beta "${beta}" --nb_samples ${nsamp} --nb_iterations ${niter} --k ${k} --seed ${seed} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/imlerun_${id}.log
exit 0
