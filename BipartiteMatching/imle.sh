#!/usr/bin/env bash

modelname=${1}
instance=${2}
temp=${3}
beta=${4}
k=${5}
nsamp=${6}
niter=${7}
lr=${8}
tag=${9}
id=${10}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python test_matching.py --model ${modelname} --instance ${instance} --num_samples ${nsamp} --beta "${beta}" --itemperature "${temp}" --target_noise_temp "${targettemp}" --k ${k}  --num_iter ${niter} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/IMLE_${id}.log
exit 0
