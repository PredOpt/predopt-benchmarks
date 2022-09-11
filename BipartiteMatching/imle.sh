#!/usr/bin/env bash

modelname=${1}
instance=${2}
inptemp=${3}
targettemp=${4}
beta=${5}
k=${6}
nsamp=${7}
niter=${8}
lr=${9}
tag=${10}
id=${11}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python test_matching_${modelname}.py --instance ${instance} --num_samples ${nsamp} --beta "${beta}" --input_noise_temp "${inptemp}" --target_noise_temp "${targettemp}" --k ${k}  --num_iter ${niter} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/IMLE_${id}.log
exit 0
