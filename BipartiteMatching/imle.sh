#!/usr/bin/env bash

modelname=${1}
instance=${2}
inptemp=${3}
targettemp=${4}
k=${5}
nsamp=${6}
niter=${7}
lr=${8}
tag=${9}
id=${10}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python test_matching_${modelname}.py --instance ${instance} --num_samples ${nsamp} --input_noise_temp "${inptemp}" --target_noise_temp "${targettemp}" --k ${k}  --num_iter ${niter} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/IMLE_${id}.log
exit 0
