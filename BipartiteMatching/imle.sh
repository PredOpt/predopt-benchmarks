#!/usr/bin/env bash

modelname=${1}
instance=${2}
inptemp=${3}
targettemp=${4}
k=${5}
niter=${6}
lr=${7}
tag=${8}
id=${9}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python test_matching_${modelname}.py --instance ${instance} --input_noise_temp "${inptemp}" --target_noise_temp "${targettemp}" --k ${k}  --num_iter ${niter} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/IMLE_${id}.log
exit 0
