#!/usr/bin/env bash

modelname=${1}
capacity=${2}
inp=${3}
targ=${4}
beta=${5}
k=${6}
num_sample=${7}
nb_iter=${8}
lr=${9}
tag=${10}
id=${11}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python testknapsack${modelname}.py --capacity ${capacity} --beta "{beta}" --input_noise_temp "${inp}" --target_noise_temp "${targ}" --k ${k}  --nb_samples ${num_sample} --nb_iterations ${nb_iter} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/imle_${id}.log
exit 0
