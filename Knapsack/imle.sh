#!/usr/bin/env bash

modelname=${1}
capacity=${2}
inp=${3}
targ=${4}
k=${5}
num_sample=${6}
nb_iter=${7}
lr=${8}
tag=${9}
id=${10}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python testknapsack${modelname}.py --capacity ${capacity} --input_noise_temp "${inp}" --target_noise_temp "${targ}" --k ${k}  --nb_samples ${num_sample} --nb_iterations ${nb_iter} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/imle_${id}.log
exit 0
