#!/usr/bin/env bash

modelname=${1}
capacity=${2}	
layer=${3}
hidden=${4}
lr=${5}
tag=${6}
id=${7}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python testknapsack${modelname}.py --capacity ${capacity} --n_layers ${layer} --n_hidden ${hidden} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/baseline_${id}.log
exit 0
