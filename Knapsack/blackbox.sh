#!/usr/bin/env bash

modelname=${1}
capacity=${2}
lambd=${3}
lr=${4}
tag=${5}
id=${6}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python testknapsack${modelname}.py --capacity ${capacity} --lambda_val "${lambda_val}" --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/blackbox_${id}.log
exit 0
