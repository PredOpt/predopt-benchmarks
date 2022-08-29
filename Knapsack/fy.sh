#!/usr/bin/env bash

modelname=${1}
capacity=${2}
sigma=${3}
nsamp=${4}
lr=${5}
tag=${6}
id=${7}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python testknapsack${modelname}.py --capacity ${capacity} --sigma "${sigma}" --num_samples ${nsamp} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/fy_${id}.log
exit 0
