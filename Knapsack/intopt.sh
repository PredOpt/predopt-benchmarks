#!/usr/bin/env bash

modelname=${1}
capacity=${2}
thr=${3}
damping=${4}
lr=${5}
tag=${6}
id=${7}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python testknapsack${modelname}.py --capacity ${capacity} --thr "${thr}" --damping "${damping}" --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/intopt_${id}.log
exit 0
