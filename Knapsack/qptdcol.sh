#!/usr/bin/env bash

modelname=${1}
capacity=${2}
mu=${3}
lr=${4}
tag=${5}
id=${6}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python testknapsack.py --model ${modelname} --capacity ${capacity} --mu "${mu}" --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/qptdcol_${id}.log
exit 0
