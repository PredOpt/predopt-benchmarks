#!/usr/bin/env bash

modelname=${1}
loss=${2}
capacity=${3}
temp=${4}
lr=${5}
tag=${6}
id=${7}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python testknapsack.py --model ${modelname} --loss ${loss} --capacity ${capacity} --tau "${temp}" --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/listwise_${id}.log
exit 0
