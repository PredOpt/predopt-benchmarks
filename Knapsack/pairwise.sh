#!/usr/bin/env bash

modelname=${1}
capacity=${2}
margin=${3}
lr=${4}
tag=${5}
id=${6}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python testknapsack${modelname}.py --capacity ${capacity} --margin "${margin}" --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/pairwise_${id}.log
exit 0
