#!/usr/bin/env bash

modelname=${1}
loss=${2}
capacity=${3}
lr=${4}
tag=${5}
id=${6}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python testknapsack.py --model ${modelname} --loss ${loss} --capacity ${capacity} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/ranking_${id}.log
exit 0
