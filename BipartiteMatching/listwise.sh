#!/usr/bin/env bash

modelname=${1}
loss=${2}
instance=${3}
temp=${4}
lr=${5}
tag=${6}
id=${7}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python test_matching.py --model ${modelname} --loss ${loss} --instance ${instance} --tau ${temp} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/Listwise_${id}.log
exit 0
