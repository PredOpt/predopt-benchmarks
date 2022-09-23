#!/usr/bin/env bash

modelname=${1}
loss=${2}
instance=${3}
lr=${4}
tag=${5}
id=${6}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python test_matching.py --model ${modelname} --loss ${loss} --instance ${instance} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/Ranking_${id}.log
exit 0
