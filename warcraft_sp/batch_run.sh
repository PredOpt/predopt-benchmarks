#!/usr/bin/env bash

modelname=${1}
instance=${2}
lr=${3}
tag= ${4}
id=${5}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python test_matching_${modelname}.py --instance ${instance} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/baseline_${id}.log
exit 0
