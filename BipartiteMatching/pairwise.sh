#!/usr/bin/env bash

modelname=${1}
instance=${2}
margin=${3}
lr=${4}
tag=${5}
id=${6}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python test_matching_${modelname}.py --instance ${instance} --margin ${margin} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/Pairwise_${id}.log
exit 0
