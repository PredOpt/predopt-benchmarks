#!/usr/bin/env bash

modelname=${1}
instance=${2}
sigma=${3}
nsamp=${4}
lr=${5}
tag=${6}
id=${7}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python test_matching.py --model ${modelname} --instance ${instance} --sigma "${sigma}" --num_samples ${nsamp} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/FY_${id}.log
exit 0
