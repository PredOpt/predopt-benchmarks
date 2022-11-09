#!/usr/bin/env bash

modelname=${1}
N=${2}
noise=${3}
deg=${4}
lr=${5}
tag=${6}
id=${7}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python TestSP.py --model ${modelname} --N ${N} --noise ${noise} --deg ${deg} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/baseline_${id}.log
exit 0
