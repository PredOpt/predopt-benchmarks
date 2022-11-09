#!/usr/bin/env bash

modelname=${1}
N=${2}
noise=${3}
deg=${4}
lambda=${5}
lr=${6}
tag=${7}
id=${8}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python TestSP.py --model ${modelname} --N ${N} --noise ${noise} --deg ${deg} --lambda_val ${lambda} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/DBB_${id}.log
exit 0
