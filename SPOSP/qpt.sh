#!/usr/bin/env bash

modelname=${1}
N=${2}
noise=${3}
deg=${4}
mu=${5}
reg=${6}
lr=${7}
tag=${8}
id=${9}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python TestSP.py --model ${modelname} --N ${N} --noise ${noise} --deg ${deg} --mu "${mu}" --regularizer ${reg} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/QPT_${id}.log
exit 0
