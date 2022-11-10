#!/usr/bin/env bash

modelname=${1}
loss=${2}
N=${3}
noise=${4}
deg=${5}
tau=${6}
growth=${7}
lr=${8}
tag=${9}
id=${10}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python TestSP.py --model ${modelname} --loss ${loss} --N ${N} --noise ${noise} --deg ${deg} --tau "${tau}" --growth ${growth} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/Listwise_${id}.log
exit 0
