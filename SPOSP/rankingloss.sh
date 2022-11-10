#!/usr/bin/env bash

modelname=${1}
loss=${2}
N=${3}
noise=${4}
deg=${5}
growth=${6}
lr=${7}
tag=${8}
id=${9}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python TestSP.py --model ${modelname} --loss ${loss} --N ${N} --noise ${noise} --deg ${deg} --growth ${growth} --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/Listwise_${id}.log
exit 0
