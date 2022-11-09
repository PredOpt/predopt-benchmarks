#!/usr/bin/env bash

modelname=${1}
N=${2}
noise=${3}
deg=${4}
temp=${5}
beta=${6}
k=${7}
nsample=${8}
niter=${9}
lr=${10}
tag=${11}
id=${12}
echo ${tag}

source ~/.bashrc
source ../warcraft_sp/warcraft_venv/bin/activate
python TestSP.py --model ${modelname} --N ${N} --noise ${noise} --deg ${deg} --temperature ${temp} --beta ${beta} --k ${k} --nb_samples ${nsample} --nb_iterations ${niter}  --lr "${lr}" --output_tag "${tag}" --index ${id}   > ./log/IMLE_${id}.log
exit 0
