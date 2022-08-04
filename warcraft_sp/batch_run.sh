#!/usr/bin/env bash

a=${1}
b=${2}
c=${3}
d=${4}


source ~/.bashrc
source warcraft_venv/bin/activate
python Test${a}.py --img_size ${b} --output_tag ${c} --index ${d}   > ./log/4aug_runs_${d}.log
exit 0
