#!/bin/bash
echo 'downloading preprocessed CORA dataset...'
gdown 1MNy9HCVkJykRbXf6XXI9D7lggF0UF8MP
tar -xvzf data.tar.gz
echo 'cleaning...'
rm data.tar.gz
cd data/
python make_cora_dataset.py
echo 'done'
