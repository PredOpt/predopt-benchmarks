#!/bin/bash
echo 'downloading preprocessed CORA dataset...'
wget -cq https://bryanwilder.github.io/files/data_decisions_benchmarks.zip && unzip -qqn data_decisions_benchmarks.zip 

echo 'done'
echo 'cleaning...'
# delete zip
rm data_decisions_benchmarks.zip 

# delete non-relevant data
rm benchmarks_release/budget_allocation_data.pickle benchmarks_release/diverse_recommendation_data.pickle benchmarks_release/readme.txt
rm benchmarks_release/make_cora_dataset.py make_cora_dataset.py
echo 'done'