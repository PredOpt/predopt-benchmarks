#!/bin/bash
echo 'downloading preprocessed CORA dataset...'
wget -cq https://bryanwilder.github.io/files/data_decisions_benchmarks.zip && unzip -qqn data_decisions_benchmarks.zip 
mkdir to_be_deleted
echo 'done'
echo 'cleaning...'
# delete zip
mv data_decisions_benchmarks.zip to_be_deleted

# delete non-relevant data
mv benchmarks_release/budget_allocation_data.pickle to_be_deleted
mv benchmarks_release/diverse_recommendation_data.pickle to_be_deleted
mv benchmarks_release/readme.txt to_be_deleted
echo 'done'
echo 'delete "to_be_deleted" dir'
