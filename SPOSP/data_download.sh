#!/usr/bin/env bash
git clone git@github.com:paulgrigas/SmartPredictThenOptimize.git
mkdir SyntheticData
julia datagen.jl
rm -rf SmartPredictThenOptimize