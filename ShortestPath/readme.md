First download the data by running
```
gdown 1_hSLqD89nk7yVcW79QvG5SzbnJ_IRmBY
```
Then extract the data by running
```
tar -xvzf ShortestPathData.tar.gz
```

Then the test_sp.py can be used to run an experiment, in the following way:
```
python test_sp.py --model IMLE --scheduler True --temperature 0.5 --nb_iterations 1 --beta 1.0 --k 5 --nb_iterations 1 --nb_samples 1 --N 1000 --noise 0.5 --deg 6 --lr 0.5
```
Please change the name of the model and the hyperaprameters accordingly.
