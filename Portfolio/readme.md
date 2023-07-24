This directory corresponds to the portfolio optimization problem.

First download the data by running
```
gdown 17P_hXJhvVsdcc1dGBGhAn5r5rkhNMsWj
```
Then extract the data by running
```
tar -xvzf PortfolioData.tar.gz
```

Then the test_sp.py can be used to run an experiment, in the following way:
```
python test_sp.py --model SPO --scheduler True --N 1000 --noise 1 --deg 16 --lr 0.05
```
Please change the name of the model and the hyperaprameters accordingly.
