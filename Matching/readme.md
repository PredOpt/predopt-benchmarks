To download the data run
```
./get_data.sh
```
This will create a folder `data/` and save the data files inside that directory.

To run an experiment use `test_matching.py` in the following way:
```
python test_matching.py --model DCOL --instance 2 --mu 100 --lr 0.01 --scheduler False
```

