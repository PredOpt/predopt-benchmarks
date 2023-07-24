The data is included in `Trainer/prices2013.dat`
There exist three instances of the scheduling problem in the director `SchedulingInstances`. 
 The first, second, and third instances contain 10, 15, and 20 tasks, respectively.

To run an experiment use `testenergy.py` in the following way:
```
python testenergy.py --model FenchelYoung --scheduler False --num_samples 1 --sigma 5 --instance 2 --lr 0.01
```
