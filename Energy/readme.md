This directory corresponds to the Energy-cost aware scheduling problem.

The data is included in `Trainer/prices2013.dat`
There exist three instances of the scheduling problem in the director `SchedulingInstances`. 
 The first, second, and third instances contain 10, 15, and 20 tasks, respectively.

To run an experiment use `testenergy.py`.
To reproduce the result of expriements run
```
python testenergy.py --scheduler True
```