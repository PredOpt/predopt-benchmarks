Here, we generate the shortest path instances as described in `Smart “predict, then optimize” (2021)` paper.

To create the instances, first create a folder named `synthetic_path` and then run `python simulate_shortest_path.py`.

* Each instance is 5x5 grid and the we have to find the shortest path problem from the southwest corner of to the northeast corner. 
With this code, we generate training set of 100, 1000 and 5000 instances.
* Like the SPO paper, we also vary the noise and the degree parameters
