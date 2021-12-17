### In this dataset, we formulate a shortest path problem on Twitter user network.
* We create the graph from ego network of one of the user.
* The Data is downloaded from http://snap.stanford.edu/data/ego-Twitter.html
* For data generation and feature engineering, please Refer to McAuley, Julian J., and Jure Leskovec. "Learning to discover social circles in ego networks." NIPS. Vol. 2012. 2012. 

* We only use the structure of the graph. But this data has no edge weights. So the edge weights are generated synthetically.

* We use the same network for all the instances. But each instance has different different start and end node. So, in the equality constraint of the form `Ax=b`, the `b` vector is different for separate instances.

* We have in total 12300 instances.
* We solve the problem using Gurobi.
