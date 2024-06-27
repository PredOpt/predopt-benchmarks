This directory corresponds to the diverse bipartite matching problem.

To download the data run
```
./get_data.sh
```
This will create a folder `data/` and save the data files inside that directory.
Alternatively, you can download the bipartite matching datset from the repository: https://doi.org/10.48804/KT2P3Z and extract the `tar.gz` file.


To run experiments use `test_matching.py`.
To reproduce the result of expriements run
```
python test_matching.py --scheduler True
```

