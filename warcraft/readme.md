This directory corresponds to the Warcraft shortest path problem/

To download the data go inside the data folder and run 
```
./data_prep.sh
```
This wil download and preprocess the data.
Then an experiment can be run using the `TestWarcraft.py` file, by running
```
python TestWarcraft.py --model ${modelname} --loss ${loss} --img_size ${imgsz} --growth ${growth} --seed ${seed} --lr "${lr}"
```
