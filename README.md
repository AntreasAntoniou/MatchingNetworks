# MatchingNetworks
An attempt at replicating the Matching Networks Paper

# Training a model
To train a model simply modify the experiment parameters in the [train_one_shot_learning_matching_network.py](https://github.com/AntreasAntoniou/MatchingNetworks/blob/master/train_one_shot_learning_matching_network.py) to match your requirements, for a one shot, 20-way experiment leave the parameters to default, for a 5-way one shot learning modify 
```
classes_per_set = 20
samples_per_class = 1
```
to 
```
classes_per_set = 5
samples_per_class = 1
```
# Acknowledgements
Special thanks to https://github.com/zergylord for his Matching Networks implementation of which parts were used for this implementation. More details at https://github.com/zergylord/oneshot

Additional thanks to my colleages https://github.com/gngdb, https://github.com/ZackHodari and https://github.com/artur-bekasov for reviewing my code and providing pointers.
