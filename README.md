# MatchingNetworks
An attempt at replicating the Matching Networks for One Shot Learning Paper within Tensorflow.

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
And then run `python train_one_shot_learning_matching_network.py`

# Features
The code supports automatic checkpointing as well as statistics saving. It uses 1200 classes for training, 300 classes for testing and 122 classes for validation. We check for a better validation score at each epoch and produce test statistics when the validation score has improved. Furthermore the number of classes and samples per class can be modified and the code will be able to handle any combinations that do not exceed the available memory. As an additional feature we have added support for full context embeddings in our implementation.

Our implementation uses the omniglot dataset, but one can easily add a new data provider and then build a new experiment by passing the data provider to the ExperimentBuilder class and the system should work with it, as long as it provides the batches in the same way as our data provider, more details can be found in data.py

# Acknowledgements
Special thanks to https://github.com/zergylord for his Matching Networks implementation of which parts were used for this implementation. More details at https://github.com/zergylord/oneshot

Additional thanks to my colleagues https://github.com/gngdb, https://github.com/ZackHodari and https://github.com/artur-bekasov for reviewing my code and providing pointers.
