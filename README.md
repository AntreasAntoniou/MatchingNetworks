# Matching Networks Tensorflow Implementation
This repo provides code that replicated the results of the Matching
Networks for One Shot Learning paper on the Omniglot dataset.

## Getting the data ready
The code in the [training script](https://github.com/AntreasAntoniou/MatchingNetworks/blob/master/train_one_shot_learning_matching_network.py)
 uses a data provider that can build a dataset directly from a folder that contains the data.
 The folder structure required for the data provider to work is:
```
 Dataset
    ||______
    |       |
 class_0 class_1 ... class_N
    |       |___________________
    |                           |
samples for class_0    samples for class_1

```
Once a dataset in the above form is build then simply using:
```
data = dataset.FolderDatasetLoader(num_of_gpus=num_gpus, batch_size=batch_size, image_height=28, image_width=28,
                                   image_channels=1,
                                   train_val_test_split=(1200/1622, 211/1622, 211/162),
                                   samples_per_iter=1, num_workers=4,
                                   data_path="path/to/dataset", name="dataset_name",
                                   index_of_folder_indicating_class=-2, reset_stored_filepaths=False,
                                   num_samples_per_class=samples_per_class, num_classes_per_set=classes_per_set)
```
Will allow one to built a data loader for Matching Networks. The data provider can be used as demonstrated in the [training script](https://github.com/AntreasAntoniou/MatchingNetworks/blob/master/train_one_shot_learning_matching_network.py).

Sampling from the data loader is as simple as:
```
for sample_id, train_sample in enumerate(self.data.get_train_batches(total_batches=total_train_batches,
                                                                            augment_images=self.data_augmentation))
```
The data provider uses parallelization as well as batch sampling while tensorflow is training a step, such that there is minimal waiting time between loading a batch and passing it to tensorflow.

## Training a model
To train a model simply modify the experiment parameters in
the [train_one_shot_learning_matching_network.py](https://github.com/AntreasAntoniou/MatchingNetworks/blob/master/train_one_shot_learning_matching_network.py)
 to match your requirements, for a one shot, 20-way experiment leave
 the parameters to default, for a 5-way one shot learning modify

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

## Features
The code supports automatic checkpointing as well as statistics saving.
 It uses 1200 classes for training, 211 classes for testing and 211 classes for validation. We save the latest 5 trained models as well as keep track
 of the models that perform best on the validation set. After training all epochs, we take the best validation model
 and produce test statistics.
 Furthermore the number of classes and samples per class can be modified and the code will be able to handle any
 combinations that do not exceed the available memory. As an additional feature we have added support for full context
 embeddings in our implementation.

Our implementation uses the omniglot dataset, but one can easily add a new data provider and then build a new
experiment by passing the data provider to the ExperimentBuilder class and the system should work with it, as long as
 it provides the batches in the same way as our data provider, more details can be found in data.py

## Acknowledgements
Special thanks to https://github.com/zergylord for his Matching Networks
 implementation of which parts were used for this implementation. More
  details at https://github.com/zergylord/oneshot

Additional thanks to my colleagues https://github.com/gngdb,
 https://github.com/ZackHodari and https://github.com/artur-bekasov
  for reviewing my code and providing pointers.
