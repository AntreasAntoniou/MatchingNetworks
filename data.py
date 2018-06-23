import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import tqdm
import concurrent.futures
import pickle
from skimage import transform

def augment_image(image, k, channels):
    if channels==1:
        image = image[:, :, 0]
    image = transform.rotate(image, angle=k, resize=False, center=None, order=1, mode='constant',
                                  cval=0, clip=True, preserve_range=False)
    if channels==1:
        image = np.expand_dims(image, axis=2)
    return image

class MatchingNetworkDatasetParallel(Dataset):
    def __init__(self, batch_size, reverse_channels, num_of_gpus, image_height, image_width, image_channels,
                 train_val_test_split, num_classes_per_set, num_samples_per_class,
                 data_path, dataset_name,  indexes_of_folders_indicating_class,
                 seed=100, reset_stored_filepaths=False, labels_as_int=False):
        """
        :param batch_size: The batch size to use for the data loader
        :param last_training_class_index: The final index for the training set, used to restrict the training set
        if needed. E.g. if training set is 1200 classes and last_training_class_index=900 then only the first 900
        classes will be used
        :param reverse_channels: A boolean indicating whether we need to reverse the colour channels e.g. RGB to BGR
        :param num_of_gpus: Number of gpus to use for training
        :param gen_batches: How many batches to use from the validation set for the end of epoch generations
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.indexes_of_folders_indicating_class = indexes_of_folders_indicating_class
        self.labels_as_int = labels_as_int
        self.train_val_test_split = train_val_test_split
        self.current_dataset_name = "train"
        self.reset_stored_filepaths = reset_stored_filepaths
        self.x_train, self.x_val, self.x_test = self.load_dataset()
        self.num_of_gpus = num_of_gpus
        self.batch_size = batch_size
        self.reverse_channels = reverse_channels
        self.image_height, self.image_width, self.image_channel = image_height, image_width, image_channels
        self.train_index = 0
        self.val_index = 0
        self.test_index = 0
        self.init_seed = {"train": seed, "val": seed, "test": seed}
        self.seed = {"train": seed, "val": seed, "test": seed}
        self.augment_images = False
        self.num_samples_per_class = num_samples_per_class
        self.num_classes_per_set = num_classes_per_set


        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datasets = {"train": self.x_train,
                         "val": self.x_val,
                         "test": self.x_test}
        self.dataset_size_dict = {"train": {key: len(self.x_train[key]) for key in list(self.x_train.keys())},
                                  "val": {key: len(self.x_val[key]) for key in list(self.x_val.keys())},
                                  "test": {key: len(self.x_test[key]) for key in list(self.x_test.keys())}}
        self.label_set = self.get_label_set()
        self.data_length = {name: np.sum([len(self.datasets[name][key])
                         for key in self.datasets[name]]) for name in self.datasets.keys()}

        print("data", self.data_length)
        #print(self.datasets)

    def load_dataset(self):
        data_image_paths, index_to_label_name_dict_file, label_to_index = self.load_datapaths()
        total_label_types = len(data_image_paths)
        print(total_label_types)
        # data_image_paths = self.shuffle(data_image_paths)
        x_train_id, x_val_id, x_test_id = int(self.train_val_test_split[0] * total_label_types), \
                                          int(np.sum(self.train_val_test_split[:2]) * total_label_types), \
                                          int(total_label_types)
        print(x_train_id, x_val_id, x_test_id)
        x_train_classes = (class_key for class_key in list(data_image_paths.keys())[:x_train_id])
        x_val_classes = (class_key for class_key in list(data_image_paths.keys())[x_train_id:x_val_id])
        x_test_classes = (class_key for class_key in list(data_image_paths.keys())[x_val_id:x_test_id])
        x_train, x_val, x_test = {class_key: data_image_paths[class_key] for class_key in x_train_classes}, \
                                 {class_key: data_image_paths[class_key] for class_key in x_val_classes}, \
                                 {class_key: data_image_paths[class_key] for class_key in x_test_classes},
        return x_train, x_val, x_test

    def load_datapaths(self):
        data_path_file = "datasets/{}.pkl".format(self.dataset_name)
        self.index_to_label_name_dict_file = "datasets/map_to_label_name_{}.pkl".format(self.dataset_name)
        self.label_name_to_map_dict_file = "datasets/label_name_to_map_{}.pkl".format(self.dataset_name)

        if self.reset_stored_filepaths == True:
            if os.path.exists(data_path_file):
                os.remove(data_path_file)
                self.reset_stored_filepaths=False

        try:
            data_image_paths = self.load_dict(data_path_file)
            label_to_index = self.load_dict(name=self.label_name_to_map_dict_file)
            index_to_label_name_dict_file = self.load_dict(name=self.index_to_label_name_dict_file)
            return data_image_paths, index_to_label_name_dict_file, label_to_index
        except:
            print("Mapped data paths can't be found, remapping paths..")
            data_image_paths, code_to_label_name, label_name_to_code = self.get_data_paths()
            self.save_dict(data_image_paths, name=data_path_file)
            self.save_dict(code_to_label_name, name=self.index_to_label_name_dict_file)
            self.save_dict(label_name_to_code, name=self.label_name_to_map_dict_file)
            return self.load_datapaths()

    def save_dict(self, obj, name):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_dict(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def load_test_image(self, filepath):
        try:
            image = cv2.imread(filepath)
            image = cv2.resize(image, dsize=(28, 28))
        except RuntimeWarning:
            os.system("convert {} -strip {}".format(filepath, filepath))
            print("converting")
            image = cv2.imread(filepath)
            image = cv2.resize(image, dsize=(28, 28))
        except:
            print("Broken image")
            os.remove(filepath)

        if image is not None:
            return filepath
        else:
            os.remove(filepath)
            return None

    def get_data_paths(self):
        print("Get images from", self.data_path)
        data_image_path_list_raw = []
        labels = set()
        for subdir, dir, files in os.walk(self.data_path):
            for file in files:
                if (".jpeg") in file.lower() or (".png") in file.lower() or (".jpg") in file.lower():
                    filepath = os.path.join(subdir, file)
                    label = self.get_label_from_path(filepath)
                    data_image_path_list_raw.append(filepath)
                    labels.add(label)

        labels = sorted(labels)
        idx_to_label_name = {idx: label for idx, label in enumerate(labels)}
        label_name_to_idx = {label: idx for idx, label in enumerate(labels)}
        data_image_path_dict = {idx: [] for idx in list(idx_to_label_name.keys())}
        with tqdm.tqdm(total=len(data_image_path_list_raw)) as pbar_error:
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                # Process the list of files, but split the work across the process pool to use all CPUs!
                for image_file in executor.map(self.load_test_image, (data_image_path_list_raw)):
                    pbar_error.update(1)
                    if image_file is not None:
                        label = self.get_label_from_path(image_file)
                        data_image_path_dict[label_name_to_idx[label]].append(image_file)


        return data_image_path_dict, idx_to_label_name, label_name_to_idx

    def get_label_set(self):
        index_to_label_name_dict_file = self.load_dict(name=self.index_to_label_name_dict_file)
        return set(list(index_to_label_name_dict_file.keys()))

    def get_index_from_label(self, label):
        label_to_index = self.load_dict(name=self.label_name_to_map_dict_file)
        return label_to_index[label]

    def get_label_from_index(self, index):
        index_to_label_name = self.load_dict(name=self.index_to_label_name_dict_file)
        return index_to_label_name[index]

    def get_label_from_path(self, filepath):
        label_bits = filepath.split("/")
        label = "_".join([label_bits[idx] for idx in self.indexes_of_folders_indicating_class])
        if self.labels_as_int:
            label = int(label)
        return label

    def load_image(self, image_path, channels):

        image = cv2.imread(image_path)[:, :, :channels]
        image = cv2.resize(image, dsize=(self.image_height, self.image_width))

        if channels==1:
            image = np.expand_dims(image, axis=2)

        return image

    def load_batch(self, batch_image_paths):

        image_batch = []
        image_paths = []

        for image_path in batch_image_paths:
            image_paths.append(image_path)

        for image_path in image_paths:
            image = self.load_image(image_path=image_path, channels=self.image_channel)
            image_batch.append(image)

        image_batch = np.array(image_batch, dtype=np.float32)
        image_batch = self.preprocess_data(image_batch)

        return image_batch

    def preprocess_data(self, x):
        """
        Preprocesses data such that their values lie in the -1.0 to 1.0 range so that the tanh activation gen output
        can work properly
        :param x: A data batch to preprocess
        :return: A preprocessed data batch
        """
        x = x / 255.0
        x = 2 * x - 1
        x_shape = x.shape
        x = np.reshape(x, (-1, x_shape[-3], x_shape[-2], x_shape[-1]))
        if self.reverse_channels is True:
            reverse_photos = np.ones(shape=x.shape)
            for channel in range(x.shape[-1]):
                reverse_photos[:, :, :, x.shape[-1] - 1 - channel] = x[:, :, :, channel]
            x = reverse_photos
        x = x.reshape(x_shape)
        # print(x.mean(), x.min(), x.max())
        return x

    def reconstruct_original(self, x):
        """
        Applies the reverse operations that preprocess_data() applies such that the data returns to their original form
        :param x: A batch of data to reconstruct
        :return: A reconstructed batch of data
        """
        x = (x + 1) / 2
        x = x * 255.0
        return x

    def shuffle(self, x):
        """
        Shuffles the data batch along it's first axis
        :param x: A data batch
        :return: A shuffled data batch
        """
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        return x

    def get_set(self, dataset_name, seed, augment_images=False):
        """
        Generates a data batch to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc
        :return: A data batch
        """
        rng = np.random.RandomState(seed)
        selected_classes = rng.choice(list(self.dataset_size_dict[dataset_name].keys()),
                                      size=self.num_classes_per_set, replace=False)
        target_class = rng.choice(selected_classes, size=1, replace=False)[0]
        k_list = rng.randint(0, 3, size=self.num_classes_per_set)
        k_dict = {selected_class: k_item for (selected_class, k_item) in zip(selected_classes, k_list)}
        episode_labels = [i for i in range(self.num_classes_per_set)]
        class_to_episode_label = {selected_class: episode_label for (selected_class, episode_label) in
                                  zip(selected_classes, episode_labels)}

        support_set_images = []
        support_set_labels = []

        for class_entry in selected_classes:
            choose_samples_list = rng.choice(self.dataset_size_dict[dataset_name][class_entry],
                                             size=self.num_samples_per_class, replace=True)
            class_image_samples = []
            class_labels = []
            for sample in choose_samples_list:
                choose_samples = self.datasets[dataset_name][class_entry][sample]
                x_class_data = self.load_batch([choose_samples])[0]
                if augment_images is True:
                    k = k_dict[class_entry]
                    x_class_data = augment_image(image=x_class_data, k=k*90, channels=self.image_channel)
                class_image_samples.append(x_class_data)
                class_labels.append(int(class_to_episode_label[class_entry]))
            support_set_images.append(class_image_samples)
            support_set_labels.append(class_labels)

        support_set_images = np.array(support_set_images, dtype=np.float32)
        support_set_labels = np.array(support_set_labels, dtype=np.int32)

        target_sample = rng.choice(self.dataset_size_dict[dataset_name][target_class], size=1,
                                         replace=True)[0]
        choose_samples = self.datasets[dataset_name][target_class][target_sample]
        target_set_image = self.load_batch([choose_samples])[0]
        if augment_images is True:
            k = k_dict[target_class]
            target_set_image = augment_image(image=target_set_image, k=k * 90, channels=self.image_channel)
        target_set_label = int(class_to_episode_label[target_class])

        return support_set_images, target_set_image, support_set_labels, target_set_label

    def __len__(self):
        total_samples = self.data_length[self.current_dataset_name]
        return total_samples

    def length(self, dataset_name):
        self.switch_set(dataset_name=dataset_name)
        return len(self)

    def set_augmentation(self, augment_images):
        self.augment_images = augment_images

    def switch_set(self, dataset_name, seed=100):
        self.current_dataset_name = dataset_name
        if dataset_name=="train":
            self.update_seed(dataset_name=dataset_name, seed=seed)

    def update_seed(self, dataset_name, seed=100):
        self.init_seed[dataset_name] = seed

    def __getitem__(self, idx):
        support_set_images, target_set_image, support_set_labels, target_set_label = \
            self.get_set(self.current_dataset_name, seed=self.init_seed[self.current_dataset_name] + idx, augment_images=self.augment_images)
        data_point = {"support_set_images": support_set_images, "target_set_image": target_set_image,
                      "support_set_labels": support_set_labels, "target_set_label": target_set_label}
        self.seed[self.current_dataset_name] = self.seed[self.current_dataset_name] + 1
        return data_point

    def reset_seed(self):
        self.seed = self.init_seed

class MatchingNetworkLoader(object):
    def __init__(self, name, num_of_gpus, batch_size, image_height, image_width, image_channels, num_classes_per_set, data_path,
                 num_samples_per_class, train_val_test_split,
                 samples_per_iter=1, num_workers=4, reverse_channels=False, seed=100, labels_as_int=False):

        self.zip_dir = "datasets/{}.zip".format(name)
        self.data_folder_dir = "datasets/{}".format(name)
        self.datasets_dir = "datasets/"
        self.num_of_gpus = num_of_gpus
        self.batch_size = batch_size
        self.samples_per_iter = samples_per_iter
        self.num_workers = num_workers
        self.total_train_iters_produced = 0

        self.dataset = self.get_dataset(batch_size, reverse_channels, num_of_gpus, image_height, image_width, image_channels,
                 train_val_test_split, num_classes_per_set, num_samples_per_class, seed=seed,
                 reset_stored_filepaths=False, data_path=data_path, labels_as_int=labels_as_int)


        self.batches_per_iter = samples_per_iter
        self.full_data_length = self.dataset.data_length

    def get_dataloader(self, shuffle=False):
        return DataLoader(self.dataset, batch_size=(self.num_of_gpus * self.batch_size * self.samples_per_iter),
                                    shuffle=shuffle, num_workers=self.num_workers, drop_last=True)

    def get_dataset(self, batch_size, reverse_channels, num_of_gpus, image_height, image_width, image_channels,
                 train_val_test_split, num_classes_per_set, num_samples_per_class, seed,
                 reset_stored_filepaths, data_path, labels_as_int):
        return NotImplementedError

    def get_train_batches(self, total_batches=-1, augment_images=False):

        if total_batches==-1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(dataset_name="train",
                                seed=self.dataset.init_seed["train"] + self.total_train_iters_produced)
        self.dataset.set_augmentation(augment_images=augment_images)
        self.total_train_iters_produced += self.dataset.data_length["train"]
        for sample_id, sample_batched in enumerate(self.get_dataloader(shuffle=True)):
            preprocess_sample = self.sample_iter_data(sample=sample_batched, num_gpus=self.dataset.num_of_gpus,
                                                      samples_per_iter=self.batches_per_iter,
                                                      batch_size=self.dataset.batch_size)
            yield preprocess_sample

    def get_val_batches(self, total_batches=-1, augment_images=False):
        if total_batches==-1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(dataset_name="val")
        self.dataset.set_augmentation(augment_images=augment_images)
        for sample_id, sample_batched in enumerate(self.get_dataloader(shuffle=False)):
            preprocess_sample = self.sample_iter_data(sample=sample_batched, num_gpus=self.dataset.num_of_gpus,
                                                      samples_per_iter=self.batches_per_iter,
                                                      batch_size=self.dataset.batch_size)
            yield preprocess_sample

    def get_test_batches(self, total_batches=-1, augment_images=False):
        if total_batches==-1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(dataset_name="test")
        self.dataset.set_augmentation(augment_images=augment_images)
        for sample_id, sample_batched in enumerate(self.get_dataloader(shuffle=False)):
            preprocess_sample = self.sample_iter_data(sample=sample_batched, num_gpus=self.dataset.num_of_gpus,
                                                      samples_per_iter=self.batches_per_iter,
                                                      batch_size=self.dataset.batch_size)
            yield preprocess_sample


    def sample_iter_data(self, sample, num_gpus, batch_size, samples_per_iter):
        output_sample = []
        for key in sample.keys():
            sample[key] = np.array(sample[key].numpy(), dtype=np.float32)
            new_shape = []
            curr_id = 1

            for i in range(len(sample[key].shape) + 2):
                if i == 0:
                    new_shape.append(samples_per_iter)
                elif i == 1:
                    new_shape.append(num_gpus)
                elif i == 2:
                    new_shape.append(batch_size)
                else:
                    new_shape.append(sample[key].shape[curr_id])
                    curr_id += 1

            output_sample.append(np.reshape(sample[key], newshape=new_shape))

        return output_sample

class FolderMatchingNetworkDatasetParallel(MatchingNetworkDatasetParallel):
    def __init__(self, name, num_of_gpus, batch_size, image_height, image_width, image_channels,
                 train_val_test_split, data_path, indexes_of_folders_indicating_class, reset_stored_filepaths,
                 num_samples_per_class, num_classes_per_set, labels_as_int, reverse_channels):

        super(FolderMatchingNetworkDatasetParallel, self).__init__(
            batch_size=batch_size, reverse_channels=reverse_channels,
            num_of_gpus=num_of_gpus, image_height=image_height,
            image_width=image_width, image_channels=image_channels,
            train_val_test_split=train_val_test_split, reset_stored_filepaths=reset_stored_filepaths,
            num_classes_per_set=num_classes_per_set, num_samples_per_class=num_samples_per_class,
            labels_as_int=labels_as_int, data_path=os.path.abspath(data_path), dataset_name=name,
            indexes_of_folders_indicating_class=indexes_of_folders_indicating_class)


class FolderDatasetLoader(MatchingNetworkLoader):
    def __init__(self, name, batch_size, image_height, image_width, image_channels, data_path, train_val_test_split,
                 num_of_gpus=1, samples_per_iter=1, num_workers=4, indexes_of_folders_indicating_class=[-2],
                 reset_stored_filepaths=False, num_samples_per_class=1, num_classes_per_set=20, reverse_channels=False,
                 seed=100, label_as_int=False):

        self.name = name
        self.indexes_of_folders_indicating_class = indexes_of_folders_indicating_class
        self.reset_stored_filepaths = reset_stored_filepaths
        super(FolderDatasetLoader, self).__init__(name, num_of_gpus, batch_size, image_height, image_width, image_channels, num_classes_per_set, data_path,
                 num_samples_per_class, train_val_test_split,
                 samples_per_iter, num_workers, reverse_channels, seed, labels_as_int=label_as_int)

    def get_dataset(self, batch_size, reverse_channels, num_of_gpus, image_height, image_width, image_channels,
                 train_val_test_split, num_classes_per_set, num_samples_per_class, seed,
                 reset_stored_filepaths, data_path, labels_as_int):
        return FolderMatchingNetworkDatasetParallel(name=self.name, num_of_gpus=num_of_gpus, batch_size=batch_size,
                                                    image_height=image_height, image_width=image_width,
                                                    image_channels=image_channels,
                                                    train_val_test_split=train_val_test_split, data_path=data_path,
                                                    indexes_of_folders_indicating_class=self.indexes_of_folders_indicating_class,
                                                    reset_stored_filepaths=self.reset_stored_filepaths,
                                                    num_samples_per_class=num_samples_per_class,
                                                    num_classes_per_set=num_classes_per_set, labels_as_int=labels_as_int,
                                                    reverse_channels=reverse_channels)
