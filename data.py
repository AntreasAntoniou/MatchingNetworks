import numpy as np
from scipy.ndimage import rotate
class OmniglotNShotDataset():
    def __init__(self, batch_size, classes_per_set=10, samples_per_class=1, seed=2591, shuffle_classes=True):

        """
        Constructs an N-Shot omniglot Dataset
        :param batch_size: Experiment batch_size
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
        """
        np.random.seed(seed)
        self.x = np.load("data.npy")
        self.x = np.reshape(self.x, newshape=(1622, 20, 28, 28, 1))
        if shuffle_classes:
            class_ids = np.arange(self.x.shape[0])
            np.random.shuffle(class_ids)
            self.x = self.x[class_ids]
        self.x_train, self.x_test, self.x_val = self.x[:1200], self.x[1200:1411], self.x[1411:]
        self.mean = np.mean(list(self.x_train) + list(self.x_val))
        self.std = np.std(list(self.x_train) + list(self.x_val))
        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        print("train_shape", self.x_train.shape, "test_shape", self.x_test.shape, "val_shape", self.x_val.shape)
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test} #original data cached

    def preprocess_batch(self, x_batch):
        """
        Normalizes our data, to have a mean of 0 and sd of 1
        """
        x_batch = (x_batch - self.mean) / self.std

        return x_batch
    def sample_new_batch(self, data_pack):
        """
        Collects 1000 batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                  data_pack.shape[3], data_pack.shape[4]), dtype=np.float32)
        support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), dtype=np.float32)
        target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]),
                            dtype=np.float32)
        target_y = np.zeros((self.batch_size,), dtype=np.float32)
        for i in range(self.batch_size):
            classes_idx = np.arange(data_pack.shape[0])
            samples_idx = np.arange(data_pack.shape[1])
            choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
            choose_label = np.random.choice(self.classes_per_set, size=1)
            choose_samples = np.random.choice(samples_idx, size=self.samples_per_class+1, replace=False)

            x_temp = data_pack[choose_classes]
            x_temp = x_temp[:, choose_samples]
            y_temp = np.arange(self.classes_per_set)
            support_set_x[i] = x_temp[:, :-1]
            support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
            target_x[i] = x_temp[choose_label, -1]
            target_y[i] = y_temp[choose_label]

        return support_set_x, support_set_y, target_x, target_y

    def get_batch(self, dataset_name, augment=False):
        """
        Gets next batch from the dataset with name.
        :param dataset_name: The name of the dataset (one of "train", "val", "test")
        :return:
        """
        x_support_set, y_support_set, x_target, y_target = self.sample_new_batch(self.datasets[dataset_name])
        if augment:
            k = np.random.randint(0, 4, size=(self.batch_size, self.classes_per_set))
            x_augmented_support_set = []
            x_augmented_target_set = []
            for b in range(self.batch_size):
                temp_class_support = []

                for c in range(self.classes_per_set):
                    x_temp_support_set = self.rotate_batch(x_support_set[b, c], axis=(1, 2), k=k[b, c])
                    if y_target[b] == y_support_set[b, c, 0]:
                        x_temp_target = self.rotate_batch(x_target[b], axis=(0, 1), k=k[b, c])

                    temp_class_support.append(x_temp_support_set)

                x_augmented_support_set.append(temp_class_support)
                x_augmented_target_set.append(x_temp_target)

            x_support_set = np.array(x_augmented_support_set)
            x_target = np.array(x_augmented_target_set)
        x_support_set = self.preprocess_batch(x_support_set)
        x_target = self.preprocess_batch(x_target)

        return x_support_set, y_support_set, x_target, y_target

    def rotate_batch(self, x_batch, axis, k):
        x_batch = rotate(x_batch, k*90, reshape=False, axes=axis, mode="nearest")
        return x_batch

    def get_train_batch(self, augment=False):

        """
        Get next training batch
        :return: Next training batch
        """
        return self.get_batch("train", augment)

    def get_test_batch(self, augment=False):

        """
        Get next test batch
        :return: Next test_batch
        """
        return self.get_batch("test", augment)

    def get_val_batch(self, augment=False):

        """
        Get next val batch
        :return: Next val batch
        """
        return self.get_batch("val", augment)
