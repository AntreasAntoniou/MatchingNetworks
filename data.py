import numpy as np
np.random.seed(2391)
class omniglot_one_shot_classification():
    def __init__(self, batch_size, classes_per_set=10, samples_per_class=1):

        self.x = np.load("data.npy")
        self.x = np.reshape(self.x, [-1, 20, 28, 28, 1])  # each of the 1600 classes has 20 examples
        self.x_train, self.x_test, self.x_val  = self.x[:1200], self.x[1200:1500], self.x[1500:]
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

        print("mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        print("train_shape", self.x_train.shape, "test_shape", self.x_test.shape, "val_shape", self.x_val.shape)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_val = (self.x_val - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        self.datasets_cache = {"train": self.get_data(self.datasets["train"]), "val": self.get_data(self.datasets["val"]), "test": self.get_data(self.datasets["test"])}

    def get_data(self, data_pack):
        n_samples = self.samples_per_class * self.classes_per_set
        data_cache = []
        for sample in range(1000):
            mb_x_i = np.zeros((self.batch_size, n_samples, 28, 28, 1))
            mb_y_i = np.zeros((self.batch_size, n_samples))
            mb_x_hat = np.zeros((self.batch_size, 28, 28, 1), dtype=np.int)
            mb_y_hat = np.zeros((self.batch_size,), dtype=np.int)
            for i in range(self.batch_size):
                ind = 0
                pinds = np.random.permutation(n_samples)
                classes = np.random.choice(data_pack.shape[0], self.classes_per_set, False)
                x_hat_class = np.random.randint(self.classes_per_set)
                for j, cur_class in enumerate(classes):  # each class
                    example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class, False)
                    for eind in example_inds:
                        mb_x_i[i, pinds[ind], :, :, :] = data_pack[cur_class][eind]
                        mb_y_i[i, pinds[ind]] = j
                        ind += 1
                    if j == x_hat_class:
                        mb_x_hat[i, :, :, :] = data_pack[cur_class][np.random.choice(data_pack.shape[1])]

                        mb_y_hat[i] = j
            data_cache.append([mb_x_i, mb_y_i, mb_x_hat, mb_y_hat])
        return data_cache

    def get_batch(self, dataset_name):
        if self.indexes[dataset_name] >= len(self.datasets_cache[dataset_name]):
            self.indexes[dataset_name] = 0
            self.datasets_cache[dataset_name] = self.get_data(self.datasets[dataset_name])
        batch_out = self.datasets_cache[dataset_name][self.indexes[dataset_name]]
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target = batch_out
        return x_support_set, y_support_set, x_target, y_target

    def get_train_batch(self):

        return self.get_batch("train")

    def get_test_batch(self):

        return self.get_batch("test")

    def get_val_batch(self):

        return self.get_batch("val")
