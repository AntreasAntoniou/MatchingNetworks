import numpy as np
import omniglot_images
np.random.seed(2391)
class omniglot_dataset():
    def __init__(self, batch_size, shuffle=True, validation_ratio=0.1, single_channel=False):
        self.x_train, self.y_train, self.x_test, self.y_test = omniglot_images.load_data(single_channel=single_channel)
        self.x, self.y = np.concatenate((self.x_train, self.x_test)), np.concatenate((self.y_train, self.y_test+np.max(self.y_train)+1))

        self.x, self.y = self.get_map_by_label(self.x, self.y)
        #print(self.x.shape)
        #self.x, self.y = self.shuffle(self.x, self.y)
        indices = np.arange(self.x.shape[1])
        np.random.shuffle(indices)
        self.x = self.x[:, indices]
        self.y = self.y[:, indices]
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self.x[:, :18], self.y[:, :18], \
                                                                                       self.x[:, 10:17], self.y[:, 10:17], \
                                                                                       self.x[:, 18:], self.y[:, 18:]
        print(np.max(self.y_train))
        print(np.min(self.y_train))
        self.x_train, self.y_train = self.break_classes(self.x_train, self.y_train)
        self.x_val, self.y_val = self.break_classes(self.x_val, self.y_val)
        self.x_test, self.y_test = self.break_classes(self.x_test, self.y_test)
        print(self.x_train.shape)
        print(np.mean(self.y_train))
        print(self.x_val.shape)
        print(np.mean(self.y_val))
        print(self.x_test.shape)
        print(np.mean(self.y_test))
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

        self.x_train = (self.x_train - self.mean)#/(self.max - self.min)
        self.x_val = (self.x_val - self.mean)# / (self.max - self.min)
        self.x_test = (self.x_test - self.mean)#/(self.max - self.min)

        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

    def shuffle(self, x, y):
        print("shuffle", x.shape)
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        return x, y

    def get_map_by_label(self, x_data, y_data):
        label_map = dict()
        for x, y in zip(x_data, y_data):

            if y in label_map:
                label_map[y].append(x)
            else:
                label_map[y] = [x]

        x_pack = []
        y_pack = []

        for key in label_map.keys():
            y_pack.append(20*[key])
            x_pack.append(label_map[key])

        y_pack = np.array(y_pack)
        x_pack = np.array(x_pack)

        return x_pack, y_pack

    def break_classes(self, x, y):
        temp_x = []
        temp_y = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                temp_x.append(x[i, j])
                temp_y.append(y[i, j])
        x = np.array(temp_x)
        y = np.array(temp_y)
        return x, y

    def get_next_train_batch(self):

        if self.train_index + self.batch_size > self.x_train.shape[0]:
            self.x_train, self.y_train = self.shuffle(self.x_train, self.y_train)
            self.train_index = 0
            return self.get_next_train_batch()
        else:
            x_batch = self.x_train[self.train_index:self.train_index + self.batch_size]
            y_batch = self.y_train[self.train_index:self.train_index + self.batch_size]
            self.train_index += 1
            return x_batch, y_batch

    def get_next_val_batch(self):

        if self.val_index + self.batch_size > self.x_val.shape[0]:
            self.val_index = 0
            return self.get_next_val_batch()
        else:
            x_batch = self.x_val[self.val_index:self.val_index+self.batch_size]
            y_batch = self.y_val[self.val_index:self.val_index + self.batch_size]
            self.val_index += 1
            return x_batch, y_batch

    def get_next_test_batch(self):

        if self.test_index + self.batch_size > self.x_test.shape[0]:
            self.test_index = 0
            return self.get_next_test_batch()
        else:
            x_batch = self.x_test[self.test_index:self.test_index+self.batch_size]
            y_batch = self.y_test[self.test_index:self.test_index + self.batch_size]
            self.test_index += 1
            return x_batch, y_batch




class omniglot_one_shot_classification():
    def __init__(self, batch_size, shuffle=True, classes_per_set=10, samples_per_class=1, single_channel=True):
        self.x_train, self.y_train, self.x_test, self.y_test = omniglot_images.load_data(single_channel=single_channel)
        self.x, self.y = np.concatenate((self.x_train, self.x_test)), np.concatenate(
            (self.y_train, self.y_test + np.max(self.y_train) + 1))
        self.x, self.y = self.get_map_by_label(self.x, self.y)
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self.x[:1200], self.y[:1200], \
                                                                                       self.x[1200:1400], self.y[1200:1400], \
                                                                                       self.x[1400:], self.y[1400:]
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

        self.x_train = (self.x_train - self.mean)# / (self.max - self.min)
        self.x_val = (self.x_val - self.mean)# / (self.max - self.min)
        self.x_test = (self.x_test - self.mean)# / (self.max - self.min)
        print(self.x_train.shape, self.x_test.shape, self.x_val.shape)
        print(self.y_train.shape, self.y_test.shape, self.y_val.shape)
        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.x_train_cache = self.get_data(self.x_train)
        self.x_val_cache = self.get_data(self.x_val)
        self.x_test_cache = self.get_data(self.x_test)
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        self.datasets_cache = {"train": self.x_train_cache, "val": self.x_val_cache, "test": self.x_test_cache}

    def shuffle(self, x):

        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        #y = y[indices]

        indices = np.arange(x.shape[1])
        np.random.shuffle(indices)
        x = x[:, indices]
        #y = y[:, indices]

        return x#, y

    def get_map_by_label(self, x_data, y_data):
        label_map = dict()
        for x, y in zip(x_data, y_data):

            if y in label_map:
                label_map[y].append(x)
            else:
                label_map[y] = [x]

        x_pack = []
        y_pack = []

        for key in label_map.keys():
            y_pack.append(20 * [key])
            x_pack.append(label_map[key])

        y_pack = np.array(y_pack)
        x_pack = np.array(x_pack)

        return x_pack, y_pack

    def convert_array_to_sample(self, data_array, data_labels):
        x = []
        y = []

        for label_samples, label in zip(data_array, data_labels):
            x.append(label_samples)
            y.append(len(label_samples)*[label])
        x = np.array(x)
        y = np.array(y)

        return x, y

    def get_data(self, x):
        x = self.shuffle(x)
        keys_available = np.arange(x.shape[0])
        support_set_images = []
        support_set_labels = []
        target_images = []
        target_labels = []
        n_possible_support_sets = int(len(keys_available) / self.classes_per_set)
        for i in range(n_possible_support_sets):
            chosen_classes = np.random.choice(keys_available, size=self.classes_per_set, replace=False)
            keys_available = [item for item in keys_available if item not in chosen_classes]

            temp_support_set_x = []
            temp_support_set_y = []

            temp_target_x = []
            temp_target_y = []

            for j in range(x.shape[1]-self.samples_per_class):
                target_class = np.random.choice(chosen_classes, replace=False)

                temp_set_images = x[chosen_classes, j:j+self.samples_per_class]
                target_index = j+self.samples_per_class+1
                if target_index>=20:
                    target_index = 0
                temp_target_images = x[target_class, target_index]
                x_support_temp, y_support_temp = self.convert_array_to_sample(temp_set_images, chosen_classes)
                x_target_temp, y_target_temp = self.convert_array_to_sample(np.array([[temp_target_images]]),
                                                                            np.array([[target_class]]))

                temp_support_set_x.append(x_support_temp)
                temp_support_set_y.append(y_support_temp)
                temp_target_x.append(x_target_temp[0])
                temp_target_y.append(y_target_temp[0])
            support_set_images.extend(temp_support_set_x)
            support_set_labels.extend(temp_support_set_y)
            target_images.extend(temp_target_x)
            target_labels.extend(temp_target_y)

        support_set_images = np.array(support_set_images)[:, :, 0]
        support_set_labels = np.array(support_set_labels)[:, :, 0]
        target_images = np.array(target_images)[:, 0]
        target_labels = np.array(target_labels)[:, 0]

        return [support_set_images, support_set_labels, target_images, target_labels]

    def get_batch(self, dataset_label):

        if self.indexes[dataset_label] + self.batch_size > self.datasets_cache[dataset_label][0].shape[0]:
            self.datasets_cache[dataset_label] = self.get_data(self.datasets[dataset_label])
            self.indexes[dataset_label] = 0
            return self.get_batch(dataset_label)
        else:
            [temp_support_set_images, temp_support_set_labels,
            temp_target_images, temp_target_labels] = self.datasets_cache[dataset_label]

            pack_index = self.indexes[dataset_label]

            x_support_batch, \
            y_support_batch, \
            x_target_batch,\
            y_target_batch = temp_support_set_images[pack_index:pack_index+self.batch_size], \
                           temp_support_set_labels[pack_index:pack_index+self.batch_size], \
                           temp_target_images[pack_index:pack_index+self.batch_size],\
                           temp_target_labels[pack_index:pack_index+self.batch_size]
            y_new_support_set = []
            y_new_target = []
            for j in range(self.batch_size):
                map_targets = set()
                for y_temp in y_support_batch[j]:
                    map_targets.add(y_temp)

                mapped_targets = dict(zip(map_targets, [i for i in range(len(map_targets))]))

                y_support_episode = []
                y_target_episode = mapped_targets[y_target_batch[j, 0]]
                for y_temp in y_support_batch[j]:
                    temp_array = np.zeros((self.classes_per_set))
                    temp_array[mapped_targets[y_temp]] = 1
                    y_support_episode.append(temp_array)
                y_new_support_set.append(y_support_episode)
                y_new_target.append(y_target_episode)
            y_support_batch = np.array(y_new_support_set)
            y_target_batch = np.array(y_new_target)
            self.indexes[dataset_label] = self.indexes[dataset_label] + 1
            #print(y_support_batch.shape)
            return x_support_batch, y_support_batch, x_target_batch, y_target_batch

    def get_train_batch(self):
        return self.get_batch("train")

    def get_test_batch(self):
        return self.get_batch("test")

    def get_val_batch(self):
        return self.get_batch("val")

