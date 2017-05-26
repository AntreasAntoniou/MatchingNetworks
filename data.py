import numpy as np
import omniglot_images
np.random.seed(230691)
class omniglot_dataset():
    def __init__(self, batch_size, shuffle=True, validation_ratio=0.1, single_channel=False):
        self.x_train, self.y_train, self.x_test, self.y_test = omniglot_images.load_data(single_channel=single_channel)
        self.x, self.y = np.concatenate((self.x_train, self.x_test)), np.concatenate((self.y_train, self.y_test+np.max(self.y_train)+1))
        self.x, self.y = self.get_map_by_label(self.x, self.y)
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self.x[:, :15], self.y[:, :15], \
                                                                                       self.x[:, 15:17], self.y[:, 15:17], \
                                                                                       self.x[:, 17:], self.y[:, 17:]
        self.x_train, self.y_train = self.break_classes(self.x_train, self.y_train)
        self.x_val, self.y_val = self.break_classes(self.x_val, self.y_val)
        self.x_test, self.y_test = self.break_classes(self.x_test, self.y_test)

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

        self.x_train = (self.x_train - self.mean)/(self.max - self.min)
        self.x_val = (self.x_val - self.mean) / (self.max - self.min)
        self.x_test = (self.x_test - self.mean)/(self.max - self.min)

        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

    def shuffle(self, x, y):

        indices = np.arange()
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
        x, y = np.reshape(x, newshape=(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])), \
                         np.reshape(y, newshape=(y.shape[0] * y.shape[1],))
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
    def __init__(self, batch_size, shuffle=True, classes_per_set=10, samples_per_class=1):
        self.x_train, self.y_train, self.x_test, self.y_test = omniglot_images.load_data(single_channel=True)
        stop_index = 15000
        self.x_train, self.y_train, self.x_val, self.y_val = self.x_train[:stop_index], self.y_train[:stop_index], self.x_train[stop_index:], self.y_train[stop_index:]

        if shuffle:
            indeces = np.arange(len(self.x_train))
            np.random.shuffle(indeces)
            self.x_train = self.x_train[indeces]
            self.y_train = self.y_train[indeces]

        self.mean = np.mean(self.x_train)

        self.x_train = self.x_train - self.mean
        self.x_test = self.x_test - self.mean
        self.x_val = self.x_val - self.mean

        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

        self.x_train = (self.x_train - self.mean)/(self.max - self.min)
        self.x_val = (self.x_val - self.mean)/(self.max - self.min)
        self.x_test = (self.x_test - self.mean)/(self.max - self.min)

        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.batch_size = batch_size

        self.train_pack = self.get_buckets_by_class(self.x_train, self.y_train)
        self.train_support_set_images, self.train_support_set_labels, \
        self.train_target_images, self.train_target_labels = self.get_data(self.train_pack)

        self.val_pack = self.get_buckets_by_class(self.x_val, self.y_val)
        self.val_support_set_images, self.val_support_set_labels, \
        self.val_target_images, self.val_target_labels = self.get_data(self.val_pack)

        self.test_pack = self.get_buckets_by_class(self.x_test, self.y_test)
        self.test_support_set_images, self.test_support_set_labels, \
        self.test_target_images, self.test_target_labels = self.get_data(self.test_pack)

        self.train_index = 0
        self.test_index = 0
        self.val_index = 0

    def get_buckets_by_class(self, x, y):
        data_pack = dict()
        for i, (x_item, y_item) in enumerate(zip(x, y)):
            try:
                data_pack[y_item].append(x_item)
            except:
                data_pack[y_item] = [x_item]
        return data_pack

    def shuffle_pack(self, pack):
        for key in pack.keys():
            items = np.array(pack[key])
            index = np.arange(len(items))
            np.random.shuffle(index)
            pack[key] = items[index]
        return pack

    def get_data(self, pack):
        temp_pack = self.shuffle_pack(pack)
        keys_available = temp_pack.keys()
        support_set_images = []
        support_set_labels = []
        target_images = []
        target_labels = []
        while len(keys_available)>self.classes_per_set:
            chosen_keys = np.random.choice(keys_available, self.classes_per_set, replace=False)

            sample_index = 0
            while sample_index<20:
                temp_support_set_images = []
                temp_support_set_targets = []
                for key in chosen_keys:
                    items = temp_pack[key]
                    images = np.array(items[sample_index:sample_index+self.samples_per_class])
                    labels = [key for i in range(self.samples_per_class)]
                    temp_support_set_images.extend(images)
                    temp_support_set_targets.extend(labels)

                target_key = np.random.choice(chosen_keys)
                target_index = np.random.choice(len(temp_pack[target_key]))
                target_image = temp_pack[target_key][target_index]
                target_label = target_key
                current_keys = np.arange(len(chosen_keys))
                np.random.shuffle(current_keys)
                keys_to_new_keys = dict(zip(chosen_keys, current_keys))
                for z in range(len(temp_support_set_targets)):
                    temp_support_set_targets[z] = keys_to_new_keys[temp_support_set_targets[z]]
                    temp_labels = np.zeros(self.classes_per_set)
                    temp_labels[temp_support_set_targets[z]] = 1
                    temp_support_set_targets[z] = temp_labels
                target_label = keys_to_new_keys[target_label]

                target_images.append(target_image)
                target_labels.append(target_label)
                support_set_images.append(temp_support_set_images)
                support_set_labels.append(temp_support_set_targets)
                sample_index = sample_index + self.samples_per_class
            keys_available = [x for x in keys_available if x not in chosen_keys]

        return support_set_images, support_set_labels, target_images, target_labels

    def get_next_train_batch(self):
        train_batch_support_set_images, train_batch_support_set_labels, \
        train_batch_target_images, train_batch_target_labels = self.train_support_set_images[self.train_index:self.train_index+self.batch_size], \
                                                               self.train_support_set_labels[self.train_index:self.train_index+self.batch_size], \
        self.train_target_images[self.train_index:self.train_index+self.batch_size], self.train_target_labels[self.train_index:self.train_index+self.batch_size]
        self.train_index = self.train_index + self.batch_size
        if self.train_index<=len(self.train_support_set_labels):
            self.train_support_set_images, self.train_support_set_labels, \
            self.train_target_images, self.train_target_labels = self.get_data(self.train_pack)
            self.train_index = 0
            train_batch_support_set_images, train_batch_support_set_labels, train_batch_target_images, \
            train_batch_target_labels = np.array(train_batch_support_set_images), \
                                       np.array(train_batch_support_set_labels), \
                                       np.array(train_batch_target_images),\
                                       np.array(train_batch_target_labels)
        return train_batch_support_set_images, train_batch_support_set_labels, train_batch_target_images, \
            train_batch_target_labels

    def get_next_test_batch(self):
        test_batch_support_set_images, test_batch_support_set_labels, \
        test_batch_target_images, test_batch_target_labels = self.test_support_set_images[self.test_index:self.test_index+self.batch_size], \
                                                               self.test_support_set_labels[self.test_index:self.test_index+self.batch_size], \
        self.test_target_images[self.test_index:self.test_index+self.batch_size], self.test_target_labels[self.test_index:self.test_index+self.batch_size]
        self.test_index = self.test_index + self.batch_size
        if self.test_index<=len(self.test_support_set_labels):
            self.test_support_set_images, self.test_support_set_labels, \
            self.test_target_images, self.test_target_labels = self.get_data(self.test_pack)
            self.test_index = 0

        return np.array(test_batch_support_set_images), np.array(test_batch_support_set_labels), np.array(test_batch_target_images), np.array(test_batch_target_labels)

    def get_next_val_batch(self):
        val_batch_support_set_images, val_batch_support_set_labels, \
        val_batch_target_images, val_batch_target_labels = self.val_support_set_images[self.val_index:self.val_index+self.batch_size], \
                                                               self.val_support_set_labels[self.val_index:self.val_index+self.batch_size], \
        self.val_target_images[self.val_index:self.val_index+self.batch_size], self.val_target_labels[self.val_index:self.val_index+self.batch_size]
        self.val_index = self.val_index + self.batch_size
        if self.val_index<=len(self.val_support_set_labels):
            self.val_support_set_images, self.val_support_set_labels, \
            self.val_target_images, self.val_target_labels = self.get_data(self.val_pack)
            self.val_index = 0

        return np.array(val_batch_support_set_images), np.array(val_batch_support_set_labels), np.array(val_batch_target_images), np.array(val_batch_target_labels)
