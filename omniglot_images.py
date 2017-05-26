import numpy as np
import os
import matplotlib.pyplot as pplt

train_images_dir = "omni_data/images_background" #to be used to re-extract data, not needed when omniglot_data npy files available
test_images_dir = "omni_data/images_evaluation"

def extract_data(data_dir, name="train"):
    import cv2
    images = []
    targets = []

    for subdir, dir, files in os.walk(data_dir):
        for file in files:
            image_path = subdir + "/" + file
            image = cv2.imread(image_path)
            image = cv2.resize(image, (32, 32))
            label = subdir.split("/")[-1] + "/" + subdir.split("/")[-2]
            if image is not None:
                images.append(image)
                targets.append(label)

    targets_set = set()

    for item in targets:
        targets_set.add(item)

    targets_dict = dict(zip(list(targets_set), [idx for idx in range(len(targets_set))]))
    targets_to_num = []
    for i, item in enumerate(targets):
        target_ = targets_dict[item]
        targets_to_num.append(target_)
    print(len(targets_set))
    print(len(images))
    images = np.array(images)
    targets = np.array(targets_to_num)

    np.save(file="omniglot_data/x_{}".format(name), arr=images)
    np.save(file="omniglot_data/y_{}".format(name), arr=targets)

def load_data(single_channel=False):

    x_train = np.load(file="omniglot_data/x_train.npy")
    y_train = np.load(file="omniglot_data/y_train.npy")
    x_test = np.load(file="omniglot_data/x_test.npy")
    y_test = np.load(file="omniglot_data/y_test.npy")

    if single_channel:
        x_train = x_train[:, :, :, 0]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        x_test = x_test[:, :, :, 0]
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    # pplt.imshow(x_train[0])
    # pplt.show()
    return x_train, y_train, x_test, y_test
