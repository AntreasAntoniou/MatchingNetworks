import tensorflow as tf
import numpy as np
import tqdm
from one_shot_learning_network import MatchingNetwork


class ExperimentBuilder:

    def __init__(self, data):
        """
        Initializes an ExperimentBuilder object. The ExperimentBuilder object takes care of setting up our experiment
        and provides helper functions such as run_training_epoch and run_validation_epoch to simplify out training
        and evaluation procedures.
        :param data: A data provider class
        """
        self.data = data

    def build_experiment(self, batch_size, classes_per_set, samples_per_class, fce, num_gpus=1, data_augmentation=True):

        """

        :param batch_size: The experiment batch size
        :param classes_per_set: An integer indicating the number of classes per support set
        :param samples_per_class: An integer indicating the number of samples per class
        :param channels: The image channels
        :param fce: Whether to use full context embeddings or not
        :return: a matching_network object, along with the losses, the training ops and the init op
        """
        height, width, channels = self.data.dataset.image_height, self.data.dataset.image_width, \
                                  self.data.dataset.image_channel #missing
        self.support_set_images = tf.placeholder(tf.float32, [num_gpus, batch_size, classes_per_set, samples_per_class, height, width,
                                                              channels], 'support_set_images')
        self.support_set_labels = tf.placeholder(tf.int32, [num_gpus, batch_size, classes_per_set, samples_per_class], 'support_set_labels')
        self.target_image = tf.placeholder(tf.float32, [num_gpus, batch_size, height, width, channels], 'target_image')
        self.target_label = tf.placeholder(tf.int32, [num_gpus, batch_size], 'target_label')
        self.training_phase = tf.placeholder(tf.bool, name='training-flag')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout-prob')
        self.current_learning_rate = 1e-03
        self.learning_rate = tf.placeholder(tf.float32, name='learning-rate-set')
        self.one_shot_omniglot = MatchingNetwork(batch_size=batch_size, support_set_images=self.support_set_images,
                                            support_set_labels=self.support_set_labels,
                                            target_image=self.target_image, target_label=self.target_label,
                                            dropout_rate=self.dropout_rate, num_channels=channels,
                                            is_training=self.training_phase, fce=fce,
                                            num_classes_per_set=classes_per_set,
                                            num_samples_per_class=samples_per_class, learning_rate=self.learning_rate)
        self.data_augmentation = data_augmentation
        summary, self.losses, self.c_error_opt_op = self.one_shot_omniglot.init_train()
        init = tf.global_variables_initializer()
        self.total_train_iter = 0
        return self.one_shot_omniglot, self.losses, self.c_error_opt_op, init

    def run_training_epoch(self, total_train_batches, sess):
        """
        Runs one training epoch
        :param total_train_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_training_categorical_crossentropy_loss and mean_training_accuracy
        """
        total_train_c_loss = []
        total_train_accuracy = []
        with tqdm.tqdm(total=total_train_batches) as pbar:
            for sample_id, train_sample in enumerate(self.data.get_train_batches(total_batches=total_train_batches,
                                                                            augment_images=self.data_augmentation)):

                support_set_images, target_set_image, support_set_labels, target_set_label = train_sample

                _, c_loss_value, acc = sess.run(
                    [self.c_error_opt_op, self.losses[self.one_shot_omniglot.classify], self.losses[self.one_shot_omniglot.dn]],
                    feed_dict={self.dropout_rate: 0.0, self.support_set_images: support_set_images[0],
                               self.support_set_labels: support_set_labels[0], self.target_image: target_set_image[0],
                               self.target_label: target_set_label[0], self.training_phase: True,
                               self.learning_rate: self.current_learning_rate})

                iter_out = "train_loss: {}, train_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)

                pbar.update(1)
                total_train_c_loss.append(c_loss_value)
                total_train_accuracy.append(acc)
                self.total_train_iter += 1
                if self.total_train_iter % 2000 == 0:
                    self.current_learning_rate /= 2
                    print("change learning rate", self.current_learning_rate)

        total_train_c_loss_mean = np.mean(total_train_c_loss)
        total_train_c_loss_std = np.std(total_train_c_loss)

        total_train_accuracy_mean = np.mean(total_train_accuracy)
        total_train_accuracy_std = np.std(total_train_accuracy)
        return total_train_c_loss_mean, total_train_c_loss_std, total_train_accuracy_mean, total_train_accuracy_std

    def run_validation_epoch(self, total_val_batches, sess):
        """
        Runs one validation epoch
        :param total_val_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_validation_categorical_crossentropy_loss and mean_validation_accuracy
        """
        total_val_c_loss = []
        total_val_accuracy = []
        with tqdm.tqdm(total=total_val_batches) as pbar:
            for sample_id, val_sample in enumerate(self.data.get_val_batches(total_batches=total_val_batches,
                                                                                 augment_images=False)):

                support_set_images, target_set_image, support_set_labels, target_set_label = val_sample

                c_loss_value, acc = sess.run(
                    [self.losses[self.one_shot_omniglot.classify],
                     self.losses[self.one_shot_omniglot.dn]],
                    feed_dict={self.dropout_rate: 0.0, self.support_set_images: support_set_images[0],
                               self.support_set_labels: support_set_labels[0], self.target_image: target_set_image[0],
                               self.target_label: target_set_label[0], self.training_phase: False,
                               self.learning_rate: self.current_learning_rate})

                iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)

                pbar.update(1)
                total_val_c_loss.append(c_loss_value)
                total_val_accuracy.append(acc)

        total_val_c_loss_mean = np.mean(total_val_c_loss)
        total_val_c_loss_std = np.std(total_val_c_loss)

        total_val_accuracy_mean = np.mean(total_val_accuracy)
        total_val_accuracy_std = np.std(total_val_accuracy)
        return total_val_c_loss_mean, total_val_c_loss_std, total_val_accuracy_mean, total_val_accuracy_std

    def run_testing_epoch(self, total_test_batches, sess):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        total_test_c_loss = []
        total_test_accuracy = []
        with tqdm.tqdm(total=total_test_batches) as pbar:
            for sample_id, test_sample in enumerate(self.data.get_test_batches(total_batches=total_test_batches,
                                                                             augment_images=False)):
                support_set_images, target_set_image, support_set_labels, target_set_label = test_sample

                c_loss_value, acc = sess.run(
                    [self.losses[self.one_shot_omniglot.classify],
                     self.losses[self.one_shot_omniglot.dn]],
                    feed_dict={self.dropout_rate: 0.0, self.support_set_images: support_set_images[0],
                               self.support_set_labels: support_set_labels[0], self.target_image: target_set_image[0],
                               self.target_label: target_set_label[0], self.training_phase: False,
                               self.learning_rate: self.current_learning_rate})

                iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)

                pbar.update(1)
                total_test_c_loss.append(c_loss_value)
                total_test_accuracy.append(acc)

        total_test_c_loss_mean = np.mean(total_test_c_loss)
        total_test_c_loss_std = np.std(total_test_c_loss)

        total_test_accuracy_mean = np.mean(total_test_accuracy)
        total_test_accuracy_std = np.std(total_test_accuracy)
        return total_test_c_loss_mean, total_test_c_loss_std, total_test_accuracy_mean, total_test_accuracy_std
