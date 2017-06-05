import tensorflow as tf
import tqdm
from one_shot_learning_network import MatchingNetwork


class ExperimentBuilder:

    def __init__(self, data):
        self.data = data

    def build_experiment(self, batch_size, classes_per_set, samples_per_class, channels, fce):

        sequence_size = classes_per_set * samples_per_class
        self.support_set_images = tf.placeholder(tf.float32, [batch_size, sequence_size, 28, 28, channels],
                                            'support_set_images')
        self.support_set_labels = tf.placeholder(tf.int32, [batch_size, sequence_size], 'support_set_labels')
        self.target_image = tf.placeholder(tf.float32, [batch_size, 28, 28, channels], 'target_image')
        self.target_label = tf.placeholder(tf.int32, [batch_size], 'target_label')
        self.training_phase = tf.placeholder(tf.bool, name='training-flag')
        self.rotate_flag = tf.placeholder(tf.bool, name='rotate-flag')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout-prob')
        self.one_shot_omniglot = MatchingNetwork(batch_size=batch_size, support_set_images=self.support_set_images,
                                            support_set_labels=self.support_set_labels,
                                            target_image=self.target_image, target_label=self.target_label,
                                            keep_prob=self.keep_prob, num_channels=channels,
                                            is_training=self.training_phase, fce=fce, rotate_flag=self.rotate_flag,
                                            num_classes_per_set=classes_per_set,
                                            num_samples_per_class=samples_per_class)

        summary, self.losses, self.c_error_opt_op = self.one_shot_omniglot.init_train()
        init = tf.global_variables_initializer()
        return self.one_shot_omniglot, self.losses, self.c_error_opt_op, init

    def run_training_epoch(self, total_train_batches, sess):
        total_c_loss = 0.
        total_accuracy = 0.
        with tqdm.tqdm(total=total_train_batches) as pbar:

            for i in range(total_train_batches):  # train epoch
                x_support_set, y_support_set, x_target, y_target = self.data.get_train_batch()
                _, c_loss_value, acc = sess.run(
                    [self.c_error_opt_op, self.losses[self.one_shot_omniglot.classify], self.losses[self.one_shot_omniglot.dn]],
                    feed_dict={self.keep_prob: 1.0, self.support_set_images: x_support_set,
                               self.support_set_labels: y_support_set, self.target_image: x_target, self.target_label: y_target,
                               self.training_phase: True, self.rotate_flag: True})

                iter_out = "train_loss: {}, train_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)

                pbar.update(1)
                total_c_loss += c_loss_value
                total_accuracy += acc

        total_c_loss = total_c_loss / total_train_batches
        total_accuracy = total_accuracy / total_train_batches
        return total_c_loss, total_accuracy

    def run_validation_epoch(self, total_val_batches, sess):
        total_val_c_loss = 0.
        total_val_accuracy = 0.

        with tqdm.tqdm(total=total_val_batches) as pbar:
            for i in range(total_val_batches):  # validation epoch
                x_support_set, y_support_set, x_target, y_target = self.data.get_val_batch()
                c_loss_value, acc = sess.run(
                    [self.losses[self.one_shot_omniglot.classify], self.losses[self.one_shot_omniglot.dn]],
                    feed_dict={self.keep_prob: 1.0, self.support_set_images: x_support_set,
                               self.support_set_labels: y_support_set, self.target_image: x_target, self.target_label: y_target,
                               self.training_phase: False, self.rotate_flag: True})

                iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)
                pbar.update(1)

                total_val_c_loss += c_loss_value
                total_val_accuracy += acc

        total_val_c_loss = total_val_c_loss / total_val_batches
        total_val_accuracy = total_val_accuracy / total_val_batches

        return total_val_c_loss, total_val_accuracy

    def run_testing_epoch(self, total_test_batches, sess):
        total_test_c_loss = 0.
        total_test_accuracy = 0.
        with tqdm.tqdm(total=total_test_batches) as pbar:
            for i in range(total_test_batches):
                x_support_set, y_support_set, x_target, y_target = self.data.get_test_batch()
                c_loss_value, acc = sess.run(
                    [self.losses[self.one_shot_omniglot.classify], self.losses[self.one_shot_omniglot.dn]],
                    feed_dict={self.keep_prob: 1.0, self.support_set_images: x_support_set,
                               self.support_set_labels: y_support_set, self.target_image: x_target,
                               self.target_label: y_target,
                               self.training_phase: False, self.rotate_flag: True})

                iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)
                pbar.update(1)

                total_test_c_loss += c_loss_value
                total_test_accuracy += acc
            total_test_c_loss = total_test_c_loss / total_test_batches
            total_test_accuracy = total_test_accuracy / total_test_batches
        return total_test_c_loss, total_test_accuracy