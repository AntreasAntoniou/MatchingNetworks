import tensorflow as tf
from one_shot_learning_wgan import *
import tensorflow.contrib.slim as slim
import data as dataset
import tqdm
from storage import *

tf.reset_default_graph()
batch_size = 50
fce = False
classes_per_sample = 20
samples_per_class = 1
channels = 1
data = dataset.omniglot_one_shot_classification(batch_size=batch_size,
                                                classes_per_set=classes_per_sample, samples_per_class=samples_per_class)
x_support_set, y_support_set, x_target, y_target = data.get_train_batch()
epochs = 200
logs_path = "one_shot_outputs/"#/disk/scratch for cdtcluster
experiment_name = "one_shot_learning_embedding_{}_{}".format(samples_per_class, classes_per_sample)
sequence_size = classes_per_sample * samples_per_class
support_set_images = tf.placeholder(tf.float32, [batch_size, sequence_size, 32, 32, channels], 'support_set_images')
support_set_labels = tf.placeholder(tf.float32, [batch_size, sequence_size, classes_per_sample], 'support_set_labels')
target_image = tf.placeholder(tf.float32, [batch_size, 32, 32, channels], 'target_image')
target_label = tf.placeholder(tf.int32, [batch_size], 'target_label')
z_inputs = tf.placeholder(tf.float32, [batch_size, 100], 'inputs')

training_phase = tf.placeholder(tf.bool, name='training-flag')
keep_prob = tf.placeholder(tf.float32, name='dropout-prob')

one_shot_omniglot = MatchingNetwork(batch_size=batch_size, support_set_images=support_set_images, support_set_labels=support_set_labels,
              target_image=target_image, target_label=target_label, keep_prob=keep_prob, num_channels=channels, is_training=training_phase, fce=fce)

summary, losses, c_error_opt_op = one_shot_omniglot.init_train()

load_from_GAN = False
load_from_classifier = False
total_train_batches = 500
total_val_batches = 100
total_test_batches = 100

init = tf.global_variables_initializer()
save_statistics(experiment_name, ["epoch", "train_c_loss", "train_c_accuracy", "val_loss", "val_accuracy", "test_c_loss", "test_c_accuracy"])

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()

    if load_from_GAN: #to initialize conv net from GAN encoder embeddings
        continue_from_epoch = 882
        checkpoint = "saved_models/{}_{}.ckpt".format("omniglot_wgan_1_channel", continue_from_epoch)
        variables_to_restore = []
        for var in  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(var)
            variables_to_restore.append(var)



        tf.logging.info('Fine-tuning from %s' % checkpoint)

        fine_tune = slim.assign_from_checkpoint_fn(
            checkpoint,
            variables_to_restore,
            ignore_missing_vars=True)
        fine_tune(sess)
    elif load_from_classifier: #to initialize network from classifier weights
        continue_from_epoch = 21
        checkpoint = "saved_models/{}_{}.ckpt".format("omniglot_classifier_4_layers", continue_from_epoch)
        variables_to_restore = []
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(var)
            variables_to_restore.append(var)

        tf.logging.info('Fine-tuning from %s' % checkpoint)

        fine_tune = slim.assign_from_checkpoint_fn(
            checkpoint,
            variables_to_restore,
            ignore_missing_vars=True)
        fine_tune(sess)
    else:
        continue_from_epoch = -1
        if continue_from_epoch != -1:
            saver.restore(sess, "saved_models/{}_{}.ckpt".format(experiment_name, continue_from_epoch))

    print("start")
    with tqdm.tqdm(total=2000) as pbar_e:
        for e in range(0, 2000):
            total_c_loss = 0.
            total_accuracy = 0.
            with tqdm.tqdm(total=total_train_batches) as pbar:
                for i in range(total_train_batches):
                    x_support_set, y_support_set, x_target, y_target = data.get_train_batch()
                    _, c_loss_value, acc = sess.run(
                        [c_error_opt_op, losses[one_shot_omniglot.classify], losses[one_shot_omniglot.dn]],
                        feed_dict={keep_prob: 0.9, support_set_images: x_support_set,
                                   support_set_labels: y_support_set, target_image: x_target, target_label: y_target,
                                   training_phase: True})

                    iter_out = "train_loss: {}, train_accuracy: {}".format(c_loss_value, acc)
                    pbar.set_description(iter_out)
                    pbar.update(1)
                    total_c_loss += c_loss_value
                    total_accuracy += acc
            save_path = saver.save(sess, "saved_models/{}_{}.ckpt".format(experiment_name, e))

            total_c_loss = total_c_loss / total_train_batches
            total_accuracy = total_accuracy / total_train_batches
            print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

            total_val_c_loss = 0.
            total_val_accuracy = 0.
            with tqdm.tqdm(total=total_val_batches) as pbar:
                for i in range(total_val_batches):
                    x_support_set, y_support_set, x_target, y_target = data.get_val_batch()
                    c_loss_value, acc = sess.run(
                        [losses[one_shot_omniglot.classify], losses[one_shot_omniglot.dn]],
                        feed_dict={keep_prob: 1.0, support_set_images: x_support_set,
                                   support_set_labels: y_support_set, target_image: x_target, target_label: y_target,
                                   training_phase: False})

                    iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss_value, acc)
                    pbar.set_description(iter_out)
                    pbar.update(1)

                    total_val_c_loss += c_loss_value
                    total_val_accuracy += acc
            total_val_c_loss = total_val_c_loss / total_val_batches
            total_val_accuracy = total_val_accuracy / total_val_batches
            print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))
            total_test_c_loss = 0.
            total_test_accuracy = 0.
            with tqdm.tqdm(total=total_test_batches) as pbar:
                for i in range(total_test_batches):
                    x_support_set, y_support_set, x_target, y_target = data.get_test_batch()
                    c_loss_value, acc = sess.run(
                        [losses[one_shot_omniglot.classify], losses[one_shot_omniglot.dn]],
                        feed_dict={keep_prob: 1.0, support_set_images: x_support_set,
                                   support_set_labels: y_support_set, target_image: x_target, target_label: y_target,
                                   training_phase: False})

                    iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value, acc)
                    pbar.set_description(iter_out)
                    pbar.update(1)

                    total_test_c_loss += c_loss_value
                    total_test_accuracy += acc
            save_path = saver.save(sess, "saved_models/{}_{}.ckpt".format(experiment_name, e))

            total_test_c_loss = total_test_c_loss / total_test_batches
            total_test_accuracy = total_test_accuracy / total_test_batches
            print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
            save_statistics(experiment_name, [e, total_c_loss, total_accuracy, total_val_c_loss, total_val_accuracy, total_test_c_loss, total_test_accuracy])
            pbar_e.update(1)
