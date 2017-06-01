import tensorflow as tf
from omniglot_networks_classifier import *
from omniglot_networks_wgan import Generator
import data as dataset
import tqdm
from storage import *

tf.reset_default_graph()
batch_size = 50
num_generations_per_sample = 1
data = dataset.omniglot_dataset(batch_size=batch_size, shuffle=True, single_channel=True)
logs_path = "omniglot_outputs/"
input_a = tf.placeholder(tf.float32, [batch_size, 32, 32, 1], 'inputs-1')
input_b_same_class = tf.placeholder(tf.float32, [batch_size, 32, 32, 1], 'inputs-2-same-class')
targets = tf.placeholder(tf.int32, [batch_size], 'targets')
use_generator = False
generator = None
if use_generator:
    generator = Generator(num_generations_per_sample, num_channels=1)
gan_stop_epoch = 1000
classification_start_epoch = 1500
train_results_gan_classifier = dict({"loss":[], "total_accuracy": []})
validation_results_gan_classifier = dict({"loss":[], "total_accuracy": []})
train_results_baseline_classifier = dict({"loss":[], "total_accuracy": []})
validation_results_baseline_classifier = dict({"loss":[], "total_accuracy": []})

training_phase = tf.placeholder(tf.bool, name='training-flag')
keep_prob = tf.placeholder(tf.float32, name='dropout-prob')

omni_classifier = omniglot_classifier(batch_size=batch_size, input_x=input_a,
              target_placeholder=targets, keep_prob=keep_prob,
              num_channels=1, n_classes=data.n_classes, is_training=training_phase, import_gan=generator, num_generations_per_sample=num_generations_per_sample)
print(data.n_classes)
summary, losses, c_error_opt_op = omni_classifier.init_train()

total_train_batches = (len(data.x_train) / batch_size)
total_val_batches = (len(data.x_val) / batch_size)
total_test_batches = len(data.x_test) / batch_size
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    continue_from_epoch = -1

    if use_generator: #to initialize conv net from GAN encoder embeddings
        continue_from_epoch = 8
        checkpoint = "saved_models/{}_{}.ckpt".format("omniglot_wgan_1_channel_64_128_256_256", continue_from_epoch)
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
    elif continue_from_epoch != -1:
        saver.restore(sess, "saved_models/{}_{}.ckpt".format("omniglot_wgan_1_channel_64_128_256_256", continue_from_epoch))

    with tqdm.tqdm(total=100) as pbar_e:
        for e in range(0, 100):

            total_c_loss = 0.
            total_accuracy = 0.
            with tqdm.tqdm(total=total_train_batches) as pbar:
                for i in range(total_train_batches):
                    x_batch, y_batch = data.get_next_train_batch()

                    _, c_loss_value, acc = sess.run(
                        [c_error_opt_op, losses[omni_classifier.c], losses["accuracy"]],
                        feed_dict={keep_prob: 0.5, input_a: x_batch,
                                   targets: y_batch, training_phase: True})
                    total_c_loss += c_loss_value
                    total_accuracy += acc
                    iter_out = "train_loss: {}, train_accuracy: {}".format(c_loss_value, acc)
                    pbar.set_description(iter_out)
                    pbar.update(1)

            total_c_loss /= total_train_batches
            total_accuracy /= total_train_batches
            save_path = saver.save(sess, "saved_models/{}_{}.ckpt".format("omniglot_classifier_4_layers", e))


            print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

            total_c_loss = 0.
            total_accuracy = 0.
            with tqdm.tqdm(total=total_train_batches) as pbar:
                for i in range(total_train_batches):
                    x_batch, y_batch = data.get_next_train_batch()

                    c_loss_value, acc = sess.run(
                        [losses[omni_classifier.c], losses["accuracy"]],
                        feed_dict={keep_prob: 1.0, input_a: x_batch,
                                   targets: y_batch, training_phase: False})
                    total_c_loss += c_loss_value
                    total_accuracy += acc
                    iter_out = "train_loss: {}, train_accuracy: {}".format(c_loss_value, acc)
                    pbar.set_description(iter_out)
                    pbar.update(1)

            total_c_loss /= total_train_batches
            total_accuracy /= total_train_batches

            print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))


            total_val_c_loss = 0.
            total_val_accuracy = 0.
            with tqdm.tqdm(total=total_val_batches) as pbar:
                for i in range(total_val_batches):
                    x_batch, y_batch = data.get_next_train_batch()
                    c_loss_value, acc = sess.run(
                        [losses[omni_classifier.c], losses["accuracy"]],
                        feed_dict={keep_prob: 1.0, input_a: x_batch,
                                   targets: y_batch, training_phase: False})
                    total_val_c_loss += c_loss_value
                    total_val_accuracy += acc
                    iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss_value, acc)
                    pbar.set_description(iter_out)
                    pbar.update(1)

            total_val_c_loss /= total_val_batches
            total_val_accuracy /= total_val_batches
            print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

            total_test_c_loss = 0.
            total_test_accuracy = 0.
            with tqdm.tqdm(total=total_test_batches) as pbar:
                for i in range(total_test_batches):
                    x_batch, y_batch = data.get_next_train_batch()
                    c_loss_value, acc = sess.run(
                        [losses[omni_classifier.c], losses["accuracy"]],
                        feed_dict={keep_prob: 1.0, input_a: x_batch,
                                   targets: y_batch, training_phase: False})
                    total_test_c_loss += c_loss_value
                    total_test_accuracy += acc
                    iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value, acc)
                    pbar.set_description(iter_out)
                    pbar.update(1)

            total_test_c_loss /= total_test_batches
            total_test_accuracy /= total_test_batches
            print("Epoch {}: test_loss: {}, t_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
            pbar_e.update(1)

