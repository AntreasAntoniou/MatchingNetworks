import tensorflow as tf
from omniglot_networks_classifier import *
import data as dataset
import tqdm
from storage import *

tf.reset_default_graph()
batch_size = 50
data = dataset.omniglot_dataset(batch_size=batch_size, shuffle=True, single_channel=True)
logs_path = "omniglot_outputs/"
input_a = tf.placeholder(tf.float32, [batch_size, 32, 32, 1], 'inputs-1')
input_b_same_class = tf.placeholder(tf.float32, [batch_size, 32, 32, 1], 'inputs-2-same-class')
targets = tf.placeholder(tf.int32, [batch_size], 'targets')

gan_stop_epoch = 1000
classification_start_epoch = 1500
train_results_gan_classifier = dict({"loss":[], "total_accuracy": []})
validation_results_gan_classifier = dict({"loss":[], "total_accuracy": []})
train_results_baseline_classifier = dict({"loss":[], "total_accuracy": []})
validation_results_baseline_classifier = dict({"loss":[], "total_accuracy": []})

training_phase = tf.placeholder(tf.bool, name='training-flag')
keep_prob = tf.placeholder(tf.float32, name='dropout-prob')

dcgan = omniglot_classifier(batch_size=batch_size, input_x=input_a,
              target_placeholder=targets, keep_prob=keep_prob, mean=data.mean, min=data.min, max=data.max,
              num_channels=3, n_classes=data.n_classes, is_training=training_phase)
print(data.n_classes)
summary, losses, c_error_opt_op = dcgan.init_train()

total_train_batches = (len(data.x_train) / batch_size)
total_val_batches = (len(data.x_val) / batch_size)
total_test_batches = len(data.x_test) / batch_size
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    saver = tf.train.Saver()
    continue_from_epoch = -1

    if continue_from_epoch != -1:
        saver.restore(sess, "saved_models/{}_{}.ckpt".format("omniglot_classifier_4_layers", continue_from_epoch))

    with tqdm.tqdm(total=100) as pbar_e:
        for e in range(0, 100):

            total_c_loss = 0.
            total_accuracy = 0.
            with tqdm.tqdm(total=total_train_batches) as pbar:
                for i in range(total_train_batches):
                    x_batch, y_batch = data.get_next_train_batch()

                    _, c_loss_value, acc = sess.run(
                        [c_error_opt_op, losses[dcgan.g], losses["accuracy"]],
                        feed_dict={keep_prob: 0.9, input_a: x_batch,
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
            with tqdm.tqdm(total=total_val_batches) as pbar:
                for i in range(total_val_batches):
                    x_batch, y_batch = data.get_next_val_batch()
                    c_loss_value, acc = sess.run(
                        [losses[dcgan.g], losses["accuracy"]],
                        feed_dict={keep_prob: 0.9, input_a: x_batch,
                                   targets: y_batch, training_phase: False})
                    total_c_loss += c_loss_value
                    total_accuracy += acc
                    iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss_value, acc)
                    pbar.set_description(iter_out)
                    pbar.update(1)

            total_c_loss /= total_val_batches
            total_accuracy /= total_val_batches
            print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_c_loss, total_accuracy))

            total_c_loss = 0.
            total_accuracy = 0.
            with tqdm.tqdm(total=total_test_batches) as pbar:
                for i in range(total_test_batches):
                    x_batch, y_batch = data.get_next_test_batch()
                    c_loss_value, acc = sess.run(
                        [losses[dcgan.g], losses["accuracy"]],
                        feed_dict={keep_prob: 0.9, input_a: x_batch,
                                   targets: y_batch, training_phase: False})
                    total_c_loss += c_loss_value
                    total_accuracy += acc
                    iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value, acc)
                    pbar.set_description(iter_out)
                    pbar.update(1)

            total_c_loss /= total_test_batches
            total_accuracy /= total_test_batches
            print("Epoch {}: test_loss: {}, t_accuracy: {}".format(e, total_c_loss, total_accuracy))
            pbar_e.update(1)

