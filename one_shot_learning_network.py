import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.nn_ops import max_pool


class BidirectionalLSTM:
    def __init__(self, layer_sizes, batch_size):
        self.reuse = False
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes

    def __call__(self, inputs, name, training=False):
        with tf.name_scope('bid-lstm' + name), tf.variable_scope('bid-lstm', reuse=self.reuse):
            fw_lstm_cells = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh) for i in range(len(self.layer_sizes))]
            bw_lstm_cells = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh) for i in range(len(self.layer_sizes))]

            outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_rnn(
                fw_lstm_cells,
                bw_lstm_cells,
                inputs,
                dtype=tf.float32
            )

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bid-lstm')
        return outputs, output_state_fw, output_state_bw


class DistanceNetwork:
    def __init__(self):
        self.reuse = False

    def __call__(self, support_set, input_image, name, training=False):
        with tf.name_scope('distance-module' + name), tf.variable_scope('distance-module', reuse=self.reuse):
            eps = 1e-10
            similarities = []
            for support_image in tf.unstack(support_set, axis=0):
                sum_support = tf.reduce_sum(tf.square(support_image), 1, keep_dims=True)
                support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support, eps, float("inf")))
                dot_product = tf.matmul(tf.expand_dims(input_image, 1), tf.expand_dims(support_image, 2))
                dot_product = tf.squeeze(dot_product, [1, ])
                cosine_similarity = dot_product * support_magnitude
                similarities.append(cosine_similarity)

        similarities = tf.concat(axis=1, values=similarities)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='distance-module')

        return similarities


class AttentionalClassify:
    def __init__(self):
        self.reuse = False
    def __call__(self, similarities, support_set_y, name, training=False):
        with tf.name_scope('attentional-classification' + name), tf.variable_scope('attentional-classification', reuse=self.reuse):
            softmax_similarities = tf.nn.softmax(similarities)
            preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities, 1), support_set_y))
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attentional-classification')
        return preds


class Classifier:
    def __init__(self, batch_size, num_channels=1):
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
    def __call__(self, conditional_input, training=False, keep_prob=1.0):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)

        conditional_input = tf.convert_to_tensor(conditional_input)
        with tf.variable_scope('g', reuse=self.reuse):

            with tf.variable_scope('conv_layers'):
                with tf.variable_scope('g_conv1'):
                    g_conv1_encoder = tf.layers.conv2d(conditional_input, 64, [3, 3], strides=(1, 1), padding='VALID')
                    g_conv1_encoder = leaky_relu(g_conv1_encoder, name='outputs')
                    g_conv1_encoder = tf.contrib.layers.batch_norm(g_conv1_encoder, is_training=training)
                    g_conv1_encoder = max_pool(g_conv1_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv1_encoder = tf.nn.dropout(g_conv1_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv2'):
                    g_conv2_encoder = tf.layers.conv2d(g_conv1_encoder, 64, [3, 3], strides=(1, 1), padding='VALID')
                    g_conv2_encoder = leaky_relu(g_conv2_encoder, name='outputs')
                    g_conv2_encoder = tf.contrib.layers.batch_norm(g_conv2_encoder, is_training=training)
                    g_conv2_encoder = max_pool(g_conv2_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv2_encoder = tf.nn.dropout(g_conv2_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv3'):
                    g_conv3_encoder = tf.layers.conv2d(g_conv2_encoder, 64, [3, 3], strides=(1, 1), padding='VALID')
                    g_conv3_encoder = leaky_relu(g_conv3_encoder, name='outputs')
                    g_conv3_encoder = tf.contrib.layers.batch_norm(g_conv3_encoder, is_training=training)
                    g_conv3_encoder = max_pool(g_conv3_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv3_encoder = tf.nn.dropout(g_conv3_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv4'):
                    g_conv4_encoder = tf.layers.conv2d(g_conv3_encoder, 64, [2, 2], strides=(1, 1), padding='VALID')
                    g_conv4_encoder = leaky_relu(g_conv4_encoder, name='outputs')
                    g_conv4_encoder = tf.contrib.layers.batch_norm(g_conv4_encoder, is_training=training)
                    g_conv4_encoder = max_pool(g_conv4_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    g_conv4_encoder = tf.nn.dropout(g_conv4_encoder, keep_prob=keep_prob)

            g_conv_encoder = g_conv4_encoder
            g_conv_encoder = tf.contrib.layers.flatten(g_conv_encoder)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return g_conv_encoder


class MatchingNetwork:
    def __init__(self, support_set_images, support_set_labels, target_image, target_label, keep_prob,
                 batch_size=100, num_channels=1, is_training=False, rotate_flag=False, fce=False, num_classes_per_set=5,
                 num_samples_per_class=1):

        self.batch_size = batch_size
        self.fce = fce
        self.g = Classifier(self.batch_size, num_channels=num_channels)
        if fce:
            self.lstm = BidirectionalLSTM(layer_sizes=[32], batch_size=self.batch_size)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        self.support_set_images = support_set_images
        self.support_set_labels = support_set_labels
        self.target_image = target_image
        self.target_label = target_label
        self.keep_prob = keep_prob
        self.is_training = is_training
        self.k = None
        self.rotate_flag = rotate_flag
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class

    def rotate_data(self, image):
        if self.k is None:
            self.k = tf.unstack(tf.random_uniform([1], minval=1, maxval=4, dtype=tf.int32, seed=None, name=None))
        image = tf.image.rot90(image, k=self.k[0])
        return image

    def rotate_batch(self, batch_images):
        shapes = map(int, list(batch_images.get_shape()))
        batch_size, x, y, c = shapes
        with tf.name_scope('augment'):
            batch_images_unpacked = tf.unstack(batch_images)
            new_images = []
            for image in batch_images_unpacked:
                rotated_batch = self.rotate_data(image)
                new_images.append(rotated_batch)
            new_images = tf.stack(new_images)
            new_images = tf.reshape(new_images, (batch_size, x, y, c))
            return new_images

    def data_augment_batch(self, batch_images):
        images = tf.cond(self.rotate_flag, lambda: self.rotate_batch(batch_images), lambda: batch_images)
        return images

    def loss(self):
        """build models, calculate losses.
        Args:
            traindata: 4-D Tensor of shape `[batch, height, width, channels]`.
        Returns:
            dict of each models' losses.
        """
        with tf.name_scope("losses"):
            self.support_set_labels = tf.one_hot(
                self.support_set_labels,
                self.num_classes_per_set)
            encoded_images = []
            for image in tf.unstack(self.support_set_images, axis=1):
                image = self.data_augment_batch(image)
                gen_encode = self.g(conditional_input=image, training=self.is_training, keep_prob=self.keep_prob)
                gen_encode = tf.contrib.layers.flatten(gen_encode)
                encoded_images.append(gen_encode)
            target_image = self.data_augment_batch(self.target_image)
            gen_encode = self.g(conditional_input=target_image, training=self.is_training, keep_prob=self.keep_prob)
            gen_encode = tf.contrib.layers.flatten(gen_encode)
            encoded_images.append(gen_encode)
            if self.fce:
                encoded_images, output_state_fw, output_state_bw = self.lstm(encoded_images, name="lstm",
                                                                             training=self.is_training)
            outputs = tf.stack(encoded_images)
            similarities = self.dn(support_set=outputs[:-1], input_image=outputs[-1], name="distance_calculation",
                                   training=self.is_training)
            preds = self.classify(similarities,
                                support_set_y=self.support_set_labels, name='classify', training=self.is_training)
            correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(self.target_label, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            crossentropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_label,
                                                                                              logits=preds))

            tf.add_to_collection('crossentropy_losses', crossentropy_loss)
            tf.add_to_collection('accuracy', accuracy)

        return {
            self.classify: tf.add_n(tf.get_collection('crossentropy_losses'), name='total_classification_loss'),
            self.dn: tf.add_n(tf.get_collection('accuracy'), name='accuracy')
        }

    def train(self, losses, learning_rate=4e-5, beta1=0.9):
        """
        Args:
            losses dict.
        Returns:
            train op.
        """
        c_opt = tf.train.AdamOptimizer(beta1=beta1, learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.fce:
                train_variables = self.lstm.variables + self.g.variables
            else:
                train_variables = self.g.variables
            c_error_opt_op = c_opt.minimize(losses[self.classify],
                                            var_list=train_variables)

        return c_error_opt_op

    def init_train(self):
        losses = self.loss()
        c_error_opt_op = self.train(losses)
        summary = tf.summary.merge_all()
        return  summary, losses, c_error_opt_op
