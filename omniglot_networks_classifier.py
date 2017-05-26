import tensorflow as tf
import tensorflow.contrib.slim as slim

class Classifier:
    def __init__(self, batch_size, num_channels=1, num_classes=100):
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.n_classes = num_classes
    def __call__(self, conditional_input, training=False):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)

        conditional_input = tf.convert_to_tensor(conditional_input)
        with tf.variable_scope('g', reuse=self.reuse):
            # reshape from inputs

            with tf.variable_scope('conv_layers'):
                with tf.variable_scope('g_conv1'):
                    g_conv1_encoder = tf.layers.conv2d(conditional_input, 64, [3, 3], strides=(2, 2), padding='SAME')
                    g_conv1_encoder = leaky_relu(g_conv1_encoder, name='outputs')
                    g_conv1_encoder = tf.contrib.layers.batch_norm(g_conv1_encoder, is_training=training)
                with tf.variable_scope('g_conv2'):
                    g_conv2_encoder = tf.layers.conv2d(g_conv1_encoder, 64, [3, 3], strides=(2, 2), padding='SAME')
                    g_conv2_encoder = leaky_relu(g_conv2_encoder, name='outputs')
                    g_conv2_encoder = tf.contrib.layers.batch_norm(g_conv2_encoder, is_training=training)
                with tf.variable_scope('g_conv3'):
                    g_conv3_encoder = tf.layers.conv2d(g_conv2_encoder, 64, [3, 3], strides=(2, 2), padding='SAME')
                    g_conv3_encoder = leaky_relu(g_conv3_encoder, name='outputs')
                    g_conv3_encoder = tf.contrib.layers.batch_norm(g_conv3_encoder, is_training=training)
                with tf.variable_scope('g_conv4'):
                    g_conv4_encoder = tf.layers.conv2d(g_conv3_encoder, 64, [3, 3], strides=(2, 2), padding='SAME')
                    g_conv4_encoder = leaky_relu(g_conv4_encoder, name='outputs')
                    g_conv4_encoder = tf.contrib.layers.batch_norm(g_conv4_encoder, is_training=training)
            g_conv_encoder = g_conv4_encoder
            g_conv_encoder = tf.contrib.layers.flatten(g_conv_encoder)
            classify = tf.layers.dense(g_conv_encoder, self.n_classes)
            classify = leaky_relu(classify, name='classify')


        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')

        return classify


class omniglot_classifier:
    def __init__(self, input_x, target_placeholder, keep_prob, mean, min, max,
                 batch_size=100, z_dim=100, num_channels=1, n_classes=100, is_training=True):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.g = Classifier(self.batch_size, num_channels=num_channels, num_classes=n_classes)
        self.input_x = input_x
        self.keep_prob = keep_prob
        self.targets = target_placeholder
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.min = tf.convert_to_tensor(min, dtype=tf.float32)
        self.max = tf.convert_to_tensor(max, dtype=tf.float32)
        self.training_phase = is_training

    def loss(self):
        """build models, calculate losses.
        Args:
            traindata: 4-D Tensor of shape `[batch, height, width, channels]`.
        Returns:
            dict of each models' losses.
        """
        with tf.name_scope("losses"):
            preds = self.g(self.data_augment_batch(self.input_x), training=self.training_phase)
            print(preds.get_shape())
            correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(self.targets, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            crossentropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=preds))

            tf.add_to_collection('crossentropy_losses', crossentropy_loss)
            tf.add_to_collection('accuracy', accuracy)

        return {self.g: tf.add_n(tf.get_collection('crossentropy_losses'), name='total_classification_loss'),
            "accuracy": tf.add_n(tf.get_collection('accuracy'), name='accuracy')}

    def rotate_data(self, image):
        # image = tf.image.random_flip_up_down(image)
        # image = tf.image.random_flip_left_right(image)
        random_variable = tf.unstack(tf.random_uniform([1], minval=1, maxval=4, dtype=tf.int32, seed=None, name=None))
        image = tf.image.rot90(image, k=random_variable[0])
        return image

    def rotate_batch(self, batch_images):
        shapes = map(int, list(batch_images.get_shape()))
        batch_size, x, y, c = shapes
        with tf.name_scope('augment'):
            batch_images_unpacked = tf.unstack(batch_images)
            new_images = []
            for image in batch_images_unpacked:
                new_images.append(self.rotate_data(image))
            new_images = tf.stack(new_images)
            new_images = tf.reshape(new_images, (batch_size, x, y, c))
            return new_images

    def data_augment_batch(self, batch_images):
        images = tf.cond(self.training_phase, lambda: self.rotate_batch(batch_images), lambda: batch_images)
        return images

    def train(self, losses, learning_rate=1e-3, beta1=0.9):
        """
        Args:
            losses dict.
        Returns:
            train op.
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            c_opt = tf.train.AdamOptimizer(beta1=beta1, learning_rate=learning_rate)
            c_error_opt_op = c_opt.minimize(losses[self.g], var_list=self.g.variables)

        return c_error_opt_op

    def init_train(self):
        losses = self.loss()
        c_error_opt_op = self.train(losses)
        summary = tf.summary.merge_all()
        return summary, losses, c_error_opt_op
