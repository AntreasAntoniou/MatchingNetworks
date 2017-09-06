import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.nn_ops import max_pool


class BidirectionalLSTM:
    def __init__(self, layer_sizes, batch_size):
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer e.g. [100, 100, 100] returns a 3 layer, 100
                                                                                                        neuron bid-LSTM
        :param batch_size: The experiments batch size
        """
        self.reuse = False
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes

    def __call__(self, inputs, name, training=False):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param inputs: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :param name: Name to give to the tensorflow op
        :param training: Flag that indicates if this is a training or evaluation stage
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        with tf.name_scope('bid-lstm' + name), tf.variable_scope('bid-lstm', reuse=self.reuse):
            with tf.variable_scope("encoder"):
                fw_lstm_cells_encoder = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                                 for i in range(len(self.layer_sizes))]
                bw_lstm_cells_encoder = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                                 for i in range(len(self.layer_sizes))]



                outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_rnn(
                    fw_lstm_cells_encoder,
                    bw_lstm_cells_encoder,
                    inputs,
                    dtype=tf.float32
                )
            print("out shape", tf.stack(outputs, axis=0).get_shape().as_list())
            with tf.variable_scope("decoder"):
                fw_lstm_cells_decoder = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                                         for i in range(len(self.layer_sizes))]
                bw_lstm_cells_decoder = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                                         for i in range(len(self.layer_sizes))]
                outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_rnn(
                    fw_lstm_cells_decoder,
                    bw_lstm_cells_decoder,
                    outputs,
                    dtype=tf.float32
                )


        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bid-lstm')
        return outputs, output_state_fw, output_state_bw


class DistanceNetwork:
    def __init__(self):
        self.reuse = False

    def __call__(self, support_set, input_image, name, training=False):
        """
        This module calculates the cosine distance between each of the support set embeddings and the target
        image embeddings.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :param name: Name of the op to appear on the graph
        :param training: Flag indicating training or evaluation (True/False)
        :return: A tensor with cosine similarities of shape [batch_size, sequence_length, 1]
        """
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
        """
        Produces pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarities of size [sequence_length, batch_size, 1]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image
                                                                            [sequence_length,  batch_size, num_classes]
        :param name: The name of the op to appear on tf graph
        :param training: Flag indicating training or evaluation stage (True/False)
        :return: Softmax pdf
        """
        with tf.name_scope('attentional-classification' + name), tf.variable_scope('attentional-classification',
                                                                                   reuse=self.reuse):
            softmax_similarities = tf.nn.softmax(similarities)
            preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities, 1), support_set_y))
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attentional-classification')
        return preds


class Classifier:
    def __init__(self, batch_size, layer_sizes, num_channels=1):
        """
        Builds a CNN to produce embeddings
        :param batch_size: Batch size for experiment
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param num_channels: Number of channels of images
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_sizes = layer_sizes
        assert len(self.layer_sizes)==4, "layer_sizes should be a list of length 4"

    def __call__(self, image_input, training=False, keep_prob=1.0):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param keep_prob: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, 64]
        """

        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)

        with tf.variable_scope('g', reuse=self.reuse):

            with tf.variable_scope('conv_layers'):
                with tf.variable_scope('g_conv1'):
                    g_conv1_encoder = tf.layers.conv2d(image_input, self.layer_sizes[0], [3, 3], strides=(1, 1),
                                                       padding='SAME')
                    g_conv1_encoder = leaky_relu(g_conv1_encoder, name='outputs')
                    g_conv1_encoder = tf.contrib.layers.batch_norm(g_conv1_encoder, updates_collections=None, decay=0.99,
                                                                   scale=True, center=True, is_training=training)
                    g_conv1_encoder = max_pool(g_conv1_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv1_encoder = tf.nn.dropout(g_conv1_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv2'):
                    g_conv2_encoder = tf.layers.conv2d(g_conv1_encoder, self.layer_sizes[1], [3, 3], strides=(1, 1),
                                                       padding='SAME')
                    g_conv2_encoder = leaky_relu(g_conv2_encoder, name='outputs')
                    g_conv2_encoder = tf.contrib.layers.batch_norm(g_conv2_encoder, updates_collections=None,
                                                                   decay=0.99,
                                                                   scale=True, center=True, is_training=training)
                    g_conv2_encoder = max_pool(g_conv2_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv2_encoder = tf.nn.dropout(g_conv2_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv3'):
                    g_conv3_encoder = tf.layers.conv2d(g_conv2_encoder, self.layer_sizes[2], [3, 3], strides=(1, 1),
                                                       padding='SAME')
                    g_conv3_encoder = leaky_relu(g_conv3_encoder, name='outputs')
                    g_conv3_encoder = tf.contrib.layers.batch_norm(g_conv3_encoder, updates_collections=None,
                                                                   decay=0.99,
                                                                   scale=True, center=True, is_training=training)
                    g_conv3_encoder = max_pool(g_conv3_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv3_encoder = tf.nn.dropout(g_conv3_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv4'):
                    g_conv4_encoder = tf.layers.conv2d(g_conv3_encoder, self.layer_sizes[3], [2, 2], strides=(1, 1),
                                                       padding='SAME')
                    g_conv4_encoder = leaky_relu(g_conv4_encoder, name='outputs')
                    g_conv4_encoder = tf.contrib.layers.batch_norm(g_conv4_encoder, updates_collections=None,
                                                                   decay=0.99,
                                                                   scale=True, center=True, is_training=training)
                    g_conv4_encoder = max_pool(g_conv4_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv4_encoder = tf.nn.dropout(g_conv4_encoder, keep_prob=keep_prob)

            g_conv_encoder = g_conv4_encoder
            g_conv_encoder = tf.contrib.layers.flatten(g_conv_encoder)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return g_conv_encoder


class MatchingNetwork:
    def __init__(self, support_set_images, support_set_labels, target_image, target_label, keep_prob,
                 batch_size=100, num_channels=1, is_training=False, learning_rate=0.001, rotate_flag=False, fce=False, num_classes_per_set=5,
                 num_samples_per_class=1):

        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, 28, 28, 1]
        :param support_set_labels: A tensor containing the support set labels [batch_size, sequence_size, 1]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, 28, 28, 1]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :param keep_prob: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the images
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the images
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        """
        self.batch_size = batch_size
        self.fce = fce
        self.g = Classifier(self.batch_size, num_channels=num_channels, layer_sizes=[64, 64, 64 ,64])
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
        self.learning_rate = learning_rate

    def loss(self):
        """
        Builds tf graph for Matching Networks, produces losses and summary statistics.
        :return:
        """
        with tf.name_scope("losses"):
            [b, num_classes, spc] = self.support_set_labels.get_shape().as_list()
            self.support_set_labels = tf.reshape(self.support_set_labels, shape=(b, num_classes * spc))
            self.support_set_labels = tf.one_hot(self.support_set_labels, self.num_classes_per_set)  # one hot encode
            encoded_images = []
            [b, num_classes, spc, h, w, c] = self.support_set_images.get_shape().as_list()
            self.support_set_images = tf.reshape(self.support_set_images, shape=(b,  num_classes*spc, h, w, c))
            for image in tf.unstack(self.support_set_images, axis=1):  #produce embeddings for support set images
                gen_encode = self.g(image_input=image, training=self.is_training, keep_prob=self.keep_prob)
                encoded_images.append(gen_encode)

            target_image = self.target_image  #produce embedding for target images
            gen_encode = self.g(image_input=target_image, training=self.is_training, keep_prob=self.keep_prob)

            encoded_images.append(gen_encode)

            if self.fce:  # Apply LSTM on embeddings if fce is enabled
                encoded_images, output_state_fw, output_state_bw = self.lstm(encoded_images, name="lstm",
                                                                             training=self.is_training)
            outputs = tf.stack(encoded_images)

            similarities = self.dn(support_set=outputs[:-1], input_image=outputs[-1], name="distance_calculation",
                                   training=self.is_training)  #get similarity between support set embeddings and target

            preds = self.classify(similarities,
                                support_set_y=self.support_set_labels, name='classify', training=self.is_training)
                                # produce predictions for target probabilities

            correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(self.target_label, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            targets = tf.one_hot(self.target_label, self.num_classes_per_set)
            crossentropy_loss = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(preds),
                                                              reduction_indices=[1]))

            tf.add_to_collection('crossentropy_losses', crossentropy_loss)
            tf.add_to_collection('accuracy', accuracy)

        return {
            self.classify: tf.add_n(tf.get_collection('crossentropy_losses'), name='total_classification_loss'),
            self.dn: tf.add_n(tf.get_collection('accuracy'), name='accuracy')
        }

    def train(self, losses):

        """
        Builds the train op
        :param losses: A dictionary containing the losses
        :param learning_rate: Learning rate to be used for Adam
        :param beta1: Beta1 to be used for Adam
        :return:
        """
        c_opt = tf.train.AdamOptimizer(beta1=0.9, learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):  # Needed for correct batch norm usage
            if self.fce:
                train_variables = self.lstm.variables + self.g.variables
            else:
                train_variables = self.g.variables
            c_error_opt_op = c_opt.minimize(losses[self.classify],
                                            var_list=train_variables)

        return c_error_opt_op

    def init_train(self):
        """
        Get all ops, as well as all losses.
        :return:
        """
        losses = self.loss()
        c_error_opt_op = self.train(losses)
        summary = tf.summary.merge_all()
        return  summary, losses, c_error_opt_op
