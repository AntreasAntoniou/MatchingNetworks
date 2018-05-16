import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.nn_ops import max_pool, leaky_relu


class g_embedding_bidirectionalLSTM:
    def __init__(self, name, layer_sizes, batch_size):
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer e.g. [100, 100, 100] returns a 3 layer, 100
                                                                                                        neuron bid-LSTM
        :param batch_size: The experiments batch size
        """
        self.reuse = False
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.name = name

    def __call__(self, inputs, training=False):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param inputs: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :param name: Name to give to the tensorflow op
        :param training: Flag that indicates if this is a training or evaluation stage
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
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

            print("g out shape", tf.stack(outputs, axis=1).get_shape().as_list())

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return outputs

class f_embedding_bidirectionalLSTM:
    def __init__(self, name, layer_size, batch_size):
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer e.g. [100, 100, 100] returns a 3 layer, 100
                                                                                                        neuron bid-LSTM
        :param batch_size: The experiments batch size
        """
        self.reuse = False
        self.batch_size = batch_size
        self.layer_size = layer_size
        self.name = name

    def __call__(self, support_set_embeddings, target_set_embeddings, K, training=False):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param inputs: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :param name: Name to give to the tensorflow op
        :param training: Flag that indicates if this is a training or evaluation stage
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        b, k, h_g_dim = support_set_embeddings.get_shape().as_list()
        b, h_f_dim = target_set_embeddings.get_shape().as_list()
        with tf.variable_scope(self.name, reuse=self.reuse):
            fw_lstm_cells_encoder = rnn.LSTMCell(num_units=self.layer_size, activation=tf.nn.tanh)
            attentional_softmax = tf.ones(shape=(b, k)) * (1.0/k)
            h = tf.zeros(shape=(b, h_g_dim)) + target_set_embeddings
            h = (h, h)
            for i in range(K):
                attentional_softmax = tf.expand_dims(attentional_softmax, axis=2)
                attented_features = support_set_embeddings * attentional_softmax
                attented_features_summed = tf.reduce_sum(attented_features, axis=1)
                x, h = fw_lstm_cells_encoder(inputs=attented_features_summed, state=h)
                attentional_softmax = tf.layers.dense(x, units=k, activation=tf.nn.softmax, reuse=self.reuse)
                self.reuse = True

        outputs = x
        print("out shape", tf.stack(outputs, axis=0).get_shape().as_list())
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        print(self.variables)
        return outputs


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
            preds = tf.squeeze(tf.matmul(tf.expand_dims(similarities, 1), support_set_y))
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attentional-classification')
        return preds


class Classifier:
    def __init__(self, name, batch_size, layer_sizes, num_channels=1):
        """
        Builds a CNN to produce embeddings
        :param batch_size: Batch size for experiment
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param num_channels: Number of channels of images
        """
        self.reuse = False
        self.name = name
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_sizes = layer_sizes
        assert len(self.layer_sizes) == 4, "layer_sizes should be a list of length 4"

    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, 64]
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            outputs = image_input
            with tf.variable_scope('conv_layers'):
                for idx, num_filters in enumerate(self.layer_sizes):
                    with tf.variable_scope('g_conv_{}'.format(idx)):
                        if idx == len(self.layer_sizes) - 1:
                            outputs = tf.layers.conv2d(outputs, num_filters, [2, 2], strides=(1, 1),
                                                       padding='VALID')
                        else:
                            outputs = tf.layers.conv2d(outputs, num_filters, [3, 3], strides=(1, 1),
                                                               padding='VALID')
                        outputs = leaky_relu(outputs)
                        outputs = tf.contrib.layers.batch_norm(outputs, updates_collections=None,
                                                                       decay=0.99,
                                                                       scale=True, center=True,
                                                                       is_training=training)
                        outputs = max_pool(outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                        #outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

            image_embedding = tf.contrib.layers.flatten(outputs)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return image_embedding


class MatchingNetwork:
    def __init__(self, support_set_images, support_set_labels, target_image, target_label, dropout_rate,
                 batch_size=100, num_channels=1, is_training=False, learning_rate=0.001, fce=False,
                 full_context_unroll_k=5, num_classes_per_set=5, num_samples_per_class=1,
                 average_per_class_embeddings=False):

        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, 28, 28, 1]
        :param support_set_labels: A tensor containing the support set labels [batch_size, sequence_size, 1]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, 28, 28, 1]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :param dropout_rate: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
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
        self.classifier = Classifier(name="classifier_net", batch_size=self.batch_size,
                            num_channels=num_channels, layer_sizes=[64, 64, 64, 64])
        if fce:
            self.g_lstm = g_embedding_bidirectionalLSTM(name="g_lstm", layer_sizes=[32], batch_size=self.batch_size)
            self.f_lstm = f_embedding_bidirectionalLSTM(name="f_attlstm", layer_size=64, batch_size=self.batch_size)

        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        self.full_context_K = full_context_unroll_k
        self.support_set_images = support_set_images
        self.support_set_labels = support_set_labels
        self.average_per_class_embeddings = average_per_class_embeddings
        self.target_image = target_image
        self.target_label = target_label
        self.dropout_rate = dropout_rate
        self.is_training = is_training
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.learning_rate = learning_rate

    def loss(self):
        """
        Builds tf graph for Matching Networks, produces losses and summary statistics.
        :return:
        """
        with tf.name_scope("losses"):
            [b, num_classes, spc] = self.support_set_labels[0].get_shape().as_list()
            self.support_set_labels = tf.reshape(self.support_set_labels[0], shape=(b, num_classes * spc))
            self.support_set_labels = tf.one_hot(self.support_set_labels, self.num_classes_per_set)  # one hot encode

            g_encoded_images = []

            [b, num_classes, spc, h, w, c] = self.support_set_images[0].get_shape().as_list()
            self.support_set_images = tf.reshape(self.support_set_images[0], shape=(b, num_classes * spc, h, w, c))

            for image in tf.unstack(self.support_set_images, axis=1):  # produce embeddings for support set images
                support_set_cnn_embed = self.classifier(image_input=image, training=self.is_training,
                                                        dropout_rate=self.dropout_rate)
                g_encoded_images.append(support_set_cnn_embed)

            if self.average_per_class_embeddings:
                g_encoded_images = tf.stack(g_encoded_images, axis=1)
                b, k, h = g_encoded_images.get_shape().as_list()
                g_encoded_images = tf.reshape(shape=(b, num_classes, spc, h))
                g_encoded_images = tf.reduce_mean(g_encoded_images, axis=2)
                self.support_set_labels = tf.reshape(self.support_set_labels, shape=(b, num_classes, spc,
                                                                                     self.num_classes_per_set))
                self.support_set_labels = tf.reduce_mean(self.support_set_labels, axis=2)

            target_image = self.target_image[0]  # produce embedding for target images

            f_encoded_image = self.classifier(image_input=target_image, training=self.is_training,
                                                   dropout_rate=self.dropout_rate)

            if self.fce:  # Apply LSTM on embeddings if fce is enabled
                g_encoded_images = self.g_lstm(g_encoded_images, training=self.is_training)
                f_encoded_image = self.f_lstm(support_set_embeddings=tf.stack(g_encoded_images, axis=1),
                                              K=self.full_context_K,
                                              target_set_embeddings=f_encoded_image, training=self.is_training)
            g_encoded_images = tf.stack(g_encoded_images, axis=0)
            similarities = self.dn(support_set=g_encoded_images, input_image=f_encoded_image, name="distance_calculation",
                                   training=self.is_training)  # get similarity between support set embeddings and target

            preds = self.classify(similarities,
                                  support_set_y=self.support_set_labels, name='classify', training=self.is_training)
            # produce predictions for target probabilities

            correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(self.target_label[0], tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            targets = tf.one_hot(self.target_label[0], self.num_classes_per_set)

            crossentropy_loss = self.crossentropy_softmax(targets=targets, outputs=preds)

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
                train_variables = self.f_lstm.variables + self.g_lstm.variables + self.classifier.variables
            else:
                train_variables = self.classifier.variables
            c_error_opt_op = c_opt.minimize(losses[self.classify],
                                            var_list=train_variables, colocate_gradients_with_ops=True)

        return c_error_opt_op

    def crossentropy_softmax(self, outputs, targets):
        normOutputs = outputs - tf.reduce_max(outputs, axis=-1)[:, None]
        logProb = normOutputs - tf.log(tf.reduce_sum(tf.exp(normOutputs), axis=-1)[:, None])
        return -tf.reduce_mean(tf.reduce_sum(targets * logProb, axis=1))

    def init_train(self):
        """
        Get all ops, as well as all losses.
        :return:
        """
        losses = self.loss()
        c_error_opt_op = self.train(losses)
        summary = tf.summary.merge_all()
        return summary, losses, c_error_opt_op
