import tensorflow as tf


class GaussianCell(tf.contrib.rnn.RNNCell):
    """
    RNN cell that  the mean and the std of a Gaussian distribution as a
    function of the hidden state.
    """
    def __init__(self, cell, event_size, reg_lambda=0.0, reuse=None):
        super(GaussianCell, self).__init__(_reuse=reuse)
        self._cell = cell
        self._event_size = event_size
        self._regulariser = tf.contrib.layers.l2_regularizer(scale=reg_lambda)
        self._layer_mu = tf.layers.Dense(self._event_size,
                                         kernel_regularizer=self._regulariser)
        self._layer_sigma = tf.layers.Dense(self._event_size, tf.nn.softplus)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return (self._event_size, self._event_size)

    def call(self, inputs, state):
        hidden, state = self._cell(inputs, state)
        mean = self._layer_mu(hidden)
        std = self._layer_sigma(hidden)
        return (mean, std), state


class GaussianConvolutionalCell(tf.contrib.rnn.RNNCell):

    def __init__(self, cell, event_size, reuse=None):
        super(GaussianConvolutionalCell, self).__init__(_reuse=reuse)
        self._cell = cell
        self._event_size = event_size

        self._conv_layer1 = tf.layers.Conv2D(8, (5, 5), strides=(1, 1),
                                             padding='valid')
        self._conv_layer2 = tf.layers.Conv2D(16, (5, 5), strides=(2, 2),
                                             padding='same')
        self._conv_layer3 = tf.layers.Conv2D(32, (5, 5), strides=(2, 2),
                                             padding='same')
        self._conv_layer4 = tf.layers.Conv2D(64, (5, 5), strides=(2, 2),
                                             padding='same')
        self._conv_layer5 = tf.layers.Conv2D(64, (5, 5), strides=(2, 2),
                                             padding='same')
        self._conv_dense = tf.layers.Dense(self._cell.state_size)

        self._deconv_dense1 = tf.layers.Dense(64)
        self._deconv_layer1 = tf.layers.Conv2DTranspose(64, (5, 5),
                                                        strides=(2, 2),
                                                        padding='same')
        self._deconv_layer2 = tf.layers.Conv2DTranspose(32, (5, 5),
                                                        strides=(2, 2),
                                                        padding='same')
        self._deconv_layer3 = tf.layers.Conv2DTranspose(16, (5, 5),
                                                        strides=(2, 2),
                                                        padding='same')
        self._deconv_layer4 = tf.layers.Conv2DTranspose(8, (5, 5),
                                                        strides=(2, 2),
                                                        padding='same')
        self._deconv_mean = tf.layers.Conv2D(1, (3, 3), strides=(1, 1),
                                             padding='valid')
        self._deconv_std = tf.layers.Conv2D(1, (3, 3), strides=(1, 1),
                                            padding='valid',
                                            activation=tf.nn.softplus)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return (self._event_size, self._event_size)

    def call(self, inputs, state):

        inputs = tf.reshape(inputs, [-1, 30, 30])
        inputs = self._conv_layer1(inputs[..., None])
        inputs = self._conv_layer2(inputs)
        inputs = self._conv_layer3(inputs)
        inputs = self._conv_layer4(inputs)
        inputs = self._conv_layer5(inputs)
        inputs = self._conv_dense(tf.layers.flatten(inputs))

        hidden, state = self._cell(inputs, state)
        hidden = self._deconv_dense1(hidden)
        hidden = tf.reshape(hidden, [-1, 2, 2, 16])
        hidden = self._deconv_layer1(hidden)
        hidden = self._deconv_layer2(hidden)
        hidden = self._deconv_layer3(hidden)
        hidden = self._deconv_layer4(hidden)
        means = tf.layers.flatten(self._deconv_mean(hidden))
        stds = tf.layers.flatten(self._deconv_std(hidden)) + 1e-5

        return (means, stds), state


class SimpleGaussianConvolutionalCell(tf.contrib.rnn.RNNCell):

    def __init__(self, cell, event_size, reuse=None):
        super(SimpleGaussianConvolutionalCell, self).__init__(_reuse=reuse)
        self._cell = cell
        self._event_size = event_size

        self._conv_layer1 = tf.layers.Conv2D(8, (5, 5), strides=(1, 1),
                                             padding='valid')
        self._conv_layer2 = tf.layers.Conv2D(16, (5, 5), strides=(2, 2),
                                             padding='same')
        self._conv_layer3 = tf.layers.Conv2D(32, (5, 5), strides=(2, 2),
                                             padding='same')
        self._conv_layer4 = tf.layers.Conv2D(64, (5, 5), strides=(2, 2),
                                             padding='same')
        self._conv_layer5 = tf.layers.Conv2D(64, (5, 5), strides=(2, 2),
                                             padding='same')
        self._conv_dense = tf.layers.Dense(self._cell.state_size)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.state_size

    def call(self, inputs, state):

        inputs = tf.reshape(inputs, [-1, 30, 30])
        inputs = self._conv_layer1(inputs[..., None])
        inputs = self._conv_layer2(inputs)
        inputs = self._conv_layer3(inputs)
        inputs = self._conv_layer4(inputs)
        inputs = self._conv_layer5(inputs)
        inputs = self._conv_dense(tf.layers.flatten(inputs))

        hidden, state = self._cell(inputs, state)

        return hidden, state


class SamplingCell(tf.contrib.rnn.RNNCell):
    """
    RNN cell that produce a sequence in autoregressive mode by sampling
    at each step.
    """

    def __init__(self, cell, output_size, reuse=None):
        super(SamplingCell, self).__init__(_reuse=reuse)
        self._cell = cell
        self._output_size = output_size

    @property
    def state_size(self):
        return (self._cell.state_size, self._output_size)

    @property
    def output_size(self):
        return (self._output_size, self._output_size, self._output_size)

    def call(self, _, state_and_sample):
        (state, sample) = state_and_sample
        (mean, std), state = self._cell(sample, state)
        sample = tf.contrib.distributions.MultivariateNormalDiag(mean,
                                                                 std).sample()
        return (mean, std, sample), (state, sample)


class BeamSamplingCell(tf.contrib.rnn.RNNCell):
    """
    RNN cell that produce a Beam Search sequence in auto-regressive mode by
    sampling and pruning at every step.
    """

    def __init__(self, cell, output_size, branch_width, reuse=None):
        super(BeamSamplingCell, self).__init__(_reuse=reuse)
        self._cell = cell
        self._output_size = output_size
        self._branch_width = branch_width

    @property
    def state_size(self):
        return (self._cell.state_size, self._output_size, 1)

    @property
    def output_size(self):
        return (self._output_size, self._output_size)

    def call(self, _, state_and_sample_and_prob):
        (state, sample, probs) = state_and_sample_and_prob
        (mean, std), state = self._cell(sample, state)
        dist = tf.contrib.distributions.MultivariateNormalDiag(mean, std)
        sample = dist.sample((self._branch_width))
        new_probs = dist.log_prob(sample)
        new_probs = tf.reshape(new_probs, [-1])
        probs = tf.reshape(tf.tile(probs[None, ..., 0],
                                   [self._branch_width, 1]),
                           [-1])
        best_probs, best_idx = tf.nn.top_k(new_probs + probs,
                                           k=self._branch_width)
        best_samples = tf.gather(tf.reshape(sample, [-1, self._output_size]),
                                 best_idx)

        return_1 = (mean, best_samples)
        return_2 = (state, best_samples, best_probs[..., None])
        return return_1, return_2
