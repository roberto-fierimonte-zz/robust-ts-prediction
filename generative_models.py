import functools
import tensorflow as tf
import numpy as np

from nn.cells import GaussianCell, GaussianConvolutionalCell, SamplingCell, \
BeamSamplingCell, SimpleGaussianConvolutionalCell, GaussianCellWithoutMeans

tfd = tf.contrib.distributions
tfn = tf.contrib.rnn
tfl = tf.linalg
dynrnn = tf.nn.dynamic_rnn
bidynrnn = tf.nn.bidirectional_dynamic_rnn


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class StandardRNN(object):
    """
    First baseline
    """

    def __init__(self, data, n_hidden, seq_length, batch_size):
        self._B = batch_size                                      # B
        self._T = seq_length                                      # T
        self._D = 1                                               # D
        self._H = n_hidden                                        # H
        self._data = data                                         # B x T x D

        self.initialise_variables
        self.compute_cost_func
        self.optimise

    @define_scope
    def initialise_variables(self):
        self._cell = tfn.GRUBlockCell(self._H)
        self._optimiser = tf.train.AdamOptimizer(0.001)

    @define_scope
    def compute_cost_func(self):
        inputs = tf.concat([tf.zeros((self._B, 1, self._D)), self._data[:, :-1]], 1)
        outputs, _ = dynrnn(self._cell, inputs=inputs, dtype=tf.float32)
        return tf.reduce_sum(tf.square(outputs - self._data))

    @define_scope
    def optimise(self):
        gradients, variables = zip(*self._optimiser.compute_gradients(self.compute_cost_func))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))


class SequenceVAE(object):
    """
    Second baseline
    """

    def __init__(self, data, n_hidden, n_samples, seq_length, batch_size):
        self._data = data                                         # B x T x D
        self._B = batch_size                                      # B
        self._T = seq_length                                      # T
        self._D = 1                                               # D
        self._H = n_hidden                                        # H
        self._Sz = n_samples                                      # Sz
        self._Z = 20                                              # Z

        self._branch_width = 5

        self.initialise_variables
        self.q_z = self.make_encoder
        self.p_y = self.make_decoder
        self.compute_elbo
        self.optimise
        self.predict_forward_beam

    @define_scope
    def initialise_variables(self):
        self._cell = tfn.GRUBlockCell(self._H)
        self._gauss_cell = GaussianCell(self._cell, self._D)
        self._beam_cell = BeamSamplingCell(self._gauss_cell, self._D, self._branch_width)
        self._post_cell = tfn.GRUBlockCell(self._H)
        self._layer = tf.layers.Dense(self._H, tf.nn.relu)
        self._optimiser = tf.train.AdamOptimizer(0.001)

    @define_scope
    def make_encoder(self):
        _, h = dynrnn(self._post_cell, self._data, dtype=tf.float32)
        means = tf.layers.dense(h, self._Z)
        stds = tf.layers.dense(h, self._Z, tf.nn.softplus) + 1e-5
        posterior_z = tfd.MultivariateNormalDiag(means, stds)
        return posterior_z

    @define_scope
    def make_kl_z(self):
        p = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        return tf.reduce_sum(tfd.kl_divergence(self.q_z, p))

    @define_scope
    def make_decoder(self):
        self._code_z = self.q_z.sample((self._Sz))
        hidden = self._layer(tf.reshape(self._code_z, [-1, self._Z]))                         # (Sz * B) x H
        data = tf.reshape(tf.tile(self._data[None, ...], (self._Sz, 1, 1, 1)), [-1, self._T, self._D])
        inputs = tf.concat([tf.zeros((self._B * self._Sz, 1, self._D)), data[:, :-1]], 1)
        (means, stds), h = dynrnn(self._gauss_cell, inputs=inputs, initial_state=hidden)
        means = tf.identity(means, name='means')
        stds = tf.identity(stds, name='stds')
        return tfd.MultivariateNormalDiag(means, stds)

    @define_scope
    def make_reconstruction(self):
        data = tf.reshape(tf.tile(self._data[None, ...], (self._Sz, 1, 1, 1)), [-1, self._T, self._D])
        return tf.reduce_sum(self.p_y.log_prob(data))/self._Sz

    @define_scope
    def compute_elbo(self):
        kl_z = self.make_kl_z
        reconstruction = self.make_reconstruction
        loss = reconstruction - kl_z
        return loss, kl_z, reconstruction

    @define_scope
    def optimise(self):
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo[0]))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def predict_forward_beam(self):
        code_z = self.q_z.sample()[:1]
        hidden = self._layer(code_z)
        inputs = self._data[:1]
        _, h = dynrnn(self._cell, inputs=inputs, initial_state=hidden)
        fake_inputs = tf.zeros((self._branch_width, 30, self._D))
        state_and_sample_and_prob = (tf.tile(h, [self._branch_width, 1]),
                                     tf.tile(inputs[:, -1], [self._branch_width, 1]),
                                     tf.zeros((self._branch_width, 1)))

        (_, samples), (_, _, probs) =\
            dynrnn(self._beam_cell, inputs=fake_inputs, initial_state=state_and_sample_and_prob)

        best_idx = tf.argmax(probs)
        return tf.gather(samples, best_idx)


class RobustARGeneratorBottleNeck(object):
    """
    Model to test
    """

    def __init__(self, data, n_hidden, n_samples, seq_length, batch_size, alpha=None, beta=None):
        self._data = data                                               # B x T x D
        self._B = batch_size                                            # B - batch size
        self._T = seq_length                                            # T - number of time steps
        self._D = 1                                                     # D - dimensionality of each time step
        self._H = n_hidden                                              # H - dimensionality of RNN hidden states
        self._Sz = n_samples                                            # Sz - number of samples from the posterior
        self._S_forw = 10                                               # S - number of samples from the forward posterior
        self._Z = 20                                                    # Z - dimensionality of the global latent z
        self._alpha0 = alpha                                            # parameter of inverse gamma prior (optional)
        self._beta0 = beta                                              # parameter of inverse gamma prior (optional)

        self.initialise_variables                                       #

        self.q_z = self.make_posterior_z                                # posterior q(z|x)
        self.p_y = self.make_transitional                               # prior p(y|z) under q(z)
        self.q_psi_y = self.make_forward_posterior_y                    # forward posterior q_psi(y|y_bar)
        self.q_psi_z = self.make_forward_posterior_z                    # forward posterior q_psi(z|y_bar)
        self.p_y_kl_forward, self.p_y_bar = self.make_rnn_forward       # prior p(y|z) and forward "prior" p(y_bar|y, z)
                                                                        # under q_psi(z|y_bar) and q_psi(y|y_bar)
        self.compute_elbo
        self.optimise

        self.compute_elbo_forward
        self.forward_E_step
        self.forward_M_step

    @define_scope
    def initialise_variables(self):
        # Initialises all variables and RNN cells
        self._cell = tfn.GRUBlockCell(self._H)
        self._gauss_cell = GaussianCell(self._cell, self._D)
        self._sampling_cell = SamplingCell(self._gauss_cell, self._D)
        self._layer = tf.layers.Dense(self._H, tf.nn.relu)

        self._post_cell_z = tfn.GRUBlockCell(self._H)

        self._most_likely_pred = tf.get_variable(shape=(1, 30, self._D), name='best_forward_sequence',
                                                 initializer=tf.constant_initializer(0))

        self._post_forward_cell_y = tfn.GRUBlockCell(self._H)
        self._gauss_forward_cell_y = GaussianCell(self._post_forward_cell_y, self._D)
        self._post_forward_cell_z = tfn.GRUBlockCell(self._H)

        self._optimiser = tf.train.AdamOptimizer(0.001)

    @define_scope
    def make_posterior_z(self):
        # Computes the posterior distribution q(z|x)
        _, h = dynrnn(self._post_cell_z, self._data, dtype=tf.float32)
        means = tf.layers.dense(h, self._Z)
        stds = tf.layers.dense(h, self._Z, tf.nn.softplus) + 1e-5
        posterior_z = tfd.MultivariateNormalDiag(means, stds)
        return posterior_z

    @define_scope
    def make_kl_z(self):
        # Computes the KL divergence from p(z) to q(z|x)
        p = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        return tf.reduce_sum(tfd.kl_divergence(self.q_z, p))

    @define_scope
    def make_transitional(self):
        # Computes the prior p(y|z) = q(y|z) under q(z|x) by passing z through dense layer, then
        # using this as initial state, outputs the distn of p(y_t|y_{<t},z) at each t
        self._code_z = self.q_z.sample((self._Sz))
        hidden = self._layer(tf.reshape(self._code_z, [-1, self._Z]))                                   # (Sz * B) x H
        state_and_sample = (hidden, tf.zeros((self._B * self._Sz, self._D)))
        inputs = tf.zeros((self._B * self._Sz, self._T, self._D))
        (means, stds, self._code_y), _ = dynrnn(self._sampling_cell, initial_state=state_and_sample, inputs=inputs)
        means = tf.identity(means, name='means')
        stds = tf.identity(stds, name='stds')
        return tfd.MultivariateNormalDiag(means, stds)

    @define_scope
    def make_reconstruction(self):
        # Computes the expected reconstruction log p(x|y) under q(y,z|x)
        data = tf.reshape(tf.tile(self._data[None, ...], (self._Sz, 1, 1, 1)), [-1, self._T, self._D])
        if not (self._alpha0 is not None and self._beta0 is not None):
            self._gen_std = tf.nn.softplus(tf.get_variable(shape=(), name='generative_std',
                                                           initializer=tf.constant_initializer(1))) + 1e-5
            gen_prob = tfd.MultivariateNormalDiag(self._code_y, scale_identity_multiplier=self._gen_std)

        else:
            alpha1 = self._alpha0 + self._B / 2
            beta1 = self._beta0 + tf.reduce_sum(tf.square(self._code_y - data)) / (2 * self._Sz)
            post = tfd.InverseGamma(alpha1, beta1)
            stds = post.sample((self._Sz * self._B, self._T, 1))
            gen_prob = tfd.MultivariateNormalDiag(self._code_y, stds)

        return tf.reduce_sum(gen_prob.log_prob(data)) / self._Sz

    @define_scope
    def compute_elbo(self):
        # Computes the ELBO used for training
        kl_z = self.make_kl_z
        reconstruction = self.make_reconstruction
        loss = reconstruction - kl_z
        return loss, kl_z, reconstruction

    @define_scope
    def optimise(self):
        # Optimses the ELBO function
        vars_modelling = [v for v in tf.trainable_variables() if v.name.startswith('make_posterior')
                          or v.name.startswith('make_transitional') or v.name.startswith('compute_elbo')]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo[0], var_list=vars_modelling))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def make_forward_posterior_y(self):
        # Computes the forward variational distribution q_psi(y|x, y_bar)

        # for q(y_{1:T}|x_{1:T},y_{T+1:T+K}), pass y_{T+1:T+K} through rnn and use
        # final hidden state for input to new rnn taking x_{1:T} as input and outputting
        # means and stds of q(y_{1:T}) - not necessarily smart implementation, seq2seq should
        # be better but need to figure out how to do that
        # _, h_qy = dynrnn(self._post_forward_cell_y, self._most_likely_pred, dtype=tf.float32)
        # (means_qy, stds_qy), _ = dynrnn(self._gauss_forward_cell_y, inputs=self._data, initial_state=h_qy)
        # return tfd.MultivariateNormalDiag(means_qy, stds_qy)
        self._cell_f = tfn.GRUBlockCell(self._H)
        self._cell_b = tfn.GRUBlockCell(self._H)
        (out_f, out_b), _ = bidynrnn(self._cell_f, self._cell_b, inputs=self._data[:1], dtype=tf.float32)
        hidden = tf.concat([out_f, out_b], 2)
        means = tf.layers.dense(hidden, self._D)
        stds = tf.layers.dense(hidden, self._D, tf.nn.softplus) + 1e-5
        return tfd.MultivariateNormalDiag(means, stds)

    @define_scope
    def make_forward_posterior_z(self):
        # Computes the forward variational distribution q_psi(z|x, y_bar)

        # concatenate x_{1:T}, y_{1:T+K} for q(z|...)
        # sample from posterior_qy
        # self._code_y_forw = tf.reshape(self.q_psi_y.sample((self._S_forw)), [-1, self._T, self._D])
        # pred = tf.tile(self._most_likely_pred[None, ...], (self._S_forw, 1, 1, 1))
        # pred = tf.reshape(pred, [-1, 30, self._D])
        # concat_q_z = tf.concat([self._code_y_forw, pred], 1)
        # _, h_qz = dynrnn(self._post_forward_cell_z, concat_q_z, dtype=tf.float32)
        # means_qz = tf.layers.dense(h_qz, self._Z)
        # means_qz = tf.reshape(means_qz, [self._S_forw, self._B, self._Z])
        # stds_qz = tf.layers.dense(h_qz, self._Z, tf.nn.softplus) + 1e-5
        # stds_qz = tf.reshape(stds_qz, [self._S_forw, self._B, self._Z])
        # return tfd.MultivariateNormalDiag(means_qz, stds_qz)
        # concat_q_z = tf.concat([self._data, self._most_likely_pred], 1)
        _, h_qz = dynrnn(self._post_forward_cell_z, self._data[:1], dtype=tf.float32)
        means_qz = tf.layers.dense(h_qz, self._Z)
        stds_qz = tf.layers.dense(h_qz, self._Z, tf.nn.softplus) + 1e-5
        return tfd.MultivariateNormalDiag(means_qz, stds_qz)

    # @define_scope
    # def make_rnn_forward(self):
    #     # Computes p(y|z) and p(y_bar|y, z) under q_psi(z) and q_psi(y) to be used in the KL divergence and
    #     # reconstruction term of the new ELBO
    #     # self._code_z_forw = self.q_psi_z.sample()
    #     # code_z = tf.reshape(self._code_z_forw, [-1, self._Z])
    #     # inputs = tf.concat([tf.zeros((self._S_forw * self._B, 1, self._D)), self._code_y_forw[:, :-1]], 1)
    #     # (means_y, stds_y), hT = dynrnn(self._gauss_cell, inputs, initial_state=self._layer(code_z))
    #     # p_y = tfd.MultivariateNormalDiag(tf.reshape(means_y, [self._S_forw, self._B, self._T, self._D]),
    #     #                                  tf.reshape(stds_y, [self._S_forw, self._B, self._T, self._D]))
    #     #
    #     # forward = tf.reshape(tf.tile(self._most_likely_pred[None, ...], [self._S_forw, 1, 1, 1]), [-1, 30, self._D])
    #     # new_inputs = tf.concat([self._code_y_forw[:, -1:], forward[:, :-1]], 1)
    #     # (means_forw, stds_forw), _ = dynrnn(self._gauss_cell, inputs=new_inputs, initial_state=hT)
    #     # p_y_bar = tfd.MultivariateNormalDiag(means_forw, stds_forw)
    #     #
    #     # return p_y, p_y_bar
    #     self._code_z_forw = self.q_psi_z.sample(self._S_forw)
    #     self._code_y_forw = self.q_psi_y.sample(self._S_forw)
    #
    #     forward = tf.tile(self._most_likely_pred, [self._S_forw, 1, 1])
    #     inputs = tf.concat([tf.zeros((self._S_forw, 1, self._D)), self._code_y_forw[:, 0], forward[:, :-1]], 1)
    #     (means_y, stds_y), _ = dynrnn(self._gauss_cell, inputs, initial_state=self._layer(self._code_z_forw[:, 0]))
    #
    #     p_y = tfd.MultivariateNormalDiag(means_y[:, :self._T], stds_y[:, :self._T])
    #     p_y_bar = tfd.MultivariateNormalDiag(means_y[:, self._T:], stds_y[:, self._T:])
    #
    #     return p_y, p_y_bar

    @define_scope
    def make_rnn_forward(self):
        # Computes p(y|z) and p(y_bar|y, z) under q_psi(z) and q_psi(y) to be used in the KL divergence and
        # reconstruction term of the new ELBO
        # self._code_z_forw = self.q_psi_z.sample()
        # code_z = tf.reshape(self._code_z_forw, [-1, self._Z])
        # inputs = tf.concat([tf.zeros((self._S_forw * self._B, 1, self._D)), self._code_y_forw[:, :-1]], 1)
        # (means_y, stds_y), hT = dynrnn(self._gauss_cell, inputs, initial_state=self._layer(code_z))
        # p_y = tfd.MultivariateNormalDiag(tf.reshape(means_y, [self._S_forw, self._B, self._T, self._D]),
        #                                  tf.reshape(stds_y, [self._S_forw, self._B, self._T, self._D]))
        #
        # forward = tf.reshape(tf.tile(self._most_likely_pred[None, ...], [self._S_forw, 1, 1, 1]), [-1, 30, self._D])
        # new_inputs = tf.concat([self._code_y_forw[:, -1:], forward[:, :-1]], 1)
        # (means_forw, stds_forw), _ = dynrnn(self._gauss_cell, inputs=new_inputs, initial_state=hT)
        # p_y_bar = tfd.MultivariateNormalDiag(means_forw, stds_forw)
        #
        # return p_y, p_y_bar
        self._code_z_forw = self.q_psi_z.sample(self._S_forw)
        self._code_y_forw = self.q_psi_y.sample(self._S_forw)

        inputs = tf.concat([tf.zeros((self._S_forw, 1, self._D)), self._code_y_forw[:, 0, :-1]], 1)
        (means_y, stds_y), hT = dynrnn(self._gauss_cell, inputs, initial_state=self._layer(self._code_z_forw[:, 0]))

        p_y = tfd.MultivariateNormalDiag(means_y, stds_y)

        fake_inputs = tf.zeros((self._S_forw, 30, self._D))
        state_and_sample = (hT, self._code_y_forw[:, 0, -1])
        (means_ybar, stds_ybar, _), _ = dynrnn(self._sampling_cell, inputs=fake_inputs, initial_state=state_and_sample)
        p_y_bar = tfd.MultivariateNormalDiag(means_ybar, stds_ybar)

        return p_y, p_y_bar

    @define_scope
    def make_forward_kl(self):
        # p_z = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        # kl_z = tfd.kl_divergence(self.q_psi_z, p_z)
        #
        # kl_y = tfd.kl_divergence(self.q_psi_y, self.p_y_kl_forward)
        #
        # return tf.reduce_sum(kl_z) / self._S_forw, tf.reduce_sum(kl_y) / self._S_forw
        p_z = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        kl_z = tfd.kl_divergence(self.q_psi_z, p_z)
        print('KL (z): ', kl_z.shape)

        kl_y = tfd.kl_divergence(self.q_psi_y, self.p_y_kl_forward)
        print('KL (y): ', kl_y.shape)

        return tf.reduce_sum(kl_z), tf.reduce_sum(kl_y) / self._S_forw

    @define_scope
    def make_forward_reconstruction(self):
        # expectation of log p(y_t|y_<t,z) for T+1:T+K
        forward = tf.tile(self._most_likely_pred, [self._S_forw, 1, 1])
        print('Reconstruction (y): ', self.p_y_bar.log_prob(forward).shape)
        exp_py = tf.reduce_sum(self.p_y_bar.log_prob(forward)) / self._S_forw

        # expectation of log p(x_t|y_t) for 1:T
        p_x = tfd.MultivariateNormalDiag(self._code_y_forw[:, 0], scale_identity_multiplier=self._gen_std)
        # data = tf.reshape(tf.tile(self._data[None, ...], (self._S_forw, 1, 1, 1)), [-1, self._T, self._D])
        data = tf.tile(self._data[:1], (self._S_forw, 1, 1))
        print('Reconstruction (x): ', p_x.log_prob(data).shape)
        exp_px = tf.reduce_sum(p_x.log_prob(data)) / self._S_forw

        return exp_px, exp_py

    @define_scope
    def compute_elbo_forward(self):
        # KL terms
        (kl_z, kl_y) = self.make_forward_kl
        # expectation of log p(y_t|y_<t,z) for T+1:T+K
        # and expectation of log p(x_t|y_t) for 1:T
        reconstruction_x, forward_reconstruction_y = self.make_forward_reconstruction

        return forward_reconstruction_y + reconstruction_x - kl_z - kl_y, forward_reconstruction_y, reconstruction_x, kl_y, kl_z

    @define_scope
    def forward_E_step(self):
        var_e = [v for v in tf.trainable_variables() if 'make_forward_posterior' in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo_forward[0], var_list=var_e))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def forward_M_step(self):
        var_m = [v for v in tf.trainable_variables() if 'best_forward_sequence' in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo_forward[0], var_list=var_m))
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def forward_optimise(self):
        vars_pred = [v for v in tf.trainable_variables() if 'forward' in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo_forward[0], var_list=vars_pred))
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))


class RobustARGeneratorMeanFieldPosterior(object):
    """
    Model to test
    """

    def __init__(self, data, n_hidden, n_samples, seq_length, batch_size, forw_init=0, alpha=None, beta=None):
        self._data = data                                               # B x T x D
        self._B = batch_size                                            # B - batch size
        self._T = seq_length                                            # T - number of time steps
        self._D = 1                                                     # D - dimensionality of each time step
        self._H = n_hidden                                              # H - dimensionality of RNN hidden states
        self._Sz = n_samples                                            # Sz - number of samples from the posterior
        self._S_forw = 10                                               # S - number of samples from the forward posterior
        self._Z = 20                                                    # Z - dimensionality of the global latent z
        self._alpha0 = alpha                                            # parameter of inverse gamma prior (optional)
        self._beta0 = beta                                              # parameter of inverse gamma prior (optional)
        self._forw_init = forw_init

        self.initialise_variables                                       #

        self.q_z = self.make_posterior_z                                # posterior q(z|x)
        self.q_y = self.make_posterior_y                                # posterior q(y|x)
        self.p_y = self.make_transitional                               # prior p(y|z) under q(z) and q(y)
        self.q_psi_y = self.make_forward_posterior_y                    # forward posterior q_psi(y|y_bar)
        self.q_psi_z = self.make_forward_posterior_z                    # forward posterior q_psi(z|y_bar)
        self.p_y_kl_forward, self.p_y_bar = self.make_rnn_forward       # prior p(y|z) and forward "prior" p(y_bar|y, z)
                                                                        # under q_psi(z|y_bar) and q_psi(y|y_bar)
        self.compute_elbo
        self.optimise

        # self.compute_elbo_forward
        # self.forward_E_step
        # self.forward_M_step

    @define_scope
    def initialise_variables(self):
        # Initialises all variables and RNN cells
        self._cell = tfn.GRUBlockCell(self._H)
        self._gauss_cell = GaussianCell(self._cell, self._D)
        self._sampling_cell = SamplingCell(self._gauss_cell, self._D)
        self._layer = tf.layers.Dense(self._H, tf.nn.relu)

        self._post_cell_z = tfn.GRUBlockCell(self._H)
        self._post_cell_f = tfn.GRUBlockCell(self._H)
        self._post_cell_b = tfn.GRUBlockCell(self._H)

        self._most_likely_pred = tf.get_variable(shape=(1, 30, self._D), name='best_forward_sequence',
                                                 initializer=tf.constant_initializer(self._forw_init))

        self._post_forward_cell_y = tfn.GRUBlockCell(self._H)
        self._gauss_forward_cell_y = GaussianCell(self._post_forward_cell_y, self._D)
        self._post_forward_cell_z = tfn.GRUBlockCell(self._H)

        self._optimiser = tf.train.AdamOptimizer(0.001)

    @define_scope
    def make_posterior_z(self):
        # Computes the posterior distribution q(z|x)
        _, h = dynrnn(self._post_cell_z, self._data, dtype=tf.float32)
        means = tf.layers.dense(h, self._Z)
        stds = tf.layers.dense(h, self._Z, tf.nn.softplus) + 1e-5
        posterior_z = tfd.MultivariateNormalDiag(means, stds)
        return posterior_z

    @define_scope
    def make_kl_z(self):
        # Computes the KL divergence from p(z) to q(z|x)
        p = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        return tf.reduce_sum(tfd.kl_divergence(self.q_z, p))

    @define_scope
    def make_posterior_y(self):
        (out_f, out_b), _ = bidynrnn(self._post_cell_f, self._post_cell_b, inputs=self._data, dtype=tf.float32)
        hidden = tf.concat([out_f, out_b], 2)
        means = tf.layers.dense(hidden, self._D)
        stds = tf.layers.dense(hidden, self._D, tf.nn.softplus) + 1e-5
        return tfd.MultivariateNormalDiag(means, stds)

    @define_scope
    def make_transitional(self):
        # Computes the prior p(y|z) = q(y|z) under q(z|x) by passing z through dense layer, then
        # using this as initial state, outputs the distn of p(y_t|y_{<t},z) at each t
        self._code_z = self.q_z.sample(self._Sz)
        self._code_y = self.q_y.sample(self._Sz)
        code_y = tf.reshape(self._code_y, [-1, self._T, self._D])

        hidden = self._layer(tf.reshape(self._code_z, [-1, self._Z]))                                   # (Sz * B) x H
        inputs = tf.concat([tf.zeros((self._B * self._Sz, 1, self._D)), code_y[:, :-1]], 1)

        (means, stds), _ = dynrnn(self._gauss_cell, initial_state=hidden, inputs=inputs)
        means = tf.identity(tf.reshape(means, [self._Sz, self._B, self._T, self._D]), name='means')
        stds = tf.identity(tf.reshape(stds, [self._Sz, self._B, self._T, self._D]), name='stds')
        return tfd.MultivariateNormalDiag(means, stds)

    @define_scope
    def make_kl_y(self):
        return tf.reduce_sum(tfd.kl_divergence(self.q_y, self.p_y)) / self._Sz

    @define_scope
    def make_reconstruction(self):
        # Computes the expected reconstruction log p(x|y) under q(y,z|x)
        data = tf.tile(self._data[None, ...], (self._Sz, 1, 1, 1))
        if not (self._alpha0 is not None and self._beta0 is not None):
            self._gen_std = tf.nn.softplus(tf.get_variable(shape=(), name='generative_std',
                                                           initializer=tf.constant_initializer(1))) + 1e-5
            gen_prob = tfd.MultivariateNormalDiag(self._code_y, scale_identity_multiplier=self._gen_std)

        else:
            alpha1 = self._alpha0 + self._B / 2
            beta1 = self._beta0 + tf.reduce_sum(tf.square(self._code_y - data)) / (2 * self._Sz)
            post = tfd.InverseGamma(alpha1, beta1)
            stds = post.sample((self._Sz, self._B, self._T, 1))
            gen_prob = tfd.MultivariateNormalDiag(self._code_y, stds)

        return tf.reduce_sum(gen_prob.log_prob(data)) / self._Sz

    @define_scope
    def compute_elbo(self):
        # Computes the ELBO used for training
        kl_z = self.make_kl_z
        kl_y = self.make_kl_y
        reconstruction = self.make_reconstruction
        loss = reconstruction - kl_z
        return loss, kl_z, kl_y, reconstruction

    @define_scope
    def optimise(self):
        # Optimses the ELBO function
        vars_modelling = [v for v in tf.trainable_variables() if v.name.startswith('make_posterior')
                          or v.name.startswith('make_transitional') or v.name.startswith('compute_elbo')]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo[0], var_list=vars_modelling))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def make_forward_posterior_y(self):
        # Computes the forward variational distribution q_psi(y|x, y_bar)

        # for q(y_{1:T}|x_{1:T},y_{T+1:T+K}), pass y_{T+1:T+K} through rnn and use
        # final hidden state for input to new rnn taking x_{1:T} as input and outputting
        # means and stds of q(y_{1:T}) - not necessarily smart implementation, seq2seq should
        # be better but need to figure out how to do that
        # _, h_qy = dynrnn(self._post_forward_cell_y, self._most_likely_pred, dtype=tf.float32)
        # (means_qy, stds_qy), _ = dynrnn(self._gauss_forward_cell_y, inputs=self._data, initial_state=h_qy)
        # return tfd.MultivariateNormalDiag(means_qy, stds_qy)
        self._cell_f = tfn.GRUBlockCell(self._H)
        self._cell_b = tfn.GRUBlockCell(self._H)
        (out_f, out_b), _ = bidynrnn(self._cell_f, self._cell_b, inputs=self._data[:1], dtype=tf.float32)
        hidden = tf.concat([out_f, out_b], 2)
        means = tf.layers.dense(hidden, self._D)
        stds = tf.layers.dense(hidden, self._D, tf.nn.softplus) + 1e-5
        return tfd.MultivariateNormalDiag(means, stds)

    @define_scope
    def make_forward_posterior_z(self):
        # Computes the forward variational distribution q_psi(z|x, y_bar)

        # concatenate x_{1:T}, y_{1:T+K} for q(z|...)
        # sample from posterior_qy
        # self._code_y_forw = tf.reshape(self.q_psi_y.sample((self._S_forw)), [-1, self._T, self._D])
        # pred = tf.tile(self._most_likely_pred[None, ...], (self._S_forw, 1, 1, 1))
        # pred = tf.reshape(pred, [-1, 30, self._D])
        # concat_q_z = tf.concat([self._code_y_forw, pred], 1)
        # _, h_qz = dynrnn(self._post_forward_cell_z, concat_q_z, dtype=tf.float32)
        # means_qz = tf.layers.dense(h_qz, self._Z)
        # means_qz = tf.reshape(means_qz, [self._S_forw, self._B, self._Z])
        # stds_qz = tf.layers.dense(h_qz, self._Z, tf.nn.softplus) + 1e-5
        # stds_qz = tf.reshape(stds_qz, [self._S_forw, self._B, self._Z])
        # return tfd.MultivariateNormalDiag(means_qz, stds_qz)
        # concat_q_z = tf.concat([self._data, self._most_likely_pred], 1)
        _, h_qz = dynrnn(self._post_forward_cell_z, self._data[:1], dtype=tf.float32)
        means_qz = tf.layers.dense(h_qz, self._Z)
        stds_qz = tf.layers.dense(h_qz, self._Z, tf.nn.softplus) + 1e-5
        return tfd.MultivariateNormalDiag(means_qz, stds_qz)

    # @define_scope
    # def make_rnn_forward(self):
    #     # Computes p(y|z) and p(y_bar|y, z) under q_psi(z) and q_psi(y) to be used in the KL divergence and
    #     # reconstruction term of the new ELBO
    #     # self._code_z_forw = self.q_psi_z.sample()
    #     # code_z = tf.reshape(self._code_z_forw, [-1, self._Z])
    #     # inputs = tf.concat([tf.zeros((self._S_forw * self._B, 1, self._D)), self._code_y_forw[:, :-1]], 1)
    #     # (means_y, stds_y), hT = dynrnn(self._gauss_cell, inputs, initial_state=self._layer(code_z))
    #     # p_y = tfd.MultivariateNormalDiag(tf.reshape(means_y, [self._S_forw, self._B, self._T, self._D]),
    #     #                                  tf.reshape(stds_y, [self._S_forw, self._B, self._T, self._D]))
    #     #
    #     # forward = tf.reshape(tf.tile(self._most_likely_pred[None, ...], [self._S_forw, 1, 1, 1]), [-1, 30, self._D])
    #     # new_inputs = tf.concat([self._code_y_forw[:, -1:], forward[:, :-1]], 1)
    #     # (means_forw, stds_forw), _ = dynrnn(self._gauss_cell, inputs=new_inputs, initial_state=hT)
    #     # p_y_bar = tfd.MultivariateNormalDiag(means_forw, stds_forw)
    #     #
    #     # return p_y, p_y_bar
    #     self._code_z_forw = self.q_psi_z.sample(self._S_forw)
    #     self._code_y_forw = self.q_psi_y.sample(self._S_forw)
    #
    #     forward = tf.tile(self._most_likely_pred, [self._S_forw, 1, 1])
    #     inputs = tf.concat([tf.zeros((self._S_forw, 1, self._D)), self._code_y_forw[:, 0], forward[:, :-1]], 1)
    #     (means_y, stds_y), _ = dynrnn(self._gauss_cell, inputs, initial_state=self._layer(self._code_z_forw[:, 0]))
    #
    #     p_y = tfd.MultivariateNormalDiag(means_y[:, :self._T], stds_y[:, :self._T])
    #     p_y_bar = tfd.MultivariateNormalDiag(means_y[:, self._T:], stds_y[:, self._T:])
    #
    #     return p_y, p_y_bar

    @define_scope
    def make_rnn_forward(self):
        # Computes p(y|z) and p(y_bar|y, z) under q_psi(z) and q_psi(y) to be used in the KL divergence and
        # reconstruction term of the new ELBO
        # self._code_z_forw = self.q_psi_z.sample()
        # code_z = tf.reshape(self._code_z_forw, [-1, self._Z])
        # inputs = tf.concat([tf.zeros((self._S_forw * self._B, 1, self._D)), self._code_y_forw[:, :-1]], 1)
        # (means_y, stds_y), hT = dynrnn(self._gauss_cell, inputs, initial_state=self._layer(code_z))
        # p_y = tfd.MultivariateNormalDiag(tf.reshape(means_y, [self._S_forw, self._B, self._T, self._D]),
        #                                  tf.reshape(stds_y, [self._S_forw, self._B, self._T, self._D]))
        #
        # forward = tf.reshape(tf.tile(self._most_likely_pred[None, ...], [self._S_forw, 1, 1, 1]), [-1, 30, self._D])
        # new_inputs = tf.concat([self._code_y_forw[:, -1:], forward[:, :-1]], 1)
        # (means_forw, stds_forw), _ = dynrnn(self._gauss_cell, inputs=new_inputs, initial_state=hT)
        # p_y_bar = tfd.MultivariateNormalDiag(means_forw, stds_forw)
        #
        # return p_y, p_y_bar
        self._code_z_forw = self.q_psi_z.sample(self._S_forw)
        self._code_y_forw = self.q_psi_y.sample(self._S_forw)

        inputs = tf.concat([tf.zeros((self._S_forw, 1, self._D)), self._code_y_forw[:, 0, :-1]], 1)
        (means_y, stds_y), hT = dynrnn(self._gauss_cell, inputs, initial_state=self._layer(self._code_z_forw[:, 0]))

        p_y = tfd.MultivariateNormalDiag(means_y, stds_y)

        fake_inputs = tf.zeros((self._S_forw, 30, self._D))
        state_and_sample = (hT, self._code_y_forw[:, 0, -1])
        (means_ybar, stds_ybar, _), _ = dynrnn(self._sampling_cell, inputs=fake_inputs, initial_state=state_and_sample)
        p_y_bar = tfd.MultivariateNormalDiag(means_ybar, stds_ybar)

        return p_y, p_y_bar

    @define_scope
    def make_forward_kl(self):
        # p_z = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        # kl_z = tfd.kl_divergence(self.q_psi_z, p_z)
        #
        # kl_y = tfd.kl_divergence(self.q_psi_y, self.p_y_kl_forward)
        #
        # return tf.reduce_sum(kl_z) / self._S_forw, tf.reduce_sum(kl_y) / self._S_forw
        p_z = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        kl_z = tfd.kl_divergence(self.q_psi_z, p_z)
        print('KL (z): ', kl_z.shape)

        kl_y = tfd.kl_divergence(self.q_psi_y, self.p_y_kl_forward)
        print('KL (y): ', kl_y.shape)

        return tf.reduce_sum(kl_z), tf.reduce_sum(kl_y) / self._S_forw

    @define_scope
    def make_forward_reconstruction(self):
        # expectation of log p(y_t|y_<t,z) for T+1:T+K
        forward = tf.tile(self._most_likely_pred, [self._S_forw, 1, 1])
        print('Reconstruction (y): ', self.p_y_bar.log_prob(forward).shape)
        exp_py = tf.reduce_sum(self.p_y_bar.log_prob(forward)) / self._S_forw

        # expectation of log p(x_t|y_t) for 1:T
        if not (self._alpha0 is not None and self._beta0 is not None):
            p_x = tfd.MultivariateNormalDiag(self._code_y_forw[:, 0], scale_identity_multiplier=self._gen_std)
            # data = tf.reshape(tf.tile(self._data[None, ...], (self._S_forw, 1, 1, 1)), [-1, self._T, self._D])
            data = tf.tile(self._data[:1], (self._S_forw, 1, 1))
        else:
            pass
        print('Reconstruction (x): ', p_x.log_prob(data).shape)
        exp_px = tf.reduce_sum(p_x.log_prob(data)) / self._S_forw

        return exp_px, exp_py

    @define_scope
    def compute_elbo_forward(self):
        # KL terms
        (kl_z, kl_y) = self.make_forward_kl
        # expectation of log p(y_t|y_<t,z) for T+1:T+K
        # and expectation of log p(x_t|y_t) for 1:T
        reconstruction_x, forward_reconstruction_y = self.make_forward_reconstruction

        return forward_reconstruction_y + reconstruction_x - kl_z - kl_y, forward_reconstruction_y, reconstruction_x, kl_y, kl_z

    @define_scope
    def forward_E_step(self):
        var_e = [v for v in tf.trainable_variables() if 'make_forward_posterior' in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo_forward[0], var_list=var_e))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def forward_M_step(self):
        var_m = [v for v in tf.trainable_variables() if 'best_forward_sequence' in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo_forward[0], var_list=var_m))
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def forward_optimise(self):
        vars_pred = [v for v in tf.trainable_variables() if 'forward' in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo_forward[0], var_list=vars_pred))
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))


class RobustARGeneratorMeanFieldPosteriorRegularizedW(RobustARGeneratorMeanFieldPosterior):


    @define_scope
    def initialise_variables(self):
        # Initialises all variables and RNN cells
        self._cell = tfn.GRUBlockCell(self._H)
        self._gauss_cell = GaussianCell(self._cell, self._D, reg_lambda=1.)
        self._sampling_cell = SamplingCell(self._gauss_cell, self._D)
        self._layer = tf.layers.Dense(self._H, tf.nn.relu)

        self._post_cell_z = tfn.GRUBlockCell(self._H)
        self._post_cell_f = tfn.GRUBlockCell(self._H)
        self._post_cell_b = tfn.GRUBlockCell(self._H)

        self._most_likely_pred = tf.get_variable(shape=(1, 30, self._D), name='best_forward_sequence',
                                                 initializer=tf.constant_initializer(self._forw_init))

        self._post_forward_cell_y = tfn.GRUBlockCell(self._H)
        self._gauss_forward_cell_y = GaussianCell(self._post_forward_cell_y, self._D)
        self._post_forward_cell_z = tfn.GRUBlockCell(self._H)

        self._optimiser = tf.train.AdamOptimizer(0.001)

    @define_scope
    def make_transitional(self):
        # Computes the prior p(y|z) = q(y|z) under q(z|x) by passing z through dense layer, then
        # using this as initial state, outputs the distn of p(y_t|y_{<t},z) at each t
        self._code_z = self.q_z.sample(self._Sz)
        self._code_y = self.q_y.sample(self._Sz)
        code_y = tf.reshape(self._code_y, [-1, self._T, self._D])

        hidden = self._layer(tf.reshape(self._code_z, [-1, self._Z]))                                   # (Sz * B) x H
        inputs = tf.concat([tf.zeros((self._B * self._Sz, 1, self._D)), code_y[:, :-1]], 1)

        (_, _), _ = self._gauss_cell(tf.zeros((self._B * self._Sz, self._D)), tf.zeros((self._B * self._Sz, self._H)))
        (means, stds), _ = dynrnn(self._gauss_cell, initial_state=hidden, inputs=inputs)
        means = tf.identity(tf.reshape(means, [self._Sz, self._B, self._T, self._D]), name='means')
        stds = tf.identity(tf.reshape(stds, [self._Sz, self._B, self._T, self._D]), name='stds')
        return tfd.MultivariateNormalDiag(means, stds)

    @define_scope
    def compute_elbo(self):
        kl_y = self.make_kl_y
        reconstruction = self.make_reconstruction

        l2_loss = tf.losses.get_regularization_loss()
        loss = reconstruction - kl_y - l2_loss
        return loss, kl_y, reconstruction


class RobustVideoARGenerator(object):
    """
    Model to test
    """

    def __init__(self, data, n_hidden, n_samples, seq_length, batch_size, alpha=None, beta=None):
        self._data = data                                               # B x T x D
        self._B = batch_size                                            # B - batch size
        self._T = seq_length                                            # T - number of time steps
        self._D = 900                                                   # D - dimensionality of each time step
        self._H = n_hidden                                              # H - dimensionality of RNN hidden states
        self._Sz = n_samples                                            # Sz - number of samples from the posterior
        self._Z = 50                                                    # Z - dimensionality of the global latent z

        self.initialise_variables                                       #

        self.q_z = self.make_posterior_z                                # posterior q(z|x)
        self.p_y = self.make_transitional                               # prior p(y|z) under q(z)
        self.q_psi_y = self.make_forward_posterior_y                    # forward posterior q_psi(y|y_bar)
        self.q_psi_z = self.make_forward_posterior_z                    # forward posterior q_psi(z|y_bar)
        self.p_y_kl_forward, self.p_y_bar = self.make_rnn_forward       # prior p(y|z) and forward "prior" p(y_bar|y, z)
                                                                        # under q_psi(z|y_bar) and q_psi(y|y_bar)
        self.compute_elbo
        self.optimise

        self.compute_elbo_forward
        self.forward_E_step
        self.forward_M_step

    @define_scope
    def initialise_variables(self):
        # Initialises all variables and RNN cells
        self._cell = tfn.GRUBlockCell(self._H)
        self._gauss_cell = GaussianConvolutionalCell(self._cell, self._D)
        self._sampling_cell = SamplingCell(self._gauss_cell, self._D)
        self._layer = tf.layers.Dense(self._H, tf.nn.relu)

        self._post_cell_z = tfn.GRUBlockCell(self._H)
        self._gauss_post_cell_z = GaussianConvolutionalCell(self._post_cell_z, self._D)

        self._most_likely_pred = tf.get_variable(shape=(self._B, 30, self._D), name='best_forward_sequence',
                                                 initializer=tf.constant_initializer(0))

        self._post_forward_cell_y = tfn.GRUBlockCell(self._H)
        self._gauss_forward_cell_y = GaussianConvolutionalCell(self._post_forward_cell_y, self._D)
        self._post_forward_cell_z = tfn.GRUBlockCell(self._H)
        self._gauss_forward_cell_z = GaussianConvolutionalCell(self._post_forward_cell_z, self._D)

        self._optimiser = tf.train.AdamOptimizer(0.001)

    @define_scope
    def make_posterior_z(self):
        # Computes the posterior distribution q(z|x)
        _, h = dynrnn(self._gauss_post_cell_z, self._data, dtype=tf.float32)
        means = tf.layers.dense(h, self._Z)
        stds = tf.layers.dense(h, self._Z, tf.nn.softplus) + 1e-5
        posterior_z = tfd.MultivariateNormalDiag(means, stds)
        return posterior_z

    @define_scope
    def make_kl_z(self):
        # Computes the KL divergence from p(z) to q(z|x)
        p = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        return tf.reduce_sum(tfd.kl_divergence(self.q_z, p))

    @define_scope
    def make_transitional(self):
        # Computes the prior p(y|z) = q(y|z) under q(z|x) by passing z through dense layer, then
        # using this as initial state, outputs the distn of p(y_t|y_{<t},z) at each t
        self._code_z = self.q_z.sample((self._Sz))
        hidden = self._layer(tf.reshape(self._code_z, [-1, self._Z]))                                   # (Sz * B) x H
        state_and_sample = (hidden, tf.zeros((self._B * self._Sz, self._D)))
        inputs = tf.zeros((self._B * self._Sz, self._T, self._D))
        (means, stds, self._code_y), _ = dynrnn(self._sampling_cell, initial_state=state_and_sample, inputs=inputs)
        means = tf.identity(means, name='means')
        stds = tf.identity(stds, name='stds')
        return tfd.MultivariateNormalDiag(means, stds)

    @define_scope
    def make_reconstruction(self):
        gen_prob = tfd.Independent(tfd.Bernoulli(tf.sigmoid(self._code_y)), 2)
        data = tf.reshape(tf.tile(self._data[None, ...], (self._Sz, 1, 1, 1)), [-1, self._T, self._D])
        return tf.reduce_sum(gen_prob.log_prob(data)) / self._Sz

    @define_scope
    def compute_elbo(self):
        # Computes the ELBO used for training
        kl_z = self.make_kl_z
        reconstruction = self.make_reconstruction
        loss = reconstruction - kl_z
        return loss, kl_z, reconstruction

    @define_scope
    def optimise(self):
        # Optimses the ELBO function
        vars_modelling = [v for v in tf.trainable_variables() if v.name.startswith('make_posterior')
                          or v.name.startswith('make_transitional') or v.name.startswith('compute_elbo')]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo[0], var_list=vars_modelling))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def make_forward_posterior_y(self):
        # Computes the forward variational distribution q_psi(y|x, y_bar)

        # for q(y_{1:T}|x_{1:T},y_{T+1:T+K}), pass y_{T+1:T+K} through rnn and use
        # final hidden state for input to new rnn taking x_{1:T} as input and outputting
        # means and stds of q(y_{1:T}) - not necessarily smart implementation, seq2seq should
        # be better but need to figure out how to do that
        _, h_qy = dynrnn(self._gauss_forward_cell_y, self._most_likely_pred, dtype=tf.float32)
        (means_qy, stds_qy), _ = dynrnn(self._gauss_forward_cell_y, inputs=self._data, initial_state=h_qy)
        return tfd.MultivariateNormalDiag(means_qy, stds_qy)

    @define_scope
    def make_forward_posterior_z(self):
        # Computes the forward variational distribution q_psi(z|x, y_bar)

        # concatenate x_{1:T}, y_{1:T+K} for q(z|...)
        # sample from posterior_qy
        code_qy = self.q_psi_y.sample()
        concat_q_z = tf.concat([code_qy, self._most_likely_pred], 1)
        _, h_qz = dynrnn(self._gauss_forward_cell_z, concat_q_z, dtype=tf.float32)
        means_qz = tf.layers.dense(h_qz, self._Z)
        stds_qz = tf.layers.dense(h_qz, self._Z, tf.nn.softplus) + 1e-5
        return tfd.MultivariateNormalDiag(means_qz, stds_qz)

    @define_scope
    def make_rnn_forward(self):
        # Computes p(y|z) and p(y_bar|y, z) under q_psi(z) and q_psi(y) to be used in the KL divergence and
        # reconstruction term of the new ELBO
        self._code_z_forw = tf.reshape(self.q_psi_z.sample((self._Sz)), [-1, self._Z])
        self._code_y_forw = tf.reshape(self.q_psi_y.sample((self._Sz)), [-1, self._T, self._D])

        inputs = tf.concat([tf.zeros((self._Sz * self._B, 1, self._D)), self._code_y_forw[:, :-1]], 1)
        (means_y, stds_y), hT = dynrnn(self._gauss_cell, inputs, initial_state=self._layer(self._code_z_forw))
        p_y = tfd.MultivariateNormalDiag(tf.reshape(means_y, [self._Sz, self._B, self._T, self._D]),
                                         tf.reshape(stds_y, [self._Sz, self._B, self._T, self._D]))

        forward = tf.reshape(tf.tile(self._most_likely_pred[None, ...], [self._Sz, 1, 1, 1]), [-1, 30, self._D])
        new_inputs = tf.concat([self._code_y_forw[:, -1:], forward[:, :-1]], 1)
        (means_forw, stds_forw), _ = dynrnn(self._gauss_cell, inputs=new_inputs, initial_state=hT)
        p_y_bar = tfd.MultivariateNormalDiag(means_forw, stds_forw)

        return p_y, p_y_bar

    @define_scope
    def make_forward_kl(self):
        p_z = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        kl_z = tfd.kl_divergence(self.q_psi_z, p_z)

        kl_y = tfd.kl_divergence(self.q_psi_y, self.p_y_kl_forward)

        return tf.reduce_sum(kl_z), tf.reduce_sum(kl_y) / self._Sz

    @define_scope
    def make_forward_reconstruction(self):
        # expectation of log p(y_t|y_<t,z) for T+1:T+K
        forward = tf.reshape(tf.tile(self._most_likely_pred[None, ...], [self._Sz, 1, 1, 1]), [-1, 30, self._D])
        exp_py = tf.reduce_sum(self.p_y_bar.log_prob(forward)) / self._Sz

        # expectation of log p(x_t|y_t) for 1:T
        p_x = tfd.Independent(tfd.Bernoulli(tf.sigmoid(self._code_y_forw)), 2)
        data = tf.reshape(tf.tile(self._data[None, ...], (self._Sz, 1, 1, 1)), [-1, self._T, self._D])
        exp_px = tf.reduce_sum(p_x.log_prob(data)) / self._Sz

        return exp_px, exp_py

    @define_scope
    def compute_elbo_forward(self):
        # KL terms
        (kl_z, kl_y) = self.make_forward_kl
        # expectation of log p(y_t|y_<t,z) for T+1:T+K
        # and expectation of log p(x_t|y_t) for 1:T
        reconstruction_x, forward_reconstruction_y = self.make_forward_reconstruction

        return forward_reconstruction_y + reconstruction_x - kl_z - kl_y

    @define_scope
    def forward_E_step(self):
        var_e = [v for v in tf.trainable_variables() if 'make_forward_posterior' in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo_forward, var_list=var_e))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def forward_M_step(self):
        var_m = [v for v in tf.trainable_variables() if 'best_forward_sequence' in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo_forward, var_list=var_m))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))


class RobustVideoARGeneratorMeanFieldPosterior(object):
    """
    Model to test
    """

    def __init__(self, data, n_hidden, n_samples, seq_length, batch_size, alpha=None, beta=None):
        self._data = data                                               # B x T x D
        self._B = batch_size                                            # B - batch size
        self._T = seq_length                                            # T - number of time steps
        self._D = 900                                                   # D - dimensionality of each time step
        self._H = n_hidden                                              # H - dimensionality of RNN hidden states
        self._Sz = n_samples                                            # Sz - number of samples from the posterior
        self._Z = 50                                                    # Z - dimensionality of the global latent z

        self.initialise_variables                                       #

        self.q_z = self.make_posterior_z                                # posterior q(z|x)
        self.q_y = self.make_posterior_y                                # posterior q(y|x)
        self.p_y = self.make_transitional                               # prior p(y|z) under q(z) and q(y)
        self.q_psi_y = self.make_forward_posterior_y                    # forward posterior q_psi(y|y_bar)
        self.q_psi_z = self.make_forward_posterior_z                    # forward posterior q_psi(z|y_bar)
        self.p_y_kl_forward, self.p_y_bar = self.make_rnn_forward       # prior p(y|z) and forward "prior" p(y_bar|y, z)
                                                                        # under q_psi(z|y_bar) and q_psi(y|y_bar)
        self.compute_elbo
        self.optimise

        self.compute_elbo_forward
        self.forward_E_step
        self.forward_M_step

    @define_scope
    def initialise_variables(self):
        # Initialises all variables and RNN cells
        self._cell = tfn.GRUBlockCell(self._H)
        self._gauss_cell = GaussianConvolutionalCell(self._cell, self._D)
        self._sampling_cell = SamplingCell(self._gauss_cell, self._D)

        self._layer = tf.layers.Dense(self._H, tf.nn.relu)

        self._post_cell_f = tfn.GRUBlockCell(self._H)
        self._gauss_post_cell_f = SimpleGaussianConvolutionalCell(self._post_cell_f, self._D)
        self._post_cell_b = tfn.GRUBlockCell(self._H)
        self._gauss_post_cell_b = SimpleGaussianConvolutionalCell(self._post_cell_b, self._D)

        self._post_cell_z = tfn.GRUBlockCell(self._H)
        self._gauss_post_cell_z = SimpleGaussianConvolutionalCell(self._post_cell_z, self._D)

        self._most_likely_pred = tf.get_variable(shape=(self._B, 30, self._D), name='best_forward_sequence',
                                                 initializer=tf.constant_initializer(0))

        self._post_forward_cell_y = tfn.GRUBlockCell(self._H)
        self._gauss_forward_cell_y = GaussianConvolutionalCell(self._post_forward_cell_y, self._D)
        self._post_forward_cell_z = tfn.GRUBlockCell(self._H)
        self._gauss_forward_cell_z = SimpleGaussianConvolutionalCell(self._post_forward_cell_z, self._D)

        self._optimiser = tf.train.AdamOptimizer(0.001)

    @define_scope
    def make_posterior_z(self):
        # Computes the posterior distribution q(z|x)
        _, h = dynrnn(self._gauss_post_cell_z, self._data, dtype=tf.float32)
        means = tf.layers.dense(h, self._Z)
        stds = tf.layers.dense(h, self._Z, tf.nn.softplus) + 1e-5
        posterior_z = tfd.MultivariateNormalDiag(means, stds)
        return posterior_z

    @define_scope
    def make_kl_z(self):
        # Computes the KL divergence from p(z) to q(z|x)
        p = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        return tf.reduce_sum(tfd.kl_divergence(self.q_z, p))

    @define_scope
    def make_posterior_y(self):
        (out_f, out_b), _ = tf.nn.bidirectional_dynamic_rnn(self._gauss_post_cell_f, self._gauss_post_cell_b,
                                                            inputs=self._data, dtype=tf.float32)

        hidden = tf.concat([out_f, out_b], 2)
        means = tf.layers.dense(hidden, self._D)
        stds = tf.layers.dense(hidden, self._D, tf.nn.softplus) + 1e-5
        return tfd.MultivariateNormalDiag(means, stds)

    @define_scope
    def make_transitional(self):
        # Computes the prior p(y|z) = q(y|z) under q(z|x) by passing z through dense layer, then
        # using this as initial state, outputs the distn of p(y_t|y_{<t},z) at each t
        self._code_z = self.q_z.sample((self._Sz))
        self._code_y = self.q_y.sample((self._Sz))
        self._code_y = tf.reshape(self._code_y, [-1, self._T, self._D])
        hidden = self._layer(tf.reshape(self._code_z, [-1, self._Z]))                                   # (Sz * B) x H
        inputs = tf.concat([tf.zeros((self._Sz * self._B, 1, self._D)), self._code_y[:, :-1]], 1)
        (means, stds), _ = dynrnn(self._gauss_cell, initial_state=hidden, inputs=inputs)
        means = tf.identity(tf.reshape(means, [self._Sz, self._B, self._T, self._D]), name='means')
        stds = tf.identity(tf.reshape(stds, [self._Sz, self._B, self._T, self._D]), name='stds')
        return tfd.MultivariateNormalDiag(means, stds)

    @define_scope
    def make_kl_y(self):
        return tf.reduce_sum(tfd.kl_divergence(self.q_y, self.p_y)) / self._Sz

    @define_scope
    def make_reconstruction(self):
        gen_prob = tfd.Independent(tfd.Bernoulli(tf.sigmoid(self._code_y)), 2)
        data = tf.reshape(tf.tile(self._data[None, ...], (self._Sz, 1, 1, 1)), [-1, self._T, self._D])
        return tf.reduce_sum(gen_prob.log_prob(data)) / self._Sz

    @define_scope
    def compute_elbo(self):
        # Computes the ELBO used for training
        kl_z = self.make_kl_z
        kl_y = self.make_kl_y
        reconstruction = self.make_reconstruction
        loss = reconstruction - kl_z - kl_y
        return loss, kl_z + kl_y, reconstruction

    @define_scope
    def optimise(self):
        # Optimses the ELBO function
        vars_modelling = [v for v in tf.trainable_variables() if v.name.startswith('make_posterior')
                          or v.name.startswith('make_transitional') or v.name.startswith('compute_elbo')]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo[0], var_list=vars_modelling))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def make_forward_posterior_y(self):
        # Computes the forward variational distribution q_psi(y|x, y_bar)

        # for q(y_{1:T}|x_{1:T},y_{T+1:T+K}), pass y_{T+1:T+K} through rnn and use
        # final hidden state for input to new rnn taking x_{1:T} as input and outputting
        # means and stds of q(y_{1:T}) - not necessarily smart implementation, seq2seq should
        # be better but need to figure out how to do that
        _, h_qy = dynrnn(self._gauss_forward_cell_y, self._most_likely_pred, dtype=tf.float32)
        (means_qy, stds_qy), _ = dynrnn(self._gauss_forward_cell_y, inputs=self._data, initial_state=h_qy)
        return tfd.MultivariateNormalDiag(means_qy, stds_qy)

    @define_scope
    def make_forward_posterior_z(self):
        # Computes the forward variational distribution q_psi(z|x, y_bar)

        # concatenate x_{1:T}, y_{1:T+K} for q(z|...)
        # sample from posterior_qy
        code_qy = self.q_psi_y.sample()
        concat_q_z = tf.concat([code_qy, self._most_likely_pred], 1)
        _, h_qz = dynrnn(self._gauss_forward_cell_z, concat_q_z, dtype=tf.float32)
        means_qz = tf.layers.dense(h_qz, self._Z)
        stds_qz = tf.layers.dense(h_qz, self._Z, tf.nn.softplus) + 1e-5
        return tfd.MultivariateNormalDiag(means_qz, stds_qz)

    @define_scope
    def make_rnn_forward(self):
        # Computes p(y|z) and p(y_bar|y, z) under q_psi(z) and q_psi(y) to be used in the KL divergence and
        # reconstruction term of the new ELBO
        self._code_z_forw = tf.reshape(self.q_psi_z.sample((self._Sz)), [-1, self._Z])
        self._code_y_forw = tf.reshape(self.q_psi_y.sample((self._Sz)), [-1, self._T, self._D])

        inputs = tf.concat([tf.zeros((self._Sz * self._B, 1, self._D)), self._code_y_forw[:, :-1]], 1)
        (means_y, stds_y), hT = dynrnn(self._gauss_cell, inputs, initial_state=self._layer(self._code_z_forw))
        p_y = tfd.MultivariateNormalDiag(tf.reshape(means_y, [self._Sz, self._B, self._T, self._D]),
                                         tf.reshape(stds_y, [self._Sz, self._B, self._T, self._D]))

        forward = tf.reshape(tf.tile(self._most_likely_pred[None, ...], [self._Sz, 1, 1, 1]), [-1, 30, self._D])
        new_inputs = tf.concat([self._code_y_forw[:, -1:], forward[:, :-1]], 1)
        (means_forw, stds_forw), _ = dynrnn(self._gauss_cell, inputs=new_inputs, initial_state=hT)
        p_y_bar = tfd.MultivariateNormalDiag(means_forw, stds_forw)

        return p_y, p_y_bar

    @define_scope
    def make_forward_kl(self):
        p_z = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        kl_z = tfd.kl_divergence(self.q_psi_z, p_z)

        kl_y = tfd.kl_divergence(self.q_psi_y, self.p_y_kl_forward)

        return tf.reduce_sum(kl_z), tf.reduce_sum(kl_y) / self._Sz

    @define_scope
    def make_forward_reconstruction(self):
        # expectation of log p(y_t|y_<t,z) for T+1:T+K
        forward = tf.reshape(tf.tile(self._most_likely_pred[None, ...], [self._Sz, 1, 1, 1]), [-1, 30, self._D])
        exp_py = tf.reduce_sum(self.p_y_bar.log_prob(forward)) / self._Sz

        # expectation of log p(x_t|y_t) for 1:T
        p_x = tfd.Independent(tfd.Bernoulli(tf.sigmoid(self._code_y_forw)), 2)
        data = tf.reshape(tf.tile(self._data[None, ...], (self._Sz, 1, 1, 1)), [-1, self._T, self._D])
        exp_px = tf.reduce_sum(p_x.log_prob(data)) / self._Sz

        return exp_px, exp_py

    @define_scope
    def compute_elbo_forward(self):
        # KL terms
        (kl_z, kl_y) = self.make_forward_kl
        # expectation of log p(y_t|y_<t,z) for T+1:T+K
        # and expectation of log p(x_t|y_t) for 1:T
        reconstruction_x, forward_reconstruction_y = self.make_forward_reconstruction

        return forward_reconstruction_y + reconstruction_x - kl_z - kl_y, forward_reconstruction_y, reconstruction_x, kl_y, kl_z

    @define_scope
    def forward_E_step(self):
        var_e = [v for v in tf.trainable_variables() if 'make_forward_posterior' in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo_forward[0], var_list=var_e))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def forward_M_step(self):
        var_m = [v for v in tf.trainable_variables() if 'best_forward_sequence' in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo_forward[0], var_list=var_m))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))


class RegularisedRobustARGenerator(object):
    """
    Model to test
    """

    def __init__(self, data, batch_size, seq_length, n_hidden, n_samples, dim_data=1, dim_z=20, future_steps=30, **kwargs):
        """
        :param data:                                                        # B x T x D - training batch
        :param batch_size:                                                  # B - batch size
        :param seq_length:                                                  # T - number of time steps
        :param n_hidden:                                                    # H - dimensionality of RNN hidden states
        :param n_samples:                                                   # Sz - number of samples from the posterior
        :param dim_z:                                                       # Z - dimensionality of the latents
        :param dim_data:                                                    # D - dimensionality of each time step
        :param future_steps:                                                # K - number of future steps for prediction
        :param kwargs:                                                      # Additional parameters
        """

        self._data = data
        self._B = batch_size
        self._T = seq_length
        self._D = dim_data
        self._Z = dim_z
        self._H = n_hidden
        self._Sz = n_samples
        self._K = future_steps
        self._w_lambda = kwargs.get('w_lambda', 0.1)                         # lambda - scale of the regulatisation term
        self._bayesian = kwargs.get('bayesian', True)                        # If we want to use Bayesian inference

        self._fixed_sigma = kwargs.get('fixed_sigma', False)                 # If we want to use the true sigma
        self._true_sigma = kwargs.get('true_sigma', 5.3)                     # The value of the true sigma

        if self._bayesian:
            mu_w = kwargs.get('mu_w', tf.zeros(self._H))                     # Prior mean for p(w)
            sigma2_w = tf.convert_to_tensor(kwargs.get('sigma2_w', 1.))      # Prior variance for p(w)
            self.p_w = tfd.MultivariateNormalDiag(mu_w, scale_identity_multiplier=tf.sqrt(sigma2_w))  # Prior term for p(w)

            self._post_mu_w = kwargs.get('mu_w', tf.zeros(self._H))           # Initial posterior mean for q(w|x)
            post_sigma2_w = tf.convert_to_tensor(kwargs.get('sigma2_w', 1.))  # Initial posterior variance scale for q(w|x)
            self._post_sigma_w = post_sigma2_w * tf.eye(self._H)              # initial posterior variance for q(w|x)

            alpha = tf.convert_to_tensor(kwargs.get('alpha', 1.5))            # Prior concentration for p(sigma2)
            beta = tf.convert_to_tensor(kwargs.get('beta', 60.))              # Prior rate for p(sigma2)
            self.p_sigma = tfd.InverseGamma(alpha, beta)

            self._post_alpha = tf.convert_to_tensor(kwargs.get('alpha', 1.5))   # Initial posterior concentration for q(sigma2|x)
            self._post_beta = tf.convert_to_tensor(kwargs.get('beta', 60.))     # Initial posterior rate for q(sigma2|x)

        self.initialise_variables                                           # Initialise all the variables

        self.q_z = self.make_posterior_z                                    # Posterior q(z|x)
        self.q_y = self.make_posterior_y                                    # Posterior q(y|x)
        self.p_y = self.make_transitional                                   # Prior p(y|z) under q(z) and q(y)
        self.p_x = self.make_generative                                     # Generative p(x|y)
        self.compute_elbo                                                   # Put all ELBO terms together

        self.E_step                                                         # Optimise ELBO wrt to q(z,y|x)
        self.M_step                                                         # Optimise ELBO wrt parameters

    @define_scope
    def initialise_variables(self):
        """
        Initialise all the variables and cells that are part of the model
        :return: None
        """
        if self._fixed_sigma:
            self._gen_std = tf.nn.softplus(tf.get_variable(shape=(), name='gen_std', trainable=False,
                                           initializer=tf.constant_initializer(self._true_sigma))) + 1e-5

        self._cell = tfn.GRUBlockCell(self._H)                                      # RNN cell for p(y|z)
        if self._bayesian:
            self._mean_bias = tf.get_variable(shape=(), name='means_bias',
                                              initializer=tf.constant_initializer(0.))
            self._gauss_cell = GaussianCellWithoutMeans(self._cell, self._D)        # Gaussian cell
        else:
            self._gauss_cell = GaussianCell(self._cell, self._D, reg_lambda=self._w_lambda)
        self._sampling_cell = SamplingCell(self._gauss_cell, self._D)               # Gaussian sampling cell

        self._post_cell_z = tfn.GRUBlockCell(self._H)                               # RNN cell for q(z|x)
        self._layer_z = tf.layers.Dense(self._H, tf.nn.relu)                        # Dense layer from z to h1

        self._post_cell_f = tfn.GRUBlockCell(self._H)                               # Forward RNN for q(y|x)
        self._post_cell_b = tfn.GRUBlockCell(self._H)                               # Backward RNN for q(y|x)
        self._post_layer_means = tf.layers.Dense(self._D)                           # Dense layer for means of q(y|x)
        self._post_layer_stds = tf.layers.Dense(self._D, tf.nn.softplus)            # Dense layer for stds of q(y|x)

        self._optimiser = tf.train.AdamOptimizer(0.001)                             # Model optimser

    @define_scope
    def make_posterior_z(self):
        """
        :return: Gaussian distribution q(z|x)
        """
        _, h = dynrnn(self._post_cell_z, self._data, dtype=tf.float32)
        means = tf.layers.dense(h, self._Z)
        stds = tf.layers.dense(h, self._Z, tf.nn.softplus) + 1e-5
        posterior_z = tfd.MultivariateNormalDiag(means, stds)
        return posterior_z

    @define_scope
    def make_kl_z(self):
        """
        :return: KL divergance KL[q(z|x) | p(z)]
        """
        p_z = tfd.MultivariateNormalDiag(np.zeros(self._Z, dtype=np.float32), np.ones(self._Z, dtype=np.float32))
        return tf.reduce_sum(tfd.kl_divergence(self.q_z, p_z))

    @define_scope
    def make_posterior_y(self):
        """
        :return: Gaussian distribution q(y|x) with Mean Field approximation
        """
        (out_f, out_b), _ = bidynrnn(self._post_cell_f, self._post_cell_b, inputs=self._data, dtype=tf.float32)
        hidden = tf.concat([out_f, out_b], 2)
        self._post_means = self._post_layer_means(hidden)
        self._post_stds = self._post_layer_stds(hidden) + 1e-5
        return tfd.MultivariateNormalDiag(self._post_means, self._post_stds)

    @define_scope
    def make_transitional(self):
        """
        :return: Gaussian distribution <p(y|z, w)> under samples from q(z|x), q(y|x) and q(w|x)
        """
        self._code_z = self.q_z.sample(self._Sz)
        self._code_y = self.q_y.sample(self._Sz)
        code_y = tf.reshape(self._code_y, [-1, self._T, self._D])
        hidden = self._layer_z(tf.reshape(self._code_z, [-1, self._Z]))
        inputs = tf.concat([tf.zeros((self._B * self._Sz, 1, self._D)), code_y[:, :-1]], 1)
        if self._bayesian:
            (self._trans_stds, self._hiddens), _ = dynrnn(self._gauss_cell, inputs=inputs, initial_state=hidden)
            hiddens = tf.reshape(self._hiddens, [self._Sz, self._B, self._T, self._H])
            self._W = tfd.MultivariateNormalFullCovariance(self._post_mu_w, self._post_sigma_w).sample()
            means = tf.tensordot(hiddens, self._W[:, None], 1) + self._mean_bias
        else:
            # This line is needed to activate the regulariser if we set w_lambda <> 0
            (_, _, _), _ = self._gauss_cell(tf.zeros((self._B * self._Sz, self._D)),
                                            tf.zeros((self._B * self._Sz, self._H)))
            (means, self._trans_stds, self._hiddens), _ = dynrnn(self._gauss_cell, inputs=inputs, initial_state=hidden)
            means = tf.reshape(means, [self._Sz, self._B, self._T, self._D])
        stds = tf.reshape(self._trans_stds, [self._Sz, self._B, self._T, self._D])
        return tfd.MultivariateNormalDiag(means, stds)

    @define_scope
    def make_kl_y(self):
        """
        :return: KL divergence KL[q(y|x) | p(y|z, w)] under q(z|x), q(y|x) and q(w|x)
        """
        return tf.reduce_sum(tfd.kl_divergence(self.q_y, self.p_y)) / self._Sz

    @define_scope
    def make_generative(self):
        """
        :return: Gaussian distribution p(x|y, sigma^2)
        """
        mean = self._code_y
        if self._fixed_sigma:
            return tfd.MultivariateNormalDiag(mean, scale_identity_multiplier=self._gen_std)
        else:
            sigma2 = tfd.InverseGamma(self._post_alpha, self._post_beta).sample(self._Sz)
            sigma2 = sigma2[:, None, None]
            return tfd.MultivariateNormalDiag(mean, scale_identity_multiplier=tf.sqrt(sigma2))

    @define_scope
    def make_reconstruction(self):
        """
        :return: <log p(x|y, sigma^2)> under q(y|x) and q(sigma^2|x)
        """
        data = tf.tile(self._data[None, ...], (self._Sz, 1, 1, 1))
        return tf.reduce_sum(self.p_x.log_prob(data)) / self._Sz

    @define_scope
    def compute_elbo(self):
        """
        :return: The Free Energy
        """
        kl_z = self.make_kl_z
        kl_y = self.make_kl_y
        reconstruction = self.make_reconstruction
        if self._bayesian:
            loss = reconstruction - kl_z - kl_y
        else:
            l2_loss = tf.losses.get_regularization_loss()
            loss = reconstruction - kl_z - kl_y - l2_loss
        return loss, kl_z, kl_y, reconstruction

    @define_scope
    def E_step(self):
        """
        Optimises the Free Energy in q(y|x) and q(z|x)
        """
        vars_e = [v for v in tf.trainable_variables() if 'make_posterior' in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo[0], var_list=vars_e))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))

    @define_scope
    def M_step(self):
        """
        Optimises the Free Energy in the model parameters
        """
        vars_m = [v for v in tf.trainable_variables() if 'make_posterior' not in v.name]
        gradients, variables = zip(*self._optimiser.compute_gradients(-self.compute_elbo[0], var_list=vars_m))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self._optimiser.apply_gradients(zip(gradients, variables))
