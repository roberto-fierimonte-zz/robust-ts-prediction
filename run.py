import tensorflow as tf
import numpy as np
import logging
import time
import os


class RunVAE(object):

    def __init__(self, data, test_data, num_iterations, num_forward_iterations, seq_length, n_samples, batch_size,
                 training_size, pre_trained, generative_model, n_hidden_py, predict_forward, experiment_path, data_loader):

        self.pre_trained = pre_trained
        self.training_data = data[:, :seq_length]
        self.test_data = test_data[:, :seq_length]
        self.num_iterations = num_iterations
        self.num_forward_iterations = num_forward_iterations

        self.seq_length = seq_length
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.training_size = training_size
        self.data_dimensionality = data.shape[-1]

        self.generative_model = generative_model

        self.predict_forward = predict_forward

        self.experiment_path = experiment_path

        self.n_hidden_py = n_hidden_py

    def train(self):
        tf.reset_default_graph()

        x_ = tf.placeholder(tf.float32, [self.batch_size, self.seq_length, self.data_dimensionality])

        gen_model = self.generative_model(data=x_, n_hidden=self.n_hidden_py, n_samples=self.n_samples, seq_length=self.seq_length,
                                          batch_size=self.batch_size)

        sess = tf.Session()
        saver = tf.train.Saver([v for v in tf.global_variables() if 'forward' not in v.name])
        start = time.clock()

        if self.pre_trained:
            saver.restore(sess, "./{}/ckpt/robust_model.ckpt".format(self.experiment_path))
            sess.run(tf.variables_initializer([v for v in tf.global_variables() if 'forward' in v.name]))

        else:
            sess.run(tf.global_variables_initializer())
            elbos = []
            kls_z = []
            kls_y = []
            llks = []
            test_elbos = []

            for i in range(self.num_iterations):

                idx = np.random.randint(0, self.training_size, self.batch_size)
                batch = self.training_data[idx, ...]

                if gen_model.__class__.__name__ == 'RobustARGeneratorMeanFieldPosterior':
                    elbo, kl_z, kl_y, llk = sess.run(gen_model.compute_elbo, {x_: batch})
                    kls_y.append(kl_y)
                else:
                    elbo, kl_z, llk = sess.run(gen_model.compute_elbo, {x_: batch})
                elbos.append(elbo)
                kls_z.append(kl_z)
                llks.append(llk)

                if i % 10 == 0:
                    test_idx = np.random.randint(0, self.test_data.shape[0], self.batch_size)
                    test_batch = self.test_data[test_idx, :self.seq_length]
                    if gen_model.__class__.__name__ == 'RobustARGeneratorMeanFieldPosterior':
                        test_elbo, _, _, _ = sess.run(gen_model.compute_elbo, {x_: test_batch})
                    else:
                        test_elbo, _, _ = sess.run(gen_model.compute_elbo, {x_: test_batch})
                    test_elbos.append(test_elbo)

                if i % 1000 == 0:
                    print('Iteration {}: ELBO = {}, KL (z) = {}, Reconstruction = {} (time taken = {})'
                          .format(i, elbo, kl_z, llk, str(time.clock() - start)))

                sess.run(gen_model.optimise, {x_: batch})

                if i % 1000 == 0:
                    saver.save(sess, './{}/ckpt/robust_model.ckpt'.format(self.experiment_path))

            np.save(os.path.join('./{}/training_curves/robust_model_elbos.npy'.format(self.experiment_path)), np.stack(elbos))
            np.save(os.path.join('./{}/training_curves/robust_model_kls_z.npy'.format(self.experiment_path)), np.stack(kls_z))
            np.save(os.path.join('./{}/training_curves/robust_model_llks.npy'.format(self.experiment_path)), np.stack(llks))
            np.save(os.path.join('./{}/training_curves/robust_model_test_elbos.npy'.format(self.experiment_path)), np.stack(test_elbos))
            np.save(os.path.join('./training_curves/robust_model_kls_y.npy'.format(self.experiment_path)), np.stack(kls_y))

        # Modelling
        if self.pre_trained:
            test_idx = np.load('./output/forward_prediction_batch_idx.npy')
            test_batch = np.load('./output/forward_prediction_batch.npy')

        else:
            test_idx = np.random.randint(0, self.test_data.shape[0], self.batch_size)
            test_batch = self.test_data[test_idx, :self.seq_length]

            np.save(os.path.join('./{}/output/forward_prediction_batch.npy'.format(self.experiment_path)), test_batch)
            np.save(os.path.join('./{}/output/forward_prediction_batch_idx.npy'.format(self.experiment_path)), test_idx)

        if self.predict_forward:
            # Forward prediction
            forw_saver = tf.train.Saver([v for v in tf.global_variables() if 'forward' in v.name])
            elbos_forward = []
            llks_y_forw = []
            llks_x_forw = []
            kls_y_forw = []
            kls_z_forw = []

            for i in range(self.num_forward_iterations):

                elbo_forward, rec_y_forw, rec_x_forw, kl_y_forw, kl_z_forw = sess.run(gen_model.compute_elbo_forward, {x_: test_batch})
                sess.run(gen_model.forward_E_step, {x_: test_batch})
                sess.run(gen_model.forward_M_step, {x_: test_batch})
                elbos_forward.append(elbo_forward)
                llks_y_forw.append(rec_y_forw)
                llks_x_forw.append(rec_x_forw)
                kls_y_forw.append(kl_y_forw)
                kls_z_forw.append(kl_z_forw)

                if i % 1000 == 0:
                    print(('Forward Prediction Iteration {}: ELBO_forward = {}, Reconstruction (y) = {}, Reconstruction (x) = {}' +
                          ', KL (y) = {}, KL (z) = {} (time taken = {})').format(i, elbo_forward, rec_y_forw, rec_x_forw,
                                                                               kl_y_forw, kl_z_forw, str(time.clock() - start)))

                    forw_saver.save(sess, './ckpt/best_forward_prediction.ckpt')

            forw_seq = sess.run(gen_model._most_likely_pred, {x_: test_batch})

            np.save(os.path.join('./{}/training_curves/robust_model_forward_elbos.npy'.format(self.experiment_path)), np.stack(elbos_forward))
            np.save(os.path.join('./{}/training_curves/robust_model_forward_llks_y.npy'.format(self.experiment_path)), np.stack(llks_y_forw))
            np.save(os.path.join('./{}/training_curves/robust_model_forward_llks_x.npy'.format(self.experiment_path)), np.stack(llks_x_forw))
            np.save(os.path.join('./{}/training_curves/robust_model_forward_kls_y.npy'.format(self.experiment_path)), np.stack(kls_y_forw))
            np.save(os.path.join('./{}/training_curves/robust_model_forward_kls_z.npy'.format(self.experiment_path)), np.stack(kls_z_forw))

            np.save(os.path.join('./{}/output/best_forward_prediction.npy'.format(self.experiment_path)), forw_seq)


class RunVAEInfiniteData(object):

    def __init__(self, data, test_data, num_iterations, num_forward_iterations, seq_length, n_samples, batch_size,
                 training_size, pre_trained, generative_model, n_hidden_py, predict_forward, experiment_path, data_loader):

        self.pre_trained = pre_trained
        self.num_iterations = num_iterations
        self.num_forward_iterations = num_forward_iterations

        self.seq_length = seq_length
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.training_size = training_size

        self.generative_model = generative_model

        self.predict_forward = predict_forward

        self.DataLoaderClass = data_loader(seq_length = seq_length, training_size = batch_size*1000)

        self.experiment_path = experiment_path

        self.n_hidden_py = n_hidden_py


    def train(self):
        #Data loader
        print("Creating dataset...")
        # DataLoaderClass = DataLoader(seq_length = seq_length, training_size = self.batch_size*1000)
        _, training_data = self.DataLoaderClass.create_dataset()
        test_data_clean, test_data = self.DataLoaderClass.create_dataset()
        data_dimensionality = training_data.shape[-1]
        print("Dataset created.")


        tf.reset_default_graph()

        x_ = tf.placeholder(tf.float32, [self.batch_size, self.seq_length, data_dimensionality])

        gen_model = self.generative_model(data=x_, n_hidden=self.n_hidden_py, n_samples=self.n_samples, seq_length=self.seq_length,
                                          batch_size=self.batch_size)

        sess = tf.Session()
        saver = tf.train.Saver([v for v in tf.global_variables() if 'forward' not in v.name])
        start = time.clock()

        if self.pre_trained:
            saver.restore(sess, "./{}/ckpt/robust_model.ckpt".format(self.experiment_path))
            sess.run(tf.variables_initializer([v for v in tf.global_variables() if 'forward' in v.name]))

        else:
            sess.run(tf.global_variables_initializer())
            elbos = []
            kls_z = []
            kls_y = []
            llks = []
            test_elbos = []

            for i in range(self.num_iterations):
                idx = np.array(range(self.batch_size*i, self.batch_size*(i+1) ))
                batch = training_data[idx, ...]


                if gen_model.__class__.__name__ == 'RobustARGeneratorMeanFieldPosterior':
                    elbo, kl_z, kl_y, llk = sess.run(gen_model.compute_elbo, {x_: batch})
                    kls_y.append(kl_y)
                else:
                    elbo, kl_z, llk = sess.run(gen_model.compute_elbo, {x_: batch})
                elbos.append(elbo)
                kls_z.append(kl_z)
                llks.append(llk)

                if i % 10 == 0:
                    test_idx = np.random.randint(0, test_data.shape[0], self.batch_size)
                    test_batch = test_data[test_idx, :self.seq_length]
                    if gen_model.__class__.__name__ == 'RobustARGeneratorMeanFieldPosterior':
                        test_elbo, _, _, _ = sess.run(gen_model.compute_elbo, {x_: test_batch})
                    else:
                        test_elbo, _, _ = sess.run(gen_model.compute_elbo, {x_: test_batch})
                    test_elbos.append(test_elbo)

                if i % 1000 == 0:
                    print('Iteration {}: ELBO = {}, KL (z) = {}, Reconstruction = {} (time taken = {})'
                          .format(i, elbo, kl_z, llk, str(time.clock() - start)))
                    _, training_data = self.DataLoaderClass.create_dataset()

                sess.run(gen_model.optimise, {x_: batch})

                if i % 1000 == 0:
                    saver.save(sess, './{}/ckpt/robust_model.ckpt'.format(self.experiment_path))

            np.save(os.path.join('./{}/training_curves/robust_model_elbos.npy'.format(self.experiment_path)), np.stack(elbos))
            np.save(os.path.join('./{}/training_curves/robust_model_kls_z.npy'.format(self.experiment_path)), np.stack(kls_z))
            np.save(os.path.join('./{}/training_curves/robust_model_llks.npy'.format(self.experiment_path)), np.stack(llks))
            np.save(os.path.join('./{}/training_curves/robust_model_test_elbos.npy'.format(self.experiment_path)), np.stack(test_elbos))
            np.save(os.path.join('./{}/training_curves/robust_model_kls_y.npy'.format(self.experiment_path)), np.stack(kls_y))

        # Modelling
        if self.pre_trained:
            test_idx = np.load('./output/forward_prediction_batch_idx.npy')
            test_batch = np.load('./output/forward_prediction_batch.npy')

        else:
            test_idx = np.random.randint(0, test_data.shape[0], self.batch_size)
            test_batch = test_data[test_idx, :self.seq_length]
            test_batch_clean = test_data_clean[test_idx, :self.seq_length]

            np.save(os.path.join('./{}/output/forward_prediction_batch.npy'.format(self.experiment_path)), test_batch)
            np.save(os.path.join('./{}/output/forward_prediction_batch_clean.npy'.format(self.experiment_path)), test_batch_clean)
            np.save(os.path.join('./{}/output/forward_prediction_batch_idx.npy'.format(self.experiment_path)), test_idx)

        if self.predict_forward:
            # Forward prediction
            forw_saver = tf.train.Saver([v for v in tf.global_variables() if 'forward' in v.name])
            elbos_forward = []
            llks_y_forw = []
            llks_x_forw = []
            kls_y_forw = []
            kls_z_forw = []

            for i in range(self.num_forward_iterations):

                elbo_forward, rec_y_forw, rec_x_forw, kl_y_forw, kl_z_forw = sess.run(gen_model.compute_elbo_forward, {x_: test_batch})
                sess.run(gen_model.forward_E_step, {x_: test_batch})
                sess.run(gen_model.forward_M_step, {x_: test_batch})
                elbos_forward.append(elbo_forward)
                llks_y_forw.append(rec_y_forw)
                llks_x_forw.append(rec_x_forw)
                kls_y_forw.append(kl_y_forw)
                kls_z_forw.append(kl_z_forw)

                if i % 1000 == 0:
                    print(('Forward Prediction Iteration {}: ELBO_forward = {}, Reconstruction (y) = {}, Reconstruction (x) = {}' +
                          ', KL (y) = {}, KL (z) = {} (time taken = {})').format(i, elbo_forward, rec_y_forw, rec_x_forw,
                                                                               kl_y_forw, kl_z_forw, str(time.clock() - start)))

                    forw_saver.save(sess, './ckpt/best_forward_prediction.ckpt')

            forw_seq = sess.run(gen_model._most_likely_pred, {x_: test_batch})

            np.save(os.path.join('./{}/training_curves/robust_model_forward_elbos.npy'.format(self.experiment_path)), np.stack(elbos_forward))
            np.save(os.path.join('./{}/training_curves/robust_model_forward_llks_y.npy'.format(self.experiment_path)), np.stack(llks_y_forw))
            np.save(os.path.join('./{}/training_curves/robust_model_forward_llks_x.npy'.format(self.experiment_path)), np.stack(llks_x_forw))
            np.save(os.path.join('./{}/training_curves/robust_model_forward_kls_y.npy'.format(self.experiment_path)), np.stack(kls_y_forw))
            np.save(os.path.join('./{}/training_curves/robust_model_forward_kls_z.npy'.format(self.experiment_path)), np.stack(kls_z_forw))

            np.save(os.path.join('./{}/output/best_forward_prediction.npy'.format(self.experiment_path)), forw_seq)


class RunRNN(RunVAE):

    def train(self):
        tf.reset_default_graph()

        x_ = tf.placeholder(tf.float32, [self.batch_size, self.seq_length, 1])

        rnn = self.generative_model(data = x_, n_hidden=30, seq_length=self.seq_length, batch_size=self.batch_size)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if self.pre_trained:
            tf.train.Saver().restore(sess, "./ckpt/rnn.ckpt")

        mses = []

        for i in range(self.num_iterations):

            start = time.clock()

            idx = np.random.randint(0, self.training_size, self.batch_size)
            batch = self.training_data[idx, ...]

            mse = sess.run(rnn.compute_cost_func, {x_: batch})
            mses.append(mse)

            print('Iteration {}: MSE = {} (time taken = {})'.format(i+1, mse, str(time.clock() - start)))

            sess.run(rnn.optimise, {x_: batch})

        tf.train.Saver().save(sess, './ckpt/rnn.ckpt')
        np.save(os.path.join('./training_curves/robust_model_mses.npy'), np.stack(mses))


class RunBayesianTimeSeriesVAE(object):

    def __init__(self, training_data, test_data, experiment_path, generative_model, logger, sess, pre_trained=False,
                 num_iterations=10000, num_forward_iterations=0, seq_length=100, n_samples=30, batch_size=20,
                 n_hidden=50, dim_z=20, **kwargs):

        self.pre_trained = pre_trained
        self.training_data = training_data[:, :seq_length]
        self.testing_data = test_data[:, :seq_length]
        self.I = num_iterations
        self.I_forw = num_forward_iterations

        self.T = seq_length
        self.S = n_samples
        self.B = batch_size
        self.N = training_data.shape[0]
        self.D = training_data.shape[-1]
        self.H = n_hidden
        self.Z = dim_z

        self.sess = sess
        self.x_ = tf.placeholder(tf.float32, [None, self.T, self.D])
        self.model = generative_model(self.x_, self.B, self.T, self.H, self.S, self.D, self.Z, future_steps=10, **kwargs)

        self.path = experiment_path
        self.logger = logger
        self.print_every = kwargs.get('print_every', 100)
        self.save_every = kwargs.get('save_every', 100)
        self.saver = tf.train.Saver([v for v in tf.global_variables() if 'forward' not in v.name])

        if self.pre_trained:
            try:
                self.saver.restore(self.sess, self.path + '/ckpt/modelling.ckpt')
                self.sess.run(tf.variables_initializer([v for v in tf.global_variables() if 'forward' in v.name]))
                self.logger.info('Model parameters loaded successfully.')

            except tf.errors.NotFoundError:
                self.logger.warning('No model to load. Run simulation without initialisation.')
                self.sess.run(tf.global_variables_initializer())

            try:
                self.elbos = np.load(os.path.join(self.path + '/training_curves/elbos.npy')).tolist()
                self.llks = np.load(os.path.join(self.path + '/training_curves/llks.npy')).tolist()
                self.kls_z = np.load(os.path.join(self.path + '/training_curves/kls_z.npy')).tolist()
                self.kls_y = np.load(os.path.join(self.path + '/training_curves/kls_y.npy')).tolist()
                self.test_elbos = np.load(os.path.join(self.path + '/training_curves/test_elbos.npy')).tolist()
                self.logger.info('Training curves loaded successfully.')

            except FileNotFoundError:
                logger.info('No training curves to restore.')
                self.elbos = []
                self.kls_z = []
                self.kls_y = []
                self.llks = []
                self.test_elbos = []

        else:
            self.sess.run(tf.global_variables_initializer())
            self.elbos = []
            self.kls_z = []
            self.kls_y = []
            self.llks = []
            self.test_elbos = []

    def bayesian_update_w(self, batch):
        prior_mean, prior_covariance, hiddens, trans_stds, code_y = \
            self.sess.run([self.model.p_w.mean(), self.model.p_w.covariance(), self.model._hiddens, self.model._trans_stds,
                           self.model._code_y], {self.x_: batch})

        scaling = self.N / self.B

        data_precision = np.diag(np.sum(np.square(hiddens / trans_stds), (0, 1))) / self.S
        post_precision = np.linalg.inv(prior_covariance) + data_precision * scaling
        post_covariance = np.linalg.inv(post_precision)
        first_order = np.reshape(hiddens / np.square(trans_stds), (self.S, self.B, self.T, self.H))
        first_order = np.mean(first_order * code_y, 0)
        first_order = np.sum(first_order, axis=(0, 1))
        post_mean = - 1 * np.matmul(post_covariance, scaling * first_order)
        self.model._post_sigma_w = tf.convert_to_tensor(post_covariance, tf.float32)
        self.model._post_mu_w = tf.convert_to_tensor(post_mean, tf.float32)

    def bayesian_update_sigma(self, batch):
        prior_alpha, prior_beta, post_means, post_stds = \
            self.sess.run([self.model.p_sigma.concentration, self.model.p_sigma.rate, self.model._post_means,
                           self.model._post_stds], {self.x_: batch})

        scaling = self.N / self.B

        post_alpha = prior_alpha + (self.B * self.T) * scaling / 2
        post_beta = np.square(batch) + (np.square(post_means) + np.square(post_stds)) - 2 * batch * post_means
        post_beta = prior_beta + np.sum(post_beta) * scaling / 2
        self.model._post_alpha = tf.convert_to_tensor(post_alpha, tf.float32)
        self.model._post_beta = tf.convert_to_tensor(post_beta, tf.float32)

    def train(self):

        self.logger.info('Experimental setup:')
        self.logger.info(
            'B = {}, S = {}, T = {}, H = {}, Z = {}, I = {}, N = {}'
                .format(self.B, self.S, self.T, self.H, self.Z, self.I, self.N))
        self.logger.info('Variational Bayes with parameters mean_w = {}, cov_w = {}, alpha = {}, beta = {}')

        self.logger.info('\n-------------------\n')
        self.logger.info('Start training...')

        try:
            for i in range(self.I):
                idx = np.random.randint(0, self.N, self.B)
                batch = self.training_data[idx]

                elbo, kl_z, kl_y, llk = self.sess.run(self.model.compute_elbo, {self.x_: batch})
                self.elbos.append(elbo)
                self.kls_z.append(kl_z)
                self.kls_y.append(kl_y)
                self.llks.append(llk)

                if i % 10 == 0:
                    test_idx = np.random.randint(0, self.testing_data.shape[0], self.BB)
                    test_batch = self.testing_data[test_idx]
                    test_elbo, _, _, _ = self.sess.run(self.model.compute_elbo, {self.x_: test_batch})
                    self.test_elbos.append(test_elbo)

                if i % self.print_every == 0:
                    self.logger.info(
                        '''Iteration: {}, ELBO: {:.2f}, KL (z): {:.2f}, KL (y): {:.2f}, Reconstruction: {:.2f}, Test ELBO: {:.2f}'''
                            .format(i, elbo, kl_z, kl_y, llk, test_elbo))

                if i % self.save_every == 0:
                    self.saver.save(self.sess, self.path + '/ckpt/modelling.ckpt')
                    np.save(os.path.join(self.path + '/training_curves/elbos.npy'), np.stack(self.elbos))
                    np.save(os.path.join(self.path + '/training_curves/llks.npy'), np.stack(self.llks))
                    np.save(os.path.join(self.path + '/training_curves/kls_z.npy'), np.stack(self.kls_z))
                    np.save(os.path.join(self.path + '/training_curves/kls_y.npy'), np.stack(self.kls_y))
                    np.save(os.path.join(self.path + '/training_curves/test_elbos.npy'), np.stack(self.test_elbos))

                self.sess.run(self.model.E_step, {self.x_: batch})
                self.sess.run(self.model.M_step, {self.x_: batch})

                self.bayesian_update_sigma(batch)
                self.bayesian_update_w(batch)

        except KeyboardInterrupt:
            self.logger.warning('Training interrupted at iteration {}.'.format(i))

        self.logger.info('Training Complete.')

        self.saver.save(self.sess, self.path + '/ckpt/modelling.ckpt')
        self.logger.info('Model parameters saved successfully.')

        np.save(os.path.join(self.path + '/training_curves/elbos.npy'), np.stack(self.elbos))
        np.save(os.path.join(self.path + '/training_curves/llks.npy'), np.stack(self.llks))
        np.save(os.path.join(self.path + '/training_curves/kls_z.npy'), np.stack(self.kls_z))
        np.save(os.path.join(self.path + '/training_curves/kls_y.npy'), np.stack(self.kls_y))
        np.save(os.path.join(self.path + '/training_curves/test_elbos.npy'), np.stack(self.test_elbos))
        self.logger.info('Training curves saved successfully.')

        self.sess.close()
