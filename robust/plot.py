import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


class TimeSeriesPlotter(object):
    def __init__(self, testing_data, testing_data_clean, batch_size,
                 seq_length, n_samples, generative_model, n_hidden_py,
                 experiment_path):

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_samples = n_samples

        self.n_hidden_py = n_hidden_py

        self.testing_data = testing_data
        self.testing_data_clean = testing_data_clean

        self.generative_model = generative_model

        self.experiment_path = experiment_path

        self.data_dimensionality = testing_data.shape[-1]

    def plot(self, show=False):
        idx = np.random.randint(0, self.testing_data.shape[0], self.batch_size)
        batch = self.testing_data[idx, :100]
        batch_clean = self.testing_data_clean[idx, :self.seq_length]

        tf.reset_default_graph()

        x_ = tf.placeholder(tf.float32, [self.batch_size, self.seq_length,
                                         self.data_dimensionality])

        gen_model = self.generative_model(data=x_, n_hidden=self.n_hidden_py,
                                          n_samples=self.n_samples,
                                          seq_length=self.seq_length,
                                          batch_size=self.batch_size)

        sess = tf.Session()
        saver = tf.train.Saver([v for v in tf.global_variables()
                                if 'forward' not in v.name])
        name = "./{}/ckpt/robust_model.ckpt".format(self.experiment_path)
        saver.restore(sess, name)
        sess.run(tf.variables_initializer([v for v in tf.global_variables()
                                           if 'forward' in v.name]))

        # Calculate paramters for batch
        prior_means = sess.run(gen_model.p_y.mean(), {x_: batch})
        post_means = sess.run(gen_model.q_y.mean(), {x_: batch})
        latents = sess.run(gen_model._code_y, {x_: batch})
        latents = np.reshape(latents, (self.n_samples, self.batch_size, self.seq_length))

        np.save(os.path.join('./{}/ts_curves/data_batch.npy'.format(self.experiment_path)), batch)
        np.save(os.path.join('./{}/ts_curves/data_batch_clean.npy'.format(self.experiment_path)), batch_clean)
        np.save(os.path.join('./{}/ts_curves/prior_means.npy'.format(self.experiment_path)), prior_means)
        np.save(os.path.join('./{}/ts_curves/post_means.npy'.format(self.experiment_path)), post_means)
        np.save(os.path.join('./{}/ts_curves/latents.npy'.format(self.experiment_path)), latents)

        # Visualise for only one sequence in batch
        i = np.random.randint(0, batch.shape[0])
        
        plt.subplots(1,1,figsize=(15,6))
        plt.plot(batch[i, :], 'b', label='Noisy Sequence')
        plt.plot(batch_clean[i, :], label='Clean Signal')
        plt.plot(prior_means[0, i], 'r-', label='Prior Means')
        plt.plot(prior_means[1, i], 'r-')
        plt.plot(prior_means[2, i], 'r-')
        plt.plot(prior_means[3, i], 'r-')
        plt.plot(latents[0, i] , 'g', label='Posterior Sample')
        plt.plot(latents[1, i] , 'g')
        plt.plot(latents[2, i] , 'g')
        plt.plot(post_means[i], 'k', label='Posterior Means')
        plt.plot([], [])
        plt.suptitle('Modelling (Robust Seq-VAE)')
        plt.legend();
        plt.savefig('{}/ts_curves/TS_plot.pdf'.format(self.experiment_path), bbox_inches='tight')
