import os
import logging
import tensorflow as tf

from datetime import datetime

# Specify one of {TimeSeriesData, BouncingBallsData, MovingMNISTData}
from robust.data import RobertoTimeSeriesData as DataLoader
from robust.plot import TimeSeriesPlotter as Plotter

from robust.models import RunVAEInfiniteData as Run
from robust.models import RegularisedRobustARGenerator as GenerativeModel


pre_trained = False
predict_forward = False

num_iterations = 1000
num_forward_iterations = 0

n_hidden = 50
seq_length = 100
n_samples = 30
batch_size = 20
dim_z = 20

model_settings = 'H_{}_B_{}_T_{}_Z_{}'.format(n_hidden, batch_size, seq_length, dim_z)
experiment_path = 'experiments/{}_{}'.format(GenerativeModel.__name__, model_settings)
# experiment_path = 'experiments/ben_test'

w_lambda = 0.
bayesian = True
fixed_sigma = False
true_sigma = 5.3
sigma2_w = 1.
alpha = 3.
beta = 10.
kwargs = {'w_lambda': w_lambda, 'bayesian': bayesian,
          'fixed_sigma': fixed_sigma, 'true_sigma': true_sigma,
          'sigma2_w': sigma2_w,
          'alpha': alpha, 'beta': beta}

if __name__ == '__main__':
    os.makedirs('./{}/training_curves/'.format(experiment_path), exist_ok=True)
    os.makedirs('./{}/ts_curves/'.format(experiment_path), exist_ok=True)
    os.makedirs('./{}/output/'.format(experiment_path), exist_ok=True)
    os.makedirs('./{}/data/'.format(experiment_path), exist_ok=True)
    os.makedirs('./{}/ckpt/'.format(experiment_path), exist_ok=True)

    if Run.__name__ == 'RunVAEInfiniteData':
        DataLoaderClass = DataLoader(seq_length=seq_length,
                                     training_size=batch_size*1000)
        data = None
        # data for plotting only:
        testing_data_clean, testing_data = DataLoaderClass.create_dataset()
        print('Data will be created inline during training.')

    elif DataLoader.__name__ in ('TimeSeriesData', 'RobertoTimeSeriesData'):
        print("Creating dataset...")
        training_size = 9000
        # also creates a test set of size training_size
        DataLoaderClass = DataLoader(seq_length=seq_length,
                                     training_size=2*training_size)
        y, y_tilde = DataLoaderClass.create_dataset()
        data = y_tilde[:training_size, :]
        testing_data = y_tilde[training_size:, :]
        # data for plotting only:
        testing_data_clean = y[training_size:, :]
        print("Dataset created.")

    elif DataLoader.__name__ == 'BouncingBallsData':
        print("Creating dataset...")
        DataLoaderClass = DataLoader()
        data, testing_data = DataLoaderClass.create_dataset()
        print("Dataset created.")

    elif DataLoader.__name__ == 'MovingMNISTData':
        print("Creating dataset...")
        training_size = 4000
        DataLoaderClass = DataLoader(training_size)
        data, testing_data = DataLoaderClass.create_dataset()
        print("Dataset created.")

    if GenerativeModel.__name__ == 'RegularisedRobustARGenerator':

        tf.reset_default_graph()
        logger = logging.getLogger('RegularisedTimeSeries')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.info('Experiment Settings: {}'.format(kwargs))
        log_fh = logging.FileHandler(
            os.path.join(experiment_path, '{}.log'.format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))))
        log_fh.setFormatter(formatter)
        logger.addHandler(log_fh)

        sess = tf.Session()

        run = Run(data, testing_data, experiment_path, GenerativeModel, logger, sess, pre_trained, num_iterations,
                  num_forward_iterations, seq_length, n_samples, batch_size, n_hidden, **kwargs)

    else:
        run = Run(data, testing_data, num_iterations, num_forward_iterations,
                  seq_length, n_samples, batch_size, training_size, pre_trained,
                  GenerativeModel, n_hidden, predict_forward, experiment_path,
                  DataLoader)
    run.train()

    plotter = Plotter(testing_data, testing_data_clean, batch_size, seq_length,
                      n_samples, GenerativeModel, n_hidden, experiment_path)
    plotter.plot()
