import os
import numpy as np
import tensorflow as tf
import scipy.linalg as la
from scipy.stats import multivariate_normal as norm
from scipy.io import loadmat
from generate_bouncing_balls import generate_bouncing_balls


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softplus(x):
    return np.log(1 + np.exp(x)) + 1e-5


class RobertoTimeSeriesData(object):
    def __init__(self, n_hidden=140, seq_length=300, training_size=1000, sigma_c=5.3, offset=50):

        self.H = n_hidden
        self.T = seq_length
        self.N = training_size
        self.sigma_c = sigma_c
        self.offset = offset

    def create_dataset(self):

        np.random.seed(100)

        A = 0.8 * np.random.randn(self.H, self.H)
        R = la.expm(A - A.T)
        B = 2 * np.random.randn(1, self.H)
        b = 0.5 * np.random.randn(self.H, 1)
        c = 0.5 * np.random.randn(self.H, 1)
        bias_h = np.random.rand(1, self.H)
        d = 0.05 * np.random.randn(self.H, 1)

        np.random.seed(0)
        h = np.zeros((self.N, self.T + 1, self.H))
        h[:, 0, :] = norm.rvs(mean=None, cov=np.eye(self.H), size=self.N)
        y = np.zeros((self.N, self.T + 1, 1))

        for i in range(1, self.T + 1):
            h[:, i, :] = np.tanh(np.dot(h[:, i - 1, :], R) + np.tanh(np.dot(y[:, i - 1], B))) + 0.7 * h[:, i - 1, :] + bias_h
            y[:, i] = (1.5 * np.dot(h[:, i, :], b) + 1 * np.tanh(np.dot(h[:, i, :], c))
                       + 1 * softplus(np.tanh(np.dot(h[:, i, :], d)) - 10) * np.random.randn(self.N, 1))

        y = y[:, 1:]
        y = y - np.mean(y)
        y_tilde = (y + self.sigma_c * np.random.randn(self.N, self.T, 1))

        return y, y_tilde

class TimeSeriesData(object):
    def __init__(self, n_hidden=140, seq_length=300, training_size=1000, sigma_c=5.3, sigma_b=1.):

        np.random.seed(1)

        self.n_hidden = n_hidden
        self.seq_length = seq_length
        self.training_size = training_size

        self.sigma_c = sigma_c
        self.sigma_b = sigma_b

    def create_dataset(self):
        h = np.zeros((self.training_size, self.seq_length + 1, self.n_hidden))
        h[:, 0, :] = norm.rvs(mean=None, cov=np.eye(self.n_hidden), size=self.training_size)
        y = np.zeros((self.training_size, self.seq_length + 1, 1))

        A = 0.08 * np.random.randn(self.n_hidden, self.n_hidden)
        R = la.expm(A - A.T)
        B = 0.2 * np.random.rand(1, self.n_hidden)
        b = 0.1 * np.random.randn(self.n_hidden, 1)
        c = 0.1 * np.random.randn(self.n_hidden, 1)
        bias_h = np.random.rand(1, self.n_hidden)
        d = 0.01 * np.random.rand(self.n_hidden, 1)

        for i in range(1, self.seq_length + 1):
            h[:, i, :] = np.tanh(np.dot(h[:, i - 1, :], R) + np.dot(y[:, i - 1], B)) + 0.7 * h[:, i - 1, :] + bias_h
            y[:, i] = (3 * np.dot(h[:, i, :], b) + 4 * np.tanh(np.dot(h[:, i, :], c))
                       + 3 * softplus(np.tanh(np.dot(h[:, i, :], d)) - 2) * np.random.randn(self.training_size, 1))

        y = y[:, 1:]
        y_tilde = (y + self.sigma_c * np.random.randn(self.training_size, self.seq_length, 1))

        return y, y_tilde


class BouncingBallsData(object):
    def __init__(self, res=30, T=100, N=4000):
        self.res = res
        self.T = T
        self.N = N

    def create_dataset(self):

        if not (os.path.isfile('./data/bouncing_balls_training_data.mat')
                and os.path.isfile('./data/bouncing_balls_testing_data.mat')):
            generate_bouncing_balls(self.res, self.T, self.N)
        else:
            pass

        training_balls = np.stack([b for b in loadmat('./data/bouncing_balls_training_data.mat')['Data'].flatten()])
        testing_balls = np.stack([b for b in loadmat('./data/bouncing_balls_testing_data.mat')['Data'].flatten()])

        return training_balls, testing_balls


class MovingMNISTData(object):
    def __init__(self, training_size, res=64, T=100, N=4000, digit_size=28, num_digits = 2, deterministic=True):
        self.training_size = training_size
        self.res = res
        self.T = T
        self.N = N
        self.digit_size = digit_size
        self.num_digits = num_digits
        self.deterministic = deterministic

        if self.training_size >= self.N:
            raise ValueError('Training size must be less than N.')

        mnist = tf.keras.datasets.mnist
        (self.train_data, _), (self.test_data, _) = mnist.load_data()

    def create_video_np_array(self, data, N):
        x = np.zeros((N,
                    self.T,
                    self.res,
                    self.res),
                    dtype=np.float32)

        for n in range(N):
            for d in range(self.num_digits):
                idx = np.random.randint(data.shape[0])
                digit = data[idx,:,:]

                #starting position
                sx = np.random.randint(self.res-self.digit_size)
                sy = np.random.randint(self.res-self.digit_size)
                #starting velocity
                dx = np.random.randint(-4, 5)
                dy = np.random.randint(-4, 5)
                for t in range(self.T):
                    if sy < 0:  #if at bottom edge
                        sy = 0
                        if self.deterministic:
                            dy = -dy
                        else:
                            dy = np.random.randint(1, 5)
                            dx = np.random.randint(-4, 5)
                    elif sy >= self.res-self.digit_size:    #if at top edge
                        sy = self.res-self.digit_size-1
                        if self.deterministic:
                            dy = -dy
                        else:
                            dy = np.random.randint(-4, 0)
                            dx = np.random.randint(-4, 5)

                    if sx < 0:  #if at left edge
                        sx = 0
                        if self.deterministic:
                            dx = -dx
                        else:
                            dx = np.random.randint(1, 5)
                            dy = np.random.randint(-4, 5)
                    elif sx >= self.res-self.digit_size:    #if at right edge
                        sx = self.res-self.digit_size-1
                        if self.deterministic:
                            dx = -dx
                        else:
                            dx = np.random.randint(-4, 0)
                            dy = np.random.randint(-4, 5)

                    x[n, t, sy:sy+self.digit_size, sx:sx+self.digit_size] += np.squeeze(digit)
                    sy += dy
                    sx += dx

        x[x>1] = 1.

        return x

    def create_dataset(self):

        return self.create_video_np_array(self.train_data, self.N), self.create_video_np_array(self.test_data, self.N - self.training_size)
