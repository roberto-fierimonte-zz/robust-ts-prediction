import os
import numpy as np
import tensorflow as tf
import scipy.linalg

from scipy.stats import multivariate_normal
from scipy.io import loadmat, savemat
from scipy import shape as scipy_shape
from scipy import size as scipy_size

SIZE = 10


def v_norm(x):
    return np.sqrt((x ** 2).sum())


def shape(A):
    if isinstance(A, np.ndarray):
        return scipy_shape(A)
    else:
        return A.shape()


def size(A):
    if isinstance(A, np.ndarray):
        return scipy_size(A)
    else:
        return A.size()


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None):
    if r is None:
        r = np.array([1.2] * n)
    if m is None:
        m = np.array([1] * n)
    # r is to be rather small.
    X = np.zeros((T, n, 2), dtype='float')
    v = np.random.randn(n, 2)
    v = v / v_norm(v) * .5
    good_config = False
    while not good_config:
        x = 2 + np.random.rand(n, 2) * 8
        good_config = True
        for i in range(n):
            for z in range(2):
                if x[i][z] - r[i] < 0:
                    good_config = False
                if x[i][z] + r[i] > SIZE:
                    good_config = False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if v_norm(x[i] - x[j]) < r[i] + r[j]:
                    good_config = False

    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t, i] = x[i]

        for mu in range(int(1 / eps)):

            for i in range(n):
                x[i] += eps * v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z] - r[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if x[i][z] + r[i] > SIZE:
                        v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n):
                for j in range(i):
                    if v_norm(x[i] - x[j]) < r[i] + r[j]:
                        # the bouncing off part:
                        w = x[i] - x[j]
                        w = w / v_norm(w)

                        v_i = np.dot(w.transpose(), v[i])
                        v_j = np.dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)

    return X


def ar(x, y, z):
    return z / 2 + np.arange(x, y, z, dtype='float')


def matricize(X, res, r=None):
    T, n = shape(X)[0:2]
    if r is None:
        r = np.array([1.2] * n)

    A = np.zeros((T, res, res), dtype='float')

    [I, J] = np.meshgrid(ar(0, 1, 1. / res) * SIZE, ar(0, 1, 1. / res) * SIZE)

    for t in range(T):
        for i in range(n):
            to_exp = ((I - X[t, i, 0])**2 + (J - X[t, i, 1])**2) / (r[i]**2)
            to_exp = -(to_exp)**4
            A[t] += np.exp(to_exp)

        A[t][A[t] > 1] = 1
    return A


def bounce_mat(res, n=2, T=128, r=None):
    if r is None:
        r = np.array([1.2] * n)
    x = bounce_n(T, n, r)
    A = matricize(x, res, r)
    return A


def bounce_vec(res, n=2, T=128, r=None, m=None):
    if r is None:
        r = np.array([1.2] * n)
    x = bounce_n(T, n, r, m)
    V = matricize(x, res, r)
    return V.reshape(T, res ** 2)

    # make sure you have this folder


def generate_bouncing_balls(res=30, T=100, N=4000):

    dat = np.empty((N), dtype=object)
    for i in range(N):
        dat[i] = bounce_vec(res=res, n=3, T=T)
    data = {}
    data['Data'] = dat
    savemat('./data/bouncing_balls_training_data.mat', data)

    N = 200
    dat = np.empty((N), dtype=object)
    for i in range(N):
        dat[i] = bounce_vec(res=res, n=3, T=T)
    data = {}
    data['Data'] = dat
    savemat('./data/bouncing_balls_testing_data.mat', data)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softplus(x):
    return np.log(1 + np.exp(x)) + 1e-5


class TimeSeriesData(object):
    def __init__(self, n_hidden=140, seq_length=300, training_size=1000,
                 sigma_c=5.3, sigma_b=1., random_seed=1):

        np.random.seed(random_seed)

        self.n_hidden = n_hidden
        self.seq_length = seq_length
        self.training_size = training_size

        self.sigma_c = sigma_c
        self.sigma_b = sigma_b

    def create_dataset(self):
        h = np.zeros((self.training_size, self.seq_length + 1, self.n_hidden))
        h[:, 0, :] = multivariate_normal.rvs(
                            mean=None,
                            cov=np.eye(self.n_hidden),
                            size=self.training_size
                                                )
        y = np.zeros((self.training_size, self.seq_length + 1, 1))

        A = 0.08 * np.random.randn(self.n_hidden, self.n_hidden)
        R = scipy.linalg.expm(A - A.T)
        B = 0.2 * np.random.rand(1, self.n_hidden)
        b = 0.1 * np.random.randn(self.n_hidden, 1)
        c = 0.1 * np.random.randn(self.n_hidden, 1)
        bias_h = np.random.rand(1, self.n_hidden)
        d = 0.01 * np.random.rand(self.n_hidden, 1)

        for i in range(1, self.seq_length + 1):
            h[:, i, :] = (np.tanh(np.dot(h[:, i - 1, :], R) +
                          np.dot(y[:, i - 1], B)) +
                          0.7 * h[:, i - 1, :] + bias_h)
            soft = softplus(np.tanh(np.dot(h[:, i, :], d)) - 2)
            random_number = np.random.randn(self.training_size, 1)
            y[:, i] = (3 * np.dot(h[:, i, :], b) +
                       4 * np.tanh(np.dot(h[:, i, :], c)) +
                       3 * soft * random_number)

        y = y[:, 1:]
        y_tilde = (y + self.sigma_c * np.random.randn(self.training_size,
                                                      self.seq_length, 1)
                   )

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
        file = './data/bouncing_balls_training_data.mat'
        training_balls = np.stack([b for b in loadmat(file)['Data'].flatten()])
        testing_balls = np.stack([b for b in loadmat(file)['Data'].flatten()])

        return training_balls, testing_balls


class MovingMNISTData(object):
    def __init__(self, training_size, res=64, T=100, N=4000, digit_size=28,
                 num_digits=2, deterministic=True):
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
        x = np.zeros((N, self.T, self.res, self.res), dtype=np.float32)

        for n in range(N):
            for d in range(self.num_digits):
                idx = np.random.randint(data.shape[0])
                digit = data[idx, :, :]

                # starting position
                sx = np.random.randint(self.res-self.digit_size)
                sy = np.random.randint(self.res-self.digit_size)
                # starting velocity
                dx = np.random.randint(-4, 5)
                dy = np.random.randint(-4, 5)
                for t in range(self.T):
                    if sy < 0:  # if at bottom edge
                        sy = 0
                        if self.deterministic:
                            dy = -dy
                        else:
                            dy = np.random.randint(1, 5)
                            dx = np.random.randint(-4, 5)
                    elif sy >= self.res-self.digit_size:  # if at top edge
                        sy = self.res-self.digit_size-1
                        if self.deterministic:
                            dy = -dy
                        else:
                            dy = np.random.randint(-4, 0)
                            dx = np.random.randint(-4, 5)

                    if sx < 0:  # if at left edge
                        sx = 0
                        if self.deterministic:
                            dx = -dx
                        else:
                            dx = np.random.randint(1, 5)
                            dy = np.random.randint(-4, 5)
                    elif sx >= self.res-self.digit_size:  # if at right edge
                        sx = self.res-self.digit_size-1
                        if self.deterministic:
                            dx = -dx
                        else:
                            dx = np.random.randint(-4, 0)
                            dy = np.random.randint(-4, 5)

                    x[n, t, sy:sy+self.digit_size,
                      sx:sx+self.digit_size] += np.squeeze(digit)
                    sy += dy
                    sx += dx

        x[x > 1] = 1.

        return x

    def create_dataset(self):
        out_1 = self.create_video_np_array(self.train_data, self.N)
        out_2 = self.create_video_np_array(
                                self.test_data, self.N - self.training_size)
        return out_1, out_2
