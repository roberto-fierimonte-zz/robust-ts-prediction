import numpy as np

from numpy import ndarray, sqrt, exp, array, empty, zeros, meshgrid, arange, dot
from scipy import shape as scipy_shape
from scipy import size as scipy_size
from scipy.io import savemat

SIZE = 10


def norm(x):
    return sqrt((x ** 2).sum())


def shape(A):
    if isinstance(A, ndarray):
        return scipy_shape(A)
    else:
        return A.shape()


def size(A):
    if isinstance(A, ndarray):
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
        r = array([1.2] * n)
    if m is None:
        m = array([1] * n)
    # r is to be rather small.
    X = zeros((T, n, 2), dtype='float')
    v = np.random.randn(n, 2)
    v = v / norm(v) * .5
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
                if norm(x[i] - x[j]) < r[i] + r[j]:
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
                    if norm(x[i] - x[j]) < r[i] + r[j]:
                        # the bouncing off part:
                        w = x[i] - x[j]
                        w = w / norm(w)

                        v_i = dot(w.transpose(), v[i])
                        v_j = dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)

    return X


def ar(x, y, z):
    return z / 2 + arange(x, y, z, dtype='float')


def matricize(X, res, r=None):
    T, n = shape(X)[0:2]
    if r is None:
        r = array([1.2] * n)

    A = zeros((T, res, res), dtype='float')

    [I, J] = meshgrid(ar(0, 1, 1. / res) * SIZE, ar(0, 1, 1. / res) * SIZE)

    for t in range(T):
        for i in range(n):
            A[t] += exp(-(((I - X[t, i, 0]) ** 2 + (J - X[t, i, 1]) ** 2) / (r[i] ** 2)) ** 4)

        A[t][A[t] > 1] = 1
    return A


def bounce_mat(res, n=2, T=128, r=None):
    if r is None:
        r = array([1.2] * n)
    x = bounce_n(T, n, r)
    A = matricize(x, res, r)
    return A


def bounce_vec(res, n=2, T=128, r=None, m=None):
    if r is None:
        r = array([1.2] * n)
    x = bounce_n(T, n, r, m)
    V = matricize(x, res, r)
    return V.reshape(T, res ** 2)

    # make sure you have this folder


def generate_bouncing_balls(res=30, T=100, N=4000):

    dat = empty((N), dtype=object)
    for i in range(N):
        dat[i] = bounce_vec(res=res, n=3, T=T)
    data = {}
    data['Data'] = dat
    savemat('./data/bouncing_balls_training_data.mat', data)

    N = 200
    dat = empty((N), dtype=object)
    for i in range(N):
        dat[i] = bounce_vec(res=res, n=3, T=T)
    data = {}
    data['Data'] = dat
    savemat('./data/bouncing_balls_testing_data.mat', data)
