import numpy as np


def normalized(X):
    return X / np.sqrt((X**2).sum(axis=1))[:, np.newaxis]


def get_nearest(q, X):
    return X[np.argmax(q.dot(X.T))]


