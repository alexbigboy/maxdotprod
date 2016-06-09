import os
import numpy as np
import pylab as plt
from maxdotprod import preprocessing
from maxdotprod import tree
from sklearn.datasets import make_blobs


def gen_data(center_x=(-2, 1), center_q=(-2, 0), sigma=2,
             n_samples=400, random_state=None):
    W, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=sigma,
                      centers=[center_x, center_q], random_state=random_state)
    X = W[y==0]
    Q = W[y==1]
    return X, Q


def get_grid_around_data(X, h=0.5, border=4):
    x_min, x_max = X[:, 0].min() - border, X[:, 0].max() + border
    y_min, y_max = X[:, 1].min() - border, X[:, 1].max() + border
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    return np.c_[xx.ravel(), yy.ravel()]


def show(X, Q, tree=None, path=None):
    fig = plt.figure(figsize=(20, 20))
    m = len(Q)
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/m) for i in range(m)]
    kwargs = dict(alpha=0.7, s=75)
    plt.scatter(X[:,0], X[:,1], c="b", label='X', **kwargs)
    plt.scatter(Q[:,0], Q[:,1], c="g", marker='s',
                edgecolors='none', label='Q', alpha=0.4, s=40)

    for q, color in zip(Q, colors):
        xx = tree.find(q) if tree is not None \
            else [preprocessing.get_nearest(q, X)]
        for x in xx:
            plt.plot([q[0], x[0]], [q[1], x[1]], 'r--', c=color)


    plt.legend()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
    return fig


if __name__ == "__main__":
    X, Q = gen_data()
    show(X, Q, tree.NaiveTree(X, Q), path="img/naive_tree.png")
    show(X, Q, path="img/maxdotprod.png")
    show(X, get_grid_around_data(X), path="img/grid_maxdotprod.png")

