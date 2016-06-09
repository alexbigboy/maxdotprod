import numpy as np
import scipy.linalg as sla
from functools import namedtuple
from sklearn.cluster import KMeans


class NaiveTree():
    Node = namedtuple("Node", ['left', 'right',
                               'indexes', 'left_sum', 'right_sum'])

    def __init__(self, X, Q, min_size=5):
        self.X = X
        self.Q = Q
        self.min_size = min_size
        n, d = self.X.shape
        m, _ = self.Q.shape
        W = sla.cholesky(self.Q.T.dot(self.Q)) if m > d else Q
        self.Z = W.dot(self.X.T).T
        self.nodes = []
        self.root = self.build(np.arange(n))[0]

    def build(self, indexes):
        n, d = self.X.shape
        if len(indexes) > self.min_size:
            ind_0, ind_1 = self.separate(indexes)
            left_node, left_sum = self.build(ind_0)
            right_node, right_sum = self.build(ind_1)
            curr_node = len(self.nodes)
            self.nodes.append(self.Node(left_node, right_node, indexes,
                                        left_sum, right_sum))
            return curr_node, left_sum + right_sum
        else:
            curr_node = len(self.nodes)
            self.nodes.append(self.Node(None, None, indexes, None, None))
            return curr_node, self.X[indexes].sum(axis=0)

    def separate(self, indexes):
        clf = KMeans(2).fit(self.Z[indexes])
        return indexes[clf.labels_==0], indexes[clf.labels_==1]

    def node(self, id_node):
        return self.nodes[id_node]

    def left(self, id_node):
        return self.nodes[id_node].left

    def right(self, id_node):
        return self.nodes[id_node].right

    def indexes(self, id_node):
        return self.nodes[id_node].indexes

    def left_sum(self, id_node):
        return self.nodes[id_node].left_sum

    def right_sum(self, id_node):
        return self.nodes[id_node].right_sum

    def find_ind(self, q):
        node = self.root
        prev_node = node
        while node is not None:
            prev_node = node
            if self.left_sum(node) is None or self.right_sum(node) is None:
                break
            node = self.left(node) if \
                    q.dot((self.left_sum(node) -
                           self.right_sum(node))[:,np.newaxis])>0 \
                else self.right(node)

        return prev_node

    def find(self, q):
        return self.X[self.nodes[self.find_ind(q)].indexes]
