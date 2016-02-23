import numpy as np
from sklearn.linear_model.base import BaseEstimator

from utils import compute_labels, log_likelihood_from_labels


class Random(BaseEstimator):
    def __init__(self, n_clusters, n_init=10):
        self.n_clusters = n_clusters
        self.n_init = n_init

    def fit(self, X):
        n_objects = X.shape[0]
        best_log_likelihood = float("-inf")
        for i in range(self.n_init):
            centers_idx = np.random.choice(n_objects, size=self.n_clusters, replace=False)
            mu = X[centers_idx, :]
            labels = compute_labels(X, mu)
            ll = log_likelihood_from_labels(X, labels)
            if ll > best_log_likelihood:
                best_log_likelihood = ll
                self.cluster_centers_ = mu.copy()
                self.labels_ = labels
