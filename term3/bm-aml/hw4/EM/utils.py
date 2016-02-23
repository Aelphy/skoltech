from numpy.testing import assert_approx_equal
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.misc import logsumexp
import numpy as np


def compute_labels(X, mu):
    """Given the centers of clusters mu compute label of each data point in X

    X is n_objects x n_features matrix
    mu is n_clusters x n_features matrix
    """
    distances = cdist(X, mu)
    labels = np.argmin(distances, axis=1)
    return labels


def log_likelihood(X, w, mu, sigma):
    """Compute the log likelihood of the data X under the mixture of Gaussians.

    X is n_objects x n_features matrix
    w is a vector of size n_clusters of the prior probabilities of the clusters
    mu is n_clusters x n_features matrix of the centers of the clusters
    sigma is n_clusters x n_features x n_features tensor with the covariance
            matrix of each cluster.
    """
    assert_approx_equal(np.sum(w), 1)
    n_objects, n_features = X.shape
    n_clusters = w.size
    log_gamma = np.zeros((n_objects, n_clusters))
    for cluster in range(n_clusters):
        log_gamma[:, cluster] = np.log(w[cluster])
        log_gamma[:, cluster] += multivariate_normal.logpdf(X, mu[cluster, :],
                                                            sigma[cluster, :, :])
    return np.sum(logsumexp(log_gamma, axis=1))


def log_likelihood_from_labels(X, labels):
    """Compute the log likelihood of the data X under the mixture of Gaussians.

    Automatically estimates the parameters of each gaussian and return the
    log likelihood.
    """
    n_objects, n_features = X.shape
    clusters_arr, counts = np.unique(labels, return_counts=True)
    n_clusters = clusters_arr.size
    w = counts.astype(float) / np.sum(counts)
    mu = np.zeros((n_clusters, n_features))
    sigma = np.zeros((n_clusters, n_features, n_features))
    for cluster_idx, cluster in enumerate(clusters_arr):
        idx = (labels == cluster)
        mu[cluster_idx, :] = np.mean(X[idx, :], axis=0)
        sigma[cluster_idx, :, :] = np.cov(X[idx, :].T)
    return log_likelihood(X, w, mu, sigma)
