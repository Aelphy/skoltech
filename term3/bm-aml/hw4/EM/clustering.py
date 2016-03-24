import numpy as np
import scipy as sp
import scipy.stats
from sklearn.linear_model.base import BaseEstimator
from utils import compute_labels, log_likelihood, log_likelihood_from_labels 

class Random(BaseEstimator):
    def __init__(self, n_clusters, n_init=10):
        self.n_clusters = n_clusters
        self.n_init = n_init

    def fit(self, X):
        n_objects = X.shape[0]
        best_log_likelihood = float('-inf')
        for i in range(self.n_init):
            centers_idx = np.random.choice(n_objects, size=self.n_clusters, replace=False)
            mu = X[centers_idx, :]
            labels = compute_labels(X, mu)
            ll = log_likelihood_from_labels(X, labels)
            if ll > best_log_likelihood:
                best_log_likelihood = ll
                self.cluster_centers_ = mu.copy()
                self.labels_ = labels
                
                
class EMInit(BaseEstimator):
    def __init__(self, n_clusters, max_iter, min_covar = 0.001, tol = 0.001, logging = False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.min_covar = min_covar
        self.tol = tol
        self.logging = logging
        
        if logging:
            self.logs = {
                'log_likelihood' : [],
                'labels' : [],
                'w' : [],
                'mu' : [],
                'sigma' : []
            }
            
    def e_step(self, X):
        n_objects, n_features = X.shape
        g = np.zeros((n_objects, self.n_clusters))
        
        for cluster in range(self.n_clusters):
            g[:, cluster] = np.log(self.w_[cluster]) + \
                            sp.stats.multivariate_normal.logpdf(X, 
                                                                self.cluster_centers_[cluster, :],
                                                                self.covars_[cluster, :, :] + self.min_covar * np.eye(n_features)
                                                               )
        norm_const = sp.misc.logsumexp(g, axis=1)
        
        for cluster in range(self.n_clusters):
            g[:, cluster] -= norm_const
            
        return g
    
    def m_step(self, X, g):
        n_objects, n_features = X.shape
        
        self.w_ = np.zeros(self.n_clusters)
        self.covars_ = np.zeros((self.n_clusters, n_features, n_features))
        self.cluster_centers_ = np.zeros((self.n_clusters, n_features))
        
        for cluster in range(self.n_clusters):
            g_k = np.exp(g[:, cluster])
            N_k = np.sum(g_k)
            self.w_[cluster] = N_k / n_objects
            self.cluster_centers_[cluster, :] = g_k.T.dot(X) / N_k
            cluster_center_k = self.cluster_centers_[cluster, :]
            
            for i in range(n_objects):
                self.covars_[cluster, :, :] += g_k[i] * np.outer((X[i] - cluster_center_k), (X[i] - cluster_center_k).T) / N_k
            
    def fit(self, X):
        n_objects, n_features = X.shape
        
        self.covars_  = np.zeros((self.n_clusters, n_features, n_features))
        self.w_ = np.tile(1.0 / self.n_clusters, self.n_clusters)

        centers_idx = np.random.choice(n_objects, size = self.n_clusters, replace = False)
        self.cluster_centers_ = X[centers_idx, :]
 
        for cluster in range(self.n_clusters):
            self.covars_[cluster :, :] = np.eye(n_features)
            
        self.ll = log_likelihood(X, self.w_, self.cluster_centers_, self.covars_)
        
        for i in range(self.max_iter):
            if self.logging:
                self.logs['log_likelihood'].append(log_likelihood(X, self.w_, self.cluster_centers_, self.covars_))
                self.logs['labels'].append(compute_labels(X, self.cluster_centers_))
                self.logs['w'].append(self.w_)
                self.logs['mu'].append(self.cluster_centers_)
                self.logs['sigma'].append(self.covars_)
                    
            ll_new = log_likelihood(X, self.w_, self.cluster_centers_, self.covars_)
            
            if i > 0 and abs(ll_new - self.ll) < self.tol:
                break
            else:
                g = self.e_step(X)
                self.m_step(X, g)
                self.ll = ll_new
                
        self.labels_ = compute_labels(X, self.cluster_centers_)

        
class EM(BaseEstimator):
    def __init__(self, n_clusters, max_iter, n_init = 10, min_covar = 0.001, tol = 0.001, logging = False):
        self.inits = [None] * n_init
        self.logging = logging
        
        for i in range(n_init):
            self.inits[i] = EMInit(n_clusters, max_iter, min_covar, tol, logging)
            
    def fit(self, X):
        ll = []
        
        for init in self.inits:
            init.fit(X)
            ll.append(init.ll)
            
        best_init = np.argmax(ll)
        
        if self.logging:
            self.logs = self.inits[best_init].logs
        
        self.w_ = self.inits[best_init].w_
        self.covars_ = self.inits[best_init].covars_
        self.cluster_centers_ = self.inits[best_init].cluster_centers_
        self.labels_ = self.inits[best_init].labels_
        
        
class SoftKMeansInit:
    def __init__(self, n_clusters, max_iter, min_covar = 0.001, tol = 0.001, logging = False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.min_covar = min_covar
        self.tol = tol
        self.logging = logging
        
        if logging:
            self.logs = {
                'log_likelihood' : [],
                'labels' : [],
                'w' : [],
                'mu' : [],
                'sigma' : []
            }
 
    def e_step(self, X):
        n_objects, n_features = X.shape
        g = np.zeros((n_objects, self.n_clusters))
        
        for cluster in range(self.n_clusters):
            g[:, cluster] = np.log(self.w_[cluster]) + \
                            sp.stats.multivariate_normal.logpdf(X, 
                                                                self.cluster_centers_[cluster, :],
                                                                self.covars_[cluster, :, :] + self.min_covar * np.eye(n_features)
                                                               )
        norm_const = sp.misc.logsumexp(g, axis=1)
        
        for cluster in range(self.n_clusters):
            g[:, cluster] -= norm_const
            
        return g
        
    def m_step(self, X, g):
        n_objects, n_features = X.shape
        
        self.w_ = np.zeros(self.n_clusters)
        self.covars_ = np.zeros((self.n_clusters, n_features, n_features))
        self.cluster_centers_ = np.zeros((self.n_clusters, n_features))
        
        for cluster in range(self.n_clusters):
            g_k = np.exp(g[:, cluster])
            N_k = np.sum(g_k)
            self.w_[cluster] = N_k / n_objects
            self.cluster_centers_[cluster, :] = g_k.T.dot(X) / N_k
            
            self.covars_[cluster, :, :] = np.eye(n_features)
            
    def fit(self,X):
        n_objects, n_features = X.shape
        
        self.covars_ = np.zeros((self.n_clusters, n_features, n_features))
        self.w_ = np.tile(1.0 / self.n_clusters, self.n_clusters)

        centers_idx = np.random.choice(n_objects, size = self.n_clusters, replace = False)
        self.cluster_centers_ = X[centers_idx, :]
 
        for cluster in range(self.n_clusters):
            self.covars_[cluster :, :] = np.eye(n_features)
            
        self.ll = log_likelihood(X, self.w_, self.cluster_centers_, self.covars_)
        
        for i in range(self.max_iter):
            if self.logging:
                self.logs['log_likelihood'].append(log_likelihood(X, self.w_, self.cluster_centers_, self.covars_))
                self.logs['labels'].append(compute_labels(X, self.cluster_centers_))
                self.logs['w'].append(self.w_)
                self.logs['mu'].append(self.cluster_centers_)
                self.logs['sigma'].append(self.covars_)
                    
            ll_new = log_likelihood(X, self.w_, self.cluster_centers_, self.covars_)
            
            if i > 0 and abs(ll_new - self.ll) < self.tol:
                break
            else:
                g = self.e_step(X)
                self.m_step(X, g)
                self.ll = ll_new
                
        self.labels_ = compute_labels(X, self.cluster_centers_)


class SoftKMeans(BaseEstimator):
    def __init__(self, n_clusters, max_iter, n_init = 10, min_covar = 0.001, tol = 0.001, logging = False):
        self.inits = [None] * n_init
        self.logging = logging
        
        for i in range(n_init):
            self.inits[i] = SoftKMeansInit(n_clusters, max_iter, min_covar, tol, logging)
            
    def fit(self, X):
        ll = []
        
        for init in self.inits:
            init.fit(X)
            ll.append(init.ll)
            
        best_init = np.argmax(ll)
        
        if self.logging:
            self.logs = self.inits[best_init].logs

        self.w_ = self.inits[best_init].w_
        self.covars_ = self.inits[best_init].covars_
        self.cluster_centers_ = self.inits[best_init].cluster_centers_
        self.labels_ = self.inits[best_init].labels_   


class KMeansInit:
    def __init__(self, n_clusters, max_iter, min_covar = 0.001, tol = 0.001, logging = False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.min_covar = min_covar
        self.tol = tol
        self.logging = logging
        
        if logging:
            self.logs = {
                'log_likelihood' : [],
                'labels' : [],
                'w' : [],
                'mu' : [],
                'sigma' : []
            }
 
    def e_step(self, X):    
        n_objects, n_features = X.shape
        p = np.zeros((n_objects, self.n_clusters))
        g = np.zeros((n_objects, self.n_clusters))
        
        for cluster in range(self.n_clusters):
            p[:, cluster] = sp.stats.multivariate_normal.logpdf(X, 
                                                                self.cluster_centers_[cluster, :],
                                                                self.covars_[cluster, :, :]
                                                               )
        norm_const = sp.misc.logsumexp(p, axis = 1)
        
        for cluster in range(self.n_clusters):
            p[:, cluster] -= norm_const
            
        labels = np.argmax(p, axis = 1)
                
        for index, prob in enumerate(labels):
            g[index, prob] = 1
            
        return g
        
    def m_step(self, X, g):
        n_objects, n_features = X.shape
        
        self.w_ = np.tile(1.0 / self.n_clusters, self.n_clusters)
        self.covars_ = np.zeros((self.n_clusters, n_features, n_features))
        self.cluster_centers_ = np.zeros((self.n_clusters, n_features))
        
        for cluster in range(self.n_clusters):
            g_k = g[:, cluster]
            N_k = np.sum(g_k)
            
            if N_k > 0:
                self.cluster_centers_[cluster, :] = g_k.T.dot(X) / N_k
            
            self.covars_[cluster, :, :] = np.eye(n_features)
            
    def fit(self,X):
        n_objects, n_features = X.shape
        
        self.covars_ = np.zeros((self.n_clusters, n_features, n_features))
        self.w_ = np.tile(1.0 / self.n_clusters, self.n_clusters)

        centers_idx = np.random.choice(n_objects, size = self.n_clusters, replace = False)
        self.cluster_centers_ = X[centers_idx, :]
 
        for cluster in range(self.n_clusters):
            self.covars_[cluster :, :] = np.eye(n_features)
            
        self.ll = log_likelihood(X, self.w_, self.cluster_centers_, self.covars_)
        
        for i in range(self.max_iter):
            if self.logging:
                self.logs['log_likelihood'].append(log_likelihood(X, self.w_, self.cluster_centers_, self.covars_))
                self.logs['labels'].append(compute_labels(X, self.cluster_centers_))
                self.logs['w'].append(self.w_)
                self.logs['mu'].append(self.cluster_centers_)
                self.logs['sigma'].append(self.covars_)
                    
            ll_new = log_likelihood(X, self.w_, self.cluster_centers_, self.covars_)
            
            if i > 0 and abs(ll_new - self.ll) < self.tol:
                break
            else:
                g = self.e_step(X)
                self.m_step(X, g)
                self.ll = ll_new
                
        self.labels_ = compute_labels(X, self.cluster_centers_)


class KMeans(BaseEstimator):
    def __init__(self, n_clusters, max_iter, n_init = 10, min_covar = 0.001, tol = 0.001, logging = False):
        self.inits = [None] * n_init
        self.logging = logging
        
        for i in range(n_init):
            self.inits[i] = KMeansInit(n_clusters, max_iter, min_covar, tol, logging)
            
    def fit(self, X):
        ll = []
        
        for init in self.inits:
            init.fit(X)
            ll.append(init.ll)
            
        best_init = np.argmax(ll)

        if self.logging:
            self.logs = self.inits[best_init].logs

        self.w_ = self.inits[best_init].w_
        self.covars_ = self.inits[best_init].covars_
        self.cluster_centers_ = self.inits[best_init].cluster_centers_
        self.labels_ = self.inits[best_init].labels_