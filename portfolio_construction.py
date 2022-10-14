import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from data_and_denoising import *
from clustering import *

def allocate_cvo(cov, mu_vec=None):
    
    # Calculating the inverse covariance matrix
    inv_cov = np.linalg.inv(cov)
    
    # Generating a vector of size of the inverted covariance matrix
    ones = np.ones(shape=(inv_cov.shape[0], 1))
    
    if mu_vec is None:  # To output the minimum variance portfolio
        mu_vec = ones
    
    # Calculating the analytical solution using CVO - weights
    w_cvo = np.dot(inv_cov, mu_vec)
    w_cvo /= np.dot(mu_vec.T, w_cvo)
    
    return w_cvo    

def optPort_nco(cov, mu=None, maxNumClusters=None):
    cov = pd.DataFrame(cov)
    if mu is not None:
        mu = pd.Series(mu[:,0])
    
    corr1 = cov2corr(cov)
    
    # Optimal partition of clusters (step 1)
    corr1, clstrs, _ = clusterKMeansBase(corr1, maxNumClusters, n_init=10)
    #wIntra = pd.DataFrame(0, index=cov.index, columns=clstrs.keys())
    w_intra_clusters = pd.DataFrame(0, index=cov.index, columns=clstrs.keys())
    for i in clstrs:
        cov_cluster = cov.loc[clstrs[i], clstrs[i]].values
        if mu is None:
            mu_cluster = None
        else: 
            mu_cluster = mu.loc[clstrs[i]].values.reshape(-1,1)
        
        #Long/Short
        #w_intra_clusters.loc[clstrs[i],i] = mc.optPort(cov_cluster, mu_cluster).flatten()
        
        # Long only: Estimating the Convex Optimization Solution in a cluster (step 2)
        w_intra_clusters.loc[clstrs[i], i] = allocate_cvo(cov_cluster, mu_cluster).flatten()        
    
    cov_inter_cluster = w_intra_clusters.T.dot(np.dot(cov, w_intra_clusters)) #reduce covariance matrix
    mu_inter_cluster = (None if mu is None else w_intra_clusters.T.dot(mu))
    
    #Long/Short
    #w_inter_clusters = pd.Series(mc.optPort(cov_inter_cluster, mu_inter_cluster).flatten(), index=cov_inter_cluster.index)
    # Long only: Optimal allocations across the reduced covariance matrix (step 3)
    w_inter_clusters = pd.Series(allocate_cvo(cov_inter_cluster, mu_inter_cluster).flatten(), index=cov_inter_cluster.index)    
    
    # Final allocations - dot-product of the intra-cluster and inter-cluster allocations (step 4)
    nco = w_intra_clusters.mul(w_inter_clusters, axis=1).sum(axis=1).values.reshape(-1,1)
    return nco