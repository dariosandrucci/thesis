import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
from denoising_and_detoning import *

def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10, debug=False):
    corr0[corr0 > 1] = 1
    dist_matrix = ((1-corr0.fillna(0))/2.)**.5
    silh_coef_optimal = pd.Series(dtype='float64') #observations matrixs
    kmeans, stat = None, None
    maxNumClusters = min(maxNumClusters, int(np.floor(dist_matrix.shape[0]/2)))
    print("maxNumClusters"+str(maxNumClusters))
    for init in range(0, n_init):
    #The [outer] loop repeats the first loop multiple times, thereby obtaining different initializations. Ref: de Prado and Lewis (2018)
    #DETECTION OF FALSE INVESTMENT STRATEGIES USING UNSUPERVISED LEARNING METHODS
        for num_clusters in tqdm(range(2, maxNumClusters+1)):
            #(maxNumClusters + 2 - num_clusters) # go in reverse order to view more sub-optimal solutions
            kmeans_ = KMeans(n_clusters=num_clusters, n_init=10) #, random_state=3425) #n_jobs=None #n_jobs=None - use all CPUs
            kmeans_ = kmeans_.fit(dist_matrix)
            silh_coef = silhouette_samples(dist_matrix, kmeans_.labels_)
            stat = (silh_coef.mean()/silh_coef.std(), silh_coef_optimal.mean()/silh_coef_optimal.std())

            # If this metric better than the previous set as the optimal number of clusters
            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh_coef_optimal = silh_coef
                kmeans = kmeans_
                if debug==True:
                    print(kmeans)
                    print(stat)
                    silhouette_avg = silhouette_score(dist_matrix, kmeans_.labels_)
                    print("For n_clusters ="+ str(num_clusters)+ "The average silhouette_score is :"+ str(silhouette_avg))
                    print("********")
    
    newIdx = np.argsort(kmeans.labels_)
    print(kmeans.labels_)

    corr1 = corr0.iloc[newIdx] #reorder rows
    corr1 = corr1.iloc[:, newIdx] #reorder columns

    clstrs = {i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() for i in np.unique(kmeans.labels_)} #cluster members
    silh_coef_optimal = pd.Series(silh_coef_optimal, index=dist_matrix.index)
    
    return corr1, clstrs, silh_coef_optimal

def optPort(cov, mu = None):
    inv = np.linalg.inv(cov) #The precision matrix: contains information about the partial correlation between variables,
    #  the covariance between pairs i and j, conditioned on all other variables (https://www.mn.uio.no/math/english/research/projects/focustat/publications_2/shatthik_barua_master2017.pdf)
    ones = np.ones(shape = (inv.shape[0], 1)) # column vector 1's
    if mu is None: 
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w) # def: w = w / sum(w) ~ w is column vector
    
    return w

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

def optPort_nco(cov, mu=None, maxNumClusters=10):
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
        w_intra_clusters.loc[clstrs[i],i] = optPort(cov_cluster, mu_cluster).flatten()
        
        # Long only: Estimating the Convex Optimization Solution in a cluster (step 2)
        #w_intra_clusters.loc[clstrs[i], i] = allocate_cvo(cov_cluster, mu_cluster).flatten()        
    
    cov_inter_cluster = w_intra_clusters.T.dot(np.dot(cov, w_intra_clusters)) #reduce covariance matrix
    mu_inter_cluster = (None if mu is None else w_intra_clusters.T.dot(mu))
    
    #Long/Short
    w_inter_clusters = pd.Series(optPort(cov_inter_cluster, mu_inter_cluster).flatten(), index=cov_inter_cluster.index)
    # Long only: Optimal allocations across the reduced covariance matrix (step 3)
    #w_inter_clusters = pd.Series(allocate_cvo(cov_inter_cluster, mu_inter_cluster).flatten(), index=cov_inter_cluster.index)    
    
    # Final allocations - dot-product of the intra-cluster and inter-cluster allocations (step 4)
    nco = w_intra_clusters.mul(w_inter_clusters, axis=1).sum(axis=1).values.reshape(-1,1)
    nco = nco.reshape(-1)
    return nco