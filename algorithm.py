import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
from denoising_and_detoning import *
from kneed import KneeLocator
from sklearn.metrics import davies_bouldin_score

def sse(matrix, maxClusters = 10, graph = True):
    sse = []
    for k in range(1, maxClusters + 1):
        kmeans = KMeans(n_clusters=k,  init= 'k-means++', n_init= 10, max_iter=300)
        kmeans.fit(matrix)
        sse.append(kmeans.inertia_)

    y = sse
    x = range(1, len(y)+1)

    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    
    if graph:
        kn.plot_knee()

    return kn.knee

def sil_score(matrix, maxClusters = 10, graph = True):
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []

    # Notice you start at 2 clusters for silhouette coefficient, because it raises an error for 1 cluster
    for k in range(2, maxClusters +1):
        kmeans = KMeans(n_clusters=k,  init= 'k-means++', n_init= 10, max_iter=300)
        kmeans.fit(matrix)
        score = silhouette_score(matrix, kmeans.labels_)
        silhouette_coefficients.append(score)

    nr_clusters = silhouette_coefficients.index(max(silhouette_coefficients)) + 2

    if graph:
        plt.plot(range(2, maxClusters + 1), silhouette_coefficients)
        plt.xticks(range(2, maxClusters + 1))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Coefficient")

    return nr_clusters

def db_score(matrix, maxClusters = 10, graph = True):
    DB_score = []

    for k in range(2, maxClusters + 1):
        kmeans = KMeans(n_clusters=k,  init= 'k-means++', n_init= 10, max_iter=300)
        kmeans.fit(matrix)
        score = davies_bouldin_score(matrix, kmeans.labels_)
        DB_score.append(score)
    
    clusters = DB_score.index(min(DB_score)) + 2

    if graph:
        plt.plot(range(2, maxClusters + 1), DB_score)
        plt.xticks(range(2, maxClusters + 1))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Davies Bouldin Score")

    return clusters

def clusterKMeansBase(matrix, nrClusters=10, n_init=10):
    
    #find range of possible clusters with SSE, Silhouette, DBScore
    nr_clusters = np.array(nrClusters)
    clst = np.unique(nr_clusters)
    silh_coef_optimal = pd.Series(dtype='float64')

    for init in range(0, n_init):
    #The [outer] loop repeats the first loop multiple times, thereby obtaining different initializations. Ref: de Prado and Lewis (2018)
    #DETECTION OF FALSE INVESTMENT STRATEGIES USING UNSUPERVISED LEARNING METHODS
        for num_clusters in tqdm(clst):
            kmeans_ = KMeans(n_clusters=num_clusters, init = "k-means++", n_init=10, max_iter=300)
            kmeans_ = kmeans_.fit(matrix)
            silh_coef = silhouette_samples(matrix, kmeans_.labels_)
            stat = (silh_coef.mean()/silh_coef.std(), silh_coef_optimal.mean()/silh_coef_optimal.std())

            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh_coef_optimal = silh_coef
                kmeans = kmeans_
                
    newIdx = np.argsort(kmeans.labels_)
    #print(kmeans.labels_)

    matrix1 = matrix.iloc[newIdx] #reorder rows
    matrix1 = matrix1.iloc[:, newIdx] #reorder columns

    clstrs = {i:matrix.columns[np.where(kmeans.labels_==i)[0]].tolist() for i in np.unique(kmeans.labels_)} #cluster members
    silh_coef_optimal = pd.Series(silh_coef_optimal, index= matrix.index)
    
    return matrix1, clstrs, silh_coef_optimal

def omega(returns, threshold):
    # Get excess return
    returns_exc = returns - threshold
    # Get sum of all values excess return above
    dfPositiveSum = sum(returns_exc[returns_exc > 0])
    # Get sum of all values excess return below
    dfNegativeSum = sum(returns_exc[returns_exc < 0])
    omega = dfPositiveSum/(-dfNegativeSum)
    return omega

def optPortOmega(df, threshold, iterations = 25000):

    df_ = np.array(df)
    weights = np.array([])
    omega_score = -9999.0
    dailyThreshold = (threshold + 1) ** np.sqrt(1/252) - 1

    for i in tqdm(range(iterations)):

        #select random weights for portfolio holdings
        weights_ = np.array(np.random.random(len(df_[0])))
        #rebalance weights to sum to 1
        weights_ /= np.sum(weights_)

        #calculate returns
        returns = np.array([np.average(x, weights = weights_) for x in df_])

        omega_score_ = omega(returns, dailyThreshold)

        if omega_score_ > omega_score:
            weights = weights_
            omega_score = omega_score_
    
    return weights 

def optPort_nco(df, cov, numClusters = 10, threshold = 0.5, nrIter = 3000, n_init = 10):

    print("\nNested Clustering Algorithm 2.1.2\n")
    print("Processing input data....")
    
    #data perparation
    cov = pd.DataFrame(cov)
    corr = cov2corr(cov)
    df_ = df
    df_.columns = list(range(len(cov.columns)))

    #clustering
    print("Clustering algorithm in progress...")
    corr1, clstrs, _ = clusterKMeansBase(corr, numClusters, n_init=n_init)

    #w intra cluster
    w_intra_clusters = pd.DataFrame(0, index=cov.index, columns=clstrs.keys())

    for i in clstrs:
        print(f"Calculating inter cluster weights for cluster {i+1} of {numClusters}...")
        w_intra_clusters.loc[clstrs[i],i] = optPortOmega(df_.loc[:,clstrs[i]], threshold, nrIter).flatten()      

    #w inter cluster
    print("Calculating inter cluster weights...")
    intra_returns = pd.DataFrame(0, index = df.index, columns = list(range(len(clstrs))))

    for i in range(len(clstrs)):
        intra_returns.loc[:,i] = pd.Series(df_.apply(lambda row: np.average(row, weights = w_intra_clusters.loc[:,i]), axis=1))

    w_inter_clusters = pd.Series(optPortOmega(intra_returns, threshold, nrIter).flatten(), index=intra_returns.columns)

    nco = w_intra_clusters.mul(w_inter_clusters, axis=1).sum(axis=1).values.reshape(-1,1)
    nco = nco.reshape(-1)

    print("Nested clustering algorithm completed sucessful!")

    return nco