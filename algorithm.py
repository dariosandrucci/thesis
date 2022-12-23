#---PACKAGES---#

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
from denoising_and_detoning import *
from data_engineering import *
from portfolio_and_backtesting import *
from kneed import KneeLocator
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta

#---FUNCTIONS---#

#ELBOW METHOD
#Use: Find optimal number of clusters based for and input matrix on the basis of the elbow method.

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

#SILHOUETTE SCORE METHOD
#Use: Find optimal number of clusters based for and input matrix on the basis of the silhouette score method.

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

#DAVIEs-BOULDIN METHOD
#Use: Find optimal number of clusters based for and input matrix on the basis of the davies-bouldin method.

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

#KMEANS
#Use: Takes in an input matrix and outputs a map of cluster assignments.

def clusterKMeansBase(matrix, nrClusters=10, n_init=10):
    
    #find range of possible clusters with SSE, Silhouette, DBScore
    nr_clusters = np.array(nrClusters)
    clst = np.unique(nr_clusters)
    silh_coef_optimal = pd.Series(dtype='float64')

    for init in range(0, n_init):
    #The [outer] loop repeats the first loop multiple times, thereby obtaining different initializations. Ref: de Prado and Lewis (2018)
    #DETECTION OF FALSE INVESTMENT STRATEGIES USING UNSUPERVISED LEARNING METHODS
        for num_clusters in clst:
            kmeans_ = KMeans(n_clusters=num_clusters, init = "random", n_init=10, max_iter=300)
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

#OMEGA SCORE
#Use: Calculates the omega score based on and returns input and a given threshold

def omega(returns, threshold):
    # Get excess return
    returns_exc = returns - threshold
    # Get sum of all values excess return above
    dfPositiveSum = sum(returns_exc[returns_exc > 0])
    # Get sum of all values excess return below
    dfNegativeSum = sum(returns_exc[returns_exc < 0])
    omega = dfPositiveSum/(-dfNegativeSum)
    return omega

#OMEGA PORTFOLIO OPTIMIZER
#Use: Returns the optimal weights for a given dataset based on the omega score and a set threshold

def optPortOmega(df, threshold, iterations = 25000, constraint = "Long-Only"):

    df_ = np.array(df)
    weights = np.array([])
    omega_score = -9999.0
    dailyThreshold = (threshold + 1) ** np.sqrt(1/252) - 1

    for i in range(iterations):

        #select random weights for portfolio holdings
        if constraint == "Long-Only":
            weights_ = np.array(np.random.random(len(df_[0])))
        else:
            weights_ = np.array(np.random.uniform(high = 10.0, low = -1.0, size = len(df_[0])))
        #rebalance weights to sum to 1
        weights_ /= np.sum(weights_)

        #calculate returns
        returns = np.array([np.average(x, weights = weights_) for x in df_])

        omega_score_ = omega(returns, dailyThreshold)

        if omega_score_ > omega_score:
            weights = weights_
            omega_score = omega_score_
    
    return weights 

#PORTFOLIO MVO
#Use: Return the optimal weights for a given dataframe based on the mean-variance optimization

def optPortMVO(returns, cov = pd.DataFrame(), nrIter = 100000, res = 4, rf = 0, cov_in = False, constraint = "Long-Only"):

    returns.columns = list(range(len(returns.columns)))
    stocks = returns.columns
    resolution = res
    simulated_weights = []
    simulated_portfolios = np.zeros((3, nrIter))
    risk_free_rate = rf
    
    returns_mean = returns.mean()
    if cov_in == False:
        returns_covariance_matrix = returns.cov()
    else:
        returns_covariance_matrix = pd.DataFrame(cov, columns = returns.columns, index = returns.columns)

    for index in range(nrIter):

    # Randomly creating the array of weight and then normalizing such that the sum equals 1
    #
        #select random weights for portfolio holdings
        if constraint == "Long-Only":
            weights_ = np.array(np.random.random(len(stocks)))
        else:
            weights_ = np.array(np.random.uniform(high = 10.0, low = -1.0, size = len(stocks)))
        #rebalance weights to sum to 1
        weights_ /= np.sum(weights_)

        simulated_weights.append(weights_)
        
        # Computing the return and volatility of the portfolio with those weights
        portfolio_return = np.sum(returns_mean.values * weights_ * 252)
        portfolio_volatility = np.sqrt(np.dot(weights_.T, np.dot(returns_covariance_matrix.values, weights_))* 252**2)

        # Store results of the simulation
        simulated_portfolios[0, index] = portfolio_return 
        simulated_portfolios[1, index] = portfolio_volatility
        simulated_portfolios[2, index] = (portfolio_return - risk_free_rate) / portfolio_volatility

    simulated_portfolios_df = pd.DataFrame(simulated_portfolios.T,columns=['retrn','stdv','sharpe'])
    highest_sharpe_position = simulated_portfolios_df['sharpe'].idxmax()
    #highest_sharpe = simulated_portfolios_df.iloc[highest_sharpe_position]
    highest_sharpe_weights = simulated_weights[highest_sharpe_position]

    return highest_sharpe_weights

#PORTFOLIO NCO
#Use: Return the optimal weights for a given dataframe based on the Nested Clustering Optimization frame work 

def optPort_nco(df, cov, numClusters = 10, threshold = 0.5, nrIter = 3000, n_init = 10, score = "omega", constraint = "Long-Only", ret_clust = False):

    #print("Nested clustering algorithm calculating ...")
    #data perparation
    corr = pd.DataFrame(cov)
    #corr = cov2corr(cov)
    df_ = df
    df_.columns = list(range(len(corr.columns)))

    #clustering
    corr1, clstrs, _ = clusterKMeansBase(corr, numClusters, n_init=n_init)

    if score == "omega":
        #w intra cluster
        w_intra_clusters = pd.DataFrame(0, index=corr.index, columns=clstrs.keys())

        for i in clstrs:
            w_intra_clusters.loc[clstrs[i],i] = optPortOmega(df_.loc[:,clstrs[i]], threshold, nrIter, constraint = constraint).flatten()      

        #w inter cluster
        intra_returns = pd.DataFrame(0, index = df.index, columns = list(range(len(clstrs))))

        for i in range(len(clstrs)):
            intra_returns.loc[:,i] = pd.Series(df_.apply(lambda row: np.average(row, weights = w_intra_clusters.loc[:,i]), axis=1))

        w_inter_clusters = pd.Series(optPortOmega(intra_returns, threshold, nrIter, constraint = constraint).flatten(), index=intra_returns.columns)

    elif score == "sharpe":
                #w intra cluster
        w_intra_clusters = pd.DataFrame(0, index=corr.index, columns=clstrs.keys())

        for i in clstrs:
            w_intra_clusters.loc[clstrs[i],i] = optPortMVO(df_.loc[:,clstrs[i]], nrIter = nrIter, cov_in = False, constraint = constraint).flatten()      

        #w inter cluster
        intra_returns = pd.DataFrame(0, index = df.index, columns = list(range(len(clstrs))))

        for i in range(len(clstrs)):
            intra_returns.loc[:,i] = pd.Series(df_.apply(lambda row: np.average(row, weights = w_intra_clusters.loc[:,i]), axis=1))

        w_inter_clusters = pd.Series(optPortMVO(intra_returns, cov_in = False, constraint = constraint).flatten(), index=intra_returns.columns)

    else:
        raise NameError('Please enter valid optimization score!')

    nco = w_intra_clusters.mul(w_inter_clusters, axis=1).sum(axis=1).values.reshape(-1,1)
    nco = nco.reshape(-1)

    #print("Calculations completed sucessful!")

    if ret_clust:
        return nco, clstrs

    else:
        return nco

#Functions to build the risk parity portfolio
#Use: Return the optimal weights for a given dataframe based on the risk-parity optimization
#Based on public RPP repository: https://gist.github.com/FJR2/f91e9660044a482ffefdcf4ce9cc98f5#file-risk_parity_strategy-py

TOLERANCE = 1e-11

def _allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

    # It returns the risk of the weights distribution
    return portfolio_risk


def _assets_risk_contribution_to_allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) \
        / portfolio_risk

    # It returns the contribution of each asset to the risk of the weights
    # distribution
    return assets_risk_contribution


def _risk_budget_objective_error(weights, args):

    # The covariance matrix occupies the first position in the variable
    covariances = args[0]

    # The desired contribution of each asset to the portfolio risk occupies the
    # second position
    assets_risk_budget = args[1]

    # We convert the weights to a matrix
    weights = np.matrix(weights)

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = \
        _assets_risk_contribution_to_allocation_risk(weights, covariances)

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = \
        np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))

    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = \
        sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]

    # It returns the calculated error
    return error


def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):

    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})

    # Optimisation process in scipy
    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})

    # Recover the weights from the optimised object
    weights = optimize_result.x

    # It returns the optimised weights
    return weights

def optPortRPP(returns):

    # We calculate the covariance matrix
    covariances = returns.cov().values

    # The desired contribution of each asset to the portfolio risk: we want all
    # asset to contribute equally
    assets_risk_budget = [1 / returns.shape[1]] * returns.shape[1]

    # Initial weights: equally weighted
    init_weights = [1 / returns.shape[1]] * returns.shape[1]

    # Optimisation process of weights
    weights = \
        _get_risk_parity_weights(covariances, assets_risk_budget, init_weights)

    # Convert the weights to a pandas Series
    #weights = pd.Series(weights, index=returns.columns, name='weight')
    weights = np.array(weights)

    # It returns the optimised weights
    return weights

#P-Parameter Tuner
#Use: Takes in a dataframe and returns the optimal parameter

def p_tuner(train_set, val_set, corr, th = 0.074, iter = 1000):

    min_ports = []
    ps = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]

    for p in tqdm(ps):
        min_matrix_ = np.array(pairwise_distances(corr, metric = "minkowski", p = p))
        nr_clusters_sse = sse(min_matrix_, graph = False)
        w_ = optPort_nco(train_set, min_matrix_, nr_clusters_sse, th, iter, 5)
        name = f"P = {p}; C = {nr_clusters_sse}"
        port = Portfolio(name, val_set, w_)
        min_ports.append(port)

    sharpes = []

    for port in min_ports:
        min_rets = port.returns
        sr = sharpe_ratio(min_rets)
        sharpes.append(sr)
    
    pv = ps[sharpes.index(max(sharpes))]

    return pv

#Portfolio NCO Rebalanced
#Use: Takes in a dataframe and returns the returns of a opimal NCO-based portfolio with mothly rebalancing

def optPort_nco_RB(df, investment_start:dt, nrIter = 5000, numClusters = 3, train_period = "1y", intervals = "monthly", ret_clust = False, th = 0.074, tuneIter = 1000):

    #find initial dates and creates variables
    returns = pd.Series(0, index = df.index)
    curren_start = investment_start
    investing = True
    clusters = [] 
    #print("Calculations for rebalanced portfolio in progress...")

    #loop over test set and get returns
    while investing == True:

        #set dates
        train_start = curren_start - relativedelta(months = 6)
        train_end = curren_start
        test_end = curren_start + relativedelta(months = 1)

        #devide data
        train_set = df.loc[train_start : train_end]
        train_set, val_set = testTrainSplit(train_set, validation_set = False, w_tt = [0.5,0.5])
        test_set = df.loc[curren_start : test_end]

        #test if test set is not empty
        if len(test_set) != 0:
            pass
        else:
            investing = False
            break
        
        #phase1: p parameter calculation
        corr_train = denoise_and_detone(train_set)
        pv = p_tuner(train_set, val_set, corr_train, th = th, iter = tuneIter)

        #phase2: calculate weights based on val set
        corr_val = denoise_and_detone(val_set)
        min_matrix = np.array(pairwise_distances(corr_val, metric = "minkowski", p = pv))

        if ret_clust:
            w, clst = optPort_nco(val_set, min_matrix, numClusters = numClusters, nrIter = nrIter, n_init = 5, score = "omega", constraint = "Long-Only", ret_clust = True, threshold = th)
            clusters.append(clst)
        else:
            w = optPort_nco(val_set, min_matrix, numClusters = numClusters, nrIter = nrIter, n_init = 5, score = "omega", constraint = "Long-Only", threshold = th)

        #invest
        returns_ = pd.Series(test_set.apply(lambda row: np.average(row, weights = w), axis=1), index = test_set.index)

        #add to data DataFrame
        returns.loc[test_set.index] = returns_

        #reset current date
        curren_start = curren_start + relativedelta(months = 1)

    returns = returns[returns != 0.0]
    #print("Calculations completed sucessfuly!")
    if ret_clust:
        return returns, clusters
    else:
        return returns