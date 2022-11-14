import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
from denoising_and_detoning import *
from kneed import KneeLocator
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from scipy.optimize import minimize


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

def optPortMVO(returns, nrIter = 100000, res = 4, rf = 0):

    stocks = returns.columns
    resolution = res
    lower_bound = 10**(-resolution)
    upper_bound = 0.4
    simulated_weights = []
    simulated_portfolios = np.zeros((3, nrIter))
    risk_free_rate = rf
    
    returns_mean = returns.mean()
    returns_covariance_matrix = returns.cov()
    
    for index in tqdm(range(nrIter)):

    # Randomly creating the array of weight and then normalizing such that the sum equals 1
    #
        #select random weights for portfolio holdings
        weights_ = np.array(np.random.random(len(stocks)))
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
    highest_sharpe = simulated_portfolios_df.iloc[highest_sharpe_position]
    highest_sharpe_weights = simulated_weights[highest_sharpe_position]

    return highest_sharpe_weights

def optPort_nco(df, cov, numClusters = 10, threshold = 0.5, nrIter = 3000, n_init = 10):

    print("\nNested Clustering Algorithm 2.1.2\n")
    print("Processing input data....")
    
    #data perparation
    corr = pd.DataFrame(cov)
    #corr = cov2corr(cov)
    df_ = df
    df_.columns = list(range(len(corr.columns)))

    #clustering
    print("Clustering algorithm in progress...")
    corr1, clstrs, _ = clusterKMeansBase(corr, numClusters, n_init=n_init)

    #w intra cluster
    w_intra_clusters = pd.DataFrame(0, index=corr.index, columns=clstrs.keys())

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