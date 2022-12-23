#---PACKAGES---#

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas_datareader as pdr
import statsmodels.api as sm
from datetime import timedelta

#---FUNCTIONS---#

#LINEAR REGRESSION
#Use: The following two function implement a linear regression function for given inputs and returns the regression metrics

def regression(X, y):

    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    alpha = est2.params[0]
    beta = est2.params[1]
    p_alpha = est2.pvalues[0]
    p_beta = est2.pvalues[1]
    r2 = est2.rsquared

    return alpha, p_alpha, beta, p_beta, r2

def reg_analysis(returns, bm = "SPY"):
        
        start  = returns.index[0] - timedelta(days = 1)
        end  = returns.index[len(returns.index)-1]
        mkt_returns = pdr.get_data_yahoo([bm], start, end)["Close"]
        mkt_returns = mkt_returns.pct_change().dropna(axis = 0)
        X = pd.DataFrame(mkt_returns)
        y = pd.DataFrame(returns)

        comb = pd.merge(X, y, how = "inner", left_index = True, right_index = True)
        X = comb.iloc[:,0]
        y = comb.iloc[:,1]

        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        alpha = est2.params[0]
        beta = est2.params[1]
        p_alpha = est2.pvalues[0]
        p_beta = est2.pvalues[1]
        r2 = est2.rsquared

        return alpha, beta, p_alpha, p_beta, r2

#PERFORMANCE METRICS
#Use: The following functions return various performance metrics for an input of asset returns

def returns_annualized(returns):
    return returns.mean() * 252

def std_annualized(returns):
    return np.std(returns) * np.sqrt(252)

def sharpe_ratio(returns):
    ret = returns.mean() * 252
    vol = np.std(returns) * np.sqrt(252)
    return ret / vol

def mdd(returns):
    return (returns - returns.rolling(len(returns), min_periods=1).max()).min()

def information_ratio(returns, benchmark_returns = "SPY"):
    start  = returns.index[0] - timedelta(days = 1)
    end  = returns.index[len(returns.index)-1]
    mkt_returns = pdr.get_data_yahoo([benchmark_returns], start, end)["Close"]
    mkt_returns = mkt_returns.pct_change().dropna(axis = 0)
    X = pd.DataFrame(mkt_returns)
    y = pd.DataFrame(returns)

    comb = pd.merge(X, y, how = "inner", left_index = True, right_index = True)
    X = comb.iloc[:,0]
    y = comb.iloc[:,1]

    mkt_returns = np.array(X).flatten()
    returns = np.array(y)

    return_difference = np.subtract(returns,mkt_returns)
    volatility = np.std(return_difference) 
    information_ratio = (return_difference.mean() * 252) / (volatility * np.sqrt(252))
    return information_ratio

def sortino_ratio(returns, N = 252, rf = 0):
    mean = returns.mean() * N -rf
    std_neg = returns[returns<0].std()*np.sqrt(N)
    return mean/std_neg

#PERSISTANCE COUNTER
#Use: Takes in a list of cluster destributions and returns for each item (destribution map) in the list a map of equal cluster members in each cluster to the destributinon given in the initial list-item 

def persistance_counter(clusters):
    nr_clusters = len(clusters[0])
    output = {}
    base_clusters = {}

    for i in range(nr_clusters):
        output[i] = []
        base_clusters[i] = np.array(clusters[0][i])

    for rb in clusters:
        cl = []
        for nr, cluster in rb.items():
            cl.append(cluster)
        for nr, cluster in rb.items():
            l = 0
            for c in cl:
                int = len(np.intersect1d(base_clusters[nr],np.array(c)))
                if int > l:
                    l = int
                else:
                    pass
            output[nr].append(l)
    return output

#---FUNCTIONS---#

#PORTFOLIO
#Use: Implements a portfolio based input data and a set of weights for the included assets

class Portfolio:

    def __init__(self, portfolio_name , data, w):
        self.w = w
        self.tickers = data.columns
        self.data = data
        self.start = data.index[0]
        self.end = data.index[len(data.index)-1]
        self.name = portfolio_name

        #checks and get weights get weights in the right format
        #if self.w.shape[0] != len(self.tickers):
        #    self.w = self.w.reshape(-1)
        
        if len(w) != len(self.tickers):
            raise NameError('weights and tickers must have the same length!')

        #getting the portfolio returns
        self.returns = pd.Series(self.data.apply(lambda row: np.average(row, weights = self.w), axis=1), index = self.data.index)
        self.cum_returns = pd.Series(np.cumprod(1 + self.returns) - 1, index = self.data.index)

        #performance metrics
        self.total_return = self.cum_returns.tail(1).values[0]
        self.ereturn = self.returns.mean()
        self.ereturn_ann = self.ereturn * 252
        self.volatility = np.std(self.returns)
        self.volatility_ann = self.volatility * np.sqrt(252)
        self.sharpe_ratio = self.ereturn / self.volatility
        self.sharpe_ratio_ann = self.ereturn_ann / self.volatility_ann


    def performance_report(self, return_values = False, bm = "SPY"):
        pr = pd.DataFrame([[0]*9], index = [self.name],columns = ["ARet", "AVol", "Alpha", "p Alpha", "Beta","Sharpe", "Sortino", "IR", "MDD"])
        ret = returns_annualized(self.returns)
        vol = std_annualized(self.returns)
        sharpe = sharpe_ratio(self.returns)
        alpha, beta, pAlpha, _, _ = reg_analysis(self.returns, bm)
        sortino = sortino_ratio(self.returns)
        ir = information_ratio(self.returns, bm)
        mdds = mdd(self.returns)
        new_col = [round(ret,4), round(vol,4), round(alpha,4), round(pAlpha,4), round(beta,4), round(sharpe,4), round(sortino,4), round(ir,4), round(mdds,4)]
        pr.loc[self.name] = new_col

        if return_values == True:
            return self.name, new_col
        else:
            print(pr)
            return None

    def performance_plot(self):
        sns.lineplot(self.cum_returns)
        plt.show()
        return None

#PORTFOLIO
#Use: Implements a portfolio based of daily returns as input data

class Portfolio_Ret:

    def __init__(self, portfolio_name , returns):
        self.start = returns.index[0]
        self.end = returns.index[len(returns.index)-1]
        self.name = portfolio_name
        
        #getting the portfolio returns
        self.returns = returns
        self.cum_returns = pd.Series(np.cumprod(1 + self.returns) - 1, index = self.returns.index)

        #performance metrics
        self.total_return = self.cum_returns.tail(1).values[0]
        self.ereturn = self.returns.mean()
        self.ereturn_ann = self.ereturn * 252
        self.volatility = np.std(self.returns)
        self.volatility_ann = self.volatility * np.sqrt(252)
        self.sharpe_ratio = self.ereturn / self.volatility
        self.sharpe_ratio_ann = self.ereturn_ann / self.volatility_ann

    def performance_report(self, return_values = False, bm = "SPY"):
        pr = pd.DataFrame([[0]*9], index = [self.name],columns = ["ARet", "AVol", "Alpha", "p Alpha", "Beta","Sharpe", "Sortino", "IR", "MDD"])
        ret = returns_annualized(self.returns)
        vol = std_annualized(self.returns)
        sharpe = sharpe_ratio(self.returns)
        alpha, beta, pAlpha, _, _ = reg_analysis(self.returns, bm)
        sortino = sortino_ratio(self.returns)
        ir = information_ratio(self.returns, bm)
        mdds = mdd(self.returns)
        new_col = [round(ret,4), round(vol,4), round(alpha,4), round(pAlpha,4), round(beta,4), round(sharpe,4), round(sortino,4), round(ir,4), round(mdds,4)]
        pr.loc[self.name] = new_col


        if return_values == True:
            return self.name, new_col
        else:
            print(pr)
            return None

    def performance_plot(self):
        sns.lineplot(self.cum_returns)
        plt.show()
        return None

#PORTFOLIO
#Use: Enables performance benchmarking bewteen different implemented portfolio classes. Outputs graphs and tables.

class PortfolioBenchmarking:

    def __init__(self, portfolios, name = "Portfolios"):
        
        self.name = name
        self.portfolios = portfolios
    
        #input checks
        fi = portfolios[0].returns.index[0]
        li = portfolios[0].returns.index[len(portfolios[0].returns.index)-1]
        for port in self.portfolios:
            if fi == port.returns.index[0] and li == port.returns.index[len(port.returns.index)-1]:
                pass
            else:
                raise NameError('Portfolios must have the same time range!')

    def performance_report(self, return_value = True, withSPY = True, bm = "SPY"):

        if withSPY: 
            inds = ["SPY"]
        else: 
            inds = []

        for port in self.portfolios:
            inds.append(port.name)

        pr = pd.DataFrame([[0]*9], index = [inds],columns = ["ARet", "AVol", "Alpha", "p Alpha", "Beta","Sharpe", "Sortino", "IR", "MDD"])

        for port in self.portfolios:
            n, v = port.performance_report(True, bm = bm)
            pr.loc[n] = v

        if withSPY:
            start = self.portfolios[0].returns.index[0]
            end = self.portfolios[0].returns.index[len(self.portfolios[0].returns)-1]
            spx_returns = pdr.get_data_yahoo(["SPY"], start, end)["Close"]
            spx_returns = spx_returns.pct_change().dropna(axis = 0)

            ret = returns_annualized(spx_returns)
            vol = std_annualized(spx_returns)
            sharpe = sharpe_ratio(spx_returns)
            alpha, beta, pAlpha, _, _ = reg_analysis(spx_returns, bm)
            sortino = sortino_ratio(spx_returns)
            ir = information_ratio(spx_returns, bm)
            mdds = mdd(spx_returns)
            new_col = [round(ret,4), round(vol,4), round(alpha,4), round(pAlpha,4), round(beta,4), round(sharpe,4), round(sortino,4), round(ir,4), round(mdds,4)]
            pr.loc["SPY"] = new_col

        if return_value:
            return pr
        
        else:
            print(pr)

    def plot_performance(self, withSPY = True, palette = "RdYlBu_r", save = ""):

        #create returns dataframe
        self.cum_returns = pd.DataFrame(index = self.portfolios[0].cum_returns.index)

        for port in self.portfolios:
            ret = pd.DataFrame(port.cum_returns, index = port.cum_returns.index, columns = [port.name])
            self.cum_returns = self.cum_returns.merge(ret, how = "inner", left_index = True, right_index = True)

        if withSPY == True:
            start = self.cum_returns.index[0]
            end = self.cum_returns.index[len(self.cum_returns.index)-1]
            spx_returns = pdr.get_data_yahoo(["SPY"], start, end)["Close"]
            spx_returns = spx_returns.pct_change().dropna(axis = 0)
            spx_cum = np.cumprod(1 + spx_returns) - 1
            spx_cum.columns = ["SPY"]
            self.cum_returns = self.cum_returns.merge(spx_cum, how = "inner", left_index = True, right_index = True)

        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        plt.figure(figsize = (10,7))
        sns.set_palette(palette)
        for i in self.cum_returns.columns:
            sns.lineplot(data = self.cum_returns.loc[:,i], label = i)
        plt.xlabel("Date")
        plt.ylabel("Cummulative Returns")
        plt.title(f"Benchmarking: {self.name}")
        if save != "":
            plt.savefig(f'figures/{save}.png')
        plt.show()
        return None