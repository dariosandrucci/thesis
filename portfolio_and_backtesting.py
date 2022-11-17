import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas_datareader as pdr
import statsmodels.api as sm

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

    def drawdown_report(self):
        pass
    
    def alpha_report(self, return_values = True):
        print("Getting benchmark data....")
        
        mkt_returns = pdr.get_data_yahoo(["SPY"], self.start, self.end)["Close"]
        mkt_returns = mkt_returns.pct_change().dropna(axis = 0)
        X = pd.DataFrame(self.returns[1:], columns = [self.name])
        y = mkt_returns

        if y.index[0] == X.index[0]:
            pass
        else:
            print("Data head has to be trimmed....")
            if y.index[0] > X.index[0]:
                X = X.iloc[1:]
            elif y.index[0] < X.index[0]:
                y = y.iloc[1:]
        
        if y.index[len(y)-1] == X.index[len(X)-1]:
            pass
        else:
            print("Data tail has to be trimmed....")
            if y.index[len(y)-1] > X.index[len(X)-1]:
                y = y.iloc[:len(y)-2]
            if y.index[len(y)-1] < X.index[len(X)-1]:
                X = X.iloc[:len(X)-2]

        print("Data was imported sucessfully!\n")

        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        alpha = est2.params[0]
        beta = est2.params[1]
        p_alpha = est2.pvalues[0]
        p_beta = est2.pvalues[1]
        r2 = est2.rsquared

        print("The model was calculated as follows:\n")
        print(f"Alpha: {alpha}")
        print(f"Beta: {beta}")
        print(f"P-Value Alpha: {p_alpha}")
        print(f"P-Value Beta: {p_beta}")
        print(f"Model R2: {r2}")

        if return_values == True:
            return alpha, beta, p_alpha, p_beta, r2
        else:
            return None

    def performance_report(self, return_values = True):
        pr = {self.name : [self.total_return, self.ereturn, self.volatility, self.ereturn_ann, self.volatility_ann, self.sharpe_ratio_ann]}
        pr = pd.DataFrame(pr, index = ["Total Return", "Expected Return", "Daily Volatility", "Annualized Returns", "Annualized Volatility", "Sharpe Ratio"])
        print(f"The performance metrics for the portfolio {self.name} are:\n")
        print(pr)
        if return_values == True:
            return pr
        else:
            return None

    def performance_plot(self):
        sns.lineplot(self.cum_returns)
        plt.show()
        return None

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

    def drawdown_report(self):
        pass
    
    def alpha_report(self, return_values = True):
        print("Getting benchmark data....")
        
        mkt_returns = pdr.get_data_yahoo(["SPY"], self.start, self.end)["Close"]
        mkt_returns = mkt_returns.pct_change().dropna(axis = 0)
        X = pd.DataFrame(self.returns[1:], columns = [self.name])
        y = mkt_returns

        if y.index[0] == X.index[0]:
            pass
        else:
            print("Data head has to be trimmed....")
            if y.index[0] > X.index[0]:
                X = X.iloc[1:]
            elif y.index[0] < X.index[0]:
                y = y.iloc[1:]
        
        if y.index[len(y)-1] == X.index[len(X)-1]:
            pass
        else:
            print("Data tail has to be trimmed....")
            if y.index[len(y)-1] > X.index[len(X)-1]:
                y = y.iloc[:len(y)-2]
            if y.index[len(y)-1] < X.index[len(X)-1]:
                X = X.iloc[:len(X)-2]

        print("Data was imported sucessfully!\n")

        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        alpha = est2.params[0]
        beta = est2.params[1]
        p_alpha = est2.pvalues[0]
        p_beta = est2.pvalues[1]
        r2 = est2.rsquared

        print("The model was calculated as follows:\n")
        print(f"Alpha: {alpha}")
        print(f"Beta: {beta}")
        print(f"P-Value Alpha: {p_alpha}")
        print(f"P-Value Beta: {p_beta}")
        print(f"Model R2: {r2}")

        if return_values == True:
            return alpha, beta, p_alpha, p_beta, r2
        else:
            return None

    def performance_report(self, return_values = True):
        pr = {self.name : [self.total_return, self.ereturn, self.volatility, self.ereturn_ann, self.volatility_ann, self.sharpe_ratio_ann]}
        pr = pd.DataFrame(pr, index = ["Total Return", "Expected Return", "Daily Volatility", "Annualized Returns", "Annualized Volatility", "Sharpe Ratio"])
        print(f"The performance metrics for the portfolio {self.name} are:\n")
        print(pr)
        if return_values == True:
            return pr
        else:
            return None

    def performance_plot(self):
        sns.lineplot(self.cum_returns)
        plt.show()
        return None

class PortfolioBenchmarking:

    def __init__(self, portfolios):
        
        self.portfolios = portfolios
    
        #input checks
        fi = portfolios[0].returns.index[0]
        li = portfolios[0].returns.index[len(portfolios[0].returns.index)-1]
        for port in self.portfolios:
            if fi == port.returns.index[0] and li == port.returns.index[len(port.returns.index)-1]:
                pass
            else:
                raise NameError('Portfolios must have the same time range!')

    def plot_performance(self, withSPY = True):

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
            spx_cum.columns = ["S&P 500 Index"]
            self.cum_returns = self.cum_returns.merge(spx_cum, how = "inner", left_index = True, right_index = True)

        sns.lineplot(self.cum_returns)
        plt.show()

        return None

    def report_returns(self):
        pass

    def report_alpha(self):
        pass

#to adapt
def max_drawdown(portfolio_data_frame, weights, time_period):
    simulated_portfolio = weights[0]*portfolio_data_frame.ix[:,0]
    for i in range(1, len(portfolio_data_frame.columns)):
        simulated_portfolio += weights[i]*portfolio_data_frame.ix[:,i]
    max_drawdown_value = float('-inf')
    for i in range(int(len(simulated_portfolio)/time_period)-1):
        biggest_variation = max(simulated_portfolio[i*time_period:(i+1)*time_period])/min(simulated_portfolio[i*time_period:(i+1)*time_period])
        if(biggest_variation > max_drawdown_value):
            max_drawdown_value = biggest_variation
    return max_drawdown_value