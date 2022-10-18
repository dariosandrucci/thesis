import numpy as np
import pandas as pd

class Portfolio:

    def __init__(self, data, tickers, w, start, end):
        self.w = w
        self.tickers = tickers
        self.data = data.loc[start: end]

        #checks and get weights get weights in the right format
        if self.w.shape[0] != len(tickers):
            self.w = self.w.reshape(-1)
        
        if len(w) != len(tickers):
            raise NameError('weights and tickers must have the same length!')

        #getting the portfolio returns
        self.returns = pd.Series(self.data.apply(lambda row: np.average(row, weights = self.w), axis=1), index = self.data.index)
        self.cum_returns = pd.Series(np.cumprod(1 + self.returns) - 1, index = self.data.index)

        #performance metrics
        self.total_return = self.cum_returns.tail(1)
        self.ereturn = self.returns.mean()
        self.ereturn_ann = self.ereturn * 252
        self.volatility = np.std(self.returns)
        self.volatility_ann = self.volatility * np.sqrt(252)
        self.sharpe_ratio = self.ereturn / self.volatility
        self.sharpe_ratio_ann = self.ereturn_ann / self.volatility_ann

    def drawdown_report(self):
        pass
    
    def aplha_report(self):
        pass

    def performance_report(self):
        pr = {}
    
    def performance_plot(self, benchmark = pd.Series()):
        pass
