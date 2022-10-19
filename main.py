from data_engineering import *
from algorithm import *
from denoising_and_detoning import *
from portfolio_and_backtesting import *

tickers = getTickers()
end = dt.now()
start = dt.fromisoformat('2012-10-15')

df = getData(start, end, tickers, downloaded = True)
df = df.dropna(axis = 0, how = "any")

train_set, val_set, test_set = testTrainSplit(df)

corr0 = getCorrMatrix(train_set)

eVal0, eVec0, eVal1, eVec1, corr1, var0 = denoiseMatrix(corr0)
corr2 = detoneMatrix(corr1, eVal1, eVec1)

w_algo = optPort_nco(corr2)
w_ew = [1/len(df.columns)]*len(df.columns)

sp500Algo = Portfolio("S&P 500 Algo", test_set, w_algo)
sp500EW = Portfolio("S&P 500 EW", test_set, w_ew)

bm = PortfolioBenchmarking([sp500Algo, sp500EW])

bm.plot_performance()

#sp500Algo.performance_plot()