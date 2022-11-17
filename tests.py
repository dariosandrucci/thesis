from data_engineering import *
from algorithm import *
from denoising_and_detoning import *
from portfolio_and_backtesting import *

tickers = getTickers()
end = dt.now()
start = dt.fromisoformat('2020-10-15')

df = getData(start, end, tickers, downloaded = True, format = "returns")
df = df.dropna(axis = 0, how = "any")

train_set, test_set = testTrainSplit(df, validation_set = False, w_tt = [0.9, 0.1])
#x = pdr.get_data_yahoo(["AAPL","A"])["Close"]
#w = optPortMVO(train_set, nrIter = 50000)
#print(w)
#print(sum(w))

#corr0 = getCorrMatrix(train_set)

#eVal0, eVec0, eVal1, eVec1, corr1, var0 = denoiseMatrix(corr0)
#corr2 = detoneMatrix(corr1, eVal1, eVec1)

#min_matrix = np.array(pairwise_distances(corr2, metric = "minkowski"))

ret = optPort_nco_RB(df, dt(2022,8,7),50)
port = Portfolio_Ret("Rets",ret)
port.performance_plot()

