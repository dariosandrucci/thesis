from data_engineering import *
from algorithm import *
from denoising_and_detoning import *
from portfolio_and_backtesting import *

tickers = getTickers()
end = dt.now()
start = dt.fromisoformat('2012-10-15')

df = getData(start, end, tickers, downloaded = True)
df = df.dropna(axis = 0, how = "any")

train_set, test_set = testTrainSplit(df, validation_set = False, w_tt = [0.9, 0.1])

corr0 = getCorrMatrix(train_set)

eVal0, eVec0, eVal1, eVec1, corr1, var0 = denoiseMatrix(corr0)
corr2 = detoneMatrix(corr1, eVal1, eVec1)

cov0 = corr2cov(corr2)
w_cov_only = optPortMVO(train_set)
print(w_cov_only)

#w_algo = optPort_nco(df, corr2)
#print(w_algo)