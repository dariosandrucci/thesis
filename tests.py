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

w = optPort_nco(train_set, corr0, 3, 0.5, 150, 3)
print(w)

