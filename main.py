from data_engineering import *
from algorithm import *
from denoising_and_detoning import *

tickers = getTickers()
end = dt.now()
start = dt.fromisoformat('2012-10-12')

df = getData(start, end, tickers, downloaded = True)

train_set, val_set, test_set = testTrainSplit(df)

print(train_set)
print(val_set)
print(test_set)