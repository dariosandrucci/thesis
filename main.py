from data_and_denoising import *

tickers = getTickers()
end = dt.now()
start = dt.fromisoformat('2012-10-12')

data = getData(start, end, tickers, downloaded = True)
corr = getCorrMatrix(data)

#method = [1,2]
#for m in method:
#    eVal0, eVec0, eVal1, eVec1, corr1, var0 = denoiseMatrix(corr, method = m)
#    plotEvalDiff(eVal0, eVal1, method = m)

eVal0, eVec0, eVal1, eVec1, corr1, var0 = denoiseMatrix(corr, method = 1)

data = detoneMatrix(corr1, eVal1, eVec1, mc = 1)

print(data)