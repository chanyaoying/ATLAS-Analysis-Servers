import pandas as pd
from datetime import datetime, timedelta
from math import *
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Statistics
from statsmodels.tsa.api import Holt, SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose

# Data reader
from pandas_datareader import data
from typing import Dict, List, Any

tickers = ['AAPL']
days_to_predict = 60
end_date = (datetime.now() - relativedelta(days=1)).strftime("%Y-%m-%d") # 1 day before as data won't be updated on yfinance
start_date = (datetime.now() - relativedelta(years=2)).strftime("%Y-%m-%d") # 2 years ago


output = []

for ticker in tickers:

    df = data.DataReader([ticker], 'yahoo', start_date, end_date)

    # train size = 80% of the dataset
    # model_train = df.iloc[:int(df.shape[0]*0.80)]

    # test size = 20%
    # valid = df.iloc[int(df.shape[0]*0.80):]

    # train size = 100%
    model_train = df

    two_years = model_train['Close']
    two_years.columns = ['predictionPrice']
    two_years['ticker'] = ticker
    two_years['date'] = two_years.index
    two_years['date'] = two_years['date'].apply(lambda epoch_time: epoch_time.strftime('%Y-%m-%d'))
    two_years = two_years.to_json(orient="records")

    output += json.loads(two_years)

    # train model
    model_arima = auto_arima(model_train["Close"], trace=False, error_action='ignore',
                                start_p=1, start_q=1, max_p=3, max_q=3,
                                suppress_warnings=True, stepwise=False, seasonal=False)

    model_arima.fit(model_train["Close"], disp=-1)

    model_predictions = pd.DataFrame(model_arima.predict(n_periods=days_to_predict), index=(
        df.index[-1] + timedelta(days=i) for i in range(1, days_to_predict + 1)))
    model_predictions.columns = ['predictionPrice']
    model_predictions['ticker'] = ticker
    model_predictions['date'] = model_predictions.index
    model_predictions['date'] = model_predictions['date'].apply(lambda epoch_time: epoch_time.strftime('%Y-%m-%d'))
    model_predictions = model_predictions.to_json(orient="records")

    output += json.loads(model_predictions)

print(output)
